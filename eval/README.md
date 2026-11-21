# Evaluating on DeepLitSearch

DeepLitSearch asks the questions a researcher would ask about the literature. Answering them means
finding several papers and reading them. It has five tasks. Every question comes with its gold
papers and a reference answer that can be checked.

The `dlsbench` package runs an agent on the questions and grades it. You supply three things: a
model endpoint (three environment variables), a key for the search backend of the harness you run,
and a key for the judge.

```bash
pip install -e '.[client]'
cp .env.example .env                       # model endpoint, search keys, judge key
python -m dlsbench.doctor --harness web    # checks the model, tool calling, search and the judge
python -m dlsbench.eval --task all --harness web
```

## 1. Install

```bash
pip install -e '.[client]'    # from this directory
```

Install in editable mode, because the runner reads the questions from `bench/` in this directory.
The client uses only the Python standard library (3.9 or newer). The `client` extra adds
`certifi`, which avoids TLS errors on machines with an outdated CA bundle (common on macOS).

## 2. Point it at your model

Set three environment variables, or put them in `.env`, which is loaded automatically. Any endpoint
that speaks the OpenAI chat-completions API **with native tool calling** works:

```bash
MODEL_NAME=gpt-5.4-mini
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=...
```

A local server only needs a different URL. vLLM must be started with tool calling enabled
(`--enable-auto-tool-choice` and the `--tool-call-parser` for your model):

```bash
MODEL_NAME=Qwen/Qwen3-32B
MODEL_BASE_URL=http://localhost:8000/v1
MODEL_API_KEY=EMPTY
```

`MODEL_NAME` also names the output files, so runs of different models do not collide.

If your model does not speak this API, subclass `Client` and override `complete()` in
[dlsbench/model.py](dlsbench/model.py), the only code that talks to the model:

```python
from dlsbench import eval, model

class MyClient(model.Client):
    def __init__(self):
        super().__init__(model="my-model", base_url="unused")   # the base class requires both

    def complete(self, messages, tools=None):
        # messages: OpenAI chat messages (user / assistant with tool_calls / tool)
        # tools:    [{"name", "description", "parameters"}] or None
        text, calls, usage = my_model(messages, tools)
        return {"text": text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.json_args} for c in calls],
                "usage": {"prompt_tokens": usage.input, "completion_tokens": usage.output}}

model._client = MyClient()
eval.main(["--task", "all", "--harness", "web"])
```

Optional request fields are sent only when you set them: `MODEL_REASONING_EFFORT`,
`MODEL_MAX_TOKENS` and `MODEL_TEMPERATURE`. If an endpoint rejects `reasoning_effort` or wants
`max_completion_tokens` instead of `max_tokens`, the client adapts after the first error.

## 3. Search backends

| `--harness` | tools | needs |
|---|---|---|
| `web` (Web-ReAct) | `google_search(query)`: top 10 Google results (title, URL, snippet) through [Serper](https://serper.dev). `visit_page(url)`: the page as text through [Jina Reader](https://jina.ai/reader), first 8,000 characters | `SERPER_API_KEY`; `JINA_API_KEY` optional |
| `s2` (S2-ReAct) | `s2_search(query, k=10)`: Semantic Scholar paper search (paperId, title, abstract, TLDR, year, venue, authors, external ids, citation counts). `s2_paper(paper_id)`: one paper's record, previews of up to 15 references and 15 citations, and its open-access full text through Jina (first 12,000 characters) | nothing required; `S2_API_KEY` and `JINA_API_KEY` optional |

Without a key, Jina and Semantic Scholar still answer, at lower rate limits. The unauthenticated
Semantic Scholar pool is shared and often returns HTTP 429; the client waits and retries, but a full
run is slow without a key. `S2_RPS` (default 1) caps the request rate across all threads. Raise it
only as far as your key allows.

A rejected key stops the run instead of feeding the agent error messages for every search. A run
in which every search fails would otherwise look like a weak model. `python -m dlsbench.doctor
--harness web|s2` checks the keys before a full run.

Native web search, where the provider runs the search inside its own API, is not included. It is
specific to each provider (for example, OpenAI's Responses API with its built-in web search tool),
so it cannot be offered through one OpenAI-compatible interface.

## 4. Run

```bash
python -m dlsbench.eval --task all --harness web --limit 3   # 3 questions per task first; grading costs money
python -m dlsbench.eval --task all --harness web             # everything; rerun the same command to resume
python -m dlsbench.eval --task 1,3 --harness s2              # some tasks only (paper numbering T1-T5)
python -m dlsbench.eval --task all --harness web --no-grade  # Recall and NDCG only, no judge calls
python -m dlsbench.eval --task all --harness web --grade-only   # grade existing records
```

`--concurrency K` sets how many questions run in parallel (default 4, or `CONCURRENCY`).
`--bench-dir DIR` reads `task1.json` ... `task5.json` from another directory, for example a newer
data release. `--out-dir DIR` moves `runs/` and `results/`.

Every question is written to its own file as soon as it finishes, so an interrupted run loses at
most the questions in flight. A question whose run failed is retried on the next run: the model
endpoint was unreachable, or a search backend rejected the key. So is an answer the judge returned
no valid verdict for.

## 5. What gets reported

| metric | tasks | meaning |
|---|---|---|
| Recall | T1-T4 | share of the gold papers that appear in the agent's `<selected_papers>` list (fuzzy title match, first 50 lines) |
| Acc | T1-T4 | share of answers the judge scores 1 |
| NDCG@10 | T5 | the `<selected_papers>` list read as a ranking: seed papers (the source and the cited paper) have relevance 2, shared references 1 |

All are reported as percentages. Each task also reports `n` (questions with a finished
record), `n_format_ok` (final response had both tags), `n_judged`, `n_judge_failed` (no valid
verdict yet; left out of Acc) and the average searches, visits and tokens per question.

Output files:

```
runs/<model>__<harness>/task<N>/<instance_id>.json   one question: the prompt, every model turn and
                                                     tool result, the parsed answer, token counts,
                                                     and its scores under "eval"
results/<model>__<harness>.json                      the per-task summary
```

The summary keeps the entries of tasks you did not run this time, so `--task 1` followed by
`--task 2` gives one file with both.

## 6. Grading

**Recall.** A gold title matches a line of `<selected_papers>` if, after lowercasing and removing
URLs, arXiv ids, DOIs and punctuation, one contains the other, or at least 70% of the gold title's
words appear in the line. Recall is the share of gold papers matched by any of the first 50 lines,
averaged over questions.

**NDCG@10.** The same title match, over the first 10 lines. A line earns the relevance of the
highest-relevance gold paper it matches that has not been matched yet, so a paper listed twice earns
once. The ideal ranking lists the two seed papers, then the shared references.

**Acc.** A binary LLM judge, `gpt-5.4-mini`, the judge used in the paper, called through the OpenAI
Responses API. It returns strict JSON `{"score": 0 or 1, "reason": ...}`. The prompts live in [dlsbench/judge.py](dlsbench/judge.py):

| task | judged against |
|---|---|
| T1 | the gold benchmark, its split or version, and up to five gold metrics |
| T2 | the gold best method, its paradigm and up to five supporting comparisons; a 1 also needs at least one supporting cross-paper comparison |
| T3 | both methods, which paper reports which ordering on which metric, and the reason for the reversal; a 1 needs a concrete cause, not "different setups" |
| T4 | the reference answer (`expected_answer`); the library, configuration or method must match |

Every request is sent the same way: the prompt is the plain-string `input`, with reasoning effort
`high` and no output cap.

The judge sees the agent's `<answer>` block. If none was parsed, it sees the whole final response
instead, so every question is judged; an empty final response scores 0.

You set only where the judge is reached: `JUDGE_API_KEY` (or `OPENAI_API_KEY`) and `JUDGE_BASE_URL`.
For an endpoint that only speaks chat-completions, set `JUDGE_API=chat`. `JUDGE_MODEL` can be
changed, but Acc is then not comparable to the paper's.

## 7. Layout and data format

```
dlsbench/
  model.py      the model adapter, the file you change
  harness.py    the agent loop, prompts and tool schemas
  tools.py      Serper, Jina and Semantic Scholar clients
  judge.py      judge prompts and the judge call
  metrics.py    recall, NDCG@10 and aggregation
  eval.py       the runner
  doctor.py     preflight checks
bench/
  task1.json ... task5.json   the questions (T1-T5)
.env.example    the settings, documented
```

Each task file is a JSON list of questions with one schema:

```
instance_id    opaque identifier
task           task name, e.g. T1_benchmark_matching
query          the question, as given to the agent
ground_truth
  retrieve     the gold papers: titles (T1-T4), or {title, role, weight} for T5, where role is
               "seed" (weight 2) or "shared_reference" (weight 1)
  answer       T1 {benchmark, split_or_version, metrics}
               T2 {best_method, paradigm, verified_findings, reasoning}
               T3 {method_a, method_b, a_beats_b, b_beats_a, reasoning}
               T4 {expected_answer, target_type, ...}
               T5 null (scored by NDCG@10 only)
```

The loader assumes only this schema, not a question count, so a newer release can replace the files
in `bench/` (or be passed with `--bench-dir`).
