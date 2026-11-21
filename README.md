<!-- TODO: add docs/assets/logo.svg and docs/assets/logo-dark.svg next to the title (as a
     <picture> element), and an overview figure docs/assets/figure1.svg under "How it works". -->
<h1 align="center">
  DeepLitSearch: A Verifiable Benchmark for<br>Researcher-Facing Scientific Literature Search
</h1>
<!-- Link the title to the arXiv page once it is public. -->

<p align="center">
  Tiansheng Hu<sup>N</sup>,
  Yilun Zhao<sup>Y</sup>,
  Varsha Kishore<sup>W,A</sup>,
  Rohan Phanse<sup>Y</sup>,
  Chen Zhao<sup>N,C</sup>,
  Doug Downey<sup>A</sup>,
  Arman Cohan<sup>Y,A</sup>
</p>
<p align="center"><sub>
  <sup>N</sup>NYU Shanghai &nbsp;
  <sup>Y</sup>Yale University &nbsp;
  <sup>W</sup>University of Washington &nbsp;
  <sup>A</sup>Allen Institute for AI &nbsp;
  <sup>C</sup>Center for Data Science, New York University
</sub></p>

<p align="center">
  <a href="https://github.com/HughieHu/DeepLitSearch/actions/workflows/ci.yml"><img src="https://github.com/HughieHu/DeepLitSearch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://huggingface.co/datasets/HughieHu/deeplitsearch-bench"><img src="https://img.shields.io/badge/data-deeplitsearch--bench-ffcc4d" alt="Benchmark data"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License">
</p>

<p align="center">
  <a href="#what-it-is">What it is</a> &middot;
  <a href="#how-it-works">How it works</a> &middot;
  <a href="#evaluate-your-model">Evaluate</a> &middot;
  <a href="#data">Data</a> &middot;
  <a href="#citation">Citation</a> &middot;
  <a href="#license">License</a>
</p>

---

```bash
git clone https://github.com/HughieHu/DeepLitSearch && cd DeepLitSearch/eval
pip install -e '.[client]'

export MODEL_NAME=gpt-5.4-mini MODEL_BASE_URL=https://api.openai.com/v1 MODEL_API_KEY=...
export SERPER_API_KEY=...      # Google search for the web harness (serper.dev); JINA_API_KEY is optional
export JUDGE_API_KEY=...       # for the gpt-5.4-mini judge

python -m dlsbench.doctor --harness web             # checks the model, tool calling, search and the judge
python -m dlsbench.eval --task all --harness web    # runs and grades all five tasks
```

## What it is

Researchers search the literature to answer specific questions. Which benchmark do papers in this
direction evaluate on? Which method is ahead? Why do two papers disagree? Answering one usually
means finding several papers and reading them closely, not just retrieving one relevant abstract.
DeepLitSearch poses such questions to search agents. Every question has its gold papers and a
reference answer that can be checked. Each of the five tasks has 300 questions:

| task | the agent must |
|---|---|
| T1 Benchmark Matching | name the benchmark (and its split or version) that papers in a described research direction evaluate on |
| T2 Best-Method Mining | identify the best method on a benchmark from comparisons spread across papers |
| T3 Conflict Solving | find two papers that rank the same pair of methods in opposite order, and explain the evaluation or setup difference behind the reversal |
| T4 Implementation Tracing | find the precedent papers for an implementation question and the evidence in their public repositories (a file, a configuration entry or a dependency) that answers it |
| T5 Cluster Mining | find a source paper, a paper it cites, and the most important references the two share |

This repository contains the evaluation suite: the two retrieval harnesses used in the paper, the
judge, and the metrics. One command runs any OpenAI-compatible model on the benchmark and grades it.

## How it works

The agent runs a ReAct loop with native function calling. It gets two tools, a search and a
reader, and at most 10 searches, 10 reads and 25 turns per question. It finishes with a list of the
papers it relied on, in `<selected_papers>`, and its answer, in `<answer>`. The harnesses differ
only in where the tools look:

| `--harness` | search tool | read tool |
|---|---|---|
| `web` (Web-ReAct) | `google_search`: top 10 Google results through Serper | `visit_page`: any URL as text through Jina Reader, first 8,000 characters |
| `s2` (S2-ReAct) | `s2_search`: Semantic Scholar paper search | `s2_paper`: a paper's Semantic Scholar record plus its open-access full text through Jina, first 12,000 characters |

The paper also evaluates native web search, where the provider runs the search inside its own API.
That setting is specific to each provider, so it is not part of this suite.

Answers are scored three ways:

| metric | tasks | how |
|---|---|---|
| Recall | T1-T4 | share of gold papers found in `<selected_papers>`, by fuzzy title match |
| Acc | T1-T4 | binary judgment by `gpt-5.4-mini` against the reference answer, with the judge prompts of the paper's Figures 8-11 |
| NDCG@10 | T5 | `<selected_papers>` read as a ranking; the source and cited papers have relevance 2, shared references 1 |

## Evaluate your model

Any endpoint that speaks the OpenAI chat-completions API with tool calling works, hosted or local.
For any other API, subclass `Client` and override `complete()` in
[`eval/dlsbench/model.py`](eval/dlsbench/model.py), which is the only code that talks to the model.

[`eval/README.md`](eval/README.md) covers setup, the search backends, grading and the output files.

## Data

The benchmark is on HuggingFace as
[`HughieHu/deeplitsearch-bench`](https://huggingface.co/datasets/HughieHu/deeplitsearch-bench), and a
copy of the questions ships in [`eval/bench/`](eval/bench/) so that the suite runs without a
download. Each question has an `instance_id`, the `query`, and a `ground_truth` with the gold papers
(`retrieve`) and the reference answer (`answer`). The format is in
[`eval/README.md`](eval/README.md#7-layout-and-data-format).

## Citation

```bibtex
@misc{hu2026deeplitsearch,
  title  = {DeepLitSearch: A Verifiable Benchmark for Researcher-Facing Scientific Literature Search},
  author = {Hu, Tiansheng and Zhao, Yilun and Kishore, Varsha and Phanse, Rohan and Zhao, Chen and Downey, Doug and Cohan, Arman},
  year   = {2026}
}
```

## License

The code is released under [Apache-2.0](LICENSE). For the data, see the dataset card on HuggingFace.
