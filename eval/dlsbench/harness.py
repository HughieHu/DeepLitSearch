"""The agent loop: Web-ReAct and S2-ReAct.

Protocol (the paper's): one user message carries the instructions, the per-task answer
instruction and the question. The model then works with provider-native function calling. Each
turn it either calls tools or writes its final response, which must contain a
<selected_papers> block (one paper per line) and an <answer> block.

Budgets per question: 10 searches, 10 visits (page or paper fetches) and 25 model turns. A tool
whose budget is spent is removed from the tool list. If the model stops calling tools without
both tags, it is asked once more for the final format (tools stay available while budget
remains). If the loop ends without a well-formed response, one last call with no tools forces it.

The prompts below are the ones used for the paper; a test pins them. Editing them makes your
numbers incomparable to the reported ones -- if you study prompt variants, say so when you report.
"""
import json
import re
import time

from . import model as gem
from .tools import ToolConfigError

MAX_SEARCHES = 10
MAX_VISITS = 10
MAX_TURNS = 25

# ---------------------------------------------------------------- prompts: Web-ReAct
WEB_SYSTEM_INSTRUCTIONS = """You are a research literature search agent. You have two tools:

  google_search(query): Search Google. Returns top-10 results (title + url + snippet). Use up to 10 searches.
  visit_page(url):      Read URL contents (paper abstracts, blogs, etc). Use up to 10 visits.

WORKFLOW:
  1. Use the tools to investigate the question.
  2. When you have enough information, OR when you have used all your budget, STOP using tools.
  3. Output your final response in EXACTLY this format (no extra text outside the tags):

<selected_papers>
- Title 1 (URL or arXiv ID)
- Title 2 (URL or arXiv ID)
- ...
</selected_papers>

<answer>
{answer_instruction}
</answer>

The <selected_papers> list MUST contain the papers you actually used to ground your answer.
Do not list papers you only saw in snippets without using them. Each line must be one paper.
"""

WEB_FORCE_FINAL_PROMPT = """Stop using tools. Output your final response NOW in EXACTLY this format (no other text):

<selected_papers>
- Title 1 (URL or arXiv ID)
- Title 2 (URL or arXiv ID)
- ...
</selected_papers>

<answer>
{answer_instruction}
</answer>"""

WEB_TOOLS = [
    {
        "name": "google_search",
        "description": "Search Google. Returns top-10 results with title, url, snippet. Up to 10 calls allowed.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Concise keyword query"}},
            "required": ["query"],
        },
    },
    {
        "name": "visit_page",
        "description": "Read URL contents. Up to 10 calls allowed.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to fetch"}},
            "required": ["url"],
        },
    },
]

# ---------------------------------------------------------------- prompts: S2-ReAct
S2_SYSTEM_INSTRUCTIONS = """You are a research literature search agent. You have two tools, both backed by the SEMANTIC SCHOLAR Academic Graph (~220M papers across all sciences, with abstracts, TLDR summaries, citation graph, and venue/year metadata):

  s2_search(query, k=10): Relevance-ranked paper search. Returns the top-k matched papers, each with paperId, title, abstract (truncated to ~1200 chars), TLDR (1-line summary), year, venue, author list, externalIds (arXiv/DOI/CorpusId), and citation counts. Up to 10 calls allowed.
  s2_paper(paper_id): Fetch full record for one paper. paper_id can be the S2 paperId from s2_search, OR a prefix-tagged ID like "DOI:10.1234/...", "ARXIV:2401.12345", or "CorpusId:266359151". Returns abstract, TLDR, fields-of-study, FULL PAPER BODY when an open-access PDF or arXiv copy is available (~12K chars of cleaned markdown via Jina extractor), plus a preview list of up to 15 references and 15 citations (titles + years). Up to 10 calls allowed.

NOTES:
  - All retrieval is grounded in Semantic Scholar. If a paper isn't in S2, you can't reach it.
  - Use s2_search to discover papers; use s2_paper to inspect one paper's full body / abstract / references / citations in depth.
  - The `body_text` field in s2_paper output is the paper's full text when available (open-access PDFs, arXiv preprints). When `body_text` is empty the paper is paywalled — fall back on the abstract.
  - paper_id values must come from s2_search results or be a tagged external ID — do not invent paperId hashes.

WORKFLOW:
  1. Use the tools to investigate the question.
  2. When you have enough information, OR when you have used all your budget, STOP using tools.
  3. Output your final response in EXACTLY this format (no extra text outside the tags):

<selected_papers>
- Title 1 (paperId or arXiv ID)
- Title 2 (paperId or arXiv ID)
- ...
</selected_papers>

<answer>
{answer_instruction}
</answer>

The <selected_papers> list MUST contain the papers you actually used to ground your answer.
Do not list papers you only saw in snippets without using them. Each line must be one paper, with its identifier (paperId, arXiv id, or DOI) in parentheses so it can be matched back.
"""

S2_FORCE_FINAL_PROMPT = """Stop using tools. Output your final response NOW in EXACTLY this format (no other text):

<selected_papers>
- Title 1 (paperId or arXiv ID)
- Title 2 (paperId or arXiv ID)
- ...
</selected_papers>

<answer>
{answer_instruction}
</answer>"""

S2_TOOLS = [
    {
        "name": "s2_search",
        "description": (
            "Search Semantic Scholar (~220M papers). Returns top-k papers with title, "
            "abstract, TLDR, year, venue, authors, externalIds, citation counts. "
            "Up to 10 calls allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Concise topic / method / dataset descriptor"},
                "k": {"type": "integer", "description": "Number of papers to return (default 10, max 100)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "s2_paper",
        "description": (
            "Fetch full record for one paper from Semantic Scholar. paper_id may be the "
            "S2 paperId (40-char hex), or a tagged external ID (DOI:..., ARXIV:..., "
            "CorpusId:...). Returns abstract, TLDR, fields-of-study, and previews of up "
            "to 15 references and 15 citations. Up to 10 calls allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string", "description": "S2 paperId or DOI:/ARXIV:/CorpusId: tagged ID"},
            },
            "required": ["paper_id"],
        },
    },
]

# ---------------------------------------------------------------- per-task answer instructions
# Keyed by the paper's task numbers T1-T5 (paper Table 14).
ANSWER_INSTRUCTIONS = {
    1: ("Identify the benchmark (or evaluation collection) the researcher should use. "
        "Give its NAME and the relevant SPLIT or VERSION if applicable. Briefly justify based "
        "on which papers use it for the same research direction."),
    2: ("Identify the BEST METHOD for this benchmark, based on cross-paper evidence. "
        "Briefly justify (which findings put it on top, what metrics)."),
    3: ("Explain the cross-paper conflict: which paper reports A>B, which reports B>A, "
        "and what evaluation/setup difference causes the reversal."),
    4: ("Identify the precedent paper(s) and point to the specific implementation evidence in "
        "their public repositories (file, configuration entry, or dependency) that answers the "
        "implementation question."),
    5: ("Identify the source paper, the cited paper, and the 3 most important shared references. "
        "Briefly explain the relationship between source and cited paper, and what the shared "
        "citations reveal about that relationship."),
}

HARNESSES = {
    "web": {"system": WEB_SYSTEM_INSTRUCTIONS, "force_final": WEB_FORCE_FINAL_PROMPT,
            "tools": WEB_TOOLS, "budget_keys": ("search", "visit")},
    "s2": {"system": S2_SYSTEM_INSTRUCTIONS, "force_final": S2_FORCE_FINAL_PROMPT,
           "tools": S2_TOOLS, "budget_keys": ("s2_search", "s2_paper")},
}


def build_question_prompt(harness, task, question):
    sys_block = HARNESSES[harness]["system"].format(answer_instruction=ANSWER_INSTRUCTIONS[task])
    return "%s\n\n=== QUESTION ===\n%s\n" % (sys_block, question)


def build_force_final(harness, task):
    return HARNESSES[harness]["force_final"].format(answer_instruction=ANSWER_INSTRUCTIONS[task])


# ---------------------------------------------------------------- final-answer parsing
_RE_SELECTED = re.compile(r"<selected_papers>([\s\S]*?)</selected_papers>", re.IGNORECASE)
_RE_ANSWER = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)


def parse_final(text):
    """(selected_papers, answer) from the LAST <selected_papers> and <answer> blocks.

    The last block, because reasoning models sometimes mention the tag names while thinking before
    they write the real final response.
    """
    sel = _RE_SELECTED.findall(text or "")
    ans = _RE_ANSWER.findall(text or "")
    sel_block = sel[-1].strip() if sel else ""
    ans_block = ans[-1].strip() if ans else ""
    selected = []
    for line in sel_block.splitlines():
        line = re.sub(r"^[-*\d.]+\s*", "", line.strip()).strip()
        if line:
            selected.append(line)
    return selected, ans_block


def _has_format(text):
    return "<answer>" in text and "<selected_papers>" in text


# ---------------------------------------------------------------- the loop
def run_episode(task, question, harness, tools, call_model=None, verbose=False):
    """Run one question. `tools` is a tools.WebTools / tools.S2Tools (or anything with the same
    `search_name`, `visit_name`, `search(args)`, `visit(args)`).

    Returns the per-question record: the parsed answer, every model turn and tool result, budget
    use and token counts. `failed` is set when the model endpoint or a tool backend broke; such a
    question is retried on the next run rather than scored.
    """
    call_model = call_model or gem.mcall
    spec = HARNESSES[harness]
    search_tool, visit_tool = spec["tools"]
    s_key, v_key = spec["budget_keys"]
    prompt = build_question_prompt(harness, task, question)

    messages = [{"role": "user", "content": prompt}]
    trace = []
    searches = visits = 0
    final_text = ""
    failed = None
    tok = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    t0 = time.time()

    def add_usage(u):
        u = u or {}
        tok["prompt_tokens"] += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        tok["output_tokens"] += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        tok["cached_tokens"] += int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        tok["reasoning_tokens"] += int(
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)

    def force_final_msg():
        return {"role": "user", "content": build_force_final(harness, task)}

    try:
        for turn in range(MAX_TURNS):
            avail = []
            if searches < MAX_SEARCHES:
                avail.append(search_tool)
            if visits < MAX_VISITS:
                avail.append(visit_tool)
            try:
                resp = call_model(messages, avail or None)
            except gem.ContextOverflow as e:
                trace.append({"turn": turn, "type": "error", "error": "context_length_exceeded: %s"
                              % str(e)[:200]})
                break
            add_usage(resp.get("usage"))
            text = resp.get("text") or ""
            tool_calls = resp.get("tool_calls") or []
            trace.append({"turn": turn, "type": "model_output", "text": text,
                          "tool_calls": tool_calls,
                          "tools_available": [t["name"] for t in avail],
                          "budget_remaining": {s_key: MAX_SEARCHES - searches,
                                               v_key: MAX_VISITS - visits}})
            if verbose:
                print("  turn %d | %s" % (turn, ", ".join(tc["name"] for tc in tool_calls)
                                          or (text[:100].replace("\n", " "))), flush=True)

            if tool_calls:
                # As in the paper's runs, only the calls go back into the history, not any text
                # the model wrote alongside them.
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}
                    for tc in tool_calls]})
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except ValueError:
                        args = {}
                    name = tc["name"]
                    if name == tools.search_name:
                        if searches >= MAX_SEARCHES:
                            result = "[ERROR] %s budget exhausted" % (
                                "search" if harness == "web" else name)
                        else:
                            result = tools.search(args)
                            searches += 1
                    elif name == tools.visit_name:
                        if visits >= MAX_VISITS:
                            result = "[ERROR] %s budget exhausted" % (
                                "visit" if harness == "web" else name)
                        else:
                            result = tools.visit(args)
                            visits += 1
                    else:
                        result = "[ERROR] unknown tool %s" % name
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    trace.append({"turn": turn, "type": "tool_result", "tool": name,
                                  "input": args, "output": result})
                continue

            final_text = text
            if _has_format(final_text):
                break
            # No tool call and no final format: ask for the format once more. The text itself
            # is not kept in the history, as in the paper's runs.
            messages.append(force_final_msg())
            trace.append({"turn": turn, "type": "force_final",
                          "reason": "missing_format" if final_text else "empty_response"})

        if not _has_format(final_text):
            messages.append(force_final_msg())
            try:
                resp = call_model(messages, None)
                add_usage(resp.get("usage"))
                final_text = resp.get("text") or final_text
            except gem.ContextOverflow as e:
                trace.append({"turn": MAX_TURNS, "type": "error",
                              "error": "context_length_exceeded: %s" % str(e)[:200]})
            trace.append({"turn": MAX_TURNS, "type": "post_loop_force_final", "text": final_text})
    except ToolConfigError as e:
        failed = "ToolConfigError: %s" % e
    except gem.EndpointError as e:
        failed = "EndpointError: %s" % e
    except Exception as e:
        failed = "%s: %s" % (type(e).__name__, str(e)[:300])

    selected, answer = parse_final(final_text)
    return {
        "prompt": prompt,
        "final_text": final_text,
        "selected_papers": selected,
        "answer": answer,
        "failed": failed,
        "stats": {
            "searches_used": searches,
            "visits_used": visits,
            "turns": len({t["turn"] for t in trace if t.get("type") == "model_output"}),
            "elapsed_sec": round(time.time() - t0, 1),
            "format_ok": bool(answer and selected),
            **tok,
        },
        "trace": trace,
    }
