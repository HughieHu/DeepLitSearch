"""Answer grading (T1-T4): a binary LLM judge.

The judge model defaults to `gpt-5.4-mini`, the paper's judge, called through the OpenAI
**Responses** API. Each answer gets one judgment: strict JSON {"score": 0 or 1, "reason": "..."}.
A call that never returns a valid verdict scores -1, which is left out of Acc and counted
separately; it is retried on the next run.

The prompts are the ones printed in the paper, Figures 8-11 (a test pins them):

    T1  TASK1_PROMPT  Figure 8   Benchmark Matching
    T2  TASK2_PROMPT  Figure 9   Best-Method Mining
    T3  TASK3_PROMPT  Figure 10  Conflict Solving
    T4  TASK4_PROMPT  Figure 11  Implementation Tracing

Every prompt is sent the same way: as the request's plain-string `input`, with reasoning effort
"high" and no output cap. T1 shows at most 5 gold metrics and T2 at most 5 verified findings.
T5 has no judge; it is scored by NDCG@10.

Settings: JUDGE_API_KEY (or OPENAI_API_KEY), JUDGE_BASE_URL (default https://api.openai.com/v1),
JUDGE_MODEL (default gpt-5.4-mini), and JUDGE_API=chat for an endpoint that only speaks
chat-completions. Changing the judge model makes Acc incomparable to the paper's.
"""
import json
import re
import threading
import time

from . import TransportError, envopt, http_request

JUDGE_MODEL = envopt("JUDGE_MODEL") or "gpt-5.4-mini"
JUDGE_BASE_URL = (envopt("JUDGE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
JUDGE_API_KEY = envopt("JUDGE_API_KEY") or envopt("OPENAI_API_KEY") or ""
JUDGE_API = (envopt("JUDGE_API") or "responses").lower()
JUDGE_TIMEOUT = 120
JUDGE_ATTEMPTS = 8

# How every judge request is sent.
#   effort             reasoning effort (None: the field is not sent)
#   max_output_tokens  output cap (None: not sent)
#   as_message         True: `input` is [{"role": "user", "content": prompt}]; False: the bare string
JUDGE_SETTINGS = {"effort": "high", "max_output_tokens": None, "as_message": False}

# ---------------------------------------------------------------- the paper's prompts
TASK1_PROMPT = """The user's research scenario was wrapped in a question. The CORRECT answer is to identify a single benchmark (with split/version when relevant). Compare the model's answer against the ground truth.

QUESTION:
{question}

GROUND TRUTH:
benchmark: {gt_benchmark}
split_or_version: {gt_split}
metrics (any subset would count): {gt_metrics}

MODEL'S ANSWER:
{answer}

Score the model's answer:
1 = answer correctly identifies the ground-truth target (right benchmark, key facts match — minor wording differences are OK).
0 = answer is wrong, names a different benchmark, or fails to identify the target.

Be strict — only score 1 if the model clearly arrived at the same answer as the ground truth.

Output strict JSON only:
{{"score": 0 or 1, "reason": "<one sentence>"}}"""


TASK2_PROMPT = """The CORRECT answer identifies the BEST METHOD on a benchmark, based on cross-paper evidence. Compare the model's answer against the ground truth.

QUESTION:
{question}

GROUND TRUTH:
best_method: {gt_best_method}
paradigm: {gt_paradigm}
verified_findings: {gt_findings}

MODEL'S ANSWER:
{answer}

Score:
1 = answer identifies the same best_method as GT, with at least one supporting cross-paper comparison (winner > loser on metric).
0 = answer names a different method or fails to identify a clear winner.

Output strict JSON only:
{{"score": 0 or 1, "reason": "<one sentence>"}}"""


TASK3_PROMPT = """The CORRECT answer explains a cross-paper conflict: which paper says A > B, which says B > A, and what evaluation/setup difference causes the reversal.

QUESTION:
{question}

GROUND TRUTH:
method_a: {gt_method_a}
method_b: {gt_method_b}
a_beats_b in paper {paper_a}: {metric_a} = {a_val_a} vs {b_val_a}
b_beats_a in paper {paper_b}: {metric_b} = {a_val_b} vs {b_val_b}
reasoning: {gt_reasoning}

MODEL'S ANSWER:
{answer}

Score:
1 = answer correctly names both methods, identifies which paper claims A > B and which claims B > A, AND explains the KEY CAUSE of the reversal (different metric, dataset split, normalization, sample size, hyper-parameter, compute budget, etc.).
0 = misses either method, mislabels the direction, or fails to identify the cause (vague "different setups" without specifying).

This task is strict — no partial credit; the explanation must point to a concrete methodological difference.

Output strict JSON only:
{{"score": 0 or 1, "reason": "<one sentence>"}}"""


TASK4_PROMPT = """You are judging a research-paper agent's answer.

QUESTION:
{question}

MODEL ANSWER:
{answer}

EXPECTED ANSWER:
{expected}

Does the model's answer convey the same core conclusion as the expected answer? Be strict — minor wording differences are OK, but the key claim (which library / config / method was used) must match.

Output strict JSON only:
{{"score": 0 or 1, "reason": "<one sentence>"}}"""


def build_judge_prompt(task, question, gt, answer):
    """The judge prompt for paper task 1, 2, 3 or 4. `gt` is the instance's `ground_truth`."""
    ans = (gt or {}).get("answer") or {}
    q = question or ""
    a = answer or ""
    if task == 1:
        return TASK1_PROMPT.format(
            question=q,
            gt_benchmark=ans.get("benchmark", ""),
            gt_split=ans.get("split_or_version", ""),
            gt_metrics=", ".join((ans.get("metrics") or [])[:5]),
            answer=a,
        )
    if task == 2:
        findings = ans.get("verified_findings") or []
        find_str = "; ".join(
            "%s > %s (%s: %s vs %s)" % (f.get("method"), f.get("beats"), f.get("metric"),
                                        f.get("winner_score"), f.get("loser_score"))
            for f in findings[:5]
        )
        return TASK2_PROMPT.format(
            question=q,
            gt_best_method=ans.get("best_method", ""),
            gt_paradigm=ans.get("paradigm", ""),
            gt_findings=find_str,
            answer=a,
        )
    if task == 3:
        ab = ans.get("a_beats_b") or {}
        ba = ans.get("b_beats_a") or {}
        return TASK3_PROMPT.format(
            question=q,
            gt_method_a=ans.get("method_a", ""),
            gt_method_b=ans.get("method_b", ""),
            paper_a=ab.get("paper_id", ""),
            metric_a=ab.get("metric", ""),
            a_val_a=ab.get("a_value", "?"),
            b_val_a=ab.get("b_value", "?"),
            paper_b=ba.get("paper_id", ""),
            metric_b=ba.get("metric", ""),
            a_val_b=ba.get("a_value", "?"),
            b_val_b=ba.get("b_value", "?"),
            gt_reasoning=ans.get("reasoning") or "",
            answer=a,
        )
    if task == 4:
        expected = ans.get("expected_answer") or "(no expected answer in GT)"
        return TASK4_PROMPT.format(question=q, answer=a, expected=expected)
    raise ValueError("task %r has no answer judge (T5 is scored by NDCG@10)" % task)


# ---------------------------------------------------------------- the call
class JudgeError(RuntimeError):
    """The judge endpoint rejected our key. Stops grading."""


_JSON_RE = re.compile(r"\{[\s\S]*?\}")
_quirks = set()
_qlock = threading.Lock()


def _body(prompt, settings):
    effort = settings.get("effort") if "no_reasoning" not in _quirks else None
    cap = settings.get("max_output_tokens")
    if JUDGE_API == "chat":
        body = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}]}
        if cap:
            body["max_completion_tokens"] = cap
        if effort:
            body["reasoning_effort"] = effort
        return "/chat/completions", body
    body = {"model": JUDGE_MODEL,
            "input": [{"role": "user", "content": prompt}] if settings.get("as_message") else prompt}
    if cap:
        body["max_output_tokens"] = cap
    if effort:
        body["reasoning"] = {"effort": effort}
    return "/responses", body


def _text(d):
    if JUDGE_API == "chat":
        ch = d.get("choices") or []
        return ((ch[0].get("message") or {}).get("content") or "") if ch else ""
    for item in d.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    return c.get("text", "")
    return ""


def parse_verdict(text):
    """{"score": 0|1|-1, "reason"} from the judge's reply: the first {...} object in the text."""
    m = _JSON_RE.search(text or "")
    if not m:
        return {"score": -1, "reason": "no JSON in judge output", "_raw": (text or "")[:300]}
    try:
        obj = json.loads(m.group())
    except ValueError as e:
        return {"score": -1, "reason": "bad JSON: %s" % e, "_raw": m.group()[:300]}
    try:
        s = int(obj.get("score"))
    except (TypeError, ValueError):
        s = -1
    if s not in (0, 1):
        s = -1
    return {"score": s, "reason": str(obj.get("reason", ""))}


def call_judge(prompt, settings=None):
    """One judgment. Returns {"score", "reason", "status"}; status is "ok" or "failed"."""
    settings = settings or JUDGE_SETTINGS
    if not JUDGE_API_KEY:
        raise JudgeError("no JUDGE_API_KEY / OPENAI_API_KEY set")
    last = None
    for attempt in range(JUDGE_ATTEMPTS):
        path, body = _body(prompt, settings)
        try:
            status, text, _h = http_request(
                JUDGE_BASE_URL + path, method="POST", body=json.dumps(body),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + JUDGE_API_KEY},
                timeout=JUDGE_TIMEOUT)
        except TransportError as e:
            status, text, last = None, "", str(e)
        if status == 200:
            try:
                v = parse_verdict(_text(json.loads(text)))
            except ValueError:
                v = {"score": -1, "reason": "judge endpoint returned non-JSON"}
            v["status"] = "ok" if v["score"] in (0, 1) else "failed"
            return v
        if status in (401, 403):
            raise JudgeError("judge endpoint rejected the key (HTTP %s %s). Set JUDGE_API_KEY."
                             % (status, text[:200]))
        if status == 400 and "reasoning" in text.lower() and "no_reasoning" not in _quirks:
            with _qlock:
                _quirks.add("no_reasoning")
            print("[judge] endpoint rejected the reasoning field; sending without it", flush=True)
            continue
        if status is not None and status != 429 and status < 500:
            return {"score": -1, "reason": "http_%s: %s" % (status, text[:150]), "status": "failed"}
        if status is not None:
            last = "http_%s" % status
        if attempt < JUDGE_ATTEMPTS - 1:
            time.sleep(min(2 ** attempt, 30))
    return {"score": -1, "reason": "all retries failed: %s" % last, "status": "failed"}


def judge_answer(task, question, gt, answer, final_text=""):
    """Grade one answer for paper task 1-4.

    The judge sees the parsed <answer> block. If none was parsed, it sees the agent's whole final
    response instead, so every question is judged; an empty final response scores 0 without a
    judge call.
    """
    text = (answer or "").strip()
    if not text:
        text = (final_text or "").strip()
        if not text:
            return {"score": 0, "reason": "empty final response; not sent to the judge", "status": "empty"}
        v = call_judge(build_judge_prompt(task, question, gt, text))
        v["judged_on"] = "final_text"
        return v
    return call_judge(build_judge_prompt(task, question, gt, answer))


def configured():
    return bool(JUDGE_API_KEY)


def describe():
    return "%s via %s (%s API)" % (JUDGE_MODEL, JUDGE_BASE_URL, JUDGE_API)

