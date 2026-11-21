"""The agent prompts, tool schemas, per-task answer instructions, budgets and truncation limits are
what the benchmark is defined by. Changing any of them makes results incomparable, so a change must
be deliberate: copy the new fingerprint from the failing assertion into FROZEN and say why in the
commit."""
import hashlib
import json

from dlsbench import harness as h
from dlsbench import tools as tl

FROZEN = "dca2d092ca8514efce736fe9f07267db4d0a046699d9157234eaebaf5bab5a7d"


def fingerprint():
    blob = {
        "web": [h.WEB_SYSTEM_INSTRUCTIONS, h.WEB_FORCE_FINAL_PROMPT, h.WEB_TOOLS],
        "s2": [h.S2_SYSTEM_INSTRUCTIONS, h.S2_FORCE_FINAL_PROMPT, h.S2_TOOLS],
        "answer": {str(k): v for k, v in h.ANSWER_INSTRUCTIONS.items()},
        "budget": [h.MAX_SEARCHES, h.MAX_VISITS, h.MAX_TURNS],
        "limits": [tl.VISIT_TRUNCATE, tl.JINA_MAX_CHARS, tl.S2_SEARCH_TRUNCATE, tl.S2_PAPER_TRUNCATE,
                   tl.ABSTRACT_TRUNC, tl.TLDR_TRUNC, tl.MAX_AUTHORS_PER_PAPER,
                   tl.MAX_REFERENCES_RETURNED, tl.MAX_CITATIONS_RETURNED,
                   tl.SEARCH_FIELDS, tl.PAPER_FIELDS],
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def test_prompts_unchanged():
    assert fingerprint() == FROZEN


def test_question_prompt_layout():
    p = h.build_question_prompt("web", 4, "How was it configured?")
    assert p.startswith("You are a research literature search agent. You have two tools:\n")
    assert "<answer>\n%s\n</answer>" % h.ANSWER_INSTRUCTIONS[4] in p
    assert p.endswith("\n\n=== QUESTION ===\nHow was it configured?\n")
    assert set(h.ANSWER_INSTRUCTIONS) == {1, 2, 3, 4, 5}
