"""The ReAct loop's control flow, in process, with a scripted model and fake tools."""
import json

from dlsbench import harness as h
from dlsbench import model as gem
from dlsbench.tools import ToolConfigError

FINAL = ("<selected_papers>\n- Paper A (arXiv:2401.00001)\n* Paper B\n1. Paper C\n</selected_papers>\n"
         "<answer>\nThe answer.\n</answer>")


class FakeTools:
    search_name, visit_name = "google_search", "visit_page"

    def __init__(self):
        self.calls = []

    def search(self, args):
        self.calls.append(("search", args))
        return "Title: t\nURL: u\nSnippet: s"

    def visit(self, args):
        self.calls.append(("visit", args))
        return "page text"


def call(name, i, **args):
    return {"id": "c%d" % i, "name": name, "arguments": json.dumps(args)}


class Script:
    """A fake model: `policy(messages, tools, n)` returns the response for call n."""

    def __init__(self, policy):
        self.policy, self.seen = policy, []

    def __call__(self, messages, tools=None):
        self.seen.append({"messages": [dict(m) for m in messages],
                          "tools": [t["name"] for t in (tools or [])]})
        return self.policy(messages, tools, len(self.seen) - 1)


def test_parse_final_takes_last_blocks_and_strips_bullets():
    text = "thinking: <selected_papers>\n- Wrong\n</selected_papers><answer>no</answer>\n" + FINAL
    sel, ans = h.parse_final(text)
    assert sel == ["Paper A (arXiv:2401.00001)", "Paper B", "Paper C"] and ans == "The answer."
    assert h.parse_final("<ANSWER>x</ANSWER>") == ([], "x")
    assert h.parse_final("") == ([], "")


def test_prompt_is_one_user_message_with_the_question():
    m = Script(lambda msgs, tools, n: {"text": FINAL, "tool_calls": [], "usage": {}})
    rec = h.run_episode(3, "Why do they disagree?", "web", FakeTools(), call_model=m)
    first = m.seen[0]["messages"]
    assert len(first) == 1 and first[0]["role"] == "user"
    assert first[0]["content"].endswith("\n\n=== QUESTION ===\nWhy do they disagree?\n")
    assert h.ANSWER_INSTRUCTIONS[3] in first[0]["content"]
    assert m.seen[0]["tools"] == ["google_search", "visit_page"]
    assert rec["stats"]["format_ok"] and rec["stats"]["turns"] == 1 and rec["failed"] is None


def test_budgets_and_turn_limit():
    """A model that searches forever: the search tool disappears after 10 calls, extra calls get
    the budget error, and after 25 turns a final no-tool call forces the answer."""
    def policy(msgs, tools, n):
        if tools is None:
            return {"text": FINAL, "tool_calls": [], "usage": {"prompt_tokens": 1}}
        return {"text": "", "tool_calls": [call("google_search", n, query="q%d" % n)],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    tools = FakeTools()
    m = Script(policy)
    rec = h.run_episode(1, "Q", "web", tools, call_model=m)
    assert rec["stats"]["searches_used"] == 10 and len(tools.calls) == 10
    assert m.seen[9]["tools"] == ["google_search", "visit_page"]
    assert m.seen[10]["tools"] == ["visit_page"]
    assert len(m.seen) == h.MAX_TURNS + 1 and m.seen[-1]["tools"] == []
    outs = [e["output"] for e in rec["trace"] if e["type"] == "tool_result"]
    assert outs[10:] == ["[ERROR] search budget exhausted"] * (h.MAX_TURNS - 10)
    assert rec["trace"][-1]["type"] == "post_loop_force_final"
    assert m.seen[-1]["messages"][-1]["content"] == h.build_force_final("web", 1)
    assert rec["answer"] == "The answer." and rec["stats"]["turns"] == h.MAX_TURNS
    assert rec["stats"]["prompt_tokens"] == h.MAX_TURNS + 1


def test_s2_budget_message_uses_tool_names():
    class S2Fake(FakeTools):
        search_name, visit_name = "s2_search", "s2_paper"

    def policy(msgs, tools, n):
        if tools is None:
            return {"text": FINAL, "tool_calls": [], "usage": {}}
        return {"text": "", "tool_calls": [call("s2_paper", n, paper_id="p")], "usage": {}}
    rec = h.run_episode(2, "Q", "s2", S2Fake(), call_model=Script(policy))
    outs = [e["output"] for e in rec["trace"] if e["type"] == "tool_result"]
    assert outs[10] == "[ERROR] s2_paper budget exhausted" and rec["stats"]["visits_used"] == 10


def test_missing_format_gets_one_more_prompt_and_text_is_not_kept():
    def policy(msgs, tools, n):
        if n == 0:
            return {"text": "Here is my answer without tags.", "tool_calls": [], "usage": {}}
        return {"text": FINAL, "tool_calls": [], "usage": {}}
    m = Script(policy)
    rec = h.run_episode(5, "Q", "s2", FakeTools(), call_model=m)
    second = m.seen[1]
    assert [x["role"] for x in second["messages"]] == ["user", "user"]
    assert second["messages"][1]["content"] == h.build_force_final("s2", 5)
    assert second["tools"] == ["s2_search", "s2_paper"]          # tools stay while budget remains
    assert [e["type"] for e in rec["trace"]] == ["model_output", "force_final", "model_output"]
    assert rec["stats"]["format_ok"]


def test_parallel_tool_calls_and_history_format():
    def policy(msgs, tools, n):
        if n == 0:
            return {"text": "ignored", "tool_calls": [call("google_search", 1, query="a"),
                                                      call("visit_page", 2, url="http://x"),
                                                      call("nope", 3)], "usage": {}}
        return {"text": FINAL, "tool_calls": [], "usage": {}}
    m = Script(policy)
    rec = h.run_episode(1, "Q", "web", FakeTools(), call_model=m)
    msgs = m.seen[1]["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] is None
    assert [tc["id"] for tc in msgs[1]["tool_calls"]] == ["c1", "c2", "c3"]
    assert [(x["role"], x["tool_call_id"]) for x in msgs[2:]] == [("tool", "c1"), ("tool", "c2"), ("tool", "c3")]
    assert msgs[4]["content"] == "[ERROR] unknown tool nope"
    assert (rec["stats"]["searches_used"], rec["stats"]["visits_used"]) == (1, 1)


def test_bad_tool_arguments_become_empty_args():
    def policy(msgs, tools, n):
        if n == 0:
            return {"text": "", "tool_calls": [{"id": "x", "name": "google_search",
                                                "arguments": "{not json"}], "usage": {}}
        return {"text": FINAL, "tool_calls": [], "usage": {}}
    tools = FakeTools()
    h.run_episode(1, "Q", "web", tools, call_model=Script(policy))
    assert tools.calls == [("search", {})]


def test_context_overflow_ends_the_episode_without_failing_it():
    def policy(msgs, tools, n):
        if n == 0:
            return {"text": "", "tool_calls": [call("google_search", 0, query="a")], "usage": {}}
        raise gem.ContextOverflow("context_length_exceeded")
    rec = h.run_episode(1, "Q", "web", FakeTools(), call_model=Script(policy))
    assert rec["failed"] is None and rec["answer"] == "" and not rec["stats"]["format_ok"]
    assert [e["type"] for e in rec["trace"]][-2:] == ["error", "post_loop_force_final"]


def test_endpoint_and_tool_config_errors_fail_the_question():
    def broken(msgs, tools=None):
        raise gem.EndpointError("HTTP 401 bad key")
    rec = h.run_episode(1, "Q", "web", FakeTools(), call_model=broken)
    assert rec["failed"].startswith("EndpointError")

    class BadTools(FakeTools):
        def search(self, args):
            raise ToolConfigError("Serper rejected the request")

    def policy(msgs, tools, n):
        return {"text": "", "tool_calls": [call("google_search", n, query="a")], "usage": {}}
    rec = h.run_episode(1, "Q", "web", BadTools(), call_model=Script(policy))
    assert rec["failed"].startswith("ToolConfigError")
