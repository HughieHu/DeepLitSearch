"""One question per task, end to end against the offline mocks: the real CLI, model client, harness,
tools, judge and metrics, in a subprocess."""
import json
import os

import pytest

import mock_servers
from conftest import run_module
from dlsbench import judge


def _read(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def mock():
    m = mock_servers.Mock()
    yield m
    m.stop()


def _summary(out_dir, harness):
    return _read(os.path.join(out_dir, "results", "mock-model__%s.json" % harness))


def _records(out_dir, harness):
    recs = {}
    base = os.path.join(out_dir, "runs", "mock-model__%s" % harness)
    for t in range(1, 6):
        d = os.path.join(base, "task%d" % t)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            recs.setdefault(t, []).append(_read(os.path.join(d, f)))
    return recs


@pytest.mark.parametrize("harness", ["web", "s2"])
def test_every_task_end_to_end(tmp_path, mock, harness):
    args = ["--task", "all", "--harness", harness, "--limit", "1", "--concurrency", "3",
            "--out-dir", str(tmp_path)]
    r = run_module("dlsbench.eval", args, mock.env(), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr

    s = _summary(tmp_path, harness)
    assert s["model"] == "mock-model" and s["harness"] == harness
    assert s["budgets"] == {"searches": 10, "visits": 10, "turns": 25}
    for t in (1, 2, 3, 4):
        row = s["tasks"]["T%d" % t]
        assert (row["n"], row["n_format_ok"], row["n_judged"]) == (1, 1, 1), row
        assert row["recall"] == 100.0 and row["acc"] == 100.0, row
    assert s["tasks"]["T5"]["ndcg@10"] == 100.0 and "acc" not in s["tasks"]["T5"]

    recs = _records(tmp_path, harness)
    for t in range(1, 6):
        (rec,) = recs[t]
        st = rec["stats"]
        assert (st["searches_used"], st["visits_used"], st["turns"]) == (1, 1, 4)
        assert st["prompt_tokens"] > 0 and st["reasoning_tokens"] == 40
        assert rec["failed"] is None and rec["answer"].startswith("MOCK-OK")
        assert any(e["type"] == "force_final" for e in rec["trace"])        # the no-tag turn
        outs = {e["tool"]: e["output"] for e in rec["trace"] if e["type"] == "tool_result"}
        if harness == "web":
            assert outs["google_search"].count("Title:") == 10
            assert len(outs["visit_page"]) == 8000 and "END" not in outs["visit_page"]
        else:
            hits = json.loads(outs["s2_search"]) if len(outs["s2_search"]) < 8000 else None
            assert len(outs["s2_search"]) <= 8000
            if hits:
                assert len(hits[0]["abstract"]) == 1200
            assert len(outs["s2_paper"]) <= 18000
            paper = outs["s2_paper"]
            assert '"body_source": "openAccessPdf"' in paper and "END" not in paper
        if t == 5:
            assert rec["eval"]["ndcg@10"] == 1.0
        else:
            assert rec["eval"]["judge"] == {"score": 1, "reason": "mock verdict", "status": "ok"}

    # T1-T4 judged with the paper's Figure 8-11 prompts, all sent the same way; T5 never.
    log = {e["header"]: (e["effort"], e["max_tokens"], e["input_is_message"]) for e in mock.state.judge_log}
    assert log == {judge.TASK1_PROMPT.splitlines()[0]: ("high", None, False),
                   judge.TASK2_PROMPT.splitlines()[0]: ("high", None, False),
                   judge.TASK3_PROMPT.splitlines()[0]: ("high", None, False),
                   "You are judging a research-paper agent's answer.": ("high", None, False)}
    assert mock.state.counts["judge"] == 4
    # Optional request fields are not sent unless configured.
    assert all(set(b) <= {"model", "tools", "tool_choice"} for b in mock.state.model_bodies)

    # Resume: a second run makes no model or judge calls and reproduces the summary.
    before = dict(mock.state.counts)
    r = run_module("dlsbench.eval", args, mock.env(), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert mock.state.counts["model"] == before["model"]
    assert mock.state.counts["judge"] == before["judge"]
    assert _summary(tmp_path, harness)["tasks"] == s["tasks"]


def test_endpoint_quirks_are_learned(tmp_path, mock):
    """An endpoint that rejects reasoning_effort and max_tokens (as some reasoning APIs do)."""
    mock.state.strict = True
    env = mock.env(MODEL_REASONING_EFFORT="high", MODEL_MAX_TOKENS="16000")
    r = run_module("dlsbench.eval", ["--task", "1", "--harness", "web", "--limit", "2",
                                     "--out-dir", str(tmp_path)], env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "adapting (no_reasoning_effort)" in r.stdout and "adapting (no_max_tokens)" in r.stdout
    row = _summary(tmp_path, "web")["tasks"]["T1"]
    assert (row["n"], row["acc"], row["recall"]) == (2, 100.0, 100.0)
    assert any(b.get("max_completion_tokens") == 16000 for b in mock.state.model_bodies)


def test_judge_failure_is_excluded_then_regraded(tmp_path, mock):
    args = ["--task", "2", "--harness", "web", "--limit", "1", "--out-dir", str(tmp_path)]
    mock.state.judge_mode = "garbage"
    r = run_module("dlsbench.eval", args, mock.env(), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    row = _summary(tmp_path, "web")["tasks"]["T2"]
    assert row["acc"] is None and row["n_judged"] == 0 and row["n_judge_failed"] == 1
    assert row["recall"] == 100.0                     # retrieval is scored regardless
    assert "no valid judge verdict" in r.stdout

    mock.state.judge_mode = "ok"
    model_calls = mock.state.counts["model"]
    r = run_module("dlsbench.eval", args + ["--grade-only"], mock.env(), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert mock.state.counts["model"] == model_calls
    row = _summary(tmp_path, "web")["tasks"]["T2"]
    assert (row["acc"], row["n_judged"], row["n_judge_failed"]) == (100.0, 1, 0)


def test_bad_judge_key_stops_grading(tmp_path, mock):
    env = mock.env(JUDGE_API_KEY="wrong")
    r = run_module("dlsbench.eval", ["--task", "1", "--harness", "web", "--limit", "1",
                                     "--out-dir", str(tmp_path)], env, tmp_path)
    assert "judge stopped" in r.stderr
    row = _summary(tmp_path, "web")["tasks"]["T1"]
    assert row["n_not_graded"] == 1 and row["acc"] is None and row["recall"] == 100.0


def test_judge_over_chat_completions(tmp_path, mock):
    r = run_module("dlsbench.eval", ["--task", "4", "--harness", "web", "--limit", "1",
                                     "--out-dir", str(tmp_path)], mock.env(JUDGE_API="chat"), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _summary(tmp_path, "web")["tasks"]["T4"]["acc"] == 100.0


def test_missing_serper_key_fails_fast(tmp_path, mock):
    r = run_module("dlsbench.eval", ["--task", "1", "--harness", "web", "--limit", "1",
                                     "--out-dir", str(tmp_path)], mock.env(SERPER_API_KEY=""), tmp_path)
    assert r.returncode != 0 and "SERPER_API_KEY" in r.stderr
    assert mock.state.counts["model"] == 0


def test_rejected_serper_key_stops_the_run(tmp_path, mock):
    r = run_module("dlsbench.eval", ["--task", "1", "--harness", "web", "--limit", "3",
                                     "--concurrency", "1", "--out-dir", str(tmp_path)],
                   mock.env(SERPER_API_KEY="revoked"), tmp_path)
    assert r.returncode == 1 and "Serper rejected" in r.stderr
    row = _summary(tmp_path, "web")["tasks"]["T1"]
    assert row["n"] == 0 and row["n_failed"] >= 1


@pytest.mark.parametrize("harness", ["web", "s2"])
def test_doctor(tmp_path, mock, harness):
    r = run_module("dlsbench.doctor", ["--harness", harness], mock.env(), tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "READY" in r.stdout and "native tool calling" in r.stdout
