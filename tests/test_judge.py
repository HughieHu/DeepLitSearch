import hashlib

import pytest

import mock_servers
from dlsbench import judge


GT = {
    1: {"answer": {"benchmark": "MatrixCity", "split_or_version": "Block_Small", "metrics": ["PSNR", "SSIM"]}},
    2: {"answer": {"best_method": "M1", "paradigm": "LLM_FINETUNED", "verified_findings": [
        {"method": "M1", "beats": "M2", "metric": "acc", "winner_score": 9.1, "loser_score": 8.0}]}},
    3: {"answer": {"method_a": "fo-sgd", "method_b": "mezo",
                   "a_beats_b": {"paper_id": "Paper A", "metric": "SST2", "a_value": "91.6", "b_value": "89.4"},
                   "b_beats_a": {"paper_id": "Paper B", "metric": "COPA", "a_value": "85.0", "b_value": "86.3"},
                   "reasoning": "different tasks"}},
    4: {"answer": {"expected_answer": "Repo X sets lr=0.1 in configs/a.yaml."}},
}

# Rendered judge prompts for fixed inputs. These are the paper's Figure 8-11 prompts; a change here
# makes Acc incomparable, so it must be deliberate: update the fingerprint and say why in the commit.
FROZEN = {
    1: "bf712d5b4f2af45ec06b080eec1e5129e48282fb6f21ec1dac609fd85a2faf9b",
    2: "2f8457cdefd2b7b4f3d6e69d4927fecdaef5744c449ba17a1c598f1bb5ac0dd9",
    3: "bc65f672092a61a162991b0bb1f0049385ae60fbdd447457248de940027e7a72",
    4: "bee3875e1debb13aa25d1ef7266c5da5c5b40b6f9620b72f748f7c0adfe108c5",
}


@pytest.mark.parametrize("task", [1, 2, 3, 4])
def test_judge_prompts_unchanged(task):
    p = judge.build_judge_prompt(task, "QUESTION TEXT", GT[task], "MODEL ANSWER TEXT")
    assert hashlib.sha256(p.encode()).hexdigest() == FROZEN[task]


def test_prompts_carry_the_paper_figures_rubrics():
    # One distinctive sentence from each of Figures 8-11.
    assert "1 = answer correctly identifies the ground-truth target (right benchmark, key facts match" \
        in judge.build_judge_prompt(1, "q", GT[1], "a")
    assert "with at least one supporting cross-paper comparison (winner > loser on metric)" \
        in judge.build_judge_prompt(2, "q", GT[2], "a")
    assert "This task is strict — no partial credit; the explanation must point to a concrete " \
        "methodological difference." in judge.build_judge_prompt(3, "q", GT[3], "a")
    assert "but the key claim (which library / config / method was used) must match." \
        in judge.build_judge_prompt(4, "q", GT[4], "a")
    for t in (1, 2, 3, 4):
        assert judge.build_judge_prompt(t, "q", GT[t], "a").endswith(
            'Output strict JSON only:\n{"score": 0 or 1, "reason": "<one sentence>"}')


def test_prompt_contents_without_truncation():
    question = "Q" * 2000 + "\nQUESTION_END"
    answer = "A" * 5000 + "\nANSWER_END"
    p = judge.build_judge_prompt(3, question, GT[3], answer)
    assert "a_beats_b in paper Paper A: SST2 = 91.6 vs 89.4" in p
    assert "b_beats_a in paper Paper B: COPA = 85.0 vs 86.3" in p
    for t in (1, 2, 3, 4):
        p = judge.build_judge_prompt(t, question, GT[t], answer)
        assert "QUESTION:\n" + question + "\n\n" in p
        assert answer + "\n\n" in p
    reasoning = "R" * 1000 + "\nREASONING_END"
    long_gt3 = {"answer": {**GT[3]["answer"], "reasoning": reasoning}}
    assert "reasoning: " + reasoning + "\n\n" in judge.build_judge_prompt(3, "q", long_gt3, "a")
    p4 = judge.build_judge_prompt(4, "q", GT[4], "a")
    assert p4.startswith("You are judging a research-paper agent's answer.\n\nQUESTION:\nq\n\nMODEL ANSWER:\na\n")
    assert "EXPECTED ANSWER:\nRepo X sets lr=0.1 in configs/a.yaml.\n\n" in p4
    expected = "E" * 2000 + "\nEXPECTED_END"
    long_gt4 = {"answer": {"expected_answer": expected}}
    assert "EXPECTED ANSWER:\n" + expected + "\n\n" in judge.build_judge_prompt(4, "q", long_gt4, "a")
    assert "(no expected answer in GT)" in judge.build_judge_prompt(4, "q", {"answer": {}}, "a")
    assert "metrics (any subset would count): PSNR, SSIM" in judge.build_judge_prompt(1, "q", GT[1], "a")
    assert "verified_findings: M1 > M2 (acc: 9.1 vs 8.0)" in judge.build_judge_prompt(2, "q", GT[2], "a")
    with pytest.raises(ValueError):
        judge.build_judge_prompt(5, "q", {}, "a")


def test_request_settings():
    assert judge.JUDGE_MODEL == "gpt-5.4-mini"
    assert judge.JUDGE_SETTINGS == {"effort": "high", "max_output_tokens": None, "as_message": False}


@pytest.mark.parametrize("text,score", [
    ('{"score": 1, "reason": "ok"}', 1),
    ('Sure. {"score": 0, "reason": "wrong"} done', 0),
    ('{"score": "1", "reason": "string score"}', 1),
    ('{"score": 3, "reason": "out of range"}', -1),
    ('{"score": 1, "reason": "has {nested} brace"}', -1),     # first {...} is cut short: invalid
    ("no json at all", -1),
])
def test_parse_verdict(text, score):
    assert judge.parse_verdict(text)["score"] == score


@pytest.mark.parametrize("task", [1, 2, 3, 4])
def test_missing_answer_falls_back_to_final_text(monkeypatch, task):
    seen = []
    monkeypatch.setattr(judge, "call_judge", lambda prompt, settings=None: (
        seen.append((prompt, settings)), {"score": 0, "reason": "r", "status": "ok"})[1])
    v = judge.judge_answer(task, "q", GT[task], "  ", final_text="The whole final response")
    assert v["judged_on"] == "final_text" and seen[-1][1] is None
    assert "The whole final response\n" in seen[-1][0]
    v = judge.judge_answer(task, "q", GT[task], "", final_text="  ")     # empty: 0, no judge call
    assert (v["score"], v["status"], len(seen)) == (0, "empty", 1)
    assert "judged_on" not in judge.judge_answer(task, "q", GT[task], "an answer", final_text="ignored")


@pytest.fixture
def live(monkeypatch):
    m = mock_servers.Mock()
    monkeypatch.setattr(judge, "JUDGE_BASE_URL", m.url + "/judge/v1")
    monkeypatch.setattr(judge, "JUDGE_API_KEY", mock_servers.JUDGE_KEY)
    monkeypatch.setattr(judge, "JUDGE_API", "responses")
    monkeypatch.setattr(judge, "_quirks", set())
    yield m
    m.stop()


def test_call_judge_against_mock(live):
    v = judge.judge_answer(2, "q", GT[2], "MOCK-OK M1 wins")
    assert v == {"score": 1, "reason": "mock verdict", "status": "ok"}
    assert live.state.judge_log[-1] == {"header": judge.TASK2_PROMPT.splitlines()[0], "effort": "high",
                                        "max_tokens": None, "input_is_message": False}
    assert judge.judge_answer(4, "q", GT[4], "something else")["score"] == 0
    assert judge.judge_answer(4, "q", GT[4], "MOCK-OK it is lr=0.1")["score"] == 1
    assert live.state.judge_log[-1] == {"header": "You are judging a research-paper agent's answer.",
                                        "effort": "high", "max_tokens": None, "input_is_message": False}
    live.state.judge_mode = "garbage"
    v = judge.judge_answer(1, "q", GT[1], "MOCK-OK")
    assert v["score"] == -1 and v["status"] == "failed"


def test_bad_key_raises(live, monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_API_KEY", "wrong")
    with pytest.raises(judge.JudgeError):
        judge.judge_answer(1, "q", GT[1], "MOCK-OK")


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(judge, "JUDGE_API_KEY", "")
    with pytest.raises(judge.JudgeError):
        judge.call_judge("prompt")
