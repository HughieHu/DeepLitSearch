"""The bundled questions load and carry the fields the harness and the scorers read. Nothing here
assumes a question count, so a new data release can be dropped into eval/bench/ as is."""
import os

import pytest

from conftest import EVAL
from dlsbench import eval as ev
from dlsbench import metrics

BENCH = os.path.join(EVAL, "bench")


@pytest.mark.parametrize("task", [1, 2, 3, 4, 5])
def test_task_file_schema(task):
    items = ev.load_task(BENCH, task)
    assert items
    for it in items:
        assert it["query"].strip()
        gold = metrics.gold_papers(it)
        assert gold, it["instance_id"]
        ans = it["ground_truth"].get("answer")
        if task == 1:
            assert ans.get("benchmark")
        elif task == 2:
            assert ans.get("best_method")
        elif task == 3:
            assert ans.get("method_a") and ans.get("method_b")
        elif task == 4:
            assert ans.get("expected_answer")
        else:
            assert {g["weight"] for g in gold} <= {1, 2} and any(g["weight"] == 2 for g in gold)


def test_task_parsing():
    assert ev.parse_tasks("all") == [1, 2, 3, 4, 5]
    assert ev.parse_tasks("3,T1,1") == [1, 3]
    with pytest.raises(Exception):
        ev.parse_tasks("6")
    assert ev.safe_name("org/Model:v1") == "org_Model_v1"
