import math

from dlsbench import metrics as M


def g(title, w=1, role="retrieve"):
    return {"title": title, "weight": w, "role": role}


def test_title_match():
    gold = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    assert M.title_match(gold, "BERT: Pre-training of Deep Bidirectional Transformers for Language "
                               "Understanding (arXiv:1810.04805)")
    assert M.title_match(gold, "bert pre-training deep bidirectional transformers language understanding https://x.org/a")
    assert not M.title_match(gold, "RoBERTa: A Robustly Optimized BERT Pretraining Approach")
    assert M.title_match("Attention Is All You Need", "Attention is all you need")
    assert not M.title_match("", "anything") and not M.title_match("x", "")
    assert M.normalize_title("Foo (doi:10.1/2) https://a.b/c arXiv:2401.12345") == "foo"


def test_recall_counts_each_gold_paper_once_and_caps_the_list():
    gold = [g("Alpha Beta Gamma Paper"), g("Delta Epsilon Zeta Paper")]
    r = M.compute_retrieval(["Alpha Beta Gamma Paper", "Alpha Beta Gamma Paper", "noise"], gold)
    assert r["recall"] == 0.5 and r["n_hit_gt"] == 1 and r["n_selected"] == 3
    late = ["filler %d" % i for i in range(50)] + ["Delta Epsilon Zeta Paper"]
    assert M.compute_retrieval(late, gold)["recall"] == 0.0            # 51st line is not read
    assert M.compute_retrieval([], gold)["recall"] == 0.0


def test_gold_papers_schema():
    inst = {"ground_truth": {"retrieve": ["A title", "", {"title": "Seed", "role": "seed", "weight": 2}]}}
    assert M.gold_papers(inst) == [g("A title"), {"title": "Seed", "role": "seed", "weight": 2}]
    assert M.gold_papers({"ground_truth": {}}) == []


# Distinct titles: the fuzzy matcher accepts 70% word overlap, so near-identical titles collide.
T5_GOLD = [g("Graph Transformers for Molecules", 2, "seed"),
           g("Diffusion Policies in Robot Learning", 2, "seed"),
           g("Contrastive Pretraining of Speech Encoders", 1, "shared_reference"),
           g("Sparse Attention for Long Documents", 1, "shared_reference"),
           g("Federated Optimization under Heterogeneity", 1, "shared_reference")]


def ndcg(sel):
    return M.ndcg_at_k(sel, T5_GOLD)["ndcg"]


def test_ndcg_ideal_and_partial():
    ideal = [x["title"] for x in T5_GOLD]
    assert ndcg(ideal) == 1.0
    assert ndcg(list(reversed(ideal))) < 1.0
    assert ndcg([]) == 0.0 and ndcg(["unrelated"] * 10) == 0.0
    idcg = sum(w / math.log2(i + 2) for i, w in enumerate([2, 2, 1, 1, 1]))
    assert abs(ndcg(["noise", "Sparse Attention for Long Documents"]) - (1 / math.log2(3)) / idcg) < 1e-12
    assert abs(ndcg(["Diffusion Policies in Robot Learning"]) - 2 / idcg) < 1e-12


def test_ndcg_no_double_credit_and_cutoff():
    once = ndcg(["Graph Transformers for Molecules"])
    assert ndcg(["Graph Transformers for Molecules", "Graph Transformers for Molecules"]) == once
    beyond = ["noise %d" % i for i in range(10)] + ["Graph Transformers for Molecules"]
    assert ndcg(beyond) == 0.0
    # a line matching both a seed and a reference earns the seed's gain
    r = M.ndcg_at_k(["Contrastive Pretraining of Speech Encoders / Graph Transformers for Molecules"], T5_GOLD)
    assert r["gains"][0] == 2.0


def test_summarize_task():
    def rec(score=None, status=None, recall=1.0, fmt=True, failed=None):
        e = {"recall": recall}
        if status:
            e["judge"] = {"score": score, "status": status}
        return {"failed": failed, "stats": {"format_ok": fmt, "searches_used": 2, "visits_used": 1},
                "eval": e}
    recs = [rec(1, "ok"), rec(0, "ok", recall=0.0), rec(0, "empty", fmt=False),
            rec(-1, "failed"), rec(failed="EndpointError: x")]
    s = M.summarize_task(1, recs, n_total=6)
    assert (s["n_total"], s["n"], s["n_failed"], s["n_format_ok"]) == (6, 4, 1, 3)
    assert abs(s["acc"] - 100.0 / 3) < 0.01 and s["n_judged"] == 3  # failed verdicts are left out
    assert s["n_judge_failed"] == 1 and "acc_all" not in s
    assert s["recall"] == 75.0 and s["searches/q"] == 2.0
    t5 = M.summarize_task(5, [{"stats": {}, "eval": {"ndcg@10": 0.5}}, {"stats": {}, "eval": {"ndcg@10": 1.0}}])
    assert t5["ndcg@10"] == 75.0 and "acc" not in t5
