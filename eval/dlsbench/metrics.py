"""Deterministic metrics: no API calls, re-runnable on existing records.

* Recall (T1-T4): the share of gold papers (`ground_truth.retrieve`) whose title fuzzily matches a
  line of the agent's <selected_papers> list. Only the first 50 lines count.
* NDCG@10 (T5): the <selected_papers> list read as a ranking, with graded relevance taken from the
  gold entries' `weight` (seed papers 2, shared references 1, anything else 0). Each gold paper
  can earn its gain once.

Both use the same fuzzy title match: lowercase, drop URLs, arXiv ids and DOIs, keep [a-z0-9 ],
then accept substring containment either way or >= 70% of the gold title's words.
"""
import math
import re

SELECTED_CAP = 50      # selected-paper lines considered for recall
NDCG_K = 10


def normalize_title(s):
    s = (s or "").lower()
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"arxiv[:\s]*\d{4}\.\d{4,6}", " ", s)
    s = re.sub(r"\bdoi[:\s]*\S+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def title_match(gt_title, selected_str):
    g = normalize_title(gt_title)
    s = normalize_title(selected_str)
    if not g or not s:
        return False
    if g in s or s in g:
        return True
    g_tokens = set(g.split())
    s_tokens = set(s.split())
    if not g_tokens:
        return False
    return len(s_tokens & g_tokens) / len(g_tokens) >= 0.7


def gold_papers(instance):
    """[{"title", "role", "weight"}] from ground_truth.retrieve.

    Schema-driven: an entry is either a title string (T1-T4) or an object with `title` and, for
    T5, `role` and `weight`.
    """
    out = []
    for e in (instance.get("ground_truth") or {}).get("retrieve") or []:
        if isinstance(e, str):
            if e.strip():
                out.append({"title": e, "role": "retrieve", "weight": 1})
        elif isinstance(e, dict) and (e.get("title") or "").strip():
            out.append({"title": e["title"], "role": e.get("role", "retrieve"),
                        "weight": e.get("weight", 1)})
    return out


def compute_retrieval(selected_papers, gold):
    """Recall / precision / F1 of the selected list against the gold titles."""
    selected = list(selected_papers or [])[:SELECTED_CAP]
    hits = []
    for g in gold:
        hit = any(title_match(g["title"], s) for s in selected)
        hits.append({"title": g["title"], "role": g.get("role"), "hit": hit})
    n_gt, n_sel = len(gold), len(selected)
    n_hit = sum(1 for h in hits if h["hit"])
    matched_selected = sum(1 for s in selected if any(title_match(g["title"], s) for g in gold))
    recall = n_hit / n_gt if n_gt else 0.0
    precision = matched_selected / n_sel if n_sel else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "n_gt": n_gt,
            "n_selected": n_sel, "n_hit_gt": n_hit, "per_gt_hits": hits}


def ndcg_at_k(selected_papers, gold, k=NDCG_K):
    """NDCG@k with graded relevance.

    Walking the first k selected lines in order, a line earns the weight of the highest-weight gold
    paper it matches that has not been matched yet (seed papers are tried before shared
    references), so listing a paper twice does not earn twice. The ideal ranking lists the gold
    papers in descending weight.
    """
    groups = sorted({float(g.get("weight") or 0) for g in gold if (g.get("weight") or 0) > 0},
                    reverse=True)
    found = [False] * len(gold)
    gains = []
    for sel in list(selected_papers or [])[:k]:
        g = 0.0
        for w in groups:
            for i, gp in enumerate(gold):
                if not found[i] and float(gp.get("weight") or 0) == w and title_match(gp["title"], sel):
                    found[i] = True
                    g = w
                    break
            if g:
                break
        gains.append(g)
    gains += [0.0] * (k - len(gains))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((float(gp.get("weight") or 0) for gp in gold), reverse=True)
    ideal = (ideal + [0.0] * k)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return {"ndcg": dcg / idcg if idcg > 0 else 0.0, "gains": gains[:k],
            "n_found": sum(found), "n_gold": len(gold)}


# ---------------------------------------------------------------- aggregation
def _pct(x):
    return None if x is None else round(100.0 * x, 2)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_task(task, records, n_total=None):
    """Aggregate one task's per-question records into the reported numbers.

    `records` are the records on disk for the selected questions. Questions whose run failed
    (model endpoint or search backend broke) are counted in `n_failed` and left out of every
    metric, as they are retried on the next run.

    Acc is the share of judge score 1 among questions with a score of 0 or 1. Questions the judge
    returned no valid verdict for are left out and counted in `n_judge_failed`.
    """
    done = [r for r in records if not r.get("failed")]
    n = len(done)
    out = {"n_total": n_total if n_total is not None else len(records), "n": n,
           "n_failed": sum(1 for r in records if r.get("failed")),
           "n_format_ok": sum(1 for r in done if (r.get("stats") or {}).get("format_ok"))}
    ev = [r.get("eval") or {} for r in done]
    if task == 5:
        out["ndcg@10"] = _pct(_mean(e["ndcg@10"] for e in ev if e.get("ndcg@10") is not None))
    else:
        out["recall"] = _pct(_mean(e["recall"] for e in ev if e.get("recall") is not None))
        judged = [e["judge"]["score"] for e in ev if (e.get("judge") or {}).get("score") in (0, 1)]
        out["acc"] = _pct(_mean(judged))
        out["n_judged"] = len(judged)
        out["n_judge_failed"] = sum(1 for e in ev if (e.get("judge") or {}).get("status") == "failed")
        out["n_not_graded"] = sum(1 for e in ev if not e.get("judge"))
    st = [r.get("stats") or {} for r in done]
    if n:
        out["searches/q"] = round(sum(s.get("searches_used", 0) for s in st) / n, 2)
        out["visits/q"] = round(sum(s.get("visits_used", 0) for s in st) / n, 2)
        out["in_tok/q"] = round(sum(s.get("prompt_tokens", 0) for s in st) / n)
        out["out_tok/q"] = round(sum(s.get("output_tokens", 0) for s in st) / n)
    return out
