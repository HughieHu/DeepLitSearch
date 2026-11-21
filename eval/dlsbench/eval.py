"""One command runs and scores a model on DeepLitSearch.

    python -m dlsbench.eval --task all --harness web

Runs every question, grades the answers, prints the table. Safe to interrupt: rerun the same command
and it picks up where it stopped. Each question is written to its own file as soon as it finishes,
and questions whose run or grading failed are retried.

    --task        all | 1 | 2 | 3 | 4 | 5, or a comma list such as 1,3   (paper task numbers)
    --harness     web (Serper + Jina) | s2 (Semantic Scholar + Jina)
    --limit N     first N questions of each task only; use it for a smoke test
    --concurrency K  questions run in parallel (default 4, or CONCURRENCY)
    --no-grade    run, and compute Recall / NDCG, but do not call the judge
    --grade-only  grade existing records without running the agent
    --bench-dir   directory holding task1.json ... task5.json (default: bench/ next to this package)
    --out-dir     where runs/ and results/ are written (default: this package's parent directory)
    --verbose     print every model turn

Output:
    runs/<model>__<harness>/task<N>/<instance_id>.json   the full record of one question
    results/<model>__<harness>.json                      per-task Recall / Acc / NDCG@10 and counts
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import TASK_NAMES, envint
from . import judge
from . import metrics
from . import model as gem
from . import tools as tl
from .harness import MAX_SEARCHES, MAX_TURNS, MAX_VISITS, run_episode

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_NAMES = ("web", "s2")


# ---------------------------------------------------------------- data
def load_task(bench_dir, task):
    """The questions of one task. Only the unified schema is assumed, never a count."""
    path = os.path.join(bench_dir, "task%d.json" % task)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or []
    seen = set()
    for it in data:
        for k in ("instance_id", "query", "ground_truth"):
            if k not in it:
                raise ValueError("%s: an instance lacks %r" % (path, k))
        if it["instance_id"] in seen:
            raise ValueError("%s: duplicate instance_id %r" % (path, it["instance_id"]))
        seen.add(it["instance_id"])
    return data


def parse_tasks(s):
    if s.strip().lower() == "all":
        return sorted(TASK_NAMES)
    out = []
    for p in s.split(","):
        p = p.strip().upper().lstrip("T")
        if not p.isdigit() or int(p) not in TASK_NAMES:
            raise argparse.ArgumentTypeError("tasks are 1-5 or 'all', got %r" % s)
        out.append(int(p))
    return sorted(set(out))


def safe_name(s):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in (s or "model"))


def record_path(run_dir, task, iid):
    return os.path.join(run_dir, "task%d" % task, safe_name(iid) + ".json")


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)          # atomic: a crash mid-write must not destroy a finished record


def is_done(rec, inst):
    return bool(rec) and not rec.get("failed") and rec.get("question") == inst["query"]


# ---------------------------------------------------------------- grading
def deterministic_eval(task, inst, rec):
    """Recall (T1-T4) or NDCG@10 (T5); no API calls."""
    gold = metrics.gold_papers(inst)
    ev = dict(rec.get("eval") or {})
    if task == 5:
        r = metrics.ndcg_at_k(rec.get("selected_papers") or [], gold)
        ev.update({"ndcg@10": r["ndcg"], "gains": r["gains"], "n_gold_found": r["n_found"],
                   "n_gold": r["n_gold"]})
    else:
        r = metrics.compute_retrieval(rec.get("selected_papers") or [], gold)
        ev.update({"recall": r["recall"], "precision": r["precision"],
                   "n_gold": r["n_gt"], "n_gold_found": r["n_hit_gt"],
                   "per_gold_hits": r["per_gt_hits"]})
    return ev


def needs_judge(task, rec):
    if task == 5 or rec.get("failed"):
        return False
    j = (rec.get("eval") or {}).get("judge")
    return not j or j.get("status") in ("failed", "no_answer")    # no_answer: left by older versions


# ---------------------------------------------------------------- report
def format_table(label, summary):
    L = ["", "=== %s ===" % label,
         "%-28s %5s %8s %7s %8s %7s %10s" % ("task", "n", "Recall", "Acc", "NDCG@10",
                                             "judged", "format_ok")]
    notes = []

    def f(v):
        return "-" if v is None else "%.1f" % v

    for key in sorted(summary):
        s = summary[key]
        t = int(key[1:])
        L.append("%-28s %5d %8s %7s %8s %7s %10s" % (
            "%s %s" % (key, TASK_NAMES[t]), s["n"], f(s.get("recall")), f(s.get("acc")),
            f(s.get("ndcg@10")), s.get("n_judged", "-"), s["n_format_ok"]))
        if s["n"] < s["n_total"]:
            notes.append("%s: %d of %d questions have no finished record%s" % (
                key, s["n_total"] - s["n"], s["n_total"],
                " (%d failed; rerun to retry)" % s["n_failed"] if s["n_failed"] else ""))
        if s.get("n_judge_failed"):
            notes.append("%s: %d answers got no valid judge verdict; left out of Acc -- rerun to "
                         "retry them" % (key, s["n_judge_failed"]))
        if s.get("n_not_graded"):
            notes.append("%s: %d answers not graded yet" % (key, s["n_not_graded"]))
    L += ["  !! " + n for n in notes]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="dlsbench.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", type=parse_tasks)
    ap.add_argument("--harness", required=True, type=str.lower, choices=HARNESS_NAMES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=envint("CONCURRENCY", 4))
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--bench-dir", default=os.path.join(HERE, "bench"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    model_name = gem.AGENT_MODEL
    if not model_name:
        sys.exit("MODEL_NAME is not set (see .env.example); it also names the output files")
    label = "%s__%s" % (safe_name(model_name), args.harness)
    run_dir = os.path.join(args.out_dir, "runs", label)
    out = os.path.join(args.out_dir, "results", label + ".json")

    bench = {}
    for t in args.task:
        items = load_task(args.bench_dir, t)
        bench[t] = items[:args.limit] if args.limit else items

    lock = threading.Lock()
    abort = threading.Event()
    abort_reason = []

    # ------------------------------------------------------------ run
    if not args.grade_only:
        if not gem.BASE_URL:
            sys.exit("MODEL_BASE_URL is not set (see .env.example)")
        try:
            tools = tl.make_tools(args.harness)
        except tl.ToolConfigError as e:
            sys.exit(str(e))
        todo = [(t, it) for t in args.task for it in bench[t]
                if not is_done(read_json(record_path(run_dir, t, it["instance_id"])), it)]
        n_all = sum(len(v) for v in bench.values())
        print("model    %s  ->  %s" % (model_name, gem.BASE_URL))
        print("harness  %s  (%s)" % (args.harness, "Serper + Jina" if args.harness == "web"
                                     else "Semantic Scholar + Jina"))
        print("tasks    %s  (%d questions, %d already done, %d to run)"
              % (", ".join("T%d" % t for t in args.task), n_all, n_all - len(todo), len(todo)))
        print("records  %s" % run_dir)

        t0 = time.time()
        n_done = [0]

        def process(t, it):
            if abort.is_set():
                return None
            res = run_episode(t, it["query"], args.harness, tools, verbose=args.verbose)
            rec = {"instance_id": it["instance_id"], "task": t, "task_name": TASK_NAMES[t],
                   "harness": args.harness, "model": model_name, "question": it["query"]}
            rec.update(res)
            err = rec.get("failed") or ""
            if err.startswith(("ToolConfigError", "EndpointError")):
                with lock:
                    if not abort.is_set():
                        abort.set()
                        abort_reason.append(err)
            write_json(record_path(run_dir, t, it["instance_id"]), rec)
            return rec

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(process, t, it) for t, it in todo]
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    print("  question crashed: %s" % str(e)[:200], flush=True)
                    continue
                if rec is None:
                    continue
                with lock:
                    n_done[0] += 1
                    if n_done[0] % 5 == 0 or n_done[0] == len(todo):
                        print("  %d/%d  %.0fs" % (n_done[0], len(todo), time.time() - t0),
                              flush=True)
        if abort_reason:
            print("\nSTOPPED: %s" % abort_reason[0][:400], file=sys.stderr)

    # ------------------------------------------------------------ grade
    loaded = {t: [] for t in args.task}          # (inst, rec or None, path)
    for t in args.task:
        for it in bench[t]:
            p = record_path(run_dir, t, it["instance_id"])
            rec = read_json(p)
            if rec and rec.get("question") != it["query"]:
                rec = None                         # a record for a different version of the question
            if rec and not rec.get("failed"):
                rec["eval"] = deterministic_eval(t, it, rec)
            loaded[t].append((it, rec, p))

    to_judge = [(t, it, rec, p) for t in args.task for it, rec, p in loaded[t]
                if rec and needs_judge(t, rec)]
    if to_judge and not args.no_grade:
        if not judge.configured():
            print("\nskipping grading: no JUDGE_API_KEY / OPENAI_API_KEY set. Recall and NDCG "
                  "below are complete; Acc needs the judge.", file=sys.stderr)
        else:
            print("\ngrading %d answers (%s)..." % (len(to_judge), judge.describe()), flush=True)
            jabort = threading.Event()
            n_graded = [0]

            def grade(t, it, rec):
                if jabort.is_set():
                    return None
                try:
                    return judge.judge_answer(t, it["query"], it["ground_truth"], rec.get("answer"),
                                              rec.get("final_text") or "")
                except judge.JudgeError as e:
                    if not jabort.is_set():
                        jabort.set()
                        print("judge stopped: %s" % e, file=sys.stderr)
                    return None

            with ThreadPoolExecutor(max_workers=max(4, args.concurrency)) as ex:
                futs = {ex.submit(grade, t, it, rec): rec for t, it, rec, _p in to_judge}
                for fut in as_completed(futs):
                    v = fut.result()
                    if v is None:
                        continue
                    rec = futs[fut]
                    with lock:
                        rec["eval"]["judge"] = v
                        n_graded[0] += 1
                        if n_graded[0] % 25 == 0:
                            print("  graded %d/%d" % (n_graded[0], len(to_judge)), flush=True)

    for t in args.task:
        for _it, rec, p in loaded[t]:
            if rec and not rec.get("failed"):
                write_json(p, rec)

    # ------------------------------------------------------------ summarize
    prev = read_json(out) or {}
    summary = dict(prev.get("tasks") or {})
    for t in args.task:
        recs = [rec for _it, rec, _p in loaded[t] if rec]
        summary["T%d" % t] = metrics.summarize_task(t, recs, n_total=len(bench[t]))
        summary["T%d" % t]["name"] = TASK_NAMES[t]
        summary["T%d" % t]["limit"] = args.limit or None
    result = {"model": model_name, "harness": args.harness, "judge_model": judge.JUDGE_MODEL,
              "budgets": {"searches": MAX_SEARCHES, "visits": MAX_VISITS, "turns": MAX_TURNS},
              "tasks": {k: summary[k] for k in sorted(summary)}}
    write_json(out, result)
    print(format_table("%s / %s" % (model_name, args.harness), result["tasks"]))
    print("\nsummary  %s\nrecords  %s" % (out, run_dir))
    return 1 if abort_reason else 0


if __name__ == "__main__":
    sys.exit(main())
