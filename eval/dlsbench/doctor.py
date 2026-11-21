"""Check your setup before spending a full run.

    python -m dlsbench.doctor [--harness web|s2]     # default web

Makes a handful of calls and says specifically what is wrong. The failure it catches most often is
a model or server without native tool calling: that does not crash, the model just answers from
memory without searching and looks weak.
"""
import argparse
import json
import sys

from . import judge
from . import model as gem
from . import tools as tl
from .harness import HARNESSES

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def _p(tag, name, detail=""):
    print("[%s] %-26s %s" % (tag, name, detail))


def check_model(harness, fails):
    if not gem.AGENT_MODEL:
        _p(BAD, "MODEL_NAME", "not set")
        fails.append("MODEL_NAME")
    else:
        _p(OK, "MODEL_NAME", gem.AGENT_MODEL)
    if not gem.BASE_URL:
        _p(BAD, "MODEL_BASE_URL", "not set")
        fails.append("MODEL_BASE_URL")
    else:
        _p(OK, "MODEL_BASE_URL", gem.BASE_URL)
    if fails:
        return
    try:
        r = gem.mcall([{"role": "user", "content": "Reply with exactly: OK"}], transient_retries=2)
        _p(OK, "model responds", repr(r["text"][:40]))
    except Exception as e:
        _p(BAD, "model responds", str(e)[:200])
        fails.append("model call")
        return
    search = HARNESSES[harness]["tools"][0]
    try:
        r = gem.mcall([{"role": "user", "content":
                        "Use the %s tool to look up the paper 'Attention Is All You Need'. "
                        "Call the tool; do not answer from memory." % search["name"]}],
                      [search], transient_retries=2)
        calls = r["tool_calls"]
        if calls and calls[0]["name"] == search["name"]:
            try:
                args = json.loads(calls[0]["arguments"] or "{}")
            except ValueError:
                args = None
            if isinstance(args, dict) and args.get("query"):
                _p(OK, "native tool calling", "%s(%r)" % (calls[0]["name"], args["query"][:50]))
            else:
                _p(WARN, "native tool calling", "called the tool with arguments %r"
                   % calls[0]["arguments"][:80])
        else:
            _p(BAD, "native tool calling",
               "no tool call came back (text: %r). The harness needs OpenAI-style `tool_calls`; "
               "for vLLM start the server with --enable-auto-tool-choice and a --tool-call-parser."
               % r["text"][:80])
            fails.append("tool calling")
    except Exception as e:
        _p(BAD, "native tool calling", str(e)[:200])
        fails.append("tool calling")


def check_web(fails):
    if not tl.SERPER_API_KEY:
        _p(BAD, "SERPER_API_KEY", "not set; get one at serper.dev")
        fails.append("SERPER_API_KEY")
    else:
        try:
            out = tl.google_search("attention is all you need transformer")
            if out.startswith("Title:"):
                _p(OK, "google_search (Serper)", "%d results" % out.count("Title:"))
            else:
                _p(BAD, "google_search (Serper)", out[:150])
                fails.append("serper")
        except tl.ToolConfigError as e:
            _p(BAD, "google_search (Serper)", str(e)[:200])
            fails.append("serper")
    if not tl.JINA_API_KEY:
        _p(WARN, "JINA_API_KEY", "not set; visit_page works without it but is rate-limited")
    out = tl.visit_page("https://arxiv.org/abs/1706.03762")
    if out.startswith("[ERROR]") or len(out) < 200:
        _p(BAD, "visit_page (Jina)", out[:150])
        fails.append("jina")
    else:
        _p(OK, "visit_page (Jina)", "%d chars" % len(out))


def check_s2(fails):
    if not tl.S2_API_KEY:
        _p(WARN, "S2_API_KEY", "not set; the shared pool is heavily rate-limited (expect 429 waits)")
    else:
        _p(OK, "S2_API_KEY", "set (S2_RPS=%s)" % tl.S2_RPS)
    if not tl.JINA_API_KEY:
        _p(WARN, "JINA_API_KEY", "not set; full-text fetch works without it but is rate-limited")
    try:
        client = tl.S2Client()
        hits = tl.s2_search(client, "attention is all you need", 3)
        if hits and not hits[0].get("_error"):
            _p(OK, "s2_search", "%d hits, first %r" % (len(hits), (hits[0].get("title") or "")[:50]))
            d = tl.s2_paper(client, hits[0]["paperId"])
            if d.get("_error"):
                _p(BAD, "s2_paper", str(d["_error"])[:150])
                fails.append("s2_paper")
            else:
                _p(OK, "s2_paper", "body_text %d chars (%s)" % (len(d.get("body_text") or ""),
                                                                 d.get("body_source") or "none"))
        else:
            _p(BAD, "s2_search", str(hits[:1])[:150])
            fails.append("s2_search")
    except tl.ToolConfigError as e:
        _p(BAD, "s2_search", str(e)[:200])
        fails.append("S2_API_KEY")


def check_judge(fails):
    if not judge.configured():
        _p(WARN, "judge", "no JUDGE_API_KEY / OPENAI_API_KEY; you can run, but Acc stays empty")
        return
    try:
        gt = {"answer": {"benchmark": "SQuAD", "split_or_version": "v1.1 dev"}}
        v = judge.call_judge(judge.build_judge_prompt(
            1, "Which benchmark should I use for extractive QA?", gt,
            "Use SQuAD v1.1 (dev set)."))
        if v["status"] == "ok":
            _p(OK, "judge", "%s -> score %s" % (judge.describe(), v["score"]))
        else:
            _p(BAD, "judge", "%s: %s" % (judge.describe(), v["reason"][:150]))
            fails.append("judge")
    except judge.JudgeError as e:
        _p(BAD, "judge", str(e)[:200])
        fails.append("judge")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dlsbench.doctor")
    ap.add_argument("--harness", default="web", type=str.lower, choices=("web", "s2"))
    args = ap.parse_args(argv)
    fails = []
    print("\n=== harness: %s ===" % args.harness)
    check_model(args.harness, fails)
    (check_web if args.harness == "web" else check_s2)(fails)
    check_judge(fails)
    print()
    if fails:
        print("NOT READY: %s" % ", ".join(fails))
        return 1
    print("READY. Smoke-test a few questions per task, then run everything:")
    print("  python -m dlsbench.eval --task all --harness %s --limit 3" % args.harness)
    print("  python -m dlsbench.eval --task all --harness %s" % args.harness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
