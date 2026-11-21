"""Offline stand-ins for every endpoint the client talks to, behind one local HTTP server:

    POST /v1/chat/completions        the model (OpenAI chat-completions with native tool calls)
    POST /judge/v1/responses         the judge (OpenAI Responses API)
    POST /judge/v1/chat/completions  the judge over chat-completions (JUDGE_API=chat)
    POST /serper                     Serper.dev search
    GET  /jina/<url>                 Jina Reader
    GET  /s2/paper/search, /s2/paper/<id>   Semantic Scholar Graph API

No network, keys or GPU needed. The mock model plays one fixed episode per question: search, then
visit, then a reply without the final tags (which must trigger the harness's force-final prompt),
then the final answer listing the question's gold papers in ideal order. It rejects a tool message
whose id does not match an earlier tool call, so the wire format is checked on every turn.
"""
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse


def _read(path):
    with open(path) as f:
        return json.load(f)


BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "bench")


def _load_bench():
    by_query = {}
    for t in range(1, 6):
        p = os.path.join(BENCH_DIR, "task%d.json" % t)
        for it in _read(p):
            by_query[it["query"].strip()] = (t, it)
    return by_query


BY_QUERY = _load_bench()
SERPER_KEY = "test-serper-key"
JUDGE_KEY = "test-judge-key"


def gold_titles(inst):
    out = []
    for e in inst["ground_truth"].get("retrieve") or []:
        if isinstance(e, str):
            out.append((e, 1))
        else:
            out.append((e["title"], e.get("weight", 1)))
    return [t for t, _w in sorted(out, key=lambda x: -x[1])]


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.strict = False                 # reject reasoning_effort and max_tokens like some APIs
        self.judge_mode = "ok"              # ok | garbage | unauthorized
        self.counts = {"model": 0, "judge": 0, "serper": 0, "jina": 0, "s2": 0}
        self.judge_log = []                 # one dict per judge request
        self.model_bodies = []

    def bump(self, k):
        with self.lock:
            self.counts[k] += 1


def _final_text(task, inst):
    sel = "\n".join("- %s (https://example.org/paper/%d)" % (t, i)
                    for i, t in enumerate(gold_titles(inst)))
    return ("<selected_papers>\n%s\n</selected_papers>\n\n<answer>\nMOCK-OK answer for %s\n</answer>"
            % (sel, inst["instance_id"]))


def model_reply(req, state):
    """Returns (status, json body)."""
    if state.strict:
        if "reasoning_effort" in req:
            return 400, {"error": {"message": "Unrecognized request argument supplied: reasoning_effort",
                                   "type": "invalid_request_error"}}
        if "max_tokens" in req:
            return 400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported "
                                              "with this model. Use 'max_completion_tokens' instead.",
                                   "type": "invalid_request_error"}}
    msgs = req.get("messages") or []
    tools = [t["function"]["name"] for t in (req.get("tools") or [])]
    ids = set()
    for m in msgs:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                ids.add(tc["id"])
        if m["role"] == "tool" and m.get("tool_call_id") not in ids:
            return 400, {"error": {"message": "tool message without a matching tool call"}}

    first = msgs[0]["content"] if msgs and isinstance(msgs[0].get("content"), str) else ""
    q = first.split("=== QUESTION ===\n", 1)[1].strip() if "=== QUESTION ===\n" in first else None
    task, inst = BY_QUERY.get(q, (None, None)) if q else (None, None)
    n_tool = sum(1 for m in msgs if m["role"] == "tool")
    last = msgs[-1] if msgs else {}

    msg = {"role": "assistant", "content": None}
    if inst is None:                                       # doctor-style probes
        if tools:
            msg["tool_calls"] = [{"id": "call_probe", "type": "function", "function": {
                "name": tools[0], "arguments": json.dumps({"query": "attention is all you need"})}}]
        else:
            msg["content"] = "OK"
    elif last.get("role") == "user" and "Stop using tools." in (last.get("content") or ""):
        msg["content"] = _final_text(task, inst)
        # a reasoning field with a draft of the tagged blocks: the parser must take the LAST ones
        msg["reasoning"] = ("Draft: <selected_papers>\n- A wrong paper\n</selected_papers>"
                            "<answer>draft</answer>")
    elif n_tool == 0 and tools:
        msg["tool_calls"] = [{"id": "call_s_%s" % inst["instance_id"], "type": "function",
                              "function": {"name": tools[0], "arguments": json.dumps(
                                  {"query": "QID:%s" % inst["instance_id"]})}}]
    elif n_tool == 1 and len(tools) > 1:
        args = ({"url": "https://example.org/%s" % inst["instance_id"]} if tools[1] == "visit_page"
                else {"paper_id": "PID-%s" % inst["instance_id"]})
        msg["tool_calls"] = [{"id": "call_v_%s" % inst["instance_id"], "type": "function",
                              "function": {"name": tools[1], "arguments": json.dumps(args)}}]
    else:
        msg["content"] = "I have enough information now."     # no tags: harness must force-final
    usage = {"prompt_tokens": 1000 + 100 * len(msgs), "completion_tokens": 50,
             "completion_tokens_details": {"reasoning_tokens": 10}}
    return 200, {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}


def judge_reply(req, state, api):
    if api == "responses":
        inp = req["input"]
        prompt = inp if isinstance(inp, str) else inp[0]["content"]
    else:
        prompt = req["messages"][0]["content"]
    with state.lock:
        state.judge_log.append({
            "header": prompt.splitlines()[0],
            "effort": (req.get("reasoning") or {}).get("effort") or req.get("reasoning_effort"),
            "max_tokens": req.get("max_output_tokens") or req.get("max_completion_tokens"),
            "input_is_message": api == "responses" and not isinstance(req["input"], str)})
    if state.judge_mode == "unauthorized":
        return 401, {"error": {"message": "Incorrect API key provided"}}
    if state.judge_mode == "garbage":
        text = "I cannot decide on this one."
    else:
        if "MODEL'S ANSWER:" in prompt:                     # T1-T3 prompts
            answer = prompt.split("MODEL'S ANSWER:", 1)[1]
        else:                                               # T4 prompt
            answer = prompt.split("MODEL ANSWER:", 1)[1].split("EXPECTED ANSWER:", 1)[0]
        ok = "MOCK-OK" in answer
        text = '{"score": %d, "reason": "mock verdict"}' % (1 if ok else 0)
    if api == "responses":
        return 200, {"output": [{"type": "reasoning", "summary": []},
                                {"type": "message", "content": [{"type": "output_text", "text": text}]}]}
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}}]}


def s2_paper_record(pid, base):
    return {"paperId": pid, "title": "Mock paper %s" % pid, "year": 2024, "venue": "MockConf",
            "authors": [{"name": "Author %d" % i} for i in range(7)],
            "externalIds": {"ArXiv": "2401.00001", "CorpusId": 1},
            "abstract": "a" * 3000, "tldr": {"text": "short summary"},
            "citationCount": 5, "influentialCitationCount": 1,
            "openAccessPdf": {"url": "%s/pdf/%s" % (base, pid)}, "fieldsOfStudy": ["Computer Science"],
            "references": [{"title": "Ref %d" % i, "year": 2020} for i in range(20)],
            "citations": [{"title": "Cit %d" % i, "year": 2025} for i in range(20)]}


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else (
                json.dumps(obj).encode() if ctype == "application/json" else obj.encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_POST(self):
            path = urlparse(self.path).path
            req = self._json()
            if path == "/v1/chat/completions":
                state.bump("model")
                with state.lock:
                    state.model_bodies.append({k: v for k, v in req.items() if k != "messages"})
                return self._send(*model_reply(req, state))
            if path in ("/judge/v1/responses", "/judge/v1/chat/completions"):
                state.bump("judge")
                if self.headers.get("Authorization") != "Bearer " + JUDGE_KEY:
                    return self._send(401, {"error": {"message": "bad key"}})
                return self._send(*judge_reply(req, state, "responses" if path.endswith("responses") else "chat"))
            if path == "/serper":
                state.bump("serper")
                if self.headers.get("X-API-KEY") != SERPER_KEY:
                    return self._send(403, {"message": "Unauthorized.", "statusCode": 403})
                q = req.get("q", "")
                return self._send(200, {"organic": [
                    {"title": "Result %d for %s" % (i, q), "link": "https://example.org/r%d" % i,
                     "snippet": "snippet %d" % i} for i in range(12)]})
            self._send(404, {"error": "no route"})

        def do_GET(self):
            u = urlparse(self.path)
            base = "http://%s:%d" % self.server.server_address[:2]
            if u.path.startswith("/jina/"):
                state.bump("jina")
                target = unquote(self.path[len("/jina/"):])
                return self._send(200, "Title: page\nURL Source: %s\n\n%sEND" % (target, "w" * 20000),
                                  "text/plain")
            if u.path == "/s2/paper/search":
                state.bump("s2")
                qs = parse_qs(u.query)
                limit = int(qs.get("limit", ["10"])[0])
                q = qs.get("query", [""])[0]
                m = re.search(r"QID:(\S+)", q)
                pid = "PID-%s" % (m.group(1) if m else "x")
                data = [s2_paper_record(pid if i == 0 else "%s-%d" % (pid, i), base) for i in range(min(limit, 12))]
                return self._send(200, {"total": len(data), "data": data})
            if u.path.startswith("/s2/paper/"):
                state.bump("s2")
                pid = unquote(u.path[len("/s2/paper/"):])
                if pid.startswith("NOTFOUND"):
                    return self._send(404, {"error": "Paper not found"})
                return self._send(200, s2_paper_record(pid, base))
            self._send(404, {"error": "no route"})

        def log_message(self, *a):
            pass
    return H


class Mock:
    def __init__(self):
        self.state = State()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def env(self, **extra):
        """Environment for a subprocess run against this mock. Every setting is given explicitly
        so a developer's own .env can never leak real endpoints or keys into a test."""
        e = dict(os.environ)
        e.update({
            "MODEL_NAME": "mock-model", "MODEL_BASE_URL": self.url + "/v1", "MODEL_API_KEY": "EMPTY",
            "MODEL_MAX_TOKENS": "", "MODEL_TEMPERATURE": "", "MODEL_REASONING_EFFORT": "",
            "SERPER_URL": self.url + "/serper", "SERPER_API_KEY": SERPER_KEY,
            "JINA_URL": self.url + "/jina", "JINA_API_KEY": "",
            "S2_BASE_URL": self.url + "/s2", "S2_API_KEY": "", "S2_RPS": "1000",
            "JUDGE_BASE_URL": self.url + "/judge/v1", "JUDGE_API_KEY": JUDGE_KEY,
            "JUDGE_MODEL": "", "JUDGE_API": "", "OPENAI_API_KEY": "", "CONCURRENCY": "",
        })
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
