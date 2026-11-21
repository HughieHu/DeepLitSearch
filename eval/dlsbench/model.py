"""THE place you change to evaluate your own model.

Any endpoint that speaks the OpenAI chat-completions API with native tool (function) calling works,
hosted or local, without a code edit:

    export MODEL_NAME=gpt-5.4-mini
    export MODEL_BASE_URL=https://api.openai.com/v1
    export MODEL_API_KEY=...

    # a local vLLM / SGLang server (start it with tool calling enabled):
    export MODEL_NAME=Qwen/Qwen3-32B
    export MODEL_BASE_URL=http://localhost:8000/v1
    export MODEL_API_KEY=EMPTY

The harness uses provider-native function calling, as the paper's runs did: the model receives a
`tools` list and answers with `tool_calls`. A model or server without tool calling cannot run it.

If your model is not OpenAI-compatible, subclass `Client` and override `complete()`; that is the
only method that touches the wire. It takes OpenAI-style chat messages plus the tool schemas and
returns {"text": str, "tool_calls": [{"id", "name", "arguments"}], "usage": {...}}.

----------------------------------------------------------------------------------------------
The harness uses only `mcall`, `ContextOverflow` and `EndpointError` from this module (eval.py and
doctor.py also read AGENT_MODEL and BASE_URL). Keep that surface if you replace it.
"""
import json
import threading
import time

from . import TransportError, envint, envopt, http_request

AGENT_MODEL = envopt("MODEL_NAME") or ""
BASE_URL = (envopt("MODEL_BASE_URL") or "").rstrip("/")
API_KEY = envopt("MODEL_API_KEY") or "EMPTY"
TIMEOUT = envint("MODEL_TIMEOUT", 300)

# Optional request fields, sent ONLY when set. The paper's runs sent no temperature and no output
# cap for the OpenAI and Azure models; models served through Together (DeepSeek-V4-Pro) got
# max_tokens=16000, because that provider's small default cut off the final answer. The GPT-5.4
# family ran with reasoning effort "high".
MAX_TOKENS = envint("MODEL_MAX_TOKENS", 0) or None
_t = envopt("MODEL_TEMPERATURE")
TEMPERATURE = float(_t) if _t is not None else None
REASONING_EFFORT = envopt("MODEL_REASONING_EFFORT")


class ContextOverflow(RuntimeError):
    """The conversation no longer fits the model's context window."""


class EndpointError(RuntimeError):
    """A permanent rejection (bad key, unknown model, malformed request). Retrying cannot help."""


_OVERFLOW_MARKERS = ("context_length_exceeded", "maximum context length", "context length is",
                     "prompt is too long", "too many tokens")


def _tool_schema(t):
    return {"type": "function", "function": t}


def _parse_choice(d):
    """Chat-completions response -> {"text", "tool_calls", "usage"}.

    Some servers return the model's reasoning in a separate `reasoning` field next to `content`.
    As in the paper's client, the two are concatenated so a final answer that ended up in either
    one is still found (the answer parser takes the LAST tagged block).
    """
    choices = d.get("choices") or []
    msg = (choices[0].get("message") or {}) if choices else {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if args is None:
            args = "{}"
        elif not isinstance(args, str):
            args = json.dumps(args)
        calls.append({"id": tc.get("id") or "", "name": fn.get("name") or "", "arguments": args})
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    text = (reasoning + "\n\n" + content).strip() if reasoning else content
    return {"text": text, "tool_calls": calls, "usage": d.get("usage") or {}}


class Client:
    """Override `complete()` for a non-OpenAI-compatible model; nothing else talks to the wire."""

    def __init__(self, model=None, base_url=None, api_key=None):
        self.model = model or AGENT_MODEL
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.api_key = api_key or API_KEY
        # "OpenAI-compatible" is a family, not a spec: reasoning models renamed `max_tokens` and
        # reject `temperature`; many servers do not know `reasoning_effort`. Learn each quirk once
        # from the 400 and carry on, instead of failing question 1.
        self.quirks = set()
        self._qlock = threading.Lock()
        if not self.model:
            raise EndpointError("MODEL_NAME is not set (see .env.example)")
        if not self.base_url:
            raise EndpointError("MODEL_BASE_URL is not set (see .env.example)")

    def _body(self, messages, tools):
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = [_tool_schema(t) for t in tools]
            body["tool_choice"] = "auto"
        if MAX_TOKENS:
            key = "max_completion_tokens" if "no_max_tokens" in self.quirks else "max_tokens"
            body[key] = MAX_TOKENS
        if TEMPERATURE is not None and "no_temperature" not in self.quirks:
            body["temperature"] = TEMPERATURE
        if REASONING_EFFORT and "no_reasoning_effort" not in self.quirks:
            body["reasoning_effort"] = REASONING_EFFORT
        return body

    def _learn(self, detail):
        """True if this 400 names a parameter we can drop or rename, so the call should be retried.

        A quirk that is ALREADY recorded still means retry: another thread learned it after this
        request was built, so the request we just sent was stale.
        """
        d = (detail or "").lower()
        if "reasoning_effort" in d or "reasoning effort" in d:
            q = "no_reasoning_effort"
        elif "max_tokens" in d and MAX_TOKENS:
            q = "no_max_tokens"
        elif "temperature" in d and TEMPERATURE is not None:
            q = "no_temperature"
        else:
            return False
        with self._qlock:
            fresh = q not in self.quirks
            self.quirks.add(q)
        if fresh:
            print("[model] endpoint rejected a parameter; adapting (%s)" % q, flush=True)
        return True

    def complete(self, messages, tools=None):
        """One chat-completions call. `tools` is a list of {"name", "description", "parameters"}."""
        for _ in range(4):
            status, text, _h = http_request(
                self.base_url + "/chat/completions", method="POST",
                body=json.dumps(self._body(messages, tools)),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key},
                timeout=TIMEOUT)
            if status == 200:
                try:
                    return _parse_choice(json.loads(text))
                except ValueError:
                    raise RuntimeError("model endpoint returned non-JSON: %r" % text[:200])
            low = text.lower()
            if status in (400, 413) and any(m in low for m in _OVERFLOW_MARKERS):
                raise ContextOverflow(text[:300])
            if status == 400 and self._learn(text):
                continue
            if status in (429,) or status >= 500:
                raise RuntimeError("HTTP %s %s" % (status, text[:300]))
            raise EndpointError(
                "model endpoint rejected the request (HTTP %s %s).\n"
                "  MODEL_NAME=%r MODEL_BASE_URL=%r\n"
                "  Check the name is one this endpoint serves, the key is valid, and the server "
                "has tool calling enabled." % (status, text[:300], self.model, self.base_url))
        raise EndpointError("endpoint kept rejecting the request body (quirks tried: %s)"
                            % sorted(self.quirks))


_client = None
_client_lock = threading.Lock()


def client():
    global _client
    with _client_lock:
        if _client is None:
            _client = Client()
    return _client


def mcall(messages, tools=None, transient_retries=12):
    """One model turn -> {"text", "tool_calls", "usage"}.

    Retries transient failures (429, 5xx, timeouts, resets) with backoff, as the paper's client
    did. Raises EndpointError at once on a permanent rejection and ContextOverflow when the
    conversation no longer fits; both are handled by the harness.
    """
    last = None
    delay = 2.0
    for attempt in range(transient_retries):
        try:
            return client().complete(messages, tools)
        except (EndpointError, ContextOverflow):
            raise
        except (TransportError, RuntimeError) as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:300])
        if attempt < transient_retries - 1:
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("model call failed after %d attempts: %s" % (transient_retries, last))
