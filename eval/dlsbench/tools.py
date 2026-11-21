"""The retrieval tools behind the two harnesses.

    Web-ReAct   google_search(query)       Serper.dev, top-10 results as title + url + snippet
                visit_page(url)            Jina Reader (r.jina.ai), first 8,000 characters
    S2-ReAct    s2_search(query, k=10)     Semantic Scholar Graph API /paper/search
                s2_paper(paper_id)         Semantic Scholar /paper/{id}, plus the open-access full
                                           text through Jina Reader (first 12,000 characters)

Output formats, truncation limits and the S2 field lists are the ones used for the paper. Each tool
returns the string the model sees. A failed call comes back to the model as an error string, as in
the paper's runs, except for a configuration error (missing or rejected key), which raises
ToolConfigError and stops the run: a run in which every search fails would look like a weak model.

Every endpoint can be overridden (SERPER_URL, JINA_URL, S2_BASE_URL); the tests point them at
local mock servers.
"""
import json
import threading
import time
from urllib.parse import quote, urlencode

from . import TransportError, envfloat, envopt, http_request

SERPER_URL = envopt("SERPER_URL") or "https://google.serper.dev/search"
SERPER_API_KEY = envopt("SERPER_API_KEY") or ""
JINA_URL = (envopt("JINA_URL") or "https://r.jina.ai").rstrip("/")
JINA_API_KEY = envopt("JINA_API_KEY") or ""          # optional; Jina works without one, more slowly
S2_BASE_URL = (envopt("S2_BASE_URL") or "https://api.semanticscholar.org/graph/v1").rstrip("/")
S2_API_KEY = envopt("S2_API_KEY") or ""              # optional; the shared pool is heavily rate-limited
S2_RPS = envfloat("S2_RPS", 1.0)                     # global S2 request rate across all threads

VISIT_TRUNCATE = 8000          # visit_page: characters of Jina output shown to the model
JINA_MAX_CHARS = 12000         # s2_paper: characters of full text kept in `body_text`
S2_SEARCH_TRUNCATE = 8000      # s2_search: characters of the JSON result shown to the model
S2_PAPER_TRUNCATE = 18000      # s2_paper: characters of the JSON result shown to the model
SERPER_TIMEOUT = 30
VISIT_TIMEOUT = 30
S2_JINA_TIMEOUT = 60
S2_TIMEOUT = 60
S2_MAX_RETRIES = 6


class ToolConfigError(RuntimeError):
    """A search backend rejected our credentials, or none were given. Stops the run."""


def _retrying(fn, attempts=3):
    """Call fn() and retry transport errors and 429/5xx a few times. Returns (status, text)."""
    last = None
    for i in range(attempts):
        try:
            status, text, _h = fn()
        except TransportError as e:
            last = e
            status, text = None, ""
        if status is not None and status != 429 and status < 500:
            return status, text
        if i < attempts - 1:
            time.sleep(2.0 * (i + 1))
    if status is None:
        raise last
    return status, text


# ---------------------------------------------------------------- Web-ReAct
def google_search(query):
    """Serper.dev search -> top-10 results, one 'Title/URL/Snippet' block each."""
    if not SERPER_API_KEY:
        raise ToolConfigError("SERPER_API_KEY is not set; the web harness needs it (serper.dev)")
    body = json.dumps({"q": (query or "")[:200], "num": 10})
    try:
        status, text = _retrying(lambda: http_request(
            SERPER_URL, method="POST", body=body,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            timeout=SERPER_TIMEOUT))
    except TransportError as e:
        return "[ERROR] %s" % e
    if status in (401, 403) or (status in (400, 402) and "credit" in text.lower()):
        raise ToolConfigError("Serper rejected the request (HTTP %s %s); check SERPER_API_KEY"
                              % (status, text[:200]))
    if status != 200:
        return "[ERROR] HTTP %s %s" % (status, text[:200])
    try:
        data = json.loads(text)
    except ValueError as e:
        return "[ERROR] %s" % e
    results = []
    for r in (data.get("organic") or [])[:10]:
        results.append("Title: %s\nURL: %s\nSnippet: %s"
                       % (r.get("title", ""), r.get("link", ""), r.get("snippet", "")))
    return "\n\n".join(results) if results else "(no results)"


def _reader_headers():
    h = {"Accept": "text/plain"}
    if JINA_API_KEY:
        h["Authorization"] = "Bearer " + JINA_API_KEY
    return h


def visit_page(url):
    """Jina Reader -> the page as text, cut to VISIT_TRUNCATE characters.

    As in the paper's runs, whatever Jina returns is passed through, including its error messages
    for pages it cannot read; only rate limits and server errors are retried.
    """
    try:
        status, text = _retrying(lambda: http_request(
            "%s/%s" % (JINA_URL, url), headers=_reader_headers(), timeout=VISIT_TIMEOUT))
    except TransportError as e:
        return "[ERROR] %s" % e
    if status in (401, 403) and JINA_API_KEY and "api key" in text.lower():
        raise ToolConfigError("Jina rejected JINA_API_KEY (HTTP %s %s)" % (status, text[:200]))
    return text[:VISIT_TRUNCATE] if len(text) > VISIT_TRUNCATE else text


def fetch_via_jina(url, max_chars=JINA_MAX_CHARS):
    """Full text of a PDF/HTML URL through Jina, for s2_paper. Empty string on any failure."""
    if not url or not url.strip():
        return ""
    try:
        status, text = _retrying(lambda: http_request(
            "%s/%s" % (JINA_URL, url.strip()), headers=_reader_headers(), timeout=S2_JINA_TIMEOUT))
    except TransportError:
        return ""
    text = text or ""
    return text[:max_chars] if len(text) > max_chars else text


# ---------------------------------------------------------------- Semantic Scholar
class TokenBucket:
    """Thread-safe token bucket: at most `rps` requests per second across all threads."""

    def __init__(self, rps):
        self.rps = max(float(rps), 0.01)
        self.capacity = max(1.0, self.rps)
        self.tokens = self.capacity
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rps)
            self.last = now
            if self.tokens < 1.0:
                time.sleep((1.0 - self.tokens) / self.rps)
                self.tokens = 0.0
                self.last = time.monotonic()
            else:
                self.tokens -= 1.0


SEARCH_FIELDS = ("title,abstract,tldr,year,venue,authors.name,externalIds,openAccessPdf,"
                 "citationCount,influentialCitationCount")
PAPER_FIELDS = ("title,abstract,tldr,year,venue,authors.name,externalIds,openAccessPdf,"
                "citationCount,influentialCitationCount,references.title,references.year,"
                "references.externalIds,citations.title,citations.year,citations.externalIds,"
                "fieldsOfStudy")


class S2Client:
    def __init__(self, api_key=None, rps=None, base_url=None):
        self.api_key = S2_API_KEY if api_key is None else api_key
        self.base_url = (base_url or S2_BASE_URL).rstrip("/")
        self._bucket = TokenBucket(S2_RPS if rps is None else rps)

    def _get(self, path, params=None):
        url = self.base_url + path + ("?" + urlencode(params) if params else "")
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        last = None
        for attempt in range(S2_MAX_RETRIES):
            self._bucket.acquire()
            try:
                status, text, h = http_request(url, headers=headers, timeout=S2_TIMEOUT)
            except TransportError as e:
                last = e
                time.sleep(min(2 ** attempt, 30))
                continue
            if status == 429 or status >= 500:
                last = "HTTP %s" % status
                ra = (h.get("retry-after") if h else None) or ""
                try:
                    wait = float(ra) if ra else min(2 ** attempt, 30)
                except ValueError:
                    wait = min(2 ** attempt, 30)
                time.sleep(max(wait, 2.0))
                continue
            if status == 404:
                return {"_status": 404, "_error": "not_found", "_path": path}
            if status in (401, 403) and self.api_key:
                raise ToolConfigError("Semantic Scholar rejected S2_API_KEY (HTTP %s %s)"
                                      % (status, text[:200]))
            if status != 200:
                return {"_error": "http_%s: %s" % (status, text[:200]), "_path": path}
            try:
                return json.loads(text)
            except ValueError as e:
                last = e
        return {"_error": "exhausted retries: %s" % last, "_path": path}

    def search(self, query, k=10):
        if not query or not query.strip():
            return {"data": [], "_error": "empty_query"}
        params = {"query": query[:512], "limit": min(max(k, 1), 100), "fields": SEARCH_FIELDS}
        return self._get("/paper/search", params)

    def paper(self, paper_id):
        """paper_id: S2 paperId, or a tagged id (DOI:, ARXIV:, CorpusId:, URL:). The DOI slash
        is escaped, the tag's colon is kept."""
        if not paper_id:
            return {"_error": "empty_paper_id"}
        return self._get("/paper/" + quote(paper_id.strip(), safe=":"), {"fields": PAPER_FIELDS})


MAX_AUTHORS_PER_PAPER = 5
MAX_REFERENCES_RETURNED = 15
MAX_CITATIONS_RETURNED = 15
ABSTRACT_TRUNC = 1200
TLDR_TRUNC = 350


def _format_external_ids(ext):
    if not ext:
        return ""
    bits = []
    if ext.get("ArXiv"):
        bits.append("arXiv:%s" % ext["ArXiv"])
    if ext.get("DOI"):
        bits.append("DOI:%s" % ext["DOI"])
    if ext.get("CorpusId"):
        bits.append("CorpusId:%s" % ext["CorpusId"])
    return ", ".join(bits)


def _truncate(s, n):
    s = s or ""
    return s if len(s) <= n else s[: n - 12] + " ...[trunc]"


def _tldr(p):
    t = p.get("tldr")
    return t.get("text") if isinstance(t, dict) else None


def _format_search_hit(p):
    authors = [(a.get("name") or "") for a in (p.get("authors") or [])][:MAX_AUTHORS_PER_PAPER]
    return {
        "paperId": p.get("paperId", ""),
        "title": p.get("title", ""),
        "year": p.get("year"),
        "venue": p.get("venue", ""),
        "authors": ", ".join(authors),
        "external": _format_external_ids(p.get("externalIds")),
        "tldr": _truncate(_tldr(p), TLDR_TRUNC),
        "abstract": _truncate(p.get("abstract"), ABSTRACT_TRUNC),
        "citations": p.get("citationCount"),
        "infl_cites": p.get("influentialCitationCount"),
    }


def s2_search(client, query, k=10):
    raw = client.search(query=query, k=k)
    if raw.get("_error"):
        return [{"_error": raw["_error"]}]
    return [_format_search_hit(p) for p in raw.get("data") or []]


def _pick_pdf_url(raw):
    """openAccessPdf.url first, then an arXiv PDF URL built from externalIds.ArXiv."""
    oa = raw.get("openAccessPdf") or {}
    if isinstance(oa, dict):
        u = (oa.get("url") or "").strip()
        if u:
            return u, "openAccessPdf"
    arxiv = ((raw.get("externalIds") or {}).get("ArXiv") or "").strip()
    if arxiv:
        return "https://arxiv.org/pdf/%s" % arxiv, "arxiv"
    return "", ""


def s2_paper(client, paper_id):
    raw = client.paper(paper_id)
    if raw.get("_error"):
        return {"_error": raw.get("_error"), "paper_id": paper_id}
    refs = raw.get("references") or []
    cits = raw.get("citations") or []
    authors = [(a.get("name") or "") for a in (raw.get("authors") or [])][:MAX_AUTHORS_PER_PAPER]
    pdf_url, body_source = _pick_pdf_url(raw)
    body_text = fetch_via_jina(pdf_url) if pdf_url else ""
    return {
        "paperId": raw.get("paperId", paper_id),
        "title": raw.get("title", ""),
        "year": raw.get("year"),
        "venue": raw.get("venue", ""),
        "authors": ", ".join(authors),
        "external": _format_external_ids(raw.get("externalIds")),
        "fields_of_study": raw.get("fieldsOfStudy") or [],
        "citations": raw.get("citationCount"),
        "infl_cites": raw.get("influentialCitationCount"),
        "tldr": _truncate(_tldr(raw), TLDR_TRUNC),
        "abstract": _truncate(raw.get("abstract"), ABSTRACT_TRUNC),
        "body_source": body_source,
        "body_text": body_text,
        "references_preview": [
            {"title": r.get("title", ""), "year": r.get("year"),
             "external": _format_external_ids(r.get("externalIds"))}
            for r in refs[:MAX_REFERENCES_RETURNED] if r.get("title")
        ],
        "citations_preview": [
            {"title": c.get("title", ""), "year": c.get("year"),
             "external": _format_external_ids(c.get("externalIds"))}
            for c in cits[:MAX_CITATIONS_RETURNED] if c.get("title")
        ],
    }


# ---------------------------------------------------------------- tool sets used by the harness
def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class WebTools:
    """google_search + visit_page. Each method takes the parsed tool arguments, returns a string."""
    search_name, visit_name = "google_search", "visit_page"

    def __init__(self):
        if not SERPER_API_KEY:
            raise ToolConfigError("SERPER_API_KEY is not set; the web harness needs it (serper.dev)")

    def search(self, args):
        return google_search(args.get("query", ""))

    def visit(self, args):
        return visit_page(args.get("url", ""))


_S2 = None
_S2_LOCK = threading.Lock()


def shared_s2_client():
    """One S2 client per process, so every thread shares the same rate budget."""
    global _S2
    with _S2_LOCK:
        if _S2 is None:
            _S2 = S2Client()
    return _S2


class S2Tools:
    search_name, visit_name = "s2_search", "s2_paper"

    def __init__(self, client=None):
        self.client = client or shared_s2_client()

    def search(self, args):
        hits = s2_search(self.client, args.get("query", ""), _as_int(args.get("k", 10), 10))
        return json.dumps(hits, ensure_ascii=False)[:S2_SEARCH_TRUNCATE]

    def visit(self, args):
        detail = s2_paper(self.client, args.get("paper_id", ""))
        return json.dumps(detail, ensure_ascii=False)[:S2_PAPER_TRUNCATE]


def make_tools(harness):
    if harness == "web":
        return WebTools()
    if harness == "s2":
        return S2Tools()
    raise ValueError("unknown harness %r (web | s2)" % harness)
