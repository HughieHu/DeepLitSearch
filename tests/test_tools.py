"""Tool backends in process against the mock server: formats, truncation and error handling."""
import json
import threading
import time

import pytest

import mock_servers
from dlsbench import tools as tl


@pytest.fixture
def live(monkeypatch):
    m = mock_servers.Mock()
    monkeypatch.setattr(tl, "SERPER_URL", m.url + "/serper")
    monkeypatch.setattr(tl, "SERPER_API_KEY", mock_servers.SERPER_KEY)
    monkeypatch.setattr(tl, "JINA_URL", m.url + "/jina")
    monkeypatch.setattr(tl, "JINA_API_KEY", "")
    yield m
    m.stop()


def test_google_search_format(live):
    out = tl.google_search("graph neural networks")
    blocks = out.split("\n\n")
    assert len(blocks) == 10
    assert blocks[0] == ("Title: Result 0 for graph neural networks\nURL: https://example.org/r0\n"
                         "Snippet: snippet 0")


def test_google_search_bad_key_raises(live, monkeypatch):
    monkeypatch.setattr(tl, "SERPER_API_KEY", "nope")
    with pytest.raises(tl.ToolConfigError):
        tl.google_search("x")
    monkeypatch.setattr(tl, "SERPER_API_KEY", "")
    with pytest.raises(tl.ToolConfigError):
        tl.WebTools()


def test_visit_page_truncates_to_8000(live):
    out = tl.visit_page("https://arxiv.org/abs/1706.03762")
    assert len(out) == 8000 and out.startswith("Title: page\nURL Source: https://arxiv.org/abs/1706.03762")


def test_unreachable_backend_returns_error_string(monkeypatch):
    monkeypatch.setattr(tl, "JINA_URL", "http://127.0.0.1:9")       # nothing listens there
    monkeypatch.setattr(tl.time, "sleep", lambda s: None)            # skip the retry backoff
    assert tl.visit_page("https://example.org").startswith("[ERROR] ")
    assert tl.fetch_via_jina("https://example.org") == ""
    assert tl.fetch_via_jina("   ") == ""


def test_s2_tools(live):
    s2 = tl.S2Client(api_key="", rps=1000, base_url=live.url + "/s2")
    t = tl.S2Tools(client=s2)
    out = t.search({"query": "QID:abc", "k": 2})
    hits = json.loads(out)
    assert [h["paperId"] for h in hits] == ["PID-abc", "PID-abc-1"]
    assert len(hits[0]["abstract"]) == 1199 and hits[0]["abstract"].endswith(" ...[trunc]")  # as the source cuts
    assert hits[0]["authors"].count(",") == 4 and hits[0]["external"] == "arXiv:2401.00001, CorpusId:1"
    assert len(t.search({"query": "QID:abc", "k": "ten"})) == 8000      # k falls back to 10; capped
    d = json.loads(t.visit({"paper_id": "PID-abc"}))
    assert d["body_source"] == "openAccessPdf" and len(d["body_text"]) == 12000
    assert len(d["references_preview"]) == 15 and len(d["citations_preview"]) == 15
    assert len(t.visit({"paper_id": "PID-abc"})) <= 18000
    assert json.loads(t.visit({"paper_id": "NOTFOUND-1"})) == {"_error": "not_found", "paper_id": "NOTFOUND-1"}
    assert json.loads(t.search({"query": "  "})) == [{"_error": "empty_query"}]
    assert json.loads(t.visit({})) == {"_error": "empty_paper_id", "paper_id": ""}


def test_s2_paper_id_is_escaped(live, monkeypatch):
    seen = []
    orig = tl.http_request

    def spy(url, **kw):
        seen.append(url)
        return orig(url, **kw)
    monkeypatch.setattr(tl, "http_request", spy)
    tl.S2Client(api_key="", rps=1000, base_url=live.url + "/s2").paper("DOI:10.1145/12345")
    assert seen[0].split("?")[0].endswith("/paper/DOI:10.1145%2F12345")


def test_token_bucket_limits_rate():
    b = tl.TokenBucket(20)
    t0 = time.monotonic()
    threads = [threading.Thread(target=b.acquire) for _ in range(30)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert time.monotonic() - t0 >= 0.4          # 20 in the bucket, 10 more at 20/s
