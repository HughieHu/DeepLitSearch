"""dlsbench: evaluate any model endpoint on DeepLitSearch."""
import os
import socket
import ssl
import urllib.error
import urllib.request

TASK_NAMES = {
    1: "Benchmark Matching",
    2: "Best-Method Mining",
    3: "Conflict Solving",
    4: "Implementation Tracing",
    5: "Cluster Mining",
}


def _load_dotenv():
    """Load .env before anything else in the package imports.

    This lives in __init__ on purpose: `model.py`, `tools.py` and `judge.py` read their settings at
    import time, so loading .env from inside eval.py would be too late. Real environment variables
    always win over the file.
    """
    for d in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        p = os.path.join(d, ".env")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                        v = v[1:-1]
                    # A blank value means "leave it at the default", which is how people comment a
                    # setting out. Setting it to "" instead would make int()/float() throw.
                    if v:
                        os.environ.setdefault(k, v)
        except OSError:
            pass
        return p
    return None


_load_dotenv()


def envint(name, default):
    """int from the environment; a blank or malformed value counts as unset."""
    v = str(os.environ.get(name, "")).strip()
    try:
        return int(v) if v else int(default)
    except ValueError:
        return int(default)


def envfloat(name, default):
    v = str(os.environ.get(name, "")).strip()
    try:
        return float(v) if v else float(default)
    except ValueError:
        return float(default)


def envopt(name):
    """The stripped value of an optional setting, or None when it is unset or blank."""
    v = str(os.environ.get(name, "")).strip()
    return v or None


def ssl_context():
    """TLS context using certifi's CA bundle when available (stale system bundles are common on macOS)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = ssl_context()


class TransportError(RuntimeError):
    """The request never got an HTTP response (DNS, refused connection, timeout, reset)."""


def http_request(url, method="GET", body=None, headers=None, timeout=60):
    """One HTTP request with the standard library. Returns (status, text, headers).

    An HTTP error status is returned, not raised, so each caller decides what a 4xx/5xx means for
    it. Only a request that got no response at all raises (TransportError).
    """
    data = body if body is None or isinstance(body, bytes) else body.encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace"), resp.headers
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return e.code, text, e.headers
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
        raise TransportError("%s: %s" % (type(e).__name__, str(e)[:200]))
