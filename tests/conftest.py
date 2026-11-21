import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "eval")
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# In-process tests must never pick up a developer's real keys from eval/.env: blank values block
# the .env loader (real environment variables win, and a blank counts as "unset" afterwards).
for _k in ("SERPER_API_KEY", "JINA_API_KEY", "S2_API_KEY", "JUDGE_API_KEY", "OPENAI_API_KEY",
           "MODEL_API_KEY", "MODEL_MAX_TOKENS", "MODEL_TEMPERATURE", "MODEL_REASONING_EFFORT",
           "JUDGE_MODEL", "JUDGE_API"):
    os.environ.setdefault(_k, "")
os.environ.setdefault("MODEL_NAME", "pytest-model")
os.environ.setdefault("MODEL_BASE_URL", "http://127.0.0.1:9/v1")      # never contacted


def run_module(module, args, env, cwd, timeout=300):
    """Run `python -m <module> <args>` in a subprocess with the package on the path."""
    env = dict(env)
    env["PYTHONPATH"] = EVAL + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", module] + list(args), cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=timeout)
