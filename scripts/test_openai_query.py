"""Throwaway sanity check for OpenAI API access. Delete when done.

Verifies:
  1. OPENAI_API_KEY is loaded into the environment (from .env via python-dotenv
     if available, otherwise expects it preset).
  2. The chat completions endpoint works (gpt-4.1).
  3. The Responses API works (gpt-5.1) — same path used by query_gpt_5 in
     prism/data/utils.py.

Run:
    python scripts/deleteme_test_openai.py
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[ok] loaded .env from {env_path}")
    else:
        print(f"[warn] no .env at {env_path}; relying on shell environment")
except ImportError:
    print("[warn] python-dotenv not installed; relying on shell environment")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[FAIL] OPENAI_API_KEY is not set")
    sys.exit(1)
print(f"[ok] OPENAI_API_KEY present (len={len(api_key)}, prefix={api_key[:7]}...)")

from openai import OpenAI

client = OpenAI()

# ---- gpt-4.1 (chat completions) ----
try:
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_completion_tokens=8,
    )
    print(f"[ok] gpt-4.1 chat: {resp.choices[0].message.content!r}")
except Exception as ex:
    print(f"[FAIL] gpt-4.1 chat: {ex}")

# ---- gpt-5.1 (responses API) ----
try:
    resp = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "user", "content": [{"type": "input_text", "text": "Reply with exactly: PONG"}]}
        ],
        text={"format": {"type": "text"}, "verbosity": "low"},
        reasoning={"effort": "none", "summary": "auto"},
    )
    print(f"[ok] gpt-5.1 responses: {resp.output_text!r}")
except Exception as ex:
    print(f"[FAIL] gpt-5.1 responses: {ex}")
