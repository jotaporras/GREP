"""End-to-end smoke test for GPTQueryClient.batch_query_gpt_5.

Submits a tiny 2-request batch to the real OpenAI Batch API and prints the
responses. Defaults to gpt-5-nano with minimal reasoning so the cost is
negligible. Use this to verify:

  1. /v1/responses is a valid batch endpoint (it's newer than chat.completions).
  2. The model accepts the reasoning/verbosity params we pass.
  3. The polling + ordering logic works end-to-end.

Note: even small batches can take a few minutes to complete. Adjust
--poll-interval if you want faster updates.

Run:
    python scripts/smoke_batch_query.py
    python scripts/smoke_batch_query.py --model gpt-5.5 --reasoning-effort low
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    print("[FAIL] OPENAI_API_KEY is not set")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prism.data.utils import GPTQueryClient


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--reasoning-effort", default="minimal")
    ap.add_argument("--poll-interval", type=int, default=15)
    args = ap.parse_args()

    queries = [
        "Reply with exactly: PING",
        "Reply with exactly: PONG",
    ]

    client = GPTQueryClient()
    responses = client.batch_query_gpt_5(
        queries,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        poll_interval=args.poll_interval,
    )

    for i, (q, r) in enumerate(zip(queries, responses)):
        print(f"[{i}] query={q!r} response={r!r}")
