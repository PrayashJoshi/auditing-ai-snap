"""Rate-limit probe: is the inference limit per key or per account, and does concurrency matter?

Tiny calls (one short prompt, 16 output tokens, `gemma-4-26b-a4b-it`), no retries: every
429 is recorded as it comes back. Phases, each after a 70 s pause so the previous window
has cleared:
  1. key 1 alone: 80 calls at concurrency 20, then one call every 5 s for 75 s
     (when does the first call succeed again?)
  2. key 2 alone: the same burst
  3. both keys at once: 80 calls each at concurrency 20
  4. key 2 first (55 calls at concurrency 20), then 5 calls on key 1 straight after
Output: `rate-probe/probe.jsonl`, one line per call. Key values are never written.

Run: uv run python experiments/smoke/probe_keys.py
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import adaption
from adaption import Adaption

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "rate-probe" / "probe.jsonl"
MODEL = "gemma-4-26b-a4b-it"
BURST, CONCURRENCY, PAUSE_S = 80, 20, 70
KEYS = {"key1": "ADAPTION_API_KEY", "key2": "ADAPTION_API_KEY_2"}

lock = threading.Lock()


def load_env() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def one(client: Adaption, key: str, phase: str, i: int, fh) -> None:
    t0 = time.time()
    rec = dict(phase=phase, key=key, i=i, model=MODEL, sdk_version=adaption.__version__,
               ts_start=datetime.now(timezone.utc).isoformat(), t_start=t0)
    try:
        raw = client.responses.with_raw_response.create(
            model=MODEL, input=f"Reply with the word OK. ({i})", max_output_tokens=16,
            temperature=0, store=False)
        resp = raw.parse()
        rec.update(http=raw.http_response.status_code, response_id=resp.id, status=resp.status,
                   rate_headers={k: v for k, v in raw.headers.items()
                                 if "ratelimit" in k.lower() or k.lower() == "retry-after"},
                   cost=resp.usage.cost if resp.usage else None)
    except adaption.APIStatusError as e:
        rec.update(http=e.status_code, error=type(e).__name__, message=str(e)[:200],
                   retry_after=e.response.headers.get("retry-after"),
                   rate_headers={k: v for k, v in e.response.headers.items()
                                 if "ratelimit" in k.lower() or k.lower() == "retry-after"})
    except (adaption.APIConnectionError, adaption.APITimeoutError) as e:
        rec.update(http=None, error=type(e).__name__, message=str(e)[:200])
    rec.update(t_end=time.time(), latency_s=round(time.time() - t0, 3))
    with lock:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()


def burst(clients: dict, phase: str, fh) -> None:
    with ThreadPoolExecutor(CONCURRENCY * len(clients)) as pool:
        for key, c in clients.items():
            for i in range(BURST):
                pool.submit(one, c, key, phase, i, fh)


def summary(phase: str) -> None:
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    rows = [r for r in rows if r["phase"] == phase]
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r["http"])
    print(phase, {k: {"ok": v.count(200), "429": v.count(429), "other": len(v) - v.count(200) - v.count(429)}
                  for k, v in by.items()}, flush=True)


def main() -> None:
    load_env()
    clients = {k: Adaption(api_key=os.environ[v], max_retries=0, timeout=60) for k, v in KEYS.items()}
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as fh:
        time.sleep(PAUSE_S)
        burst({"key1": clients["key1"]}, "1_key1_burst", fh)
        summary("1_key1_burst")
        for i in range(15):  # recovery: one call every 5 s
            one(clients["key1"], "key1", "1_key1_recovery", i, fh)
            time.sleep(5)
        summary("1_key1_recovery")
        time.sleep(PAUSE_S)
        burst({"key2": clients["key2"]}, "2_key2_burst", fh)
        summary("2_key2_burst")
        time.sleep(PAUSE_S)
        burst(clients, "3_both_keys_burst", fh)
        summary("3_both_keys_burst")
        time.sleep(PAUSE_S)
        with ThreadPoolExecutor(CONCURRENCY) as pool:
            for i in range(55):
                pool.submit(one, clients["key2"], "key2", "4_key2_then_key1", i, fh)
        for i in range(5):
            one(clients["key1"], "key1", "4_key2_then_key1", 100 + i, fh)
        summary("4_key2_then_key1")


if __name__ == "__main__":
    main()
