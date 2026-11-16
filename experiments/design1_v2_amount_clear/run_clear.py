"""Clear-label twin of the amount-only arm, Design 1 v2 reader experiment: runner. The default prints the plan
and a credit estimate: zero calls. Calls happen only behind --run.

    PYTHONPATH=src python experiments/design1_v2_amount_clear/run_clear.py         # plan only, no calls
    PYTHONPATH=src python experiments/design1_v2_amount_clear/run_clear.py --run   # 144 calls

The three frozen readers (`deepseek-v4-flash`, `qwen3.5-397b`, `gemma-4-31b-it`), repeat 1 only, on the 48 inputs
of `clear-inputs.jsonl` (built by `build_clear.py`): 144 calls, job ids tagged `design1v2amountclear`. There is no
dry run: these readers were validated on prompts identical except for one credit amount. Reader instructions,
tool schema, token cap, temperature and call function are the frozen runner's, imported unchanged and read-only
(`run_design1_v2.py` is loaded as a module and its `main` is never run). Nothing outside this directory is
written by this file. Results go to `clear-results.jsonl` in the frozen record format; run settings are written
to `clear-run-config.json` before any call.

Two paired comparisons, within each reader, household by household: (1) with the frozen repeat 1 `stray_clear`
records (the amount effect under a clear label); (2) with the amount arm's `stray_amount` records (the label
effect at omission-sized amounts). The main rate is a flag on either income group (the reader hypothesis 3
definition in `freeze.md` section 8); the target-group rate is reported beside it.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing the frozen runner must not leave bytecode files behind

import argparse
import hashlib
import importlib.util
import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from snapqc import llm

HERE = Path(__file__).resolve().parent
FROZEN_DIR = HERE.parent / "design1_v2"
AMOUNT_DIR = HERE.parent / "design1_v2_amount"
_spec = importlib.util.spec_from_file_location("run_design1_v2", FROZEN_DIR / "run_design1_v2.py")
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)  # module-level code only; main() is not called
run_design1 = frozen.run_design1

MODELS = list(frozen.MODELS)  # the three frozen readers
REPEATS = [1]
TAG = "design1v2amountclear"
N_INPUTS = 48
STATE = "stray_clear_amount"
COMPARE = [("stray_clear", FROZEN_DIR / "results.jsonl"),          # amount effect under a clear label
           ("stray_amount", AMOUNT_DIR / "amount-results.jsonl")]   # label effect at omission-sized amounts

CLEAR_INPUTS_PATH = HERE / "clear-inputs.jsonl"
RESULTS_PATH = HERE / "clear-results.jsonl"
CONFIG_PATH = HERE / "clear-run-config.json"
AMOUNT_MAP_PATH = AMOUNT_DIR / "amount-map.csv"

EXPECT_MODELS = ["deepseek-v4-flash", "qwen3.5-397b", "gemma-4-31b-it"]
EXPECT_INSTRUCTIONS_SHA = "faa22e9fc6f7b58e"
EXPECT_TOOL_SHA = "d7bf54860554aac4"
EXPECT_MAX_OUTPUT_TOKENS = 8192
EXPECT_INPUTS_SHA_PREFIX = "9b84d771"
EXPECT_AMOUNT_MAP_SHA = "212f603759d749fb4f9f144af9851cdfeeaf8185f5ceb500df09a770f84eb46b"
EXPECT_CLEAR_INPUTS_SHA = "054983c3ef7ce938c67a4b81edc502600dc88001ee266355888c76c6f9b6ee34"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen() -> tuple[str, str, str]:
    """Stop on any drift from the frozen readers, instructions, tool, token cap or inputs, from the amount map,
    or from the twin inputs as built. Returns (frozen inputs, amount map, twin inputs) sha256."""
    assert MODELS == EXPECT_MODELS, f"frozen readers {MODELS}"
    assert llm.sha(frozen.INSTRUCTIONS) == EXPECT_INSTRUCTIONS_SHA, "reader instructions differ from the frozen run"
    assert llm.tool_sha(frozen.TOOL) == EXPECT_TOOL_SHA, "tool schema differs from the frozen run"
    assert frozen.MAX_OUTPUT_TOKENS == EXPECT_MAX_OUTPUT_TOKENS, "token cap differs from the frozen run"
    sha = file_sha256(frozen.build_inputs.INPUTS_PATH)
    assert sha.startswith(EXPECT_INPUTS_SHA_PREFIX), f"inputs.jsonl sha256 {sha} differs from the frozen inputs"
    sha_map = file_sha256(AMOUNT_MAP_PATH)
    assert sha_map == EXPECT_AMOUNT_MAP_SHA, f"amount-map.csv sha256 {sha_map} differs from the committed map"
    sha_clear = file_sha256(CLEAR_INPUTS_PATH)
    assert sha_clear == EXPECT_CLEAR_INPUTS_SHA, f"clear-inputs.jsonl sha256 {sha_clear} differs from the built file"
    return sha, sha_map, sha_clear


def load_clear_inputs() -> list[dict]:
    inputs = run_design1.read_jsonl(CLEAR_INPUTS_PATH)
    frozen_ids = {x["input_id"] for x in frozen.load_inputs()}
    assert len(inputs) == N_INPUTS and len({x["input_id"] for x in inputs}) == N_INPUTS
    assert all(x["cls"] == "zero_stray" and x["evidence_state"] == STATE for x in inputs)
    assert all(x["reader_input_sha"] == llm.sha(x["reader_input"]) for x in inputs)
    assert not frozen_ids & {x["input_id"] for x in inputs}, "twin input ids collide with frozen ids"
    return inputs


def estimate(n_calls_per_reader: int) -> pd.DataFrame:
    """Credits per reader: mean logged cost per call in the frozen results.jsonl, times the calls planned."""
    recs = [r for r in run_design1.latest(run_design1.read_jsonl(frozen.RESULTS_PATH)).values()
            if r.get("ok_http") and r.get("cost") is not None]
    rows = []
    for m in MODELS:
        c = [r["cost"] for r in recs if r["reader"] == m]
        per = statistics.mean(c)
        rows.append({"reader": m, "frozen_calls": len(c), "mean_credits_per_call": round(per, 6),
                     "calls": n_calls_per_reader, "credits": round(per * n_calls_per_reader, 4)})
    return pd.DataFrame(rows)


def comparison_rows() -> dict[str, dict[str, int]]:
    """Repeat 1 records per reader in each comparison cell."""
    out = {}
    for state, path in COMPARE:
        recs = run_design1.latest(run_design1.read_jsonl(path)).values()
        out[state] = dict(Counter(r["reader"] for r in recs if r["evidence_state"] == state and r["repeat"] == 1))
    return out


def snapshot_config(inputs_sha: str, map_sha: str, clear_sha: str) -> None:
    CONFIG_PATH.write_text(json.dumps({
        "stage": "main", "models": MODELS, "repeats": REPEATS, "tag": TAG,
        "max_output_tokens": frozen.MAX_OUTPUT_TOKENS, "concurrency": run_design1.CONCURRENCY,
        "temperature": llm.TEMPERATURE,
        "instructions_sha": llm.sha(frozen.INSTRUCTIONS), "tool_sha": llm.tool_sha(frozen.TOOL),
        "inputs_sha256": inputs_sha, "amount_map_sha256": map_sha, "clear_inputs_sha256": clear_sha,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "builder_sha256": file_sha256(HERE / "build_clear.py"),
        "frozen_runner_sha256": file_sha256(FROZEN_DIR / "run_design1_v2.py"),
        "compare_with": [
            {"evidence_state": "stray_clear", "repeat": 1, "results": "design1_v2/results.jsonl",
             "reads": "amount effect under a clear label"},
            {"evidence_state": "stray_amount", "repeat": 1, "results": "design1_v2_amount/amount-results.jsonl",
             "reads": "label effect at omission-sized amounts"}],
        "main_rate": "flag on either income group, paired by household within each reader",
        "also_reported": ["target-group flag"],
    }, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="make model calls (never the default)")
    args = ap.parse_args()

    inputs_sha, map_sha, clear_sha = check_frozen()
    inputs = load_clear_inputs()
    jobs = frozen.build_jobs(inputs, MODELS, REPEATS, tag=TAG)
    assert len(jobs) == 144 and len({j["job_id"] for j in jobs}) == 144, len(jobs)

    print(f"frozen inputs.jsonl sha256 {inputs_sha}")
    print(f"amount-map.csv sha256 {map_sha}")
    print(f"clear-inputs.jsonl sha256 {clear_sha}")
    print(f"runner sha256 {file_sha256(Path(__file__).resolve())}")
    print(f"instructions sha {llm.sha(frozen.INSTRUCTIONS)}, tool sha {llm.tool_sha(frozen.TOOL)}, "
          f"max_output_tokens {frozen.MAX_OUTPUT_TOKENS}, temperature {llm.TEMPERATURE}, "
          f"concurrency {run_design1.CONCURRENCY}")
    print(f"jobs: {len(jobs)} {dict(Counter(j['reader'] for j in jobs))} (repeat 1, {len(inputs)} inputs, tag {TAG})")
    print(f"comparison rows (repeat 1) per reader: {comparison_rows()}")
    est = estimate(len(inputs) * len(REPEATS))
    print(est.to_string(index=False))
    print(f"total credits: {est['credits'].sum():.3f} (mean frozen cost per call per reader)")

    if not args.run:
        print("No calls made (pass --run to call).")
        return 0

    # Everything below makes model calls.
    snapshot_config(inputs_sha, map_sha, clear_sha)
    run_design1.run_llm_jobs(jobs, RESULTS_PATH)
    run_design1.report_calls(RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
