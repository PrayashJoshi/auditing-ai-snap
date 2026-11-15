"""Added readers for the Design 1 v2 reader experiment. The default prints the plan and a credit estimate:
zero calls. Calls happen only behind --run.

    PYTHONPATH=src python experiments/design1_v2_added/run_added.py                  # plan only, no calls
    PYTHONPATH=src python experiments/design1_v2_added/run_added.py --run --dry-run  # dry run (96 calls)
    PYTHONPATH=src python experiments/design1_v2_added/run_added.py --run            # main run (retained readers only)

Three readers are added after the frozen run's dry run, outside reader hypotheses 1 to 3:
`glm-5.3-flash` and `gemma-4-26b-a4b-it` (264 inputs, repeats 1 and 2) and `qwen3.8-27b` (264 inputs, repeat 1).
Inputs, reader instructions, tool schema, token cap and call function are the frozen runner's, imported unchanged
and read-only (`run_design1_v2.py` is loaded as a module and its `main` is never run). Nothing in
`experiments/design1_v2/` is written by this file.

Dry run (repeat 1 only, 32 inputs per reader): the frozen dry run's 24 inputs (first 3 of each of the 8 class and
state cells) plus 8 more: the first input with an unearned target group in each of those 8 cells, in file order.
The frozen dry run did not include unearned-target inputs. Validity is the frozen rule (`reader_output_valid`).
A reader is retained when at least 22 of the 24 frozen-dry-run inputs are valid (the frozen 90% bar). The 8
unearned inputs are counted and reported separately and do not decide retention. The main run calls only readers
retained in `added-dryrun-validity.csv`, and refuses to start without that file. Results go to
`added-results.jsonl` in the frozen record format, with job ids tagged `design1v2added`.
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
_spec = importlib.util.spec_from_file_location("run_design1_v2", FROZEN_DIR / "run_design1_v2.py")
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)  # module-level code only; main() is not called
run_design1 = frozen.run_design1

# readers added: model -> repeats
ADDED = {"glm-5.3-flash": [1, 2], "gemma-4-26b-a4b-it": [1, 2], "qwen3.8-27b": [1]}
MAIN_TAG = "design1v2added"
DRY_TAG = "design1v2added-dryrun"
MIN_VALID_FROZEN24 = 22  # of the 24 frozen-dry-run inputs

DRYRUN_PATH = HERE / "added-dryrun.jsonl"
DRYRUN_VALIDITY_PATH = HERE / "added-dryrun-validity.csv"
DRYRUN_CONFIG_PATH = HERE / "added-dryrun-config.json"
RESULTS_PATH = HERE / "added-results.jsonl"
CONFIG_PATH = HERE / "added-run-config.json"
PROBE_PATH = HERE / "probe-reachability.jsonl"

EXPECT_INSTRUCTIONS_SHA = "faa22e9fc6f7b58e"
EXPECT_TOOL_SHA = "d7bf54860554aac4"
EXPECT_INPUTS_SHA_PREFIX = "9b84d771"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen() -> str:
    """Stop on any drift from the frozen instructions, tool or inputs. Returns the inputs sha256."""
    assert llm.sha(frozen.INSTRUCTIONS) == EXPECT_INSTRUCTIONS_SHA, "reader instructions differ from the frozen run"
    assert llm.tool_sha(frozen.TOOL) == EXPECT_TOOL_SHA, "tool schema differs from the frozen run"
    sha = file_sha256(frozen.build_inputs.INPUTS_PATH)
    assert sha.startswith(EXPECT_INPUTS_SHA_PREFIX), f"inputs.jsonl sha256 {sha} differs from the frozen inputs"
    return sha


def dry_inputs(inputs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(the frozen dry run's 24 inputs, the 8 extra unearned-target inputs), both in cell order."""
    frozen24 = frozen.dry_run_inputs(inputs)
    extra = []
    for cell in frozen.build_inputs.EXPECTED_CELLS:
        cands = [x for x in inputs if (x["cls"], x["evidence_state"]) == cell and x["target_group"] == "unearned"]
        assert cands, f"no unearned-target input in cell {cell}"
        extra.append(cands[0])
    ids24 = {x["input_id"] for x in frozen24}
    assert len(frozen24) == 24 and len(extra) == 8
    assert all(x["target_group"] == "unearned" for x in extra)
    assert not ids24 & {x["input_id"] for x in extra}, "extra inputs overlap the frozen dry-run inputs"
    return frozen24, extra


def main_jobs(inputs: list[dict], models: list[str]) -> list[dict]:
    jobs = []
    for m in models:
        jobs += frozen.build_jobs(inputs, [m], ADDED[m], tag=MAIN_TAG)
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    return jobs


def dry_jobs(inputs: list[dict]) -> list[dict]:
    frozen24, extra = dry_inputs(inputs)
    jobs = []
    for m in ADDED:
        jobs += frozen.build_jobs(frozen24 + extra, [m], [1], tag=DRY_TAG)
    assert len(jobs) == 96 and len({j["job_id"] for j in jobs}) == 96
    return jobs


def dry_run_validity(records: list[dict], frozen24_ids: set[str], extra_ids: set[str]) -> pd.DataFrame:
    rows = []
    for m in ADDED:
        recs = [r for r in records if r.get("reader") == m]
        ok = {r["input_id"]: frozen.reader_output_valid(r) for r in recs}
        k24 = sum(v for i, v in ok.items() if i in frozen24_ids)
        k8 = sum(v for i, v in ok.items() if i in extra_ids)
        k = k24 + k8
        rows.append({"reader": m, "calls": len(recs), "valid": k,
                     "valid_share": round(k / len(recs), 4) if recs else float("nan"),
                     "calls_frozen24": sum(i in frozen24_ids for i in ok), "valid_frozen24": k24,
                     "calls_unearned8": sum(i in extra_ids for i in ok), "valid_unearned8": k8,
                     "min_valid_frozen24": MIN_VALID_FROZEN24,
                     "retained": sum(i in frozen24_ids for i in ok) == 24 and k24 >= MIN_VALID_FROZEN24})
    return pd.DataFrame(rows)


def estimate(inputs: list[dict], dry: list[dict]) -> pd.DataFrame:
    """Credits per reader: mean logged cost per ok call in the 8-input reachability probe (same instructions, tool
    and token cap), times the calls planned. No calls are made here."""
    if not PROBE_PATH.exists():
        print(f"probe file {PROBE_PATH.name} not found: no credit estimate")
        return pd.DataFrame()
    probe = [r for r in run_design1.read_jsonl(PROBE_PATH) if r.get("ok_http") and r.get("cost") is not None]
    chars = {x["input_id"]: len(x["reader_input"]) for x in inputs}
    probe_ids = sorted({r["input_id"] for r in probe})
    rows = []
    for m, reps in ADDED.items():
        recs = [r for r in probe if r["reader"] == m]
        per = statistics.mean(r["cost"] for r in recs)
        n_main = len(inputs) * len(reps)
        rows.append({"reader": m, "probe_calls": len(recs), "mean_credits_per_call": round(per, 6),
                     "main_calls": n_main, "main_credits": round(per * n_main, 3),
                     "dry_calls": 32, "dry_credits": round(per * 32, 3)})
    df = pd.DataFrame(rows)
    mean_all = statistics.mean(chars.values())
    mean_probe = statistics.mean(chars[i] for i in probe_ids)
    print(f"probe inputs: {len(probe_ids)} (first input of each cell, earned target), mean {mean_probe:.0f} chars; "
          f"all {len(inputs)} inputs mean {mean_all:.0f} chars ({mean_all / mean_probe:.2f}x the probe's); "
          f"dry-run inputs mean {statistics.mean(chars[x['input_id']] for x in dry):.0f}")
    print("caveat: each per-call mean comes from 8 probe calls on longer-than-average inputs; cost also depends on output "
          "and reasoning tokens, which vary by input. Treat the figures as rough, probably slightly high on input cost.")
    return df


def snapshot_config(path: Path, stage: str, models: list[str], inputs_sha: str) -> None:
    path.write_text(json.dumps({
        "stage": stage, "models": models, "repeats_per_model": {m: ADDED[m] for m in models},
        "tag": MAIN_TAG if stage == "main" else DRY_TAG,
        "max_output_tokens": frozen.MAX_OUTPUT_TOKENS, "concurrency": run_design1.CONCURRENCY,
        "temperature": llm.TEMPERATURE,
        "instructions_sha": llm.sha(frozen.INSTRUCTIONS), "tool_sha": llm.tool_sha(frozen.TOOL),
        "inputs_sha256": inputs_sha, "runner_sha256": file_sha256(Path(__file__).resolve()),
        "frozen_runner_sha256": file_sha256(FROZEN_DIR / "run_design1_v2.py"),
        "min_valid_frozen24": MIN_VALID_FROZEN24,
    }, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="make model calls (never the default)")
    ap.add_argument("--dry-run", action="store_true", help="with --run: the 96-call dry run only")
    args = ap.parse_args()

    inputs_sha = check_frozen()
    inputs = frozen.load_inputs()
    assert len(inputs) == 264
    frozen24, extra = dry_inputs(inputs)
    main_all = main_jobs(inputs, list(ADDED))
    dry = dry_jobs(inputs)

    print(f"inputs.jsonl sha256 {inputs_sha}")
    print(f"instructions sha {llm.sha(frozen.INSTRUCTIONS)}, tool sha {llm.tool_sha(frozen.TOOL)}")
    print(f"main jobs: {len(main_all)} {dict(Counter(j['reader'] for j in main_all))}")
    print(f"dry-run jobs: {len(dry)} {dict(Counter(j['reader'] for j in dry))} (repeat 1; {len(frozen24)} frozen-dry-run inputs + {len(extra)} unearned inputs per reader)")
    print("dry-run inputs, frozen 24:", [x["input_id"] for x in frozen24])
    print("dry-run inputs, unearned 8:", [x["input_id"] for x in extra])
    for m, reps in ADDED.items():
        print(f"expected calls {m}: main {len(inputs) * len(reps)} (repeats {reps}), dry run 32")
    est = estimate(inputs, frozen24 + extra)
    if len(est):
        print(est.to_string(index=False))
        print(f"total credits, main if all retained: {est['main_credits'].sum():.2f}; dry run: {est['dry_credits'].sum():.3f}")

    if not args.run:
        print("No calls made (pass --run to call).")
        return 0

    # Everything below makes model calls.
    if args.dry_run:
        snapshot_config(DRYRUN_CONFIG_PATH, "dry-run", list(ADDED), inputs_sha)
        run_design1.run_llm_jobs(dry, DRYRUN_PATH)
        recs = list(run_design1.latest(run_design1.read_jsonl(DRYRUN_PATH)).values())
        v = dry_run_validity(recs, {x["input_id"] for x in frozen24}, {x["input_id"] for x in extra})
        v.to_csv(DRYRUN_VALIDITY_PATH, index=False)
        print(v.to_string(index=False))
        return 0
    assert DRYRUN_VALIDITY_PATH.exists(), "run the dry run first (--run --dry-run)"
    v = pd.read_csv(DRYRUN_VALIDITY_PATH).set_index("reader")
    retained = [m for m in ADDED if m in v.index and bool(v.loc[m, "retained"])]
    dropped = [m for m in ADDED if m not in retained]
    print(f"retained after dry run: {retained}; dropped: {dropped}")
    if not retained:
        print("No reader retained: no calls made.")
        return 0
    snapshot_config(CONFIG_PATH, "main", retained, inputs_sha)
    run_design1.run_llm_jobs([j for j in main_all if j["reader"] in retained], RESULTS_PATH)
    run_design1.report_calls(RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
