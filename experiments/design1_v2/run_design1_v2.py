"""Design 1 v2 runner. The default builds the inputs, runs the preflight and prints the job plan and cost
estimate: zero calls. Calls happen only behind --run.

    PYTHONPATH=src python experiments/design1_v2/run_design1_v2.py                # build + preflight only
    PYTHONPATH=src python experiments/design1_v2/run_design1_v2.py --run --dry-run  # dry run (72 calls)
    PYTHONPATH=src python experiments/design1_v2/run_design1_v2.py --run            # main run (1,584 calls)

The call, retry and logging code is the first reader run's (`run_design1.run_llm_jobs`, which wraps
`snapqc.llm.call`), imported unchanged; so are its reader instructions and the `report_flags` tool.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from snapqc import forms, llm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("build_inputs", HERE / "build_inputs.py")
build_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_inputs)
run_design1 = build_inputs.run_design1

MODELS = ["deepseek-v4-flash", "qwen3.5-397b", "gemma-4-31b-it"]
REPEATS = [1, 2]
INSTRUCTIONS = run_design1.READER_INSTRUCTIONS
TOOL = run_design1.reader_tool(True)
MAX_OUTPUT_TOKENS = run_design1.MAX_OUTPUT_TOKENS
DRY_RUN_PER_CELL = 3  # first 3 households (input order) of each of the 8 class x state cells: 24 inputs
DRY_RUN_MIN_VALID = 0.90

RESULTS_PATH = HERE / "results.jsonl"
DRYRUN_PATH = HERE / "dryrun.jsonl"
DRYRUN_VALIDITY_PATH = HERE / "dryrun-validity.csv"
CONFIG_PATH = HERE / "run-config.json"
COST_PATH = HERE / "cost-estimate.csv"
V1_RESULTS = ROOT / "experiments" / "design1" / "results.jsonl"
SMOKE_GEMMA = ROOT / "experiments" / "smoke" / "gemma-4-31b-it.jsonl"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> list[dict]:
    return run_design1.read_jsonl(build_inputs.INPUTS_PATH)


def build_jobs(inputs: list[dict], models: list[str], repeats: list[int], tag: str = "design1v2") -> list[dict]:
    jobs = []
    for x in inputs:
        for model in models:
            for rep in repeats:
                jobs.append({
                    "job_id": f"{tag}|{x['input_id']}|{model}|r{rep}",
                    "input_id": x["input_id"], "row": x["row"], "cls": x["cls"], "target_group": x["target_group"],
                    "evidence_state": x["evidence_state"], "is_omission": x["is_omission"], "repeat": rep,
                    "reader": model, "app_monthly": x["app_monthly"], "reader_input_sha": x["reader_input_sha"],
                    "_model": model, "_instructions": INSTRUCTIONS, "_input": x["reader_input"], "_tool": TOOL,
                })
    return jobs


def dry_run_inputs(inputs: list[dict]) -> list[dict]:
    out = []
    for cell in build_inputs.EXPECTED_CELLS:
        out += [x for x in inputs if (x["cls"], x["evidence_state"]) == cell][:DRY_RUN_PER_CELL]
    return out


# --- Validity -------------------------------------------------------------------------------------------
def reader_output_valid(rec: dict) -> bool:
    """Valid reader output: the call completed with exactly one report_flags call whose arguments pass the
    tool schema (`llm.call` sets `valid`), and every flag names one of the four groups at most once."""
    if not rec.get("valid"):
        return False
    flags = (rec.get("parsed") or {}).get("flags")
    if not isinstance(flags, list):
        return False
    groups = [f.get("group") for f in flags if isinstance(f, dict)]
    return len(groups) == len(flags) and all(g in forms.GROUPS for g in groups) and len(set(groups)) == len(groups)


def dry_run_validity(records: list[dict], models: list[str]) -> pd.DataFrame:
    """Share of valid reader output per model in the dry run; a model under 90% is dropped and reported."""
    rows = []
    for m in models:
        recs = [r for r in records if r.get("reader") == m]
        k = sum(reader_output_valid(r) for r in recs)
        share = k / len(recs) if recs else float("nan")
        rows.append({"reader": m, "calls": len(recs), "valid": k, "valid_share": round(share, 4),
                     "threshold": DRY_RUN_MIN_VALID, "retained": bool(recs) and share >= DRY_RUN_MIN_VALID})
    return pd.DataFrame(rows)


# --- Cost estimate --------------------------------------------------------------------------------------
def cost_estimate(n_inputs: int, mean_chars: float) -> pd.DataFrame:
    """Credits per model from logged per-call costs, no calls. deepseek and qwen: mean cost per call in the
    first reader run. gemma-4-31b-it was not in that run: mean cost of its report_flags calls in the model smoke
    test, whose inputs are shorter; the input-length ratio is given beside it."""
    v1 = [r for r in run_design1.read_jsonl(V1_RESULTS) if r.get("cost") is not None]
    v1_inputs = {x["input_id"]: len(x["reader_input"]) for x in run_design1.read_jsonl(ROOT / "experiments" / "design1" / "inputs.jsonl")}
    smoke = [r for r in run_design1.read_jsonl(SMOKE_GEMMA) if r.get("cost") is not None and r.get("tool_name") == "report_flags"]
    rows = []
    n_calls = n_inputs * len(REPEATS)
    for m in MODELS:
        if m == "gemma-4-31b-it":
            src, recs = "model smoke test, report_flags calls", smoke
            chars = None
        else:
            src, recs = "first reader run", [r for r in v1 if r["reader"] == m]
            chars = statistics.mean(v1_inputs[r["input_id"]] for r in recs)
        per_call = statistics.mean(r["cost"] for r in recs)
        mean_in_tok = statistics.mean(r.get("input_tokens") or 0 for r in recs)
        rows.append({"reader": m, "source": src, "logged_calls": len(recs), "mean_credits_per_call": per_call,
                     "mean_input_tokens_logged": round(mean_in_tok, 1),
                     "mean_input_chars_logged": None if chars is None else round(chars, 1),
                     "mean_input_chars_this_run": round(mean_chars, 1),
                     "calls_this_run": n_calls, "credits_estimate": per_call * n_calls,
                     "dry_run_calls": DRY_RUN_PER_CELL * 8, "dry_run_credits_estimate": per_call * DRY_RUN_PER_CELL * 8})
    df = pd.DataFrame(rows)
    total = {"reader": "all", "source": "", "calls_this_run": int(df["calls_this_run"].sum()),
             "credits_estimate": float(df["credits_estimate"].sum()),
             "dry_run_calls": int(df["dry_run_calls"].sum()), "dry_run_credits_estimate": float(df["dry_run_credits_estimate"].sum())}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def snapshot_config(models: list[str], inputs_sha: str) -> None:
    CONFIG_PATH.write_text(json.dumps({
        "models": models, "repeats": REPEATS, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "concurrency": run_design1.CONCURRENCY, "temperature": llm.TEMPERATURE,
        "instructions_sha": llm.sha(INSTRUCTIONS), "instructions": INSTRUCTIONS,
        "tool_sha": llm.tool_sha(TOOL), "tool": TOOL, "inputs_sha256": inputs_sha,
        "dry_run_per_cell": DRY_RUN_PER_CELL, "dry_run_min_valid": DRY_RUN_MIN_VALID,
    }, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="make model calls (never the default)")
    ap.add_argument("--dry-run", action="store_true", help="with --run: the 72-call dry run only")
    args = ap.parse_args()

    # Build and preflight; the rebuilt inputs must equal the frozen inputs.jsonl byte for byte.
    inputs, hh, _ = build_inputs.build()
    build_inputs.preflight(inputs, hh)
    sha_built = hashlib.sha256(build_inputs.serialize(inputs).encode()).hexdigest()
    sha_file = file_sha256(build_inputs.INPUTS_PATH)
    assert sha_built == sha_file, f"inputs.jsonl ({sha_file}) differs from a fresh build ({sha_built})"
    on_disk = load_inputs()
    assert [x["input_id"] for x in on_disk] == [x["input_id"] for x in inputs]

    jobs = build_jobs(on_disk, MODELS, REPEATS)
    dry = build_jobs(dry_run_inputs(on_disk), MODELS, [1], tag="design1v2-dryrun")
    assert len(jobs) == 1584 and len({j["job_id"] for j in jobs}) == 1584, len(jobs)
    assert len(dry) == 72
    mean_chars = statistics.mean(len(x["reader_input"]) for x in on_disk)
    cost = cost_estimate(len(on_disk), mean_chars)
    print(f"inputs.jsonl sha256 {sha_file}")
    print(f"instructions sha {llm.sha(INSTRUCTIONS)}, tool sha {llm.tool_sha(TOOL)}")
    print(f"jobs: {len(jobs)} main ({Counter(j['reader'] for j in jobs)}), {len(dry)} dry run")
    print(cost.to_string(index=False))

    if not args.run:
        cost.to_csv(COST_PATH, index=False)
        print("No calls made (pass --run to call).")
        return 0

    # Everything below makes model calls.
    if args.dry_run:
        snapshot_config(MODELS, sha_file)
        run_design1.run_llm_jobs(dry, DRYRUN_PATH)
        recs = list(run_design1.latest(run_design1.read_jsonl(DRYRUN_PATH)).values())
        v = dry_run_validity(recs, MODELS)
        v.to_csv(DRYRUN_VALIDITY_PATH, index=False)
        print(v.to_string(index=False))
        return 0
    assert DRYRUN_VALIDITY_PATH.exists(), "run the dry run first (--run --dry-run)"
    v = pd.read_csv(DRYRUN_VALIDITY_PATH)
    retained = [m for m in MODELS if bool(v.set_index("reader").loc[m, "retained"])]
    dropped = [m for m in MODELS if m not in retained]
    print(f"retained after dry run: {retained}; dropped: {dropped}")
    snapshot_config(retained, sha_file)
    run_design1.run_llm_jobs([j for j in jobs if j["reader"] in retained], RESULTS_PATH)
    run_design1.report_calls(RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
