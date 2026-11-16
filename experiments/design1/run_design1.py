"""Design 1: Reader Omission Detection Diagnostic.

Builds 144 reader inputs (12 pilot households x 4 trace levels x 3 application states),
verifies pre-LLM invariants and deterministic baselines (`frozen_checker`, `checker_plus_2_rules`),
runs the 2 LLM readers (288 calls total) resumable by `job_id`, and writes `summary.csv`.

Run:
    uv run python experiments/design1/run_design1.py build
    uv run python experiments/design1/run_design1.py run
    uv run python experiments/design1/run_design1.py score
    uv run python experiments/design1/run_design1.py all
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from snapqc import checker, forms, llm, packets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "experiments" / "pilot"

# Import sua_line and reader_tool from experiments/pilot/run_pilot.py without modifying it.
_spec = importlib.util.spec_from_file_location("run_pilot", PILOT / "run_pilot.py")
run_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pilot)
sua_line = run_pilot.sua_line
reader_tool = run_pilot.reader_tool

MODELS = ["deepseek-v4-flash", "qwen3.5-397b"]
MAX_OUTPUT_TOKENS = 8192
CONCURRENCY = 8

TARGET_EARNED = [5744, 14542, 28402, 42865, 42777]
TARGET_UNEARNED = [28066, 30966, 21739, 43456, 41978, 34791, 30269]
HOUSEHOLD_TARGETS: dict[int, str] = {
    **{r: "earned" for r in TARGET_EARNED},
    **{r: "unearned" for r in TARGET_UNEARNED},
}
ALL_ROWS = TARGET_EARNED + TARGET_UNEARNED

# Level 3 split: 6 clear (3 earned + 3 unearned) / 6 ambiguous (2 earned + 4 unearned).
# First 3 of each group in prompt order are clear; remaining 2 + 4 are ambiguous.
# Notably this places 30269 (which has STATE CHILD SUPPORT DISBURSEMENT UNIT) in ambiguous
# so its description is masked to ACH CREDIT ELECTRONIC DEPOSIT, and ensures all 6 ambiguous
# households have TRANSFER FROM SAVINGS > $50 in the truthful twin.
L3_CLEAR = {5744, 14542, 28402, 28066, 30966, 21739}
L3_AMBIGUOUS = {42865, 42777, 43456, 41978, 34791, 30269}

TRACE_LEVELS = ["none", "budget_gap", "net_deposits", "wage_match"]
LEVEL_NUM = {"none": 1, "budget_gap": 2, "net_deposits": 3, "wage_match": 4}
STATES = ["omission_zero", "omission_not_provided", "truthful_twin"]
STATE_MARKER = {
    "omission_zero": "$0",
    "omission_not_provided": "not_provided",
    "truthful_twin": "twin",
}

LEAD_TEXT = {
    "earned": "\nAgency cross-match lead (IEVS): state wage records show recent quarterly wages for a household member.",
    "unearned": "\nAgency cross-match lead (IEVS/SSA): external benefit records show recent monthly benefit payments for a household member.",
}

READER_INSTRUCTIONS = """You review SNAP applications for a state agency. You see the application's monthly total per fact group (whole dollars; "not provided" means the filer abstained or left the group blank without entering a dollar amount), the household's retrieved records, and any agency cross-match leads.
Your goal is to flag every fact group that likely contains a costly error or omission:
1. Contradiction or unsupported claim: a reported non-zero amount lacks supporting records or materially conflicts with the records.
2. Likely omission: a group reported as $0 or "not provided" is contradicted by records, recurring unexplained deposits, housing/utility outflows that exceed reported income, or an agency cross-match lead.
Only flag groups where the file gives concrete reason to doubt the reported status; do not flag a group solely because records are absent if the application reports $0 and nothing in the file conflicts with $0.
Call report_flags exactly once, giving for each flag your confidence from 0 to 1 that the group needs review and a one-line reason. Use an empty list if nothing warrants review."""

INPUTS_PATH = HERE / "inputs.jsonl"
RESULTS_PATH = HERE / "results.jsonl"
SUMMARY_PATH = HERE / "summary.csv"
CONFIG_PATH = HERE / "run-config.json"

LEAK_RE = re.compile(
    r"\b(verified|verif\w*|qc|correct\w*|true|errors?|actual\w*|filed|oracle|ledger|facts?)\b|\bf\d{2}\b",
    re.IGNORECASE,
)
APP_VALUE_RE = re.compile(r"^\$\d{1,3}(,\d{3})*$|^not provided$")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def latest(records: list[dict]) -> dict[str, dict]:
    out = {}
    for r in records:
        if r.get("ok_http") or r["job_id"] not in out:
            out[r["job_id"]] = r
    return out


def load_pilot_data() -> tuple[dict[int, dict], dict[int, dict]]:
    df = pd.read_csv(PILOT / "households.csv")
    hh = {int(r["row"]): r for r in df.to_dict("records")}
    pk = {}
    for line in (PILOT / "packets.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        if p["variant"] == "complete":
            pk[int(p["row"])] = p
    return hh, pk


def infer_fact_groups(complete_packet: dict) -> dict[str, str]:
    """Map each fact id (f01, ...) in a complete packet to its fact group via checker._group_of."""
    fact_groups: dict[str, str] = {}
    all_fids = {fid for fid in complete_packet["sources"].values() if fid}
    for d in complete_packet["documents"]:
        fid = complete_packet["sources"].get(d["id"])
        if fid is None:
            continue
        g = checker._group_of(d)
        if g is not None:
            if fid in fact_groups:
                assert fact_groups[fid] == g, f"Conflicting groups for {fid}: {fact_groups[fid]} vs {g}"
            fact_groups[fid] = g
    assert all_fids == set(fact_groups), f"Unmapped fact ids: {all_fids - set(fact_groups)}"
    return fact_groups


def render_reader_input(app_monthly: dict, clean_packet: dict, target: str, trace_level: str) -> str:
    base = checker.reader_view(app_monthly, clean_packet) + "\n" + sua_line(clean_packet["sua"]) + "\n"
    if trace_level == "wage_match":
        lead = LEAD_TEXT[target]
        marker = "\nState utility allowances:"
        assert marker in base
        base = base.replace(marker, lead + "\n" + marker, 1)
    return base


def build_single_input(row: int, target: str, hh_row: dict, complete_packet: dict,
                       fact_groups: dict[str, str], trace_level: str, state: str) -> dict:
    verified = {g: float(hh_row[f"verified_{g}"]) for g in forms.GROUPS}
    clarity = "clear" if row in L3_CLEAR else "ambiguous"
    is_omission = state != "truthful_twin"
    marker = STATE_MARKER[state]

    # Build app_monthly and document filter
    docs_in = copy.deepcopy(complete_packet["documents"])
    sources = complete_packet["sources"]

    def doc_group(d: dict) -> str | None:
        fid = sources.get(d["id"])
        return fact_groups[fid] if fid else None

    if trace_level == "none":
        app_monthly = {**verified, "shelter": 0.0, "utilities": 0.0}
        app_monthly[target] = forms.ESCAPE if state == "omission_not_provided" else 0.0
        kept = [d for d in docs_in if doc_group(d) not in (target, "shelter", "utilities")]

    elif trace_level == "budget_gap":
        if is_omission:
            app_monthly = dict(verified)
            app_monthly[target] = forms.ESCAPE if state == "omission_not_provided" else 0.0
            kept = [d for d in docs_in if doc_group(d) != target]
        else:
            app_monthly = dict(verified)
            kept = docs_in

    elif trace_level == "net_deposits":
        app_monthly = {**verified, "shelter": 0.0, "utilities": 0.0}
        if is_omission:
            app_monthly[target] = forms.ESCAPE if state == "omission_not_provided" else 0.0
            kept = [
                d for d in docs_in
                if (doc_group(d) not in (target, "shelter", "utilities"))
                or (doc_group(d) == target and d["kind"] == "transaction")
            ]
            if clarity == "ambiguous":
                for d in kept:
                    if doc_group(d) == target and d["kind"] == "transaction":
                        d["data"]["Description"] = "ACH CREDIT ELECTRONIC DEPOSIT"
        else:
            app_monthly[target] = 0.0
            kept = [d for d in docs_in if doc_group(d) not in (target, "shelter", "utilities")]
            if clarity == "ambiguous":
                replaced = 0
                for d in kept:
                    if d["kind"] == "transaction" and "TRANSFER FROM SAVINGS" in d["data"]["Description"]:
                        d["data"]["Description"] = "ACH CREDIT ELECTRONIC TRANSFER"
                        replaced += 1
                assert replaced == 1, f"Row {row}: expected 1 TRANSFER FROM SAVINGS, found {replaced}"

    elif trace_level == "wage_match":
        app_monthly = {**verified, "shelter": 0.0, "utilities": 0.0}
        if is_omission:
            app_monthly[target] = forms.ESCAPE if state == "omission_not_provided" else 0.0
            kept = [d for d in docs_in if doc_group(d) not in (target, "shelter", "utilities")]
        else:
            kept = [d for d in docs_in if doc_group(d) not in ("shelter", "utilities")]

    else:
        raise ValueError(f"Unknown trace_level: {trace_level}")

    # Renumber kept documents cleanly as doc-01, doc-02, ...
    for i, d in enumerate(kept, 1):
        d["id"] = f"doc-{i:02d}"

    # Strip metadata (sources, missing_facts, undocumented_groups, etc.)
    clean_packet = {
        "row": int(complete_packet["row"]),
        "state": complete_packet["state"],
        "sua": copy.deepcopy(complete_packet["sua"]),
        "account": complete_packet["account"],
        "window": list(complete_packet["window"]),
        "documents": kept,
    }

    text = render_reader_input(app_monthly, clean_packet, target, trace_level)
    omitted_monthly = {**verified, target: 0.0}
    omission_impact = abs(packets.benefit_of(omitted_monthly, hh_row) - packets.benefit_of(verified, hh_row))

    return {
        "input_id": f"{row}|{trace_level}|{state}",
        "row": row,
        "target_group": target,
        "trace_level": trace_level,
        "level_num": LEVEL_NUM[trace_level],
        "state": state,
        "is_omission": is_omission,
        "marker": marker,
        "l3_clarity": clarity,
        "app_monthly": app_monthly,
        "verified_monthly": verified,
        "omission_impact": float(omission_impact),
        "packet": clean_packet,
        "reader_input": text,
        "reader_input_sha": llm.sha(text),
    }


def build_all_inputs() -> list[dict]:
    hh, pk = load_pilot_data()
    inputs = []
    for row in ALL_ROWS:
        target = HOUSEHOLD_TARGETS[row]
        hh_row = hh[row]
        complete_packet = pk[row]
        fact_groups = infer_fact_groups(complete_packet)
        for trace_level in TRACE_LEVELS:
            for state in STATES:
                inputs.append(build_single_input(row, target, hh_row, complete_packet, fact_groups, trace_level, state))
    return inputs


# --- Deterministic Baselines -------------------------------------------------------------------------------
def _is_zero_or_missing(v) -> bool:
    return v == forms.ESCAPE or v is None or float(v) == 0.0


def _num_val(v) -> float:
    return 0.0 if (v == forms.ESCAPE or v is None) else float(v)


def run_frozen_checker(inp: dict, hh_row: dict) -> list[dict]:
    return checker.check(inp["app_monthly"], inp["packet"], hh_row)


def _matching_supporting_doc(desc: str, docs: list[dict], app_monthly: dict, packet: dict) -> bool:
    """Whether a credit transaction description is already accounted for by a supporting document
    (pay_stub for PAYROLL DEP, award_letter/benefit_notice for benefit deposits, or reported/oracle-documented
    STATE CHILD SUPPORT DISBURSEMENT UNIT)."""
    upper = desc.upper()
    if upper.startswith("PAYROLL DEP "):
        employer = upper[len("PAYROLL DEP "):].strip()
        return any(
            d["kind"] == "pay_stub" and d["data"].get("Employer", "").upper() == employer
            for d in docs
        )
    if desc == packets.CHILD_SUPPORT_LABEL:
        # Child support has no award letter in packets.py; checker._group_of maps it directly to "unearned".
        # It is matched if unearned is reported (> $0) and documented in the packet.
        return (not _is_zero_or_missing(app_monthly.get("unearned"))) and ("unearned" in checker.documented_groups(packet))
    if upper in ("SOC SEC BENEFIT SSA", "SSI BENEFIT SSA"):
        want_benefit = "Social Security benefits" if upper.startswith("SOC SEC") else "Supplemental Security Income (SSI)"
        return any(d["kind"] == "award_letter" and d["data"].get("Benefit") == want_benefit for d in docs)
    if upper.endswith(" PAYMENT"):
        payer = upper[:-len(" PAYMENT")].strip()
        return any(d["kind"] == "benefit_notice" and d["data"].get("From", "").upper() == payer for d in docs)
    return False


def run_checker_plus_2_rules(inp: dict, hh_row: dict) -> list[dict]:
    app = inp["app_monthly"]
    packet = inp["packet"]
    text = inp["reader_input"]
    docs = packet["documents"]
    documented = checker.documented_groups(packet)

    by_group: dict[str, list[str]] = {}
    for f in checker.check(app, packet, hh_row):
        by_group.setdefault(f["group"], []).append(f["reason"])

    # Rule A (budget gap): observed lease monthly rent + sum of utility_bill amounts > total reported monthly income
    lease_rent = sum(float(d["data"]["Monthly rent"]) for d in docs if d["kind"] == "lease")
    util_bills = sum(float(d["data"]["Amount due"]) for d in docs if d["kind"] == "utility_bill")
    reported_income = _num_val(app.get("earned")) + _num_val(app.get("unearned"))
    if lease_rent + util_bills > reported_income:
        for g in ("earned", "unearned"):
            if _is_zero_or_missing(app.get(g)) and g not in documented:
                by_group.setdefault(g, []).append(
                    f"Rule A (budget gap): observed housing ${lease_rent + util_bills:,.2f} > reported income ${reported_income:,.0f}"
                )

    # Rule B (unmatched lead/deposit)
    # 1. Agency cross-match lead
    for line in text.splitlines():
        if "Agency cross-match lead" in line:
            if "IEVS/SSA" in line or "external benefit records" in line:
                if _is_zero_or_missing(app.get("unearned")):
                    by_group.setdefault("unearned", []).append("Rule B (lead): unmatched IEVS/SSA benefit lead")
            elif "IEVS" in line or "state wage records" in line:
                if _is_zero_or_missing(app.get("earned")):
                    by_group.setdefault("earned", []).append("Rule B (lead): unmatched IEVS wage lead")

    # 2. >= 2 same-description credit transactions > $50 (excluding TRANSFER FROM SAVINGS and REFUND)
    # without corresponding pay stubs/award letters
    desc_counts: Counter[str] = Counter()
    for d in docs:
        if d["kind"] != "transaction":
            continue
        amt = float(d["data"]["Amount"])
        desc = str(d["data"]["Description"])
        upper = desc.upper()
        if amt <= 50.0:
            continue
        if "TRANSFER FROM SAVINGS" in upper or "REFUND" in upper:
            continue
        if _matching_supporting_doc(desc, docs, app, packet):
            continue
        desc_counts[desc] += 1

    unmatched_recurring = [desc for desc, c in desc_counts.items() if c >= 2]
    if unmatched_recurring:
        for g in ("earned", "unearned"):
            if _is_zero_or_missing(app.get(g)):
                by_group.setdefault(g, []).append(
                    f"Rule B (deposits): >=2 unmatched credit txns >$50 ({', '.join(sorted(unmatched_recurring))})"
                )

    return [{"group": g, "reason": "; ".join(reasons)} for g, reasons in by_group.items()]


# --- Pre-LLM Verification ---------------------------------------------------------------------------------
def verify_inputs(inputs: list[dict], hh: dict[int, dict]) -> None:
    assert len(inputs) == 144, f"Expected 144 inputs, got {len(inputs)}"
    assert len({x["input_id"] for x in inputs}) == 144

    for x in inputs:
        p = x["packet"]
        text = x["reader_input"]
        # 1. No metadata in packet dict
        assert not ({"sources", "missing_facts", "undocumented_groups", "notes", "filer_input", "variant"} & set(p)), x["input_id"]
        # 2. Clean sequential doc-01..doc-NN
        expected_ids = [f"doc-{i:02d}" for i in range(1, len(p["documents"]) + 1)]
        assert [d["id"] for d in p["documents"]] == expected_ids, x["input_id"]
        # 3. Leak scan on reader_input and field names
        hits = [m.group(0) for m in LEAK_RE.finditer(text)]
        assert not hits, f"Metadata leak in {x['input_id']}: {hits}"
        field_names = " ".join(sorted({k for d in p["documents"] for k in [*d["data"], d["kind"], d["connector"]]}))
        assert not LEAK_RE.search(field_names), f"Field name leak in {x['input_id']}"
        # 4. Format guard on application lines
        app_lines = text.split("\n\nDocuments:")[0].splitlines()[1:]
        assert [l.split(": ")[0] for l in app_lines] == [checker.GROUP_LABEL[g] for g in forms.GROUPS], x["input_id"]
        assert all(APP_VALUE_RE.match(l.split(": ", 1)[1]) for l in app_lines), x["input_id"]
        # 5. Omission benefit impact > $56
        assert x["omission_impact"] > 56.0, f"Impact <= $56 for {x['input_id']}: {x['omission_impact']}"

    # 6. Level 2 housing > reported income for 12/12 omissions (and <= for 12/12 twins)
    l2_om = [x for x in inputs if x["trace_level"] == "budget_gap" and x["is_omission"]]
    l2_tw = [x for x in inputs if x["trace_level"] == "budget_gap" and not x["is_omission"]]
    assert len(l2_om) == 24 and len(l2_tw) == 12
    for x in l2_om:
        docs = x["packet"]["documents"]
        housing = sum(d["data"]["Monthly rent"] for d in docs if d["kind"] == "lease") + sum(
            d["data"]["Amount due"] for d in docs if d["kind"] == "utility_bill"
        )
        rep_inc = _num_val(x["app_monthly"]["earned"]) + _num_val(x["app_monthly"]["unearned"])
        assert housing > rep_inc, f"L2 omission {x['input_id']}: housing {housing} <= rep_inc {rep_inc}"
    for x in l2_tw:
        docs = x["packet"]["documents"]
        housing = sum(d["data"]["Monthly rent"] for d in docs if d["kind"] == "lease") + sum(
            d["data"]["Amount due"] for d in docs if d["kind"] == "utility_bill"
        )
        rep_inc = _num_val(x["app_monthly"]["earned"]) + _num_val(x["app_monthly"]["unearned"])
        assert housing <= rep_inc, f"L2 twin {x['input_id']}: housing {housing} > rep_inc {rep_inc}"

    # 7. Level 3 split verification
    l3 = [x for x in inputs if x["trace_level"] == "net_deposits"]
    clear_rows = {x["row"] for x in l3 if x["l3_clarity"] == "clear"}
    ambig_rows = {x["row"] for x in l3 if x["l3_clarity"] == "ambiguous"}
    assert len(clear_rows) == 6 and len(ambig_rows) == 6
    assert sum(HOUSEHOLD_TARGETS[r] == "earned" for r in clear_rows) == 3
    assert sum(HOUSEHOLD_TARGETS[r] == "unearned" for r in clear_rows) == 3
    assert sum(HOUSEHOLD_TARGETS[r] == "earned" for r in ambig_rows) == 2
    assert sum(HOUSEHOLD_TARGETS[r] == "unearned" for r in ambig_rows) == 4

    print("ALL 144 INPUT VERIFICATION ASSERTIONS PASSED.", flush=True)


def print_samples(inputs: list[dict]) -> None:
    print("\n=== SAMPLE INPUTS (1 per trace_level x state) ===")
    # Pick 5744 (earned, clear) for clear sample, or vary across levels to show clear & ambiguous
    sample_row_by_level = {
        "none": 5744,
        "budget_gap": 14542,
        "net_deposits": 42865,  # ambiguous earned
        "wage_match": 30966,    # unearned
    }
    for lvl in TRACE_LEVELS:
        row = sample_row_by_level[lvl]
        for st in STATES:
            x = next(i for i in inputs if i["row"] == row and i["trace_level"] == lvl and i["state"] == st)
            print(f"\n--- [{x['input_id']}] target={x['target_group']} clarity={x['l3_clarity']} ---")
            print(x["reader_input"].rstrip())


# --- LLM Execution ----------------------------------------------------------------------------------------
def snapshot_config() -> None:
    tool = reader_tool(True)
    cfg = {
        "models": MODELS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "concurrency": CONCURRENCY,
        "temperature": llm.TEMPERATURE,
        "instructions_sha": llm.sha(READER_INSTRUCTIONS),
        "instructions": READER_INSTRUCTIONS,
        "tool_sha": llm.tool_sha(tool),
        "tool": tool,
        "target_earned": TARGET_EARNED,
        "target_unearned": TARGET_UNEARNED,
        "l3_clear": sorted(L3_CLEAR),
        "l3_ambiguous": sorted(L3_AMBIGUOUS),
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=1) + "\n")


def build_jobs(inputs: list[dict]) -> list[dict]:
    tool = reader_tool(True)
    jobs = []
    for x in inputs:
        for model in MODELS:
            jobs.append({
                "job_id": f"design1|{x['input_id']}|{model}",
                "input_id": x["input_id"],
                "row": x["row"],
                "target_group": x["target_group"],
                "trace_level": x["trace_level"],
                "level_num": x["level_num"],
                "state": x["state"],
                "is_omission": x["is_omission"],
                "marker": x["marker"],
                "l3_clarity": x["l3_clarity"],
                "reader": model,
                "app_monthly": x["app_monthly"],
                "reader_input_sha": x["reader_input_sha"],
                "_model": model,
                "_instructions": READER_INSTRUCTIONS,
                "_input": x["reader_input"],
                "_tool": tool,
            })
    return jobs


def run_llm_jobs(jobs: list[dict], out: Path) -> None:
    done = {j for j, r in latest(read_jsonl(out)).items() if r.get("ok_http")}
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"{out.name}: {len(jobs)} jobs, {len(jobs) - len(todo)} already answered, {len(todo)} to run", flush=True)
    if not todo:
        return
    cl = llm.client()
    lock = threading.Lock()
    n = [0]
    with out.open("a") as fh, ThreadPoolExecutor(CONCURRENCY) as pool:
        def work(job):
            meta = {k: v for k, v in job.items() if not k.startswith("_")}
            rec = llm.call(cl, job["_model"], job["_instructions"], job["_input"], job["_tool"],
                           MAX_OUTPUT_TOKENS, meta)
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                n[0] += 1
                if n[0] % 20 == 0 or n[0] == len(todo):
                    print(f"  {n[0]}/{len(todo)} done", flush=True)
            return rec
        list(pool.map(work, todo))


def report_calls(path: Path) -> None:
    recs = list(latest(read_jsonl(path)).values())
    if not recs:
        print("No LLM results yet.")
        return
    trunc = sum(
        r.get("status") == "incomplete" or (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
        for r in recs
    )
    lat = [r["latency_s"] for r in recs if r.get("ok_http")]
    tok = {k: sum(r.get(k) or 0 for r in recs) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
    wall = max(r["t_end"] for r in recs) - min(r["t_start"] for r in recs)
    print(
        f"\n{path.name}: {len(recs)} jobs, {sum(bool(r['valid']) for r in recs)} valid, "
        f"{trunc} truncated, {sum(r['retries'] for r in recs)} retries, wall {wall / 60:.1f} min, "
        f"credits {sum(r.get('cost') or 0 for r in recs):.4f}, median latency "
        f"{statistics.median(lat) if lat else float('nan'):.1f}s, tokens {tok}"
    )
    by = Counter((r["reader"], bool(r["valid"])) for r in recs)
    print("  valid by model:", dict(sorted(by.items(), key=str)))
    for r in recs:
        if not r["valid"]:
            print(f"  INVALID {r['job_id']}: {r.get('invalid_reason')}")


# --- Scoring ----------------------------------------------------------------------------------------------
def evaluate_all_readers(inputs: list[dict], hh: dict[int, dict], llm_recs: list[dict]) -> pd.DataFrame:
    """Return one row per (reader, input_id) for all 4 readers (if LLM results present) or 2 baselines."""
    records = []
    # Deterministic baselines
    for x in inputs:
        hh_row = hh[x["row"]]
        for reader_name, fn in [("frozen_checker", run_frozen_checker), ("checker_plus_2_rules", run_checker_plus_2_rules)]:
            fl = fn(x, hh_row)
            flagged = {f["group"] for f in fl}
            records.append({
                "reader": reader_name,
                "input_id": x["input_id"],
                "row": x["row"],
                "target_group": x["target_group"],
                "trace_level": x["trace_level"],
                "level_num": x["level_num"],
                "state": x["state"],
                "is_omission": x["is_omission"],
                "marker": x["marker"],
                "l3_clarity": x["l3_clarity"],
                "valid": True,
                "flagged": sorted(flagged),
                "n_flags": len(flagged),
                "exact_target_hit": int(x["target_group"] in flagged) if x["is_omission"] else np.nan,
                "any_income_hit": int(bool(flagged & {"earned", "unearned"})) if x["is_omission"] else np.nan,
                "twin_any_flag": int(len(flagged) > 0) if not x["is_omission"] else np.nan,
                "twin_income_flag": int(bool(flagged & {"earned", "unearned"})) if not x["is_omission"] else np.nan,
                "reasons": "; ".join(f"{f['group']}: {f['reason']}" for f in fl),
                "confidences": {},
            })

    # LLM readers
    inp_by_id = {x["input_id"]: x for x in inputs}
    for r in llm_recs:
        x = inp_by_id[r["input_id"]]
        parsed = r.get("parsed") or {}
        flags_list = parsed.get("flags") or []
        flagged = {f["group"] for f in flags_list if isinstance(f, dict) and "group" in f}
        confs = {f["group"]: f.get("confidence") for f in flags_list if isinstance(f, dict) and "group" in f}
        reasons = "; ".join(
            f"{f.get('group')}: {f.get('reason', '')}" for f in flags_list if isinstance(f, dict)
        )
        records.append({
            "reader": r["reader"],
            "input_id": x["input_id"],
            "row": x["row"],
            "target_group": x["target_group"],
            "trace_level": x["trace_level"],
            "level_num": x["level_num"],
            "state": x["state"],
            "is_omission": x["is_omission"],
            "marker": x["marker"],
            "l3_clarity": x["l3_clarity"],
            "valid": bool(r.get("valid")),
            "flagged": sorted(flagged),
            "n_flags": len(flagged),
            "exact_target_hit": int(x["target_group"] in flagged) if x["is_omission"] else np.nan,
            "any_income_hit": int(bool(flagged & {"earned", "unearned"})) if x["is_omission"] else np.nan,
            "twin_any_flag": int(len(flagged) > 0) if not x["is_omission"] else np.nan,
            "twin_income_flag": int(bool(flagged & {"earned", "unearned"})) if not x["is_omission"] else np.nan,
            "reasons": reasons,
            "confidences": confs,
        })

    return pd.DataFrame(records)


def build_summary_table(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Compute hit rates, twin FPR, and net separation per (reader, trace_level, marker) + L3 clear/ambig."""
    readers_present = [r for r in ["frozen_checker", "checker_plus_2_rules", *MODELS] if r in df_eval["reader"].unique()]
    rows = []

    twins = df_eval[~df_eval["is_omission"]]
    omissions = df_eval[df_eval["is_omission"]]

    for reader in readers_present:
        for lvl in TRACE_LEVELS:
            tw = twins[(twins["reader"] == reader) & (twins["trace_level"] == lvl)]
            twin_any_fpr = float(tw["twin_any_flag"].mean())
            twin_inc_fpr = float(tw["twin_income_flag"].mean())
            for marker in ["$0", "not_provided"]:
                om = omissions[(omissions["reader"] == reader) & (omissions["trace_level"] == lvl) & (omissions["marker"] == marker)]
                exact_hit = float(om["exact_target_hit"].mean())
                any_hit = float(om["any_income_hit"].mean())
                rows.append({
                    "reader": reader,
                    "trace_level": lvl,
                    "subset": "all",
                    "marker": marker,
                    "n_omission": len(om),
                    "n_twin": len(tw),
                    "exact_target_hits": int(om["exact_target_hit"].sum()),
                    "exact_target_hit": round(exact_hit, 4),
                    "any_income_hits": int(om["any_income_hit"].sum()),
                    "any_income_hit": round(any_hit, 4),
                    "twin_any_flags": int(tw["twin_any_flag"].sum()),
                    "twin_any_fpr": round(twin_any_fpr, 4),
                    "twin_income_flags": int(tw["twin_income_flag"].sum()),
                    "twin_income_fpr": round(twin_inc_fpr, 4),
                    "net_sep_exact_vs_any_fpr": round(exact_hit - twin_any_fpr, 4),
                    "net_sep_any_vs_any_fpr": round(any_hit - twin_any_fpr, 4),
                    "net_sep_exact_vs_inc_fpr": round(exact_hit - twin_inc_fpr, 4),
                    "net_sep_any_vs_inc_fpr": round(any_hit - twin_inc_fpr, 4),
                })

            # Level 3 clear vs ambiguous breakdown
            if lvl == "net_deposits":
                for clarity in ["clear", "ambiguous"]:
                    tw_c = tw[tw["l3_clarity"] == clarity]
                    tw_c_any = float(tw_c["twin_any_flag"].mean())
                    tw_c_inc = float(tw_c["twin_income_flag"].mean())
                    for marker in ["$0", "not_provided"]:
                        om_c = omissions[
                            (omissions["reader"] == reader)
                            & (omissions["trace_level"] == lvl)
                            & (omissions["l3_clarity"] == clarity)
                            & (omissions["marker"] == marker)
                        ]
                        ex_c = float(om_c["exact_target_hit"].mean())
                        an_c = float(om_c["any_income_hit"].mean())
                        rows.append({
                            "reader": reader,
                            "trace_level": lvl,
                            "subset": clarity,
                            "marker": marker,
                            "n_omission": len(om_c),
                            "n_twin": len(tw_c),
                            "exact_target_hits": int(om_c["exact_target_hit"].sum()),
                            "exact_target_hit": round(ex_c, 4),
                            "any_income_hits": int(om_c["any_income_hit"].sum()),
                            "any_income_hit": round(an_c, 4),
                            "twin_any_flags": int(tw_c["twin_any_flag"].sum()),
                            "twin_any_fpr": round(tw_c_any, 4),
                            "twin_income_flags": int(tw_c["twin_income_flag"].sum()),
                            "twin_income_fpr": round(tw_c_inc, 4),
                            "net_sep_exact_vs_any_fpr": round(ex_c - tw_c_any, 4),
                            "net_sep_any_vs_any_fpr": round(an_c - tw_c_any, 4),
                            "net_sep_exact_vs_inc_fpr": round(ex_c - tw_c_inc, 4),
                            "net_sep_any_vs_inc_fpr": round(an_c - tw_c_inc, 4),
                        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "run", "score", "all"])
    ap.add_argument("--print-samples", action="store_true")
    args = ap.parse_args()

    hh, _ = load_pilot_data()

    if args.action in ("build", "all"):
        inputs = build_all_inputs()
        verify_inputs(inputs, hh)
        INPUTS_PATH.write_text("".join(json.dumps(x) + "\n" for x in inputs))
        print(f"Wrote {len(inputs)} inputs to {INPUTS_PATH}")
        if args.print_samples:
            print_samples(inputs)
        # Verify deterministic baselines BEFORE running any LLM calls
        df_base = evaluate_all_readers(inputs, hh, [])
        base_summary = build_summary_table(df_base)
        print("\n=== DETERMINISTIC BASELINES (PRE-LLM CHECK) ===")
        print(base_summary[base_summary["subset"] == "all"][[
            "reader", "trace_level", "marker", "exact_target_hits", "any_income_hits",
            "twin_any_flags", "net_sep_exact_vs_any_fpr", "net_sep_any_vs_any_fpr"
        ]].to_string(index=False))
        print("\n=== DETERMINISTIC BASELINES (LEVEL 3 CLEAR VS AMBIGUOUS) ===")
        print(base_summary[base_summary["subset"] != "all"][[
            "reader", "subset", "marker", "exact_target_hits", "any_income_hits",
            "twin_any_flags", "net_sep_exact_vs_any_fpr"
        ]].to_string(index=False))

    if args.action in ("run", "all"):
        inputs = read_jsonl(INPUTS_PATH)
        verify_inputs(inputs, hh)
        snapshot_config()
        jobs = build_jobs(inputs)
        run_llm_jobs(jobs, RESULTS_PATH)
        report_calls(RESULTS_PATH)

    if args.action in ("score", "all"):
        inputs = read_jsonl(INPUTS_PATH)
        llm_recs = list(latest(read_jsonl(RESULTS_PATH)).values())
        df_eval = evaluate_all_readers(inputs, hh, llm_recs)
        summary = build_summary_table(df_eval)
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"\nSaved {SUMMARY_PATH}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
