"""Design 1 v2 comparator: the frozen checker plus the budget-gap rule.

The budget-gap rule is the first reader run's budget-gap rule, with the same logic and thresholds: when observed
lease rent plus the utility-bill amounts exceed the reported monthly income (earned plus unearned), every income
group that is reported as $0 (or missing) and has no document in the packet gets a flag. The first run's second
rule (unmatched lead or deposit) is not part of this comparator.

That rule sits inside `run_design1.run_checker_plus_2_rules` and cannot be called alone, so it is re-implemented
here. `verify_against_run_design1` proves the two agree on every record: for each input, this comparator's flag
reasons per group equal the first run's flag reasons with the second rule's text removed.

    PYTHONPATH=src python experiments/design1_v2/comparators_v2.py    # equivalence check on all 264 inputs
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from snapqc import checker

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_inputs", HERE / "build_inputs.py")
build_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_inputs)
run_design1 = build_inputs.run_design1

_is_zero_or_missing = run_design1._is_zero_or_missing
_num_val = run_design1._num_val


def run_checker_plus_budget_gap_rule(inp: dict, hh_row: dict) -> list[dict]:
    app = inp["app_monthly"]
    packet = inp["packet"]
    docs = packet["documents"]
    documented = checker.documented_groups(packet)

    by_group: dict[str, list[str]] = {}
    for f in checker.check(app, packet, hh_row):
        by_group.setdefault(f["group"], []).append(f["reason"])

    # Budget gap: observed lease monthly rent + sum of utility_bill amounts > total reported monthly income
    lease_rent = sum(float(d["data"]["Monthly rent"]) for d in docs if d["kind"] == "lease")
    util_bills = sum(float(d["data"]["Amount due"]) for d in docs if d["kind"] == "utility_bill")
    reported_income = _num_val(app.get("earned")) + _num_val(app.get("unearned"))
    if lease_rent + util_bills > reported_income:
        for g in ("earned", "unearned"):
            if _is_zero_or_missing(app.get(g)) and g not in documented:
                by_group.setdefault(g, []).append(
                    f"Rule A (budget gap): observed housing ${lease_rent + util_bills:,.2f} > reported income ${reported_income:,.0f}"
                )

    return [{"group": g, "reason": "; ".join(reasons)} for g, reasons in by_group.items()]


def _reasons_without_second_rule(flags: list[dict]) -> dict[str, list[str]]:
    """Per group, the reason parts of the first run's flags that are not the second rule's."""
    out: dict[str, list[str]] = {}
    for f in flags:
        parts = [p for p in f["reason"].split("; ") if not p.startswith("Rule B")]
        if parts:
            out[f["group"]] = parts
    return out


def verify_against_run_design1() -> dict:
    inputs = [json.loads(line) for line in build_inputs.INPUTS_PATH.read_text().splitlines() if line.strip()]
    hh = pd.read_csv(build_inputs.HOUSEHOLDS_PATH, index_col="row")
    n = n_gap = 0
    for x in inputs:
        row = hh.loc[x["row"]]
        mine = {f["group"]: f["reason"].split("; ") for f in run_checker_plus_budget_gap_rule(x, row)}
        theirs = _reasons_without_second_rule(run_design1.run_checker_plus_2_rules(x, row))
        assert mine == theirs, (x["row"], mine, theirs)
        n += 1
        n_gap += any("Rule A" in r for rs in mine.values() for r in rs)
    return {"inputs_compared": n, "inputs_with_budget_gap_flag": n_gap, "identical": True}


if __name__ == "__main__":
    print(verify_against_run_design1())
