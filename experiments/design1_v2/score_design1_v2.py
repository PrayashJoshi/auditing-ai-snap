"""Design 1 v2 scorer. Zero calls: reads inputs.jsonl and, when it exists, results.jsonl.

Household is the unit. Repeat 1 is primary; agreement between repeats 1 and 2 is reported beside it.
Primary metrics are on the target group: hit rate on omissions, false-flag rate on true zeros (and on truthful
documented households). Flags on any group are secondary. Wilson 95% intervals; the difference in reader
hypothesis 1 uses Newcombe's hybrid score interval, built from the two Wilson intervals. Reader hypothesis 3 is
scored on the income-group flag (either income group), with the target-group rate reported beside it; hypotheses 1
and 2 stay on the target group. Deterministic
comparators (the frozen checker, and the checker plus the budget-gap rule of the first reader run) are scored on the
same 264 inputs. No precision at balanced prevalence and no caseload precision is computed.

    PYTHONPATH=src python experiments/design1_v2/score_design1_v2.py              # score
    PYTHONPATH=src python experiments/design1_v2/score_design1_v2.py --self-test  # scorer test on fake outputs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("run_design1_v2", HERE / "run_design1_v2.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
run_design1 = runner.run_design1
_cspec = importlib.util.spec_from_file_location("comparators_v2", HERE / "comparators_v2.py")
comparators_v2 = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(comparators_v2)

Z = 1.959963984540054
COMPARATORS = {"frozen_checker": run_design1.run_frozen_checker,
               "checker_plus_budget_gap_rule": comparators_v2.run_checker_plus_budget_gap_rule}
INCOME = {"earned", "unearned"}
HYP1_CLS, HYP1_NEG = ("omission", "none"), ("zero_rent", "none")
HYP2_CELL, HYP2_MIN = ("omission", "lead"), 0.90
HYP3_CELL, HYP3_MIN_RATE, HYP3_MIN_READERS = ("zero_stray", "stray_ambiguous"), 0.20, 2

SCORES_PATH = HERE / "scores.csv"
AGREEMENT_PATH = HERE / "repeat-agreement.csv"
HYPOTHESES_PATH = HERE / "hypotheses.csv"
COMPARATORS_PATH = HERE / "comparators.csv"
REFERENCE_PATH = HERE / "reference-false-flag-rates.csv"


# --- Intervals ------------------------------------------------------------------------------------------
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, mid - half), 1.0 if k == n else min(1.0, mid + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 with Newcombe's hybrid score 95% interval (independent samples)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


# --- Flags ----------------------------------------------------------------------------------------------
def households() -> dict[int, pd.Series]:
    hh = pd.read_csv(runner.build_inputs.HOUSEHOLDS_PATH, index_col="row")
    return {int(r): h for r, h in hh.iterrows()}


def comparator_flags(inputs: list[dict], hh: dict[int, pd.Series]) -> pd.DataFrame:
    rows = []
    for x in inputs:
        for name, fn in COMPARATORS.items():
            fl = fn(x, hh[x["row"]])
            rows.append(_flag_row(x, name, 1, True, {f["group"] for f in fl}))
    return pd.DataFrame(rows)


def reader_flags(inputs: list[dict], records: list[dict]) -> pd.DataFrame:
    by_id = {x["input_id"]: x for x in inputs}
    rows = []
    for r in records:
        x = by_id[r["input_id"]]
        valid = runner.reader_output_valid(r)
        flagged = {f["group"] for f in (r.get("parsed") or {}).get("flags") or [] if isinstance(f, dict)} if valid else set()
        rows.append(_flag_row(x, r["reader"], int(r["repeat"]), valid, flagged))
    return pd.DataFrame(rows)


def _flag_row(x: dict, reader: str, repeat: int, valid: bool, flagged: set) -> dict:
    return {"reader": reader, "repeat": repeat, "input_id": x["input_id"], "row": x["row"], "cls": x["cls"],
            "evidence_state": x["evidence_state"], "target_group": x["target_group"], "valid": valid,
            "flagged": ",".join(sorted(flagged)), "target_flag": int(x["target_group"] in flagged),
            "income_flag": int(bool(flagged & INCOME)), "any_flag": int(bool(flagged))}


# --- Scores ---------------------------------------------------------------------------------------------
def cell_scores(flags: pd.DataFrame, repeat: int = 1) -> pd.DataFrame:
    """One row per reader x class x state: target-group rate (hit rate on omissions, false-flag rate otherwise)
    with its Wilson interval; income-group and any-group rates beside it. Invalid outputs are left out of the
    denominator and counted."""
    out = []
    f = flags[flags["repeat"] == repeat]
    for (reader, cls, st), g in f.groupby(["reader", "cls", "evidence_state"], sort=False):
        v = g[g["valid"]]
        n = len(v)
        row = {"reader": reader, "repeat": repeat, "cls": cls, "evidence_state": st,
               "metric": "hit_rate" if cls == "omission" else "false_flag_rate",
               "households": len(g), "invalid": int((~g["valid"]).sum()), "n": n}
        for col in ("target_flag", "income_flag", "any_flag"):
            k = int(v[col].sum())
            lo, hi = wilson(k, n)
            name = col.replace("_flag", "")
            row |= {f"{name}_k": k, f"{name}_rate": (k / n) if n else float("nan"), f"{name}_lo": lo, f"{name}_hi": hi}
        out.append(row)
    return pd.DataFrame(out)


def repeat_agreement(flags: pd.DataFrame) -> pd.DataFrame:
    """Share of households whose repeat 1 and repeat 2 agree, on the target flag and on the whole flag set.
    Only households with valid output in both repeats count."""
    r1 = flags[flags["repeat"] == 1]
    r2 = flags[flags["repeat"] == 2][["reader", "input_id", "valid", "flagged", "target_flag"]]
    both = r1.merge(r2, on=["reader", "input_id"], suffixes=("", "_2"), how="inner")
    both = both[both["valid"] & both["valid_2"]]
    out = []
    for (reader, cls, st), g in both.groupby(["reader", "cls", "evidence_state"], sort=False):
        n = len(g)
        kt = int((g["target_flag"] == g["target_flag_2"]).sum())
        ks = int((g["flagged"] == g["flagged_2"]).sum())
        out.append({"reader": reader, "cls": cls, "evidence_state": st, "n_pairs": n,
                    "target_agree_k": kt, "target_agree": kt / n, "set_agree_k": ks, "set_agree": ks / n})
    return pd.DataFrame(out)


def hypotheses(scores: pd.DataFrame, readers: list[str]) -> pd.DataFrame:
    """The three reader hypotheses on repeat 1, target group, for the readers given (those retained after
    the dry run)."""
    s = scores[scores["repeat"] == 1].set_index(["reader", "cls", "evidence_state"])
    rows = []
    for r in readers:
        a, b = s.loc[(r, *HYP1_CLS)], s.loc[(r, *HYP1_NEG)]
        d, lo, hi = newcombe_diff(int(a["target_k"]), int(a["n"]), int(b["target_k"]), int(b["n"]))
        rows.append({"hypothesis": "reader hypothesis 1", "reader": r, "estimate": d, "lo": lo, "hi": hi,
                     "rule": "95% interval of hit rate at none minus false-flag rate on rent true zeros includes 0",
                     "holds": bool(lo <= 0 <= hi)})
    for r in readers:
        a = s.loc[(r, *HYP2_CELL)]
        rows.append({"hypothesis": "reader hypothesis 2", "reader": r, "estimate": a["target_rate"],
                     "lo": a["target_lo"], "hi": a["target_hi"], "rule": "hit rate at lead >= 0.90",
                     "holds": bool(a["target_rate"] >= HYP2_MIN)})
    meets = []
    for r in readers:
        a = s.loc[(r, *HYP3_CELL)]
        ok = bool(a["income_rate"] >= HYP3_MIN_RATE)
        meets.append(ok)
        rows.append({"hypothesis": "reader hypothesis 3 (per reader)", "reader": r, "estimate": a["income_rate"],
                     "lo": a["income_lo"], "hi": a["income_hi"], "target_rate_beside": a["target_rate"],
                     "rule": "income-group false-flag rate on stray-credit true zeros, ambiguous label, >= 0.20",
                     "holds": ok})
    rows.append({"hypothesis": "reader hypothesis 3", "reader": f"{sum(meets)} of {len(readers)}",
                 "estimate": float(sum(meets)), "lo": np.nan, "hi": np.nan,
                 "rule": "at least 2 readers meet the per-reader rule", "holds": bool(sum(meets) >= HYP3_MIN_READERS)})
    return pd.DataFrame(rows)


def reference_rates() -> pd.DataFrame:
    """False-flag rate a reader would need for 80% precision at 80% recall, at the two prevalence figures,
    and the Wilson upper bound with zero flags in 48 negatives. Arithmetic only, no reader output."""
    rows = []
    for label, p in (("known rebuilt omissions among $0 filings", 0.0046), ("all-cause upper bracket", 0.0247)):
        rows.append({"item": f"false-flag rate for 80% precision at 80% recall, prevalence {p:.2%} ({label})",
                     "value": 0.8 * p * (1 / 0.8 - 1) / (1 - p)})
    rows.append({"item": "Wilson 95% upper bound, 0 flags in 48 negatives", "value": wilson(0, 48)[1]})
    return pd.DataFrame(rows)


def score(inputs: list[dict], records: list[dict], hh: dict, readers: list[str] | None = None) -> dict:
    comp = comparator_flags(inputs, hh)
    flags = pd.concat([comp, reader_flags(inputs, records)], ignore_index=True) if records else comp
    scores = pd.concat([cell_scores(flags, 1), cell_scores(flags[flags["reader"].isin(set(r["reader"] for r in records))], 2)],
                       ignore_index=True) if records else cell_scores(flags, 1)
    present = sorted({r["reader"] for r in records}, key=lambda m: runner.MODELS.index(m) if m in runner.MODELS else 99)
    readers = present if readers is None else readers
    return {"flags": flags, "scores": scores,
            "agreement": repeat_agreement(flags[flags["reader"].isin(present)]) if records else pd.DataFrame(),
            "hypotheses": hypotheses(scores, readers) if readers else pd.DataFrame(),
            "comparators": cell_scores(comp, 1)}


# --- Self-test on fake reader outputs -------------------------------------------------------------------
def _fake_records(inputs: list[dict]) -> list[dict]:
    """Four fake readers with known answers. perfect: flags the target on omissions except at none.
    all: flags the target everywhere. silent: flags nothing. noisy: seeded random flags, a different draw per
    repeat, with every 10th repeat-1 output invalid."""
    rng = np.random.default_rng(7)
    recs = []
    for x in inputs:
        for rep in (1, 2):
            tg = x["target_group"]
            for name, flags, valid in (
                ("fake_perfect", [tg] if x["is_omission"] and x["evidence_state"] != "none" else [], True),
                ("fake_all", [tg, "shelter"], True),
                ("fake_silent", [], True),
                ("fake_noisy", [g for g in ("earned", "unearned", "shelter") if rng.random() < 0.3], True),
            ):
                rec = {"input_id": x["input_id"], "reader": name, "repeat": rep, "valid": valid,
                       "parsed": {"flags": [{"group": g, "confidence": 0.9, "reason": "test"} for g in flags]}}
                recs.append(rec)
    noisy_r1 = [r for r in recs if r["reader"] == "fake_noisy" and r["repeat"] == 1]
    for r in noisy_r1[::10]:
        r["valid"] = False
    return recs


def self_test() -> None:
    assert abs(wilson(0, 48)[1] - 0.0741) < 1e-3, wilson(0, 48)
    lo, hi = wilson(12, 24)
    assert abs(lo - 0.3138) < 1e-3 and abs(hi - 0.6862) < 1e-3, (lo, hi)
    d, lo, hi = newcombe_diff(0, 24, 0, 48)
    assert d == 0 and lo < 0 < hi
    d, lo, hi = newcombe_diff(24, 24, 0, 48)
    assert d == 1 and lo > 0.8 and hi == 1.0
    # A dropped repeat-2 duplicate or extra group must not pass validity.
    assert not runner.reader_output_valid({"valid": True, "parsed": {"flags": [{"group": "earned"}, {"group": "earned"}]}})
    assert not runner.reader_output_valid({"valid": False, "parsed": {"flags": []}})
    assert runner.reader_output_valid({"valid": True, "parsed": {"flags": []}})

    inputs = runner.load_inputs()
    hh = households()
    recs = _fake_records(inputs)
    fake = ["fake_perfect", "fake_all", "fake_silent", "fake_noisy"]
    out = score(inputs, recs, hh, readers=fake)
    s = out["scores"][out["scores"]["repeat"] == 1].set_index(["reader", "cls", "evidence_state"])
    for st, want in (("none", 0.0), ("deposits_clear", 1.0), ("deposits_ambiguous", 1.0), ("lead", 1.0)):
        assert s.loc[("fake_perfect", "omission", st), "target_rate"] == want
        assert s.loc[("fake_perfect", "omission", st), "n"] == 24
    for cell in (("zero_rent", "none"), ("zero_stray", "stray_clear"), ("zero_stray", "stray_ambiguous"),
                 ("truthful_documented", "documented")):
        assert s.loc[("fake_perfect", *cell), "target_rate"] == 0.0
        assert s.loc[("fake_all", *cell), "target_rate"] == 1.0
        assert s.loc[("fake_all", *cell), "any_rate"] == 1.0
        assert s.loc[("fake_silent", *cell), "any_rate"] == 0.0
    assert s.loc[("fake_perfect", "zero_rent", "none"), "n"] == 48
    # Invalid outputs leave the denominator.
    nz = out["scores"][(out["scores"]["reader"] == "fake_noisy") & (out["scores"]["repeat"] == 1)]
    assert nz["invalid"].sum() == 264 // 10 + (264 % 10 > 0) and (nz["n"] + nz["invalid"] == nz["households"]).all()
    # Repeat agreement: deterministic fakes agree fully, the noisy one does not.
    a = out["agreement"].set_index(["reader", "cls", "evidence_state"]).sort_index()
    assert (a.loc["fake_perfect", "set_agree"] == 1.0).all() and (a.loc["fake_all", "target_agree"] == 1.0).all()
    assert (a.loc["fake_noisy", "set_agree"] < 1.0).any()
    h = out["hypotheses"].set_index(["hypothesis", "reader"]).sort_index()
    assert h.loc[("reader hypothesis 1", "fake_perfect"), "holds"]          # 0 - 0, interval spans 0
    assert h.loc[("reader hypothesis 1", "fake_silent"), "holds"]
    assert h.loc[("reader hypothesis 1", "fake_all"), "holds"]              # 1 - 1
    assert h.loc[("reader hypothesis 2", "fake_perfect"), "holds"] and not h.loc[("reader hypothesis 2", "fake_silent"), "holds"]
    assert not h.loc[("reader hypothesis 3 (per reader)", "fake_perfect"), "holds"]
    assert h.loc[("reader hypothesis 3 (per reader)", "fake_all"), "holds"]
    # 1 + possibly the noisy reader (about 30% target flags) of 4 meet the rule.
    n_meet = int(h.loc[("reader hypothesis 3 (per reader)",), "holds"].sum())
    assert h.loc[("reader hypothesis 3", f"{n_meet} of 4"), "holds"] == (n_meet >= 2)
    # A reader that over-flags at none but not on rent true zeros fails hypothesis 1.
    over = [dict(r, reader="fake_over", parsed={"flags": [{"group": x["target_group"], "confidence": 1, "reason": "t"}]
                                                if x["is_omission"] else []})
            for r, x in ((r, next(i for i in inputs if i["input_id"] == r["input_id"])) for r in recs
                         if r["reader"] == "fake_silent")]
    h2 = score(inputs, over, hh)["hypotheses"].set_index(["hypothesis", "reader"])
    assert not h2.loc[("reader hypothesis 1", "fake_over"), "holds"]
    # Round trip through a results file, as the scorer reads it.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "results.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
        back = run_design1.read_jsonl(p)
        assert score(inputs, back, hh, readers=fake)["scores"].equals(out["scores"])
    print("SCORER SELF-TEST PASSED on", len(recs), "fake reader outputs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    inputs = runner.load_inputs()
    hh = households()
    records = list(run_design1.latest(run_design1.read_jsonl(runner.RESULTS_PATH)).values())
    retained = None
    if runner.DRYRUN_VALIDITY_PATH.exists():
        v = pd.read_csv(runner.DRYRUN_VALIDITY_PATH)
        retained = [m for m in v["reader"] if bool(v.set_index("reader").loc[m, "retained"])]
        records = [r for r in records if r["reader"] in retained]
    out = score(inputs, records, hh, readers=retained)
    out["comparators"].to_csv(COMPARATORS_PATH, index=False)
    reference_rates().to_csv(REFERENCE_PATH, index=False)
    cols = ["reader", "cls", "evidence_state", "metric", "n", "target_k", "target_rate", "target_lo", "target_hi",
            "income_k", "any_k"]
    print(out["comparators"][cols].to_string(index=False))
    print(reference_rates().to_string(index=False))
    if records:
        out["scores"].to_csv(SCORES_PATH, index=False)
        out["agreement"].to_csv(AGREEMENT_PATH, index=False)
        out["hypotheses"].to_csv(HYPOTHESES_PATH, index=False)
        print(out["hypotheses"].to_string(index=False))
    else:
        print("No reader results yet: comparators only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
