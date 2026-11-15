"""Post hoc analyses of Design 1 v2, computed after all results were seen. Zero model calls.

Five sections, one CSV each, written beside this script:
  1. flags-by-rule-state.csv          reader flags when the observable budget-gap rule fires vs when it is silent
  2. r1-lease-matched.csv             hit rate at omission/none minus false-flag rate at zero_rent/none, households with rent
  3. twoxtwo-by-reported-income.csv   the label x amount 2x2 on the 48 stray-credit true zeros, split by reported income
  4. any-group-false-flags.csv        any-group flag rate on zero_rent/none and the groups flagged
  5. qwen38-recurrence.csv            qwen3.8-27b on omission/deposits_ambiguous vs zero_stray/stray_ambiguous

Records: the frozen run (`experiments/design1_v2/results.jsonl`), the added-readers run
(`experiments/design1_v2_added/added-results.jsonl`), the amount-only arm and the clear-label twin. Validity is
the frozen `reader_output_valid`; the latest record per job is used; repeat 1 throughout. The frozen scorer is
loaded as a module, as `experiments/design1_v2_added/score_results.py` does, and its `main` is never run.

    PYTHONPATH=src python experiments/design1_v2_posthoc/posthoc.py

Every known count below is asserted at the end of its section.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # loading the frozen scorer must not leave bytecode files behind

import importlib.util
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, fisher_exact

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
_spec = importlib.util.spec_from_file_location("score_design1_v2", EXP / "design1_v2" / "score_design1_v2.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)  # module-level code only; main() is not called
runner = scorer.runner
run_design1 = scorer.run_design1
INCOME = scorer.INCOME

FROZEN = ["deepseek-v4-flash", "qwen3.5-397b", "gemma-4-31b-it"]
ADDED = ["glm-5.3-flash", "gemma-4-26b-a4b-it", "qwen3.8-27b"]
READERS = FROZEN + ADDED
RUNS = {
    "frozen": (EXP / "design1_v2/inputs.jsonl", EXP / "design1_v2/results.jsonl"),
    "added": (EXP / "design1_v2/inputs.jsonl", EXP / "design1_v2_added/added-results.jsonl"),
    "amount": (EXP / "design1_v2_amount/amount-inputs.jsonl", EXP / "design1_v2_amount/amount-results.jsonl"),
    "clear": (EXP / "design1_v2_amount_clear/clear-inputs.jsonl", EXP / "design1_v2_amount_clear/clear-results.jsonl"),
}


def read(path):
    return run_design1.read_jsonl(path)


INPUTS = read(RUNS["frozen"][0])
HH = pd.read_csv(EXP / "design1_v2" / "households.csv")


def run_flags(name: str) -> pd.DataFrame:
    inputs_path, results_path = RUNS[name]
    latest = list(run_design1.latest(read(results_path)).values())
    f = scorer.reader_flags(read(inputs_path), latest)
    f["run"] = name
    return f


def features(inputs: list[dict]) -> dict:
    out = {}
    for x in inputs:
        docs, app = x["packet"]["documents"], x["app_monthly"]
        rent = sum(float(e["data"]["Monthly rent"]) for e in docs if e["kind"] == "lease")
        bills = sum(float(e["data"]["Amount due"]) for e in docs if e["kind"] == "utility_bill")
        income = (app["earned"] or 0) + (app["unearned"] or 0)
        out[x["input_id"]] = {"reported_income": income, "any_income": int(income > 0), "has_lease": int(rent > 0),
                              "obs_gap": rent + bills - income}
    return out


FLAGS = pd.concat([run_flags(r) for r in RUNS], ignore_index=True)
FEAT = {}
for _name, (_ip, _rp) in RUNS.items():
    FEAT.update(features(read(_ip)))
for _col in ("any_income", "has_lease", "obs_gap"):
    FLAGS[_col] = [FEAT[i][_col] for i in FLAGS.input_id]

# The comparator `checker_plus_budget_gap_rule`: lease rent plus utility bills above reported income, and the
# income group reported as $0 with no document. Its target flag is the rule's verdict for the household.
_comp = scorer.comparator_flags(INPUTS, {int(r): h for r, h in HH.set_index("row").iterrows()})
_rule = _comp[_comp.reader == "checker_plus_budget_gap_rule"][["input_id", "target_flag"]].rename(columns={"target_flag": "rule_fires"})
FLAGS = FLAGS.merge(_rule, on="input_id", how="left")

F1 = FLAGS[FLAGS.repeat == 1]
NONE = F1[(F1.evidence_state == "none") & F1.run.isin(["frozen", "added"])]


def k_of(s, col):
    return int(s[col].sum()), len(s)


def fisher(a, b, c, d):
    return fisher_exact([[a, b], [c, d]])[1]


# --- 1. Flags by rule state -----------------------------------------------------------------------------
# At evidence state "none", valid outputs only: the target-group flag when the rule fires vs when it is silent.
rows = []
for scope, classes in [("omission and zero_rent pooled", ["omission", "zero_rent"]), ("omission", ["omission"]), ("zero_rent", ["zero_rent"])]:
    for r in READERS:
        s = NONE[(NONE.reader == r) & NONE.cls.isin(classes) & NONE.valid]
        fire, quiet = s[s.rule_fires == 1], s[s.rule_fires == 0]
        (k1, n1), (k0, n0) = k_of(fire, "target_flag"), k_of(quiet, "target_flag")
        rows.append({"scope": scope, "reader": r, "rule_fires_k": k1, "rule_fires_n": n1,
                     "rule_silent_k": k0, "rule_silent_n": n0,
                     "fisher_p": fisher(k1, n1 - k1, k0, n0 - k0) if n1 and n0 else float("nan")})
sec1 = pd.DataFrame(rows)
sec1.to_csv(HERE / "flags-by-rule-state.csv", index=False)
print("1. flags by rule state\n", sec1.to_string(index=False))
p = sec1[sec1.scope == "omission and zero_rent pooled"].set_index("reader")
for r, (k1, n1, k0, n0) in {"glm-5.3-flash": (12, 14, 0, 53), "gemma-4-31b-it": (6, 19, 0, 52), "qwen3.8-27b": (2, 19, 0, 53)}.items():
    got = tuple(int(p.loc[r, c]) for c in ("rule_fires_k", "rule_fires_n", "rule_silent_k", "rule_silent_n"))
    assert got == (k1, n1, k0, n0), (r, got)
assert round(p.loc["gemma-4-31b-it", "fisher_p"], 4) == 0.0002 and round(p.loc["qwen3.8-27b", "fisher_p"], 2) == 0.07


# --- 2. Hit-rate gap on households with rent -------------------------------------------------------------
# Gap = target hit rate at omission/none minus target false-flag rate at zero_rent/none, Newcombe interval as in
# the frozen scorer. Lease-matched: only households whose packet has a lease with rent above $0.
rows = []
for r in READERS:
    s = NONE[(NONE.reader == r) & NONE.valid]
    for label, t in [("all households", s), ("households with rent", s[s.has_lease == 1])]:
        o, z = t[t.cls == "omission"], t[t.cls == "zero_rent"]
        (ko, no), (kz, nz) = k_of(o, "target_flag"), k_of(z, "target_flag")
        d, lo, hi = scorer.newcombe_diff(ko, no, kz, nz)
        rows.append({"reader": r, "sample": label, "omission_k": ko, "omission_n": no, "zero_rent_k": kz, "zero_rent_n": nz,
                     "gap": d, "lo": lo, "hi": hi})
sec2 = pd.DataFrame(rows)
sec2.to_csv(HERE / "r1-lease-matched.csv", index=False)
print("\n2. gap, all vs households with rent\n", sec2.round(3).to_string(index=False))
m = sec2[sec2["sample"] == "households with rent"].set_index("reader")
g31 = m.loc["gemma-4-31b-it"]
assert (g31.omission_k, g31.omission_n, g31.zero_rent_k, g31.zero_rent_n) == (4, 13, 2, 47)
for r, (d, lo, hi) in {"gemma-4-31b-it": (0.265, 0.058, 0.536), "qwen3.8-27b": (0.154, 0.021, 0.422), "glm-5.3-flash": (0.624, 0.332, 0.810)}.items():
    assert (round(m.loc[r, "gap"], 3), round(m.loc[r, "lo"], 3), round(m.loc[r, "hi"], 3)) == (d, lo, hi), (r, m.loc[r, ["gap", "lo", "hi"]].tolist())


# --- 3. Label x amount 2x2 by reported income -----------------------------------------------------------
# The 48 stray-credit true zeros in four cells: clear label, small amount (stray_clear); clear, large
# (stray_clear_amount); ambiguous, small (stray_ambiguous); ambiguous, large (stray_amount). Split by whether the
# application reports any income (earned plus unearned above $0). Interaction per household:
# (ambiguous large - ambiguous small) - (clear large - clear small) on the either-income flag; exact sign test
# on the nonzero values. The sign counts use every household (an invalid output counts as no flag); the
# `_valid_only` counts keep households whose four outputs are all valid.
CELLS = {"stray_clear": "clear_small", "stray_clear_amount": "clear_large", "stray_ambiguous": "ambiguous_small", "stray_amount": "ambiguous_large"}
S = F1[(F1.cls == "zero_stray") & F1.evidence_state.isin(CELLS) & F1.reader.isin(FROZEN)]
rows = []
for r in FROZEN:
    sr = S[S.reader == r]
    wide = sr.pivot(index="row", columns="evidence_state", values="income_flag")
    ok = sr.pivot(index="row", columns="evidence_state", values="valid").all(axis=1)
    az = sr.drop_duplicates("row").set_index("row").any_income
    for half, flag in [("some income", 1), ("zero income", 0), ("all 48", None)]:
        keep = wide.index if flag is None else az[az == flag].index
        row = {"reader": r, "half": half, "households": len(keep),
               "households_without_lease": int((sr.drop_duplicates("row").set_index("row").loc[keep].has_lease == 0).sum())}
        for st, name in CELLS.items():
            for col in ("income_flag", "target_flag"):
                t = sr[(sr.evidence_state == st) & sr.row.isin(keep) & sr.valid]
                row[f"{name}_{'either' if col == 'income_flag' else 'target'}_k"] = int(t[col].sum())
            row[f"{name}_n_valid"] = int(sr[(sr.evidence_state == st) & sr.row.isin(keep)].valid.sum())
        for suffix, mask in [("", pd.Series(True, index=wide.index)), ("_valid_only", ok)]:
            w = wide[wide.index.isin(keep) & mask.reindex(wide.index)]
            c = (w.stray_amount - w.stray_ambiguous) - (w.stray_clear_amount - w.stray_clear)
            pos, neg = int((c > 0).sum()), int((c < 0).sum())
            row[f"interaction_positive{suffix}"], row[f"interaction_negative{suffix}"] = pos, neg
            row[f"sign_test_p{suffix}"] = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
        rows.append(row)
sec3 = pd.DataFrame(rows)
sec3.to_csv(HERE / "twoxtwo-by-reported-income.csv", index=False)
print("\n3. 2x2 by reported income\n", sec3.round(4).T.to_string())
some = sec3[sec3.half == "some income"].set_index("reader")
zero = sec3[sec3.half == "zero income"].set_index("reader")
assert (some.loc["deepseek-v4-flash", "ambiguous_small_either_k"], some.loc["deepseek-v4-flash", "ambiguous_large_either_k"]) == (11, 23)
assert (some.loc["qwen3.5-397b", "ambiguous_small_either_k"], some.loc["qwen3.5-397b", "ambiguous_large_either_k"]) == (9, 16)
assert int(some.loc["deepseek-v4-flash", "households"]) == 24 and int(zero.loc["deepseek-v4-flash", "households"]) == 24
assert [(int(some.loc[r, "interaction_positive"]), int(some.loc[r, "interaction_negative"])) for r in FROZEN] == [(12, 0), (9, 2), (7, 0)]
assert [(int(zero.loc[r, "interaction_positive"]), int(zero.loc[r, "interaction_negative"])) for r in FROZEN] == [(1, 0), (2, 1), (3, 0)]
assert int(zero.loc["deepseek-v4-flash", "households_without_lease"]) == 22


# --- 4. Any-group false flags on zero_rent/none ---------------------------------------------------------
rows = []
for r in READERS:
    s = NONE[(NONE.reader == r) & (NONE.cls == "zero_rent") & NONE.valid]
    groups = Counter(g for fl in s.flagged for g in fl.split(",") if g)
    k, n = k_of(s, "any_flag")
    rows.append({"reader": r, "any_flag_k": k, "n_valid": n, "any_flag_rate": k / n,
                 "groups_flagged": "; ".join(f"{g} {c}" for g, c in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])))})
sec4 = pd.DataFrame(rows)
sec4.to_csv(HERE / "any-group-false-flags.csv", index=False)
print("\n4. any-group false flags, zero_rent/none\n", sec4.to_string(index=False))
got = {r.reader: (r.any_flag_k, r.n_valid) for r in sec4.itertuples()}
for r, kn in {"gemma-4-26b-a4b-it": (48, 48), "qwen3.8-27b": (42, 48), "gemma-4-31b-it": (36, 47), "deepseek-v4-flash": (1, 45)}.items():
    assert got[r] == kn, (r, got[r])


# --- 5. qwen3.8-27b: the same behaviour on two cells ----------------------------------------------------
Q = F1[F1.reader == "qwen3.8-27b"]
rows = []
for cls, st in [("omission", "deposits_ambiguous"), ("zero_stray", "stray_ambiguous")]:
    c = Q[(Q.cls == cls) & (Q.evidence_state == st)]
    v = c[c.valid]
    non_income = int(sum(bool(set(fl.split(",")) - INCOME - {""}) for fl in v.flagged))
    non_target = int(sum(bool(set(fl.split(",")) - {"", t}) for fl, t in zip(v.flagged, v.target_group)))
    rows.append({"cell": f"{cls}/{st}", "households": len(c), "valid": len(v), "valid_share": len(v) / len(c),
                 "either_income_flag_k": int(v.income_flag.sum()), "target_flag_k": int(v.target_flag.sum()),
                 "any_group_flag_k": int(v.any_flag.sum()),
                 "flags_a_non_income_group_k": non_income, "flags_a_non_target_group_k": non_target})
sec5 = pd.DataFrame(rows)
sec5.to_csv(HERE / "qwen38-recurrence.csv", index=False)
print("\n5. qwen3.8-27b recurrence\n", sec5.to_string(index=False))
assert (sec5.iloc[0].either_income_flag_k, sec5.iloc[0].valid) == (23, 24)
assert (sec5.iloc[1].either_income_flag_k, sec5.iloc[1].valid) == (6, 48)
assert (sec5.iloc[1].any_group_flag_k, sec5.iloc[1].valid) == (26, 48)
print("\nall known counts reproduced")
