"""Design 2 v2: the three filer hypotheses, computed from the re-score output (no model calls).

Reads scored_household_v2.csv (one row per call) and scored_main_v2.csv (one row per call x filed earned line),
both written by rescore_design2.py, and writes hypotheses.csv: one row per hypothesis x part x model x repeat,
with k, n, share, Wilson 95% interval, the threshold and pass/fail. Repeat 1 is primary (it carries the
verdict); repeat 2 is computed the same way and reported beside it. The rules are those in freeze.md.

Run: PYTHONPATH=src python experiments/design2_v2/hypotheses.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.design2_v2.rescore_design2 import NUMERIC_CLASSES, OUT, TOL_GROUP, wilson

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "hypotheses.csv"
PRIMARY_REPEAT = 1
MODELS = ["deepseek-v4-flash", "qwen3.5-397b", "gemma-4-31b-it"]
REPEATS = [1, 2]


def copies_stub(line: dict) -> bool:
    """A numeric line copies the stale stub: its per-period value is the stale gross, or its monthly amount is
    the stale gross at the true cadence (both within $5)."""
    if line["copied_value_kind"] == "stale_gross":
        return True
    return (pd.notna(line["line_monthly"]) and pd.notna(line["stale_naive_e"])
            and abs(float(line["line_monthly"]) - float(line["stale_naive_e"])) <= TOL_GROUP)


def result(hyp: str, part: str, model: str, repeat: int, k: int, n: int, op: str, threshold: float) -> dict:
    share = k / n if n else None
    lo, hi = wilson(k, n)
    if n == 0:
        verdict = "not evaluable"
    elif op == "<=":
        verdict = "pass" if share <= threshold else "fail"
    else:
        verdict = "pass" if share >= threshold else "fail"
    return {"hypothesis": hyp, "part": part, "model": model, "repeat": repeat,
            "primary": repeat == PRIMARY_REPEAT, "k": k, "n": n, "share": share, "wilson_lo": lo, "wilson_hi": hi,
            "rule": f"share {op} {threshold:.2f}", "verdict": verdict}


def evaluate(hh: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    hh = hh[hh["turn"] == 1]
    lines = lines[(lines["turn"] == 1) & (lines["numeric_line"] == True)]  # noqa: E712
    out = []
    for model in MODELS:
        for rep in REPEATS:
            h = hh[(hh["model"] == model) & (hh["repeat"] == rep)]
            ln = lines[(lines["model"] == model) & (lines["repeat"] == rep)]

            # filer hypothesis 1: invented numbers at most 5% of numeric filings (all levels, both schemas)
            num = h[h["household_class"].isin(NUMERIC_CLASSES)]
            out.append(result("filer hypothesis 1", "invented share of numeric filings", model, rep,
                              int((num["household_class"] == "invented").sum()), len(num), "<=", 0.05))

            # filer hypothesis 2: at "none", required gives $0 in >= 90%, provenance abstains in >= 80%
            req = h[(h["level"] == "none") & (h["schema"] == "required")]
            out.append(result("filer hypothesis 2", "none, required: zero", model, rep,
                              int((req["household_class"] == "zero").sum()), len(req), ">=", 0.90))
            prov = h[(h["level"] == "none") & (h["schema"] == "provenance")]
            out.append(result("filer hypothesis 2", "none, provenance: abstain", model, rep,
                              int((prov["household_class"] == "abstain").sum()), len(prov), ">=", 0.80))

            # filer hypothesis 3: at "stale stub", >= 90% of filings copy the stub (both schemas), and >= 90% of
            # derived-wrong provenance lines carry retrieved_record
            st = h[h["level"] == "stale_stub"]
            st_lines = ln[ln["level"] == "stale_stub"]
            copied = 0
            for jid, cls in zip(st["job_id"], st["household_class"]):
                jl = st_lines[st_lines["job_id"] == jid].to_dict("records")
                copied += bool(jl) and cls == "derived_wrong" and all(copies_stub(x) for x in jl)
            out.append(result("filer hypothesis 3", "stale_stub: filing copies the stub", model, rep,
                              int(copied), len(st), ">=", 0.90))
            dw = st_lines[(st_lines["schema"] == "provenance") & (st_lines["line_class"] == "derived_wrong")]
            out.append(result("filer hypothesis 3", "stale_stub, provenance: derived_wrong line tagged retrieved_record",
                              model, rep, int((dw["line_source_tag"] == "retrieved_record").sum()), len(dw), ">=", 0.90))
    return pd.DataFrame(out)


def main() -> None:
    hh = pd.read_csv(OUT["household"])
    lines = pd.read_csv(OUT["lines"])
    table = evaluate(hh, lines)
    table.to_csv(OUT_PATH, index=False)
    primary = table[table["primary"]]
    for (hyp, model), g in primary.groupby(["hypothesis", "model"], sort=False):
        verdict = "pass" if (g["verdict"] == "pass").all() else (
            "not evaluable" if (g["verdict"] == "not evaluable").any() and not (g["verdict"] == "fail").any() else "fail")
        parts = "; ".join(f"{r['part']} {r['k']}/{r['n']}" for r in g.to_dict("records"))
        print(f"{hyp}, {model} (repeat {PRIMARY_REPEAT}): {verdict} ({parts})")
    print(f"\nWrote {len(table)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
