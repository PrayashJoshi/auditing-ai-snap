"""Scoring and summary generation for Design 2 (no LLM calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pilot.run_pilot import decode_strings, latest, load_households, read_jsonl
from snapqc import checker, forms, packets

HERE = Path(__file__).resolve().parent
INPUTS_PATH = HERE / "inputs.jsonl"
RESULTS_PATH = HERE / "results.jsonl"
SUMMARY_PATH = HERE / "summary.csv"
SCORED_MAIN_PATH = HERE / "scored_main.csv"
SCORED_SIDE_A_PATH = HERE / "scored_side_a.csv"
SCORED_SIDE_B_PATH = HERE / "scored_side_b.csv"

TOL_GROUP = 5.0
TOL_BENEFIT = 56.0
TAG_ORDER = ["retrieved_record", "applicant_stated", "agent_inferred"]
EARNED_FIELDS = [("employment", "wages_before_tax"), ("self_employment", "monthly_income_before_expenses")]


def earned_provenance(app: dict) -> tuple[str | None, str | None, list[str]]:
    """Return (tag_strongest, tag_weakest, cited_doc_ids) for earned fields carrying a number > 0."""
    fields = [item.get(k) for sec, k in EARNED_FIELDS for item in (app.get(sec) or [])]
    fields = [f for f in fields if isinstance(f, dict) and isinstance(f.get("value"), (int, float)) and f["value"] > 0]
    if not fields:
        return None, None, []
    tags = {f.get("source") for f in fields}
    strongest = next((t for t in TAG_ORDER if t in tags), None)
    weakest = next((t for t in reversed(TAG_ORDER) if t in tags), None)
    doc_ids = sorted({i for f in fields for i in (f.get("document_ids") or [])})
    return strongest, weakest, doc_ids


def classify_earned(filed, verified: float, naive: float | None) -> str:
    if filed == forms.ESCAPE:
        return "abstain"
    if filed is None or float(filed) == 0.0:
        return "zero"
    v = float(filed)
    if abs(v - verified) <= TOL_GROUP:
        return "correct"
    if naive is not None and abs(v - naive) <= TOL_GROUP:
        return "naive"
    return "other"


def is_truncated(rec: dict) -> bool:
    return rec.get("status") == "incomplete" or (rec.get("incomplete_details") or {}).get("reason") == "max_output_tokens"


def score_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = {r["input_id"]: r for r in read_jsonl(INPUTS_PATH)}
    recs = latest(read_jsonl(RESULTS_PATH))
    hh = load_households()

    main_rows = []
    side_a_rows = []
    side_b_rows = []

    for jid, rec in recs.items():
        arm = rec["arm"]
        trunc = is_truncated(rec)
        parsed = rec.get("parsed")
        repaired = False
        monthly = None
        if isinstance(parsed, dict):
            app, repaired = decode_strings(parsed)
        else:
            app = None

        if arm == "main":
            inp = inputs[f"main|{rec['row']}|{rec['level']}"]
            pkt = inp["packet"]
            h = hh[rec["row"]]
            qc_app = {g: float(h[f"verified_{g}"]) for g in forms.GROUPS}
            qc_ben = packets.benefit_of(qc_app, h)
            if app is not None:
                try:
                    monthly = forms.to_monthly(app, pkt["sua"])
                except Exception:
                    try:
                        monthly = forms.to_monthly({**app, "utilities_paid": None}, pkt["sua"])
                    except Exception:
                        monthly = None

            filed_earned = monthly.get("earned") if monthly else None
            outcome = classify_earned(filed_earned, inp["verified"], inp["naive"]) if monthly else "invalid"

            if monthly and filed_earned not in (None, forms.ESCAPE):
                signed_err = float(filed_earned) - inp["verified"]
                filed_ben = packets.benefit_of({**qc_app, "earned": float(filed_earned)}, h)
                ben_err = filed_ben - qc_ben
            elif monthly and filed_earned == forms.ESCAPE:
                signed_err = None
                filed_ben = packets.benefit_of({**qc_app, "earned": forms.ESCAPE}, h)
                ben_err = filed_ben - qc_ben
            else:
                signed_err = None
                ben_err = None

            tag_strong, tag_weak, cited_ids = (None, None, [])
            if rec["schema"] == "provenance" and app is not None:
                tag_strong, tag_weak, cited_ids = earned_provenance(app)

            truth_ids = {did for did, fid in pkt["sources"].items() if fid in set(inp["wage_fact_ids"])}
            cited_set = set(cited_ids)
            prec = len(cited_set & truth_ids) / len(cited_set) if cited_set else np.nan
            recall = len(cited_set & truth_ids) / len(truth_ids) if truth_ids else np.nan

            oracle_earned = checker.oracle(pkt)["earned"]
            laundered = (
                tag_strong == "retrieved_record"
                and filed_earned not in (None, forms.ESCAPE)
                and float(filed_earned) > 0
                and not (oracle_earned > 0 and abs(float(filed_earned) - oracle_earned) <= TOL_GROUP)
            )

            main_rows.append({
                "job_id": jid,
                "row": rec["row"],
                "level": rec["level"],
                "schema": rec["schema"],
                "turn": rec["turn"],
                "model": rec["filer"],
                "valid": bool(rec.get("valid")),
                "invalid_reason": rec.get("invalid_reason"),
                "truncated": trunc,
                "repaired": repaired,
                "output_tokens": rec.get("output_tokens"),
                "verified": inp["verified"],
                "naive": inp["naive"],
                "true_source": inp["true_source"],
                "filed_earned": filed_earned,
                "outcome": outcome,
                "signed_dollar_error": signed_err,
                "qc_benefit": qc_ben,
                "benefit_error": ben_err,
                "benefit_over_56": abs(ben_err) > TOL_BENEFIT if ben_err is not None else False,
                "tag_strongest": tag_strong,
                "tag_weakest": tag_weak,
                "cited_ids": ",".join(cited_ids),
                "n_cited": len(cited_ids),
                "n_truth_ids": len(truth_ids),
                "cited_precision": prec,
                "cited_recall": recall,
                "laundered": laundered,
            })

        elif arm == "side_a":
            util_val = None
            util_outcome = "invalid"
            tag_u = None
            cited_u = []
            if app is not None:
                u = app.get("utilities_paid")
                if isinstance(u, dict):
                    util_val = u.get("value")
                    tag_u = u.get("source")
                    cited_u = u.get("document_ids") or []
                else:
                    util_val = u
                if util_val == forms.ESCAPE:
                    util_outcome = "escape"
                elif isinstance(util_val, list):
                    util_outcome = "empty_list" if len(util_val) == 0 else "nonempty_list"
            side_a_rows.append({
                "job_id": jid,
                "row": rec["row"],
                "condition": rec["condition"],
                "model": rec["filer"],
                "valid": bool(rec.get("valid")),
                "truncated": trunc,
                "repaired": repaired,
                "util_outcome": util_outcome,
                "util_value": json.dumps(util_val),
                "tag": tag_u,
                "cited_ids": ",".join(cited_u),
            })

        elif arm == "side_b":
            inp = inputs[f"side_b|{rec['row']}|{rec['condition']}"]
            pkt = inp["packet"]
            if app is not None:
                try:
                    monthly = forms.to_monthly(app, pkt["sua"])
                except Exception:
                    try:
                        monthly = forms.to_monthly({**app, "utilities_paid": None}, pkt["sua"])
                    except Exception:
                        monthly = None
            filed_earned = monthly.get("earned") if monthly else None
            tag_strong, tag_weak, cited_ids = earned_provenance(app) if app is not None else (None, None, [])
            truth_ids = {did for did, fid in pkt["sources"].items() if fid in set(inp["wage_fact_ids"])}
            cited_set = set(cited_ids)
            target = inp["verified"] if rec["condition"] == "original" else inp["swapped_target"]
            within_5 = (
                filed_earned not in (None, forms.ESCAPE)
                and abs(float(filed_earned) - target) <= TOL_GROUP
            )
            cites_wage_id = bool(cited_set & truth_ids)
            all_cited_wage = bool(cited_set) and cited_set.issubset(truth_ids)
            side_b_rows.append({
                "job_id": jid,
                "row": rec["row"],
                "condition": rec["condition"],
                "model": rec["filer"],
                "valid": bool(rec.get("valid")),
                "truncated": trunc,
                "repaired": repaired,
                "verified": inp["verified"],
                "target": target,
                "filed_earned": filed_earned,
                "within_5_of_target": within_5,
                "tag_strongest": tag_strong,
                "tag_weakest": tag_weak,
                "cited_ids": ",".join(cited_ids),
                "cites_wage_id": cites_wage_id,
                "all_cited_wage": all_cited_wage,
            })

    df_main = pd.DataFrame(main_rows)
    df_a = pd.DataFrame(side_a_rows)
    df_b = pd.DataFrame(side_b_rows)

    df_main.to_csv(SCORED_MAIN_PATH, index=False)
    df_a.to_csv(SCORED_SIDE_A_PATH, index=False)
    df_b.to_csv(SCORED_SIDE_B_PATH, index=False)

    # Summary table by model x turn x schema x level
    if not df_main.empty:
        summary = (
            df_main.groupby(["model", "turn", "schema", "level"], dropna=False)
            .agg(
                n=("row", "count"),
                valid=("valid", "sum"),
                truncated=("truncated", "sum"),
                repaired=("repaired", "sum"),
                abstain=("outcome", lambda s: (s == "abstain").sum()),
                zero=("outcome", lambda s: (s == "zero").sum()),
                correct=("outcome", lambda s: (s == "correct").sum()),
                naive=("outcome", lambda s: (s == "naive").sum()),
                other=("outcome", lambda s: (s == "other").sum()),
                unsupported=("outcome", lambda s: s.isin(["naive", "other"]).sum()),
                ben_over_56=("benefit_over_56", "sum"),
                laundered=("laundered", "sum"),
            )
            .reset_index()
        )
        summary.to_csv(SUMMARY_PATH, index=False)
    return df_main, df_a, df_b


if __name__ == "__main__":
    df_main, df_a, df_b = score_all()
    print(f"Scored main={len(df_main)}, side_a={len(df_a)}, side_b={len(df_b)}")
    if not df_main.empty:
        print("\n=== Decision Rule Check (Turn 1, net_deposits or stale_stub, naive or other) ===")
        sub = df_main[(df_main["turn"] == 1) & (df_main["level"].isin(["net_deposits", "stale_stub"])) & (df_main["outcome"].isin(["naive", "other"]))]
        for model, grp in sub.groupby("model"):
            hh_count = grp["row"].nunique()
            print(f"  {model}: {hh_count}/8 households ({sorted(grp['row'].unique())}), total cells={len(grp)}")
