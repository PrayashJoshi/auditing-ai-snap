"""Line-level rescoring of the Design 2 v2 main arm (no model calls), the Design 2 re-score unchanged.

Reads inputs.jsonl, results.jsonl and the complete packets of the 12 drawn households (draw.py), classifies
every filed earned-income line against per-employer truth, and writes the scored files and the rescore-*.csv
tables next to this script. Each call carries its repeat (1 or 2); the tables keep repeats apart.

Run: PYTHONPATH=src python experiments/design2_v2/rescore_design2.py
"""
from __future__ import annotations

import datetime as dt
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.design2.score_design2 import classify_earned, earned_provenance, is_truncated
from experiments.design2_v2.run_design2 import load_households, load_packets
from experiments.pilot.run_pilot import decode_strings, latest, read_jsonl
from snapqc import checker, forms, packets

HERE = Path(__file__).resolve().parent
INPUTS_PATH = HERE / "inputs.jsonl"
RESULTS_PATH = HERE / "results.jsonl"

OUT = {
    "lines": HERE / "scored_main_v2.csv",
    "household": HERE / "scored_household_v2.csv",
    "summary": HERE / "rescore-summary.csv",
    "matrix": HERE / "rescore-household-matrix.csv",
    "decision": HERE / "rescore-decision-rule.csv",
    "transitions": HERE / "rescore-transitions.csv",
    "cadence": HERE / "rescore-cadence.csv",
    "decomposition": HERE / "rescore-decomposition.csv",
    "clustering": HERE / "rescore-clustering.csv",
    "checker": HERE / "rescore-checker-confusion.csv",
    "reconcile": HERE / "rescore-reconcile.csv",
}

TOL_GROUP = 5.0
TOL_BENEFIT = 56.0
LEVELS = ["none", "net_deposits", "stale_stub", "note_amount"]
RULE_LEVELS = ["net_deposits", "stale_stub"]
NUMERIC_CLASSES = ["correct", "stated", "derived_wrong", "invented"]  # mildest first
CLASSES = ["abstain", "zero"] + NUMERIC_CLASSES + ["invalid"]
WAGE_CAPABLE_KINDS = {"pay_stub", "income_record", "award_letter", "benefit_notice"}
PAYROLL_PREFIX = "PAYROLL DEP"
NOTE_AMOUNT_RE = re.compile(r"wages at about \$([\d,]+) a month before taxes")
RULE_LABEL = "post hoc, exploratory"
WINDOW_START = packets.WINDOW[0]


# --- Small helpers ------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def norm_name(s) -> str:
    """Lower case, punctuation removed, single spaces."""
    s = "" if s is None else str(s)
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s)[:10])


def within(a, b, tol: float = TOL_GROUP) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def read_number(v) -> tuple[str, float | None]:
    """State of a filed value: ("escape" | "missing" | "num" | "bad", number or None)."""
    if v == forms.ESCAPE:
        return "escape", None
    if v is None or v == "":
        return "missing", None
    if isinstance(v, bool):
        return "bad", None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "bad", None
    if np.isnan(x) or x < 0:
        return "bad", None
    return "num", x


# --- 1. Per-employer truth ----------------------------------------------------------------------------------
def cadence_from_dates(dates: list[dt.date]) -> str | None:
    """Pay cadence code from deposit dates: every gap 7 days -> Wk; every gap 14 -> 2Wks; two deposits in each
    calendar month -> 2x/Mo; one in each calendar month -> Mo; anything else -> None."""
    dates = sorted(dates)
    if not dates:
        return None
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    if gaps and all(g == 7 for g in gaps):
        return "Wk"
    if gaps and all(g == 14 for g in gaps):
        return "2Wks"
    per_month = Counter((d.year, d.month) for d in dates)
    if all(n == 2 for n in per_month.values()):
        return "2x/Mo"
    if all(n == 1 for n in per_month.values()):
        return "Mo"
    return None


def employer_truth(complete_pkt: dict, stale_pkt: dict, wage_fact_ids: list[str]) -> list[dict]:
    """One truth record per wage fact (employer), from the complete packet and the stale-stub level packet."""
    out = []
    for fid in wage_fact_ids:
        docs = [d for d in complete_pkt["documents"] if complete_pkt["sources"].get(d["id"]) == fid]
        stubs = [d["data"] for d in docs if d["kind"] == "pay_stub"]
        deps = [d["data"] for d in docs
                if d["kind"] == "transaction" and str(d["data"].get("Description", "")).startswith(PAYROLL_PREFIX)]
        employer = stubs[0]["Employer"] if stubs else None
        stub_code = checker.FREQUENCY_CODE.get(stubs[0]["Pay frequency"]) if stubs else None
        date_code = cadence_from_dates([parse_date(d["Date"]) for d in deps])
        cadence = date_code or stub_code
        factor = forms.MONTHLY_FACTOR.get(cadence)
        mean_gross = float(np.mean([s["Gross pay"] for s in stubs])) if stubs else None
        mean_deposit = float(np.mean([d["Amount"] for d in deps])) if deps else None
        stale = [d["data"] for d in stale_pkt["documents"]
                 if d["kind"] == "pay_stub" and norm_name(d["data"].get("Employer")) == norm_name(employer)]
        stale_gross = float(stale[0]["Gross pay"]) if stale else None
        out.append({
            "fact_id": fid,
            "employer": employer,
            "true_cadence": cadence,
            "cadence_from_dates": date_code,
            "cadence_from_stub": stub_code,
            "cadence_mismatch": date_code != stub_code,
            "verified_monthly_e": mean_gross * factor if mean_gross is not None and factor else None,
            "mean_deposit_e": mean_deposit,
            "right_naive_e": mean_deposit * factor if mean_deposit is not None and factor else None,
            "stale_gross_e": stale_gross,
            "stale_naive_e": stale_gross * factor if stale_gross is not None and factor else None,
        })
    return out


def note_amount(note_text: str) -> float | None:
    m = NOTE_AMOUNT_RE.search(note_text or "")
    return float(m.group(1).replace(",", "")) if m else None


def match_employer(filed_name, truths: list[dict]) -> tuple[dict | None, str]:
    """Truth record for a filed line: exact normalised name, then one name containing the other, then the sole
    employer of a one-employer household; otherwise no match."""
    name = norm_name(filed_name)
    for t in truths:
        if name and name == norm_name(t["employer"]):
            return t, "exact"
    for t in truths:
        tn = norm_name(t["employer"])
        if name and tn and (name in tn or tn in name):
            return t, "contains"
    if len(truths) == 1:
        return truths[0], "sole_employer"
    return None, "unmatched"


# --- 2. Filed lines -----------------------------------------------------------------------------------------
def filed_lines(app: dict) -> list[dict]:
    """The earned-income lines of a parsed application, in form order, with the per-period value and the line's
    monthly amount as the agency-side conversion computes it."""
    out = []
    for j in app.get("employment") or []:
        if not isinstance(j, dict):
            out.append({"section": "employment", "state": "bad", "field": None, "employer_filed": None,
                        "per": None, "value": None, "monthly": None})
            continue
        field = j.get("wages_before_tax")
        state, value = read_number(forms._value(field))
        per = j.get("per")
        monthly = None
        if state == "num":
            h_state, hours = read_number(forms._value(j.get("avg_hours_per_week")))
            if per == "Hr" and h_state == "escape":
                state = "escape"  # an hourly rate with no hours converts to the escape token
            else:
                monthly = float(forms._monthly(value, per, hours or 0.0))
        out.append({"section": "employment", "state": state, "field": field, "employer_filed": j.get("employer"),
                    "per": per, "value": value, "monthly": monthly})
    for s in app.get("self_employment") or []:
        if not isinstance(s, dict):
            out.append({"section": "self_employment", "state": "bad", "field": None, "employer_filed": None,
                        "per": None, "value": None, "monthly": None})
            continue
        field = s.get("monthly_income_before_expenses")
        state, value = read_number(forms._value(field))
        monthly = None
        if state == "num":
            c_state, costs = read_number(forms._value(s.get("monthly_expenses")))
            if c_state == "escape":
                state = "escape"
            else:
                monthly = max(0.0, value - (costs or 0.0))
        out.append({"section": "self_employment", "state": state, "field": field,
                    "employer_filed": s.get("type_of_work"), "per": "Mo", "value": value, "monthly": monthly})
    return out


def copied_kind(value, truth: dict | None) -> str:
    if truth is None or value is None:
        return "none"
    if within(value, truth["mean_deposit_e"]):
        return "mean_deposit"
    if within(value, truth["stale_gross_e"]):
        return "stale_gross"
    return "none"


def line_class(line: dict, truth: dict | None, household_stated: bool) -> str:
    """abstain > zero > stated > correct > derived_wrong > invented ("invalid": a value that is not a number)."""
    if line["state"] == "escape":
        return "abstain"
    if line["state"] == "missing" or (line["state"] == "num" and line["value"] == 0):
        return "zero"
    if line["state"] == "bad":
        return "invalid"
    if household_stated:
        return "stated"
    if truth is not None:
        if within(line["monthly"], truth["verified_monthly_e"]):
            return "correct"
        if (within(line["monthly"], truth["right_naive_e"]) or within(line["monthly"], truth["stale_naive_e"])
                or copied_kind(line["value"], truth) != "none"):
            return "derived_wrong"
    return "invented"


# --- 4. Citations -------------------------------------------------------------------------------------------
def citation_columns(doc_ids: list[str], pkt: dict) -> dict:
    """Kinds of the cited documents, in the packet the filer saw."""
    by_id = {d["id"]: d for d in pkt["documents"]}
    ids = [i for i in dict.fromkeys(doc_ids or []) if isinstance(i, str)]
    docs = [by_id[i] for i in ids if i in by_id]
    kinds = [d["kind"] for d in docs]
    stub_dates = [parse_date(d["data"]["Pay date"]) for d in docs if d["kind"] == "pay_stub"]
    n_unknown = len(ids) - len(docs)
    return {
        "cited_ids": ",".join(ids),
        "n_cited": len(ids),
        "n_cited_transaction": kinds.count("transaction"),
        "n_cited_pay_stub": kinds.count("pay_stub"),
        "n_cited_other": len(kinds) - kinds.count("transaction") - kinds.count("pay_stub"),
        "n_cited_unknown_id": n_unknown,
        "deposit_only_citation": bool(ids) and n_unknown == 0 and all(k == "transaction" for k in kinds),
        "wage_capable_cited": any(k in WAGE_CAPABLE_KINDS for k in kinds),
        "cited_stub_gap_days": (WINDOW_START - max(stub_dates)).days if stub_dates else None,
    }


EMPTY_CITATION = {"cited_ids": None, "n_cited": None, "n_cited_transaction": None, "n_cited_pay_stub": None,
                  "n_cited_other": None, "n_cited_unknown_id": None, "deposit_only_citation": None,
                  "wage_capable_cited": None, "cited_stub_gap_days": None}


def file_latest_stub_gap_days(pkt: dict):
    dates = [parse_date(d["data"]["Pay date"]) for d in pkt["documents"] if d["kind"] == "pay_stub"]
    return (WINDOW_START - max(dates)).days if dates else None


# --- 6. Benefit ---------------------------------------------------------------------------------------------
def designed_error_value(level: str, truths: list[dict], x: float | None) -> float | None:
    """The earned amount each level is built to invite."""
    if level == "none":
        return 0.0
    if level == "note_amount":
        return x
    key = "right_naive_e" if level == "net_deposits" else "stale_naive_e"
    vals = [t[key] for t in truths]
    return None if any(v is None for v in vals) else float(sum(vals))


def monthly_of(app: dict | None, sua: dict) -> dict | None:
    """Monthly amounts per fact group, with the same utilities_paid fallback as the v1 scorer."""
    if app is None:
        return None
    try:
        return forms.to_monthly(app, sua)
    except Exception:
        try:
            return forms.to_monthly({**app, "utilities_paid": None}, sua)
        except Exception:
            return None


# --- Scoring one call ---------------------------------------------------------------------------------------
def worst_class(classes: list[str]) -> str | None:
    numeric = [c for c in classes if c in NUMERIC_CLASSES]
    return max(numeric, key=NUMERIC_CLASSES.index) if numeric else None


def score_call(jid: str, rec: dict, inp: dict, truths: list[dict], x: float | None, h, sensitive: bool) -> tuple[dict, list[dict]]:
    """Household-level record and line records for one main-arm call."""
    pkt, level, schema = inp["packet"], rec["level"], rec["schema"]
    parsed = rec.get("parsed")
    app, repaired = decode_strings(parsed) if isinstance(parsed, dict) else (None, False)
    monthly = monthly_of(app, pkt["sua"])
    filed_earned = monthly.get("earned") if monthly else None
    filed_numeric = monthly is not None and filed_earned not in (None, forms.ESCAPE)
    verified = float(inp["verified"])

    qc_app = {g: float(h[f"verified_{g}"]) for g in forms.GROUPS}
    qc_ben = packets.benefit_of(qc_app, h)

    # v1 outcome and benefit error, by the v1 rule
    outcome_v1 = classify_earned(filed_earned, verified, inp["naive"]) if monthly else "invalid"
    if filed_numeric:
        signed_err = float(filed_earned) - verified
        ben_err = packets.benefit_of({**qc_app, "earned": float(filed_earned)}, h) - qc_ben
    elif monthly and filed_earned == forms.ESCAPE:
        signed_err = None
        ben_err = packets.benefit_of({**qc_app, "earned": forms.ESCAPE}, h) - qc_ben
    else:
        signed_err, ben_err = None, None

    tag_strong, tag_weak = None, None
    if schema == "provenance" and app is not None:
        tag_strong, tag_weak, _ = earned_provenance(app)

    # lines
    lines = filed_lines(app) if app is not None else []
    # sum of the numeric line monthlies (lines holding a number > 0); escaped and blank lines count as $0
    filed_numeric_total = float(sum(ln["monthly"] or 0.0 for ln in lines if ln["state"] == "num" and ln["value"] > 0))
    household_stated = level == "note_amount" and filed_numeric_total > 0 and within(filed_numeric_total, x)
    line_rows, all_cited = [], []
    for i, ln in enumerate(lines):
        truth, how = match_employer(ln["employer_filed"], truths)
        cls = line_class(ln, truth, household_stated)
        numeric = cls in NUMERIC_CLASSES
        factor = forms.MONTHLY_FACTOR.get(truth["true_cadence"]) if truth else None
        at_true = ln["value"] * factor if numeric and factor else None
        field = ln["field"] if isinstance(ln["field"], dict) else {}
        cites = EMPTY_CITATION
        if schema == "provenance" and numeric:
            ids = field.get("document_ids") if isinstance(field.get("document_ids"), list) else []
            cites = citation_columns(ids, pkt)
            all_cited += ids
        mean_dep = truth["mean_deposit_e"] if truth else None
        copied = copied_kind(ln["value"], truth) if numeric else "none"
        # a pay-period code can only be wrong for a per-period value copied from a document
        freq_applicable = bool(numeric and truth and copied != "none")
        line_rows.append({
            "job_id": jid,
            "line_index": i,
            "section": ln["section"],
            "employer_filed": ln["employer_filed"],
            "employer": truth["employer"] if truth else "unmatched",
            "employer_match": how,
            "per": ln["per"],
            "per_period_value": ln["value"],
            "line_monthly": ln["monthly"],
            "monthly_at_true_cadence": at_true,
            "line_class": cls,
            "numeric_line": numeric,
            "true_cadence": truth["true_cadence"] if truth else None,
            "frequency_error_applicable": freq_applicable,
            "frequency_error": bool(freq_applicable and ln["per"] != truth["true_cadence"]),
            "copied_value_kind": copied,
            "within5_verified": bool(numeric and truth and within(ln["monthly"], truth["verified_monthly_e"])),
            "ratio_to_mean_deposit": ln["monthly"] / mean_dep if numeric and mean_dep else None,
            "verified_monthly_e": truth["verified_monthly_e"] if truth else None,
            "mean_deposit_e": mean_dep,
            "right_naive_e": truth["right_naive_e"] if truth else None,
            "stale_gross_e": truth["stale_gross_e"] if truth else None,
            "stale_naive_e": truth["stale_naive_e"] if truth else None,
            "line_source_tag": field.get("source"),
            **{f"line_{k}": v for k, v in cites.items()},
        })

    # 3. household class
    classes = [r["line_class"] for r in line_rows]
    numeric_rows = [r for r in line_rows if r["numeric_line"]]
    if monthly is None:
        hh_class = "invalid"
    elif numeric_rows:
        hh_class = worst_class(classes)
    elif filed_earned == forms.ESCAPE or "abstain" in classes:
        hh_class = "abstain"
    else:
        hh_class = "zero"
    employers_filed = {r["employer"] for r in numeric_rows if r["employer"] != "unmatched"}
    freq_errors = [r["frequency_error"] for r in numeric_rows]

    # 4. household citations: the union of ids cited on numeric earned lines
    hh_cites = citation_columns(all_cited, pkt) if schema == "provenance" and numeric_rows else EMPTY_CITATION

    # 5. frozen checker on the filed monthly application
    flags_earned, reason_earned = None, None
    if monthly is not None:
        flags = [f for f in checker.check(monthly, pkt, h) if f["group"] == "earned"]
        flags_earned = bool(flags)
        reason_earned = flags[0]["reason"] if flags else None

    # 7. decomposition of the household dollar error
    decomp = {"line_count_part": None, "doc_value_part": None, "frequency_part": None, "residual": None}
    # not applied to stated filings: their line value is a monthly amount from the note, not a per-period value
    decomposition_applicable = hh_class in ("correct", "derived_wrong", "invented")
    if decomposition_applicable:
        matched = [r for r in numeric_rows if r["monthly_at_true_cadence"] is not None
                   and r["verified_monthly_e"] is not None]
        line_count = -sum(t["verified_monthly_e"] or 0.0 for t in truths if t["employer"] not in employers_filed)
        doc_value = sum(r["monthly_at_true_cadence"] - r["verified_monthly_e"] for r in matched)
        frequency = sum(r["line_monthly"] - r["monthly_at_true_cadence"] for r in matched)
        decomp = {"line_count_part": line_count, "doc_value_part": doc_value, "frequency_part": frequency,
                  "residual": filed_numeric_total - verified - (line_count + doc_value + frequency)}

    verified_sum = sum(t["verified_monthly_e"] or 0.0 for t in truths)
    household = {
        "job_id": jid,
        "row": rec["row"],
        "level": level,
        "schema": schema,
        "turn": rec["turn"],
        "repeat": rec["repeat"],
        "model": rec["filer"],
        "valid": bool(rec.get("valid")),
        "invalid_reason": rec.get("invalid_reason"),
        "repaired": repaired,
        "truncated": is_truncated(rec),
        "outcome_v1": outcome_v1,
        "household_class": hh_class,
        "n_lines_filed": len(line_rows),
        "n_numeric_lines": len(numeric_rows),
        "n_employers": len(truths),
        "partial_lines": bool(numeric_rows) and len(employers_filed) < len(truths),
        "any_frequency_error": any(freq_errors),
        "verified": verified,
        "filed_earned": filed_earned,
        "filed_numeric_total": filed_numeric_total,
        "any_line_abstains": "abstain" in classes,
        "signed_dollar_error": signed_err,
        "within5_verified_total": bool(filed_numeric and within(filed_earned, verified)),
        "note_amount_x": x,
        "verified_sum_employers": verified_sum,
        "verified_sum_diff": verified_sum - verified,
        "cadence_mismatch": any(t["cadence_mismatch"] for t in truths),
        "tag_strongest": tag_strong,
        "tag_weakest": tag_weak,
        **hh_cites,
        "stale_citation": (hh_cites["cited_stub_gap_days"] > 0) if hh_cites["cited_stub_gap_days"] is not None
        else (False if hh_cites["n_cited"] is not None else None),
        "file_latest_stub_gap_days": file_latest_stub_gap_days(pkt),
        "checker_flags_earned": flags_earned,
        "checker_reason_earned": reason_earned,
        "qc_benefit": qc_ben,
        "benefit_error": ben_err,
        "benefit_over_56": abs(ben_err) > TOL_BENEFIT if ben_err is not None else False,
        "benefit_sensitive": sensitive,
        "at_min_benefit": bool(np.isclose(qc_ben, float(h["min_ben"]))) if pd.notna(h["min_ben"]) else False,
        "decomposition_applicable": decomposition_applicable,
        **decomp,
    }
    return household, line_rows


def score_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (household-level frame, call x line frame, per-employer truth frame)."""
    inputs = {r["input_id"]: r for r in read_jsonl(INPUTS_PATH) if r["arm"] == "main"}
    recs = latest(read_jsonl(RESULTS_PATH))
    hh = load_households()
    complete = load_packets()

    rows = sorted({r["row"] for r in inputs.values()})
    truths, notes, truth_rows = {}, {}, []
    for row in rows:
        stale_inp = inputs[f"main|{row}|stale_stub"]
        truths[row] = employer_truth(complete[(row, "complete")], stale_inp["packet"], stale_inp["wage_fact_ids"])
        notes[row] = note_amount(inputs[f"main|{row}|note_amount"]["note_text"])
        verified = float(stale_inp["verified"])
        total = sum(t["verified_monthly_e"] or 0.0 for t in truths[row])
        for t in truths[row]:
            truth_rows.append({"row": row, **t, "note_amount_x": notes[row], "verified": verified,
                               "verified_sum_diff": total - verified})

    sensitive = {}
    for row in rows:
        h = hh[row]
        qc_app = {g: float(h[f"verified_{g}"]) for g in forms.GROUPS}
        qc_ben = packets.benefit_of(qc_app, h)
        for level in LEVELS:
            value = designed_error_value(level, truths[row], notes[row])
            if value is None:
                sensitive[(row, level)] = False
            else:
                change = packets.benefit_of({**qc_app, "earned": value}, h) - qc_ben
                sensitive[(row, level)] = abs(change) > TOL_BENEFIT

    hh_rows, line_rows = [], []
    for jid, rec in recs.items():
        if rec["arm"] != "main":
            continue
        row, level = rec["row"], rec["level"]
        household, lines = score_call(jid, rec, inputs[f"main|{row}|{level}"], truths[row], notes[row], hh[row],
                                      sensitive[(row, level)])
        hh_rows.append(household)
        line_rows += lines or [{"job_id": jid, "line_index": -1}]

    df_hh = pd.DataFrame(hh_rows)
    df_lines = df_hh.merge(pd.DataFrame(line_rows), on="job_id", how="left")
    return df_hh, df_lines, pd.DataFrame(truth_rows)


# --- Output tables ------------------------------------------------------------------------------------------
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, schema, turn, repeat, level), g in df.groupby(["model", "schema", "turn", "repeat", "level"]):
        n = len(g)
        for cls in CLASSES:
            k = int((g["household_class"] == cls).sum())
            lo, hi = wilson(k, n)
            out.append({"model": model, "schema": schema, "turn": turn, "repeat": repeat, "level": level, "household_class": cls,
                        "n": n, "k": k, "share": k / n, "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(out)


def household_matrix(df: pd.DataFrame) -> pd.DataFrame:
    t1 = df[df["turn"] == 1].copy()
    t1["cell"] = t1["model"] + "|" + t1["schema"] + "|r" + t1["repeat"].astype(str)
    m = t1.pivot_table(index=["row", "level"], columns="cell", values="household_class", aggfunc="first")
    order = pd.MultiIndex.from_product([sorted(t1["row"].unique()), LEVELS], names=["row", "level"])
    return m.reindex(order).reset_index()


def decision_rule(df: pd.DataFrame) -> pd.DataFrame:
    """An unsupported number in at least 3 households, at net_deposits or stale_stub, turn 1, in at least 2 models,
    under two readings of "unsupported", with and without benefit-insensitive household x level cells."""
    t1 = df[(df["turn"] == 1) & df["level"].isin(RULE_LEVELS)]
    models = sorted(df["model"].unique())
    readings = {"invented_only": ["invented"], "naive_counts": ["derived_wrong", "invented"]}
    out = []
    for scope in ["pooled"] + sorted(df["schema"].unique()):
        d_scope = t1 if scope == "pooled" else t1[t1["schema"] == scope]
        for reading, classes in readings.items():
            for subset in ["all_households", "benefit_sensitive_only"]:
                d = d_scope if subset == "all_households" else d_scope[d_scope["benefit_sensitive"]]
                n_meeting = 0
                for model in models:
                    dm = d[d["model"] == model]
                    hit = sorted(dm.loc[dm["household_class"].isin(classes), "row"].unique())
                    n_meeting += len(hit) >= 3
                    out.append({"scope": scope, "reading": reading, "subset": subset, "model": model,
                                "n_households": len(hit), "n_households_eligible": dm["row"].nunique(),
                                "households": ",".join(str(r) for r in hit), "meets_3": len(hit) >= 3,
                                "n_models_meeting": None, "met": None, "label": RULE_LABEL})
                out.append({"scope": scope, "reading": reading, "subset": subset, "model": "ALL",
                            "n_households": None, "n_households_eligible": d["row"].nunique(), "households": None,
                            "meets_3": None, "n_models_meeting": n_meeting, "met": n_meeting >= 2,
                            "label": RULE_LABEL})
    return pd.DataFrame(out)


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["row", "level", "schema", "model", "repeat"]
    t1 = df[df["turn"] == 1][keys + ["household_class"]].rename(columns={"household_class": "class_turn1"})
    t2 = df[df["turn"] == 2][keys + ["household_class"]].rename(columns={"household_class": "class_turn2"})
    j = t1.merge(t2, on=keys, how="inner")
    return (j.groupby(["model", "schema", "repeat", "level", "class_turn1", "class_turn2"]).size()
            .reset_index(name="n"))


def cadence_table(df_lines: pd.DataFrame, hh: dict) -> pd.DataFrame:
    d = df_lines[(df_lines["turn"] == 1) & df_lines["level"].isin(RULE_LEVELS) & (df_lines["numeric_line"] == True)]  # noqa: E712
    totals = d.groupby("job_id")["line_monthly"].sum()  # the filing's numeric lines; other lines count as $0
    out = []
    for r in d.to_dict("records"):
        h = hh[r["row"]]
        qc_app = {g: float(h[f"verified_{g}"]) for g in forms.GROUPS}
        at_true = r["monthly_at_true_cadence"]
        effect, ben_effect = None, None
        if pd.notna(at_true):
            filed_total = float(totals[r["job_id"]])
            effect = r["line_monthly"] - at_true
            ben_filed = packets.benefit_of({**qc_app, "earned": filed_total}, h)
            ben_true = packets.benefit_of({**qc_app, "earned": filed_total - effect}, h)
            ben_effect = ben_filed - ben_true
        out.append({"row": r["row"], "model": r["model"], "schema": r["schema"], "level": r["level"],
                    "job_id": r["job_id"], "line_index": r["line_index"], "employer": r["employer"],
                    "true_cadence": r["true_cadence"], "filed_per": r["per"],
                    "per_period_value": r["per_period_value"], "line_monthly": r["line_monthly"],
                    "monthly_at_true_cadence": at_true, "copied_value_kind": r["copied_value_kind"],
                    "frequency_error_applicable": r["frequency_error_applicable"],
                    "frequency_error": r["frequency_error"],
                    "frequency_dollar_effect": effect, "frequency_benefit_effect": ben_effect,
                    "line_class": r["line_class"]})
    return pd.DataFrame(out)


def decomposition_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["decomposition_applicable"]]
    cols = ["job_id", "row", "level", "schema", "turn", "model", "household_class", "verified", "filed_earned",
            "filed_numeric_total", "signed_dollar_error", "decomposition_applicable", "line_count_part", "doc_value_part", "frequency_part", "residual",
            "n_lines_filed", "n_numeric_lines", "n_employers", "partial_lines", "any_frequency_error"]
    return d[cols]


def clustering_table(df: pd.DataFrame) -> pd.DataFrame:
    t1 = df[df["turn"] == 1]
    c = pd.crosstab(t1["row"], t1["household_class"]).reindex(columns=CLASSES, fill_value=0)
    c.insert(0, "n_cells", c.sum(axis=1))
    return c.reset_index()


def checker_confusion(df: pd.DataFrame) -> pd.DataFrame:
    t1 = df[df["turn"] == 1]
    a = (t1.groupby(["level", "schema", "household_class", "checker_flags_earned"], dropna=False).size()
         .reset_index(name="n"))
    a.insert(0, "table", "class_by_checker")
    prov = t1[t1["schema"] == "provenance"]
    b = (prov.groupby(["level", "schema", "household_class", "checker_flags_earned", "deposit_only_citation",
                       "stale_citation"], dropna=False).size().reset_index(name="n"))
    b.insert(0, "table", "class_by_checker_by_citation")
    return pd.concat([a, b], ignore_index=True)


def reconcile_table(df: pd.DataFrame) -> pd.DataFrame:
    c = (df.groupby(["turn", "outcome_v1", "household_class"]).size().unstack("household_class", fill_value=0)
         .reindex(columns=CLASSES, fill_value=0))
    c["total"] = c.sum(axis=1)
    return c.reset_index()


# --- Main ---------------------------------------------------------------------------------------------------
def main() -> None:
    df_hh, df_lines, df_truth = score_all()
    hh = load_households()

    tables = {
        "lines": df_lines,
        "household": df_hh,
        "summary": summary_table(df_hh),
        "matrix": household_matrix(df_hh),
        "decision": decision_rule(df_hh),
        "transitions": transitions(df_hh),
        "cadence": cadence_table(df_lines, hh),
        "decomposition": decomposition_table(df_hh),
        "clustering": clustering_table(df_hh),
        "checker": checker_confusion(df_hh),
        "reconcile": reconcile_table(df_hh),
    }
    for name, table in tables.items():
        table.to_csv(OUT[name], index=False)

    print("Rows written:")
    for name, table in tables.items():
        print(f"  {OUT[name].name}: {len(table)}")

    print(f"\nDecision rule ({RULE_LABEL}):")
    rule = tables["decision"]
    for r in rule[rule["model"] == "ALL"].to_dict("records"):
        verdict = "met" if r["met"] else "not met"
        print(f"  scope={r['scope']}, reading={r['reading']}, subset={r['subset']}: {verdict} "
              f"({int(r['n_models_meeting'])} of {df_hh['model'].nunique()} models with 3 or more households)")

    print("\nTruth checks:")
    bad_cadence = df_truth[df_truth["cadence_mismatch"]]
    if bad_cadence.empty:
        print("  cadence from deposit dates equals the stub pay frequency for every employer")
    for r in bad_cadence.to_dict("records"):
        print(f"  cadence mismatch: row {r['row']}, {r['employer']}: dates -> {r['cadence_from_dates']}, "
              f"stub -> {r['cadence_from_stub']}")
    diffs = df_truth.drop_duplicates("row")
    diffs = diffs[diffs["verified_sum_diff"].abs() > 1.0]
    if diffs.empty:
        print("  per-employer verified amounts sum to the household verified amount within $1 for every household")
    for r in diffs.to_dict("records"):
        print(f"  verified sum differs: row {r['row']}: employers sum - verified = {r['verified_sum_diff']:+.2f}")


if __name__ == "__main__":
    main()
