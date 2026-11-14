"""Scoring the pilot's filer and reader outputs (notebook 04, parts 4 to 8). No model calls.

Filer outputs are scored per application x fact group against the verified monthly amount of the group (the fact
ledger) and the packet the filer saw. Reader outputs are scored against the group in error (human-filed
applications) or against the groups whose filed amount moves the benefit by more than $56 (AI-filed
applications). The call-budget calculator counts calls for a plan of filer and reader cells.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, TOLERANCE
from .forms import ESCAPE, MONTHLY_FACTOR
from .packets import round_50

PILOT = ROOT / "experiments" / "pilot"
TOL_BENEFIT = TOLERANCE  # dollars: a benefit error above $56 counts (per rq.md)
TOL_GROUP = 5  # dollars: a filed monthly group amount within $5 of the reference counts as the same amount

OUTCOMES = ["A", "Z", "R", "M", "S", "G"]
OUTCOME_LABEL = {"A": "abstained", "Z": "blank or $0", "R": "right (within $5)", "M": "documented, number off",
                 "S": "note-B amount (within $5)", "G": "undocumented, other number"}
ZERO_FACT = ["no fact: $0", "no fact: abstained", "no fact: number"]


def runner():
    """The pilot runner as a module (its repair of JSON-encoded fields and its reader-input text)."""
    spec = importlib.util.spec_from_file_location("run_pilot", PILOT / "run_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def final_records(path: Path) -> list[dict]:
    """One record per job id: the last answered one (an unanswered record only if the job was never answered)."""
    out = {}
    for r in read_jsonl(path):
        if r.get("ok_http") or r["job_id"] not in out:
            out[r["job_id"]] = r
    return list(out.values())


# --- Filers -----------------------------------------------------------------------------------------------
def classify(filed, verified: float, undocumented: bool, note_b: float | None) -> str:
    """Outcome of one application x fact group (see OUTCOME_LABEL and ZERO_FACT)."""
    if verified == 0:
        return "no fact: abstained" if filed == ESCAPE else "no fact: $0" if float(filed) == 0 else "no fact: number"
    if filed == ESCAPE:
        return "A"
    if float(filed) == 0:
        return "Z"
    if not undocumented:
        return "R" if abs(float(filed) - verified) <= TOL_GROUP else "M"
    return "S" if note_b is not None and abs(float(filed) - note_b) <= TOL_GROUP else "G"


def m_sublabel(filed: float, group: str, packet: dict) -> str:
    """Why a documented earned amount is off, where the pay stubs tell: net pay used for gross; the stub gross
    converted with another frequency's factor; the gross of the stubs in the two-month window averaged per month.
    Anything else (and every other group) is "other"."""
    stubs = [d["data"] for d in packet["documents"] if d["kind"] == "pay_stub"]
    if group != "earned" or not stubs:
        return "other"
    jobs = {}
    for s in stubs:
        jobs.setdefault((s["Employee"], s["Employer"]), []).append(s)
    code = {"Weekly": "Wk", "Biweekly": "2Wks", "Semi-monthly": "2x/Mo", "Monthly": "Mo"}
    net = sum(np.mean([s["Net pay"] for s in ss]) * MONTHLY_FACTOR[code[ss[0]["Pay frequency"]]] for ss in jobs.values())
    window = sum(s["Gross pay"] for s in stubs) / 2
    other = [sum(ss[0]["Gross pay"] * f for ss in jobs.values()) for f in (4.0, 52 / 12, 4.33, 2.0, 26 / 12, 1.0)]
    if abs(filed - net) <= TOL_GROUP:
        return "net for gross"
    if abs(filed - window) <= TOL_GROUP:
        return "averaging over the window"
    if any(abs(filed - x) <= TOL_GROUP for x in other):
        return "wrong period or frequency"
    return "other"


def g_sublabel(filed: float, documented_part: float, note_b: float | None) -> str:
    """What an undocumented-group number looks like: only the documented part of the group (the missing facts
    left off), a whole multiple of the note-B amount, or other."""
    if documented_part > 0 and abs(filed - documented_part) <= TOL_GROUP:
        return "documented part only"
    if note_b:
        k = round(filed / note_b)
        if k >= 2 and abs(filed - k * note_b) <= TOL_GROUP:
            return f"{k} x the note-B amount"
    return "other"


def note_b_amount(facts: list[dict], packet: dict, group: str) -> tuple[float, float | None]:
    """(documented part of the group, the amount note B implies for the group: documented part + the missing
    facts' total rounded to $50 as note B states it). None when the group has no missing fact."""
    missing = [f for f in facts if f["group"] == group and f["fact"] in packet["missing_facts"]]
    total = sum(f["amount"] for f in facts if f["group"] == group)
    documented = total - sum(f["amount"] for f in missing)
    return documented, (documented + round_50(sum(f["amount"] for f in missing))) if missing else None


GROUP_FIELDS = {"earned": [("employment", "wages_before_tax"), ("self_employment", "monthly_income_before_expenses"),
                           ("self_employment", "monthly_expenses")],
                "unearned": [("additional_income", "amount")], "shelter": [("housing_expenses", "amount")]}
TAG_ORDER = ["retrieved_record", "applicant_stated", "agent_inferred"]  # strongest claim first


def group_tag(app: dict, group: str) -> tuple[str | None, list[str]]:
    """Provenance tag of a group: the strongest claim among its fields that carry a number above $0 (or, for
    utilities, a non-empty list of types): retrieved_record if any such field says so, else applicant_stated,
    else agent_inferred. Returns (tag, document ids cited by those fields); (None, []) if no field qualifies."""
    if group == "utilities":
        f = app.get("utilities_paid")
        fields = [f] if isinstance(f, dict) and isinstance(f.get("value"), list) and f["value"] else []
    else:
        fields = [item.get(k) for sec, k in GROUP_FIELDS[group] for item in app.get(sec) or []]
        fields = [f for f in fields if isinstance(f, dict) and isinstance(f.get("value"), (int, float)) and f["value"] > 0]
    tags = {f.get("source") for f in fields}
    tag = next((t for t in TAG_ORDER if t in tags), None)
    return tag, sorted({i for f in fields for i in f.get("document_ids") or []})


def cited_documents(ids: list[str], packet: dict, fact_group: dict, group: str) -> str:
    """Whether the cited document ids exist in the packet and belong to the group (packet `sources`: doc -> fact)."""
    if not ids:
        return "no ids cited"
    known = [i for i in ids if i in packet["sources"]]
    if not known:
        return "ids not in the packet"
    if any(fact_group.get(packet["sources"][i]) == group for i in known):
        return "cites a document of this group"
    return "cites only documents of other groups or neutral ones"


# --- Readers ----------------------------------------------------------------------------------------------
def flagged(rec: dict) -> set[str]:
    parsed = rec.get("parsed") or {}
    return {f["group"] for f in parsed.get("flags") or []}


def confidences(rec: dict) -> dict:
    parsed = rec.get("parsed") or {}
    return {f["group"]: f.get("confidence") for f in parsed.get("flags") or []}


def paired_ratio(num_a, den_a, num_b, den_b, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Difference of two ratios of sums, sum(num_a) / sum(den_a) - sum(num_b) / sum(den_b), over households
    (one array entry per household), with a 95% percentile interval from a household bootstrap: households are
    drawn with replacement and both ratios are computed on the same draw. A draw with a zero denominator is
    dropped."""
    arrays = [np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b)]

    def diff(idx):
        na, da, nb, db = (x[idx].sum(axis=-1) for x in arrays)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((da > 0) & (db > 0), na / da - nb / db, np.nan)

    k = len(arrays[0])
    idx = np.random.default_rng(seed).integers(0, k, size=(n, k))
    lo, hi = np.nanpercentile(diff(idx), [2.5, 97.5])
    return float(diff(np.arange(k))), float(lo), float(hi)


# --- Format guard -----------------------------------------------------------------------------------------
def application_lines(reader_input: str) -> list[str]:
    return reader_input.split("\n\nDocuments:")[0].splitlines()[1:]


# --- Budget -----------------------------------------------------------------------------------------------
def budget(filer_cells: list[dict], reader_cells: list[dict], n_readers: int, rereads: int,
           rates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calls and hours for a plan.

    filer_cells: dicts with label, households, packets, notes (default 1), filers -> calls = product.
    reader_cells: dicts with label, households, packets, filers (1 for human-filed) -> applications; each is read
    once by each of `n_readers` readers. rereads: extra reader calls. rates: label -> calls per hour."""
    rows = [{"task": "filer", "cell": c["label"],
             "calls": c["households"] * c["packets"] * c.get("notes", 1) * c["filers"]} for c in filer_cells]
    rows += [{"task": "reader", "cell": c["label"],
              "calls": c["households"] * c["packets"] * c["filers"] * n_readers} for c in reader_cells]
    rows.append({"task": "reader", "cell": "re-reads", "calls": rereads})
    cells = pd.DataFrame(rows)
    total = cells.groupby("task").calls.sum()
    total["all"] = cells.calls.sum()
    hours = pd.DataFrame({f"hours at {k}": total / v for k, v in rates.items()})
    return cells, pd.concat([total.rename("calls"), hours], axis=1)


