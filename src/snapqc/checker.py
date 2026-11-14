"""Reader view, packet oracle and the deterministic checker.

`reader_view` is the one serializer every application goes through before a reader sees it, whoever filed it
and whatever schema it used. `oracle` reads a packet's rendered documents with fixed rules and returns the
monthly amount per fact group they document. `check` flags a group that the application touches and no document
supports, or whose documented amount would move the benefit by more than $56.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .data import TOLERANCE
from .forms import ESCAPE, GROUPS, MONTHLY_FACTOR, utility_amount
from .packets import CHILD_SUPPORT_LABEL, EARNED_TYPES, INCOME_LABEL, N_MONTHS, PAY_FREQUENCY, benefit_of, render_documents

GROUP_LABEL = {"earned": "Earned income", "unearned": "Unearned income", "shelter": "Shelter costs",
               "utilities": "Utilities"}
FREQUENCY_CODE = {v: k for k, v in PAY_FREQUENCY.items()}
EARNED_INCOME_RECORDS = {INCOME_LABEL[t] for t in EARNED_TYPES}  # income records counted as earned income


def whole_dollars(x) -> str:
    """One number format for every filer: whole dollars, half up, thousands separator; the escape token and an
    empty group read "not provided"."""
    if x == ESCAPE or x is None:
        return "not provided"
    return f"${int(math.floor(float(x) + 0.5)):,}"


def reader_view(application_monthly: dict, packet: dict) -> str:
    """The text a reader sees: the monthly total per fact group in a fixed order, then the packet documents.
    Source tags, escape tokens and per-line detail are never shown."""
    lines = ["Application (monthly amounts):"]
    lines += [f"{GROUP_LABEL[g]}: {whole_dollars(application_monthly.get(g))}" for g in GROUPS]
    return "\n".join(lines) + "\n\nDocuments:\n\n" + render_documents(packet) + "\n"


def _group_of(doc: dict) -> str | None:
    """The fact group a document documents under the oracle rules (None: supporting or neutral)."""
    k, data = doc["kind"], doc["data"]
    if k == "pay_stub":
        return "earned"
    if k in ("award_letter", "benefit_notice"):
        return "unearned"
    if k == "income_record":
        return "earned" if data["Income"] in EARNED_INCOME_RECORDS else "unearned"
    if k == "transaction" and data["Description"] == CHILD_SUPPORT_LABEL:
        return "unearned"
    if k == "lease":
        return "shelter"
    if k == "utility_bill":
        return "utilities"
    return None


def documented_groups(packet: dict) -> set[str]:
    return {g for d in packet["documents"] if (g := _group_of(d))}


def oracle(packet: dict) -> dict:
    """Monthly amount per fact group from the packet's documents only:
    pay stubs -> mean gross pay per (employee, employer) x the monthly factor of the stated pay frequency;
    award letters and benefit notices -> the monthly amount stated; income records -> the average monthly
    amount (earned or unearned by the income named); child-support disbursement deposits -> their total over
    the months the bank feed covers; lease -> monthly rent; utility bills -> the state's allowance tier for the
    services billed. Payroll deposits, benefit deposits and rent payments only support these and are not added."""
    out = dict.fromkeys(GROUPS, 0.0)
    stubs = defaultdict(list)
    services = set()
    for d in packet["documents"]:
        k, data = d["kind"], d["data"]
        if k == "pay_stub":
            stubs[(data["Employee"], data["Employer"], data["Pay frequency"])].append(data["Gross pay"])
        elif k == "award_letter":
            out["unearned"] += data["Monthly benefit amount"]
        elif k == "benefit_notice":
            out["unearned"] += data["Monthly amount"]
        elif k == "income_record":
            out[_group_of(d)] += data["Average monthly amount"]
        elif k == "transaction" and data["Description"] == CHILD_SUPPORT_LABEL:
            out["unearned"] += data["Amount"] / N_MONTHS
        elif k == "lease":
            out["shelter"] += data["Monthly rent"]
        elif k == "utility_bill":
            services.add(data["Service"])
    for (_, _, freq), gross in stubs.items():
        out["earned"] += sum(gross) / len(gross) * MONTHLY_FACTOR[FREQUENCY_CODE[freq]]
    out["utilities"] = utility_amount(services, packet["sua"])
    return {g: round(v, 2) for g, v in out.items()}


def check(application_monthly: dict, packet: dict, household) -> list[dict]:
    """Flagged groups with a reason. A group is in scope if the application reports a non-zero amount or
    "not provided" for it, or the packet documents it. In scope and no document -> flagged; documented amount
    put in place of the application's amount changes the benefit by more than $56 -> flagged."""
    app = {g: application_monthly.get(g) for g in GROUPS}
    documented, docs = documented_groups(packet), oracle(packet)
    base = benefit_of(app, household)
    flags = []
    for g in GROUPS:
        v = app[g]
        if not (v == ESCAPE or (v not in (None, ESCAPE) and float(v) != 0) or g in documented):
            continue
        if g not in documented:
            flags.append({"group": g, "reason": "no document"})
            continue
        change = benefit_of({**app, g: docs[g]}, household) - base
        if abs(change) > TOLERANCE:
            flags.append({"group": g, "reason": f"documented amount changes the benefit by ${change:+,.0f}"})
    return flags
