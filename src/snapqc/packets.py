"""Pilot households and their packets.

A packet is what an applicant's agent could retrieve for one household: pay stubs from a payroll connector,
transactions from a bank connector and documents from a drive search. Every amount in it is rendered from the
household's QC-verified values (the fact ledger); nothing in it says which value is the verified one.

Steps: `pilot_sample` draws the 20 households; `build_case` turns one QC row into its ledger and its complete set
of records (seeded); `make_packet` keeps the records of one variant (complete, faithful, faithful_linkability);
`note` writes the applicant's note (version A or B); `filer_input` and `render_documents` give the text a
filer or a reader sees; `human_filed` is the application as the household filed it.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from .calc import benefit, inputs
from .data import RESULTS, SINGLE_SUA_STATES, TOLERANCE
from .forms import ESCAPE, GROUPS, MONTHLY_FACTOR, sua_tiers

SEED = 20261114
STRATA = {"earned": 5, "shelter": 5, "unearned": 3, "utilities": 3}  # error households per fact group
N_COMPARISON = 4
MIN_IN_151, MIN_FROM_CASE_RECORD, MIN_CLIENT_IPV = 4, 2, 2
FROM_CASE_RECORD = [1, 2]  # DISCOV: from the case record, not automated match (1) or automated match (2)
CLIENT_IPV = [3, 4]  # AGENCY: information withheld (3) or incorrect information provided (4) by client, IPV referral
MAX_DRAWS = 10_000
VARIANTS = ["complete", "faithful", "faithful_linkability"]
GROUP_AMOUNT = {"earned": "earned", "unearned": "unearned", "shelter": "rent", "utilities": "util"}  # calc argument
FIXED_AMOUNTS = ["depcare", "medical", "child_support"]  # verified, outside the four groups: fixed calculator inputs
PARAMETERS = ["std_ded", "homeless_ded", "shelter_cap", "uncapped", "benmax", "min_ben", "size"]

# Person-level income types (persons 1 to 18) that add up to FSEARN and FSUNEARN, as in 01 part 3d.
EARNED_TYPES = ["WAGES", "SLFEMP", "OTHERN", "RENTMINC"]
UNEARNED_TYPES = ["SOCSEC", "VET", "SSI", "UNEMP", "WCOMP", "OTHGOV", "CONT", "DEEM", "GA", "EDLOAN", "OTHUN", "TANF",
                  "CSUPRT", "ENERGY", "WGESUP", "DIVER", "FOSTER", "INTER", "ANNUITY", "PENSION", "OLDAGE", "SURV",
                  "STRIKE", "RENTINC", "ALIMNY", "GOVROY", "GOVDIV", "GOVINTER", "TRUST", "UNK"]

# Plain-language name of each income type, used in notes and documents. Written from the variable names;
# not quoted from the codebook.
INCOME_LABEL = {
    "WAGES": "wages", "SLFEMP": "self-employment income (net of business costs)", "OTHERN": "pay for odd jobs",
    "RENTMINC": "income from renting out a room", "SOCSEC": "Social Security benefits", "VET": "veterans benefits",
    "SSI": "Supplemental Security Income (SSI)", "UNEMP": "unemployment benefits",
    "WCOMP": "workers' compensation", "OTHGOV": "a government benefit", "CONT": "help with money from a relative",
    "DEEM": "money from a relative's income", "GA": "general assistance", "EDLOAN": "student aid",
    "OTHUN": "other income", "TANF": "cash assistance (TANF)", "CSUPRT": "child support",
    "ENERGY": "energy assistance", "WGESUP": "a wage supplement", "DIVER": "a diversion payment",
    "FOSTER": "foster care payments", "INTER": "interest and dividends", "ANNUITY": "annuity payments",
    "PENSION": "a pension", "OLDAGE": "old-age assistance", "SURV": "survivor benefits", "STRIKE": "strike benefits",
    "RENTINC": "rental income", "ALIMNY": "alimony", "GOVROY": "royalties", "GOVDIV": "dividends",
    "GOVINTER": "interest", "TRUST": "trust income", "UNK": "other income",
}

# Which record a source leaves where an agent can retrieve it. No tax connector.
#   payroll: pay stubs from the payroll connector, net pay deposited in the bank feed.
#   award_letter: award letter in the drive and a monthly bank deposit (Social Security, SSI).
#   state_disbursement: monthly bank deposit labelled as a state disbursement (child support); no letter.
#   benefit_notice: our choice for the other agency or payer benefits not named in the design (unemployment,
#     TANF, veterans and similar): the payer's notice in the drive and a monthly bank deposit.
#   none: self-employment, odd jobs, room rental, contributions, deemed, other and unknown income leave no
#     linkable record. The complete packet still carries a monthly income record for them in the drive (so the
#     complete packet documents every fact); the linkability variant drops it.
LINK = {"WAGES": "payroll", "SOCSEC": "award_letter", "SSI": "award_letter", "CSUPRT": "state_disbursement",
        **{t: "none" for t in ["SLFEMP", "OTHERN", "RENTMINC", "CONT", "DEEM", "OTHUN", "UNK"]}}
LINK.update({t: "benefit_notice" for t in EARNED_TYPES + UNEARNED_TYPES if t not in LINK})
PAYER = {"SOCSEC": "Social Security Administration", "SSI": "Social Security Administration",
         "UNEMP": "State unemployment insurance agency", "TANF": "State human services department",
         "GA": "County general assistance office", "VET": "Department of Veterans Affairs",
         "WCOMP": "Workers' compensation insurer", "PENSION": "Pension plan administrator",
         "FOSTER": "State child welfare agency"}
DEPOSIT_LABEL = {"SOCSEC": "SOC SEC BENEFIT SSA", "SSI": "SSI BENEFIT SSA",
                 "CSUPRT": "STATE CHILD SUPPORT DISBURSEMENT UNIT"}
CHILD_SUPPORT_LABEL = DEPOSIT_LABEL["CSUPRT"]

# Utility types a household pays under each allowance tier (types from the form, lines 875-876). "actual
# costs" (UTIL matching no tier amount) is kept out of the pilot: the form asks only which utilities are paid,
# so it cannot carry an actual-cost amount.
TIER_TYPES = {"HCSUA": ["Heat", "Electricity"], "LUA": ["Electricity", "Water/Sewer"], "TUA": ["Phone"],
              "actual costs": ["Electricity", "Water/Sewer"]}

WINDOW = (dt.date(2026, 9, 1), dt.date(2026, 10, 31))  # two months of records
N_MONTHS = 2
PAY_FREQUENCY = {"Wk": "Weekly", "2Wks": "Biweekly", "2x/Mo": "Semi-monthly", "Mo": "Monthly"}
NO_STATE_WAGE_TAX = {"Alaska", "Florida", "Nevada", "New Hampshire", "South Dakota", "Tennessee", "Texas",
                     "Washington", "Wyoming"}
FICA = 0.0765

FIRST = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Dana", "Robin",
         "Jesse", "Drew", "Cameron", "Reese", "Hayden", "Rowan", "Emerson", "Sage", "Kendall", "Logan", "Parker",
         "Skyler", "Blair", "Elliot", "Finley", "Harper", "Lane", "Marlow", "Noel"]
LAST = ["Rivera", "Chen", "Okafor", "Novak", "Patel", "Lindqvist", "Moreau", "Haddad", "Kowalski", "Mendez",
        "Osei", "Brennan", "Tanaka", "Duarte", "Ferreira", "Ibrahim", "Larsen", "Nguyen", "Quinlan", "Sato"]
EMPLOYERS = ["Northside Logistics", "Harbor View Care Center", "Bluebird Diner", "Summit Cleaning Services",
             "Riverside Grocery Co.", "Pinecrest Warehouse", "Lakeshore Home Health", "Metro Parcel Services",
             "Greenway Landscaping", "Cedar Hill School District", "Brightpath Call Center", "Oak Street Bakery"]
LANDLORDS = ["Maple Court Properties", "Elm Ridge Apartments LLC", "Willow Park Housing", "Stonegate Rentals",
             "Birchwood Property Management", "Fairview Homes LLC"]
STREETS = ["Maple Ave", "Lincoln St", "Hillcrest Dr", "Park Blvd", "River Rd", "Washington St", "Oak Ln"]
WORK = ["house cleaning", "hair braiding", "rideshare driving", "lawn care", "home daycare", "furniture repair"]
STORES = ["Fresh Market", "Corner Grocery", "Save-Mart Foods", "Neighborhood Pantry Market"]
RETAILERS = ["Home Goods Outlet", "City Pharmacy", "Value Clothing"]
UTILITY_COMPANY = {"Heat": "Regional Gas Co.", "Electricity": "Metro Electric", "Water/Sewer": "City Water Department",
                   "Phone": "Lineworks Telephone"}
BILL_RANGE = {"Heat": (35, 140), "Electricity": (55, 160), "Water/Sewer": (30, 80), "Phone": (25, 70)}


# --- Sample -----------------------------------------------------------------------------------------------
def nz(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def utility_tier(df: pd.DataFrame, tiers: pd.DataFrame) -> pd.Series:
    """Each household's utility tier, inferred by matching its verified UTIL to its state's tier amounts:
    HCSUA, LUA, TUA, "none" (UTIL $0) or "actual costs" (no tier amount matches)."""
    util = df["UTIL"].fillna(0)
    t = tiers.reindex(df["STATENAME"]).set_axis(df.index)
    out = pd.Series("actual costs", index=df.index)
    for tier in reversed(t.columns):  # HCSUA wins a tie
        out[(t[tier] - util).abs() < 0.5] = tier
    out[util == 0] = "none"
    return out


def qc_benefit(df: pd.DataFrame) -> pd.Series:
    """calc.benefit on the verified amounts (the QC-correct benefit)."""
    return pd.Series(benefit(**inputs(df)), index=df.index)


def _draw_errors(arm: pd.DataFrame, rng) -> pd.Index:
    picks = [rng.choice(np.sort(arm.index[arm["group"] == g]), n, replace=False) for g, n in STRATA.items()]
    return pd.Index(np.concatenate(picks))


def pilot_sample(df: pd.DataFrame, arm: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """The 16 error and 4 comparison households, and a log of the draw.

    Error households: stratified by fact group (STRATA) from the human arm. A draw must hold at least 4 of the
    151, 2 with DISCOV1 in FROM_CASE_RECORD and 2 with AGENCY1 in CLIENT_IPV; if it does not, the next seed is
    tried (seed, seed + 1, ...). Comparison households: single-row-SUA states, outside the arm, AMTERR $56 or
    less, no MFIP or SSI-CAP unit, drawn with `seed`. Both exclude households where calc.benefit on the
    verified amounts misses FSBEN by more than $1, and households whose utility tier is "actual costs".
    """
    calc_ok = (qc_benefit(df) - df["FSBEN"]).abs() <= 1
    tier = utility_tier(df, sua_tiers(df))
    log = {}
    pool = arm[arm["group"].isin(STRATA)]
    log["error pool (four groups)"] = len(pool)
    log["error pool: calculator misses FSBEN"] = int((~calc_ok[pool.index]).sum())
    log["error pool: actual utility costs"] = int((calc_ok[pool.index] & (tier[pool.index] == "actual costs")).sum())
    pool = pool[calc_ok[pool.index] & (tier[pool.index] != "actual costs")]
    agency = df["AGENCY1"]
    for draw in range(1, MAX_DRAWS + 1):
        rows = _draw_errors(pool, np.random.default_rng(seed + draw - 1))
        counts = {"in_151": int(pool.loc[rows, "in_151"].sum()),
                  "DISCOV1 1 or 2": int(pool.loc[rows, "DISCOV1"].isin(FROM_CASE_RECORD).sum()),
                  "AGENCY1 3 or 4": int(agency[rows].isin(CLIENT_IPV).sum())}
        if (counts["in_151"] >= MIN_IN_151 and counts["DISCOV1 1 or 2"] >= MIN_FROM_CASE_RECORD
                and counts["AGENCY1 3 or 4"] >= MIN_CLIENT_IPV):
            break
    else:
        raise RuntimeError(f"no draw met the constraints in {MAX_DRAWS} seeds")
    log |= {"draws": draw, "seed used": seed + draw - 1, **counts}

    comp = df[df["STATENAME"].isin(SINGLE_SUA_STATES) & ~df.index.isin(arm.index) & (df["AMTERR"].fillna(0) <= TOLERANCE)
              & (df["MN_FIP"].fillna(0) != 1) & (df["SSI_CAP"].fillna(0) == 0)]
    log["comparison pool"] = len(comp)
    log["comparison pool: calculator misses FSBEN"] = int((~calc_ok[comp.index]).sum())
    log["comparison pool: actual utility costs"] = int((calc_ok[comp.index] & (tier[comp.index] == "actual costs")).sum())
    comp = comp[calc_ok[comp.index] & (tier[comp.index] != "actual costs")]
    comp_rows = pd.Index(np.random.default_rng(seed).choice(np.sort(comp.index), N_COMPARISON, replace=False))

    rows = rows.append(comp_rows)
    q = df.loc[rows]
    calc_in = inputs(q)
    out = pd.DataFrame({
        "kind": ["error"] * len(rows.difference(comp_rows, sort=False)) + ["comparison"] * N_COMPARISON,
        "group": arm["group"].reindex(rows).fillna(""),
        "in_151": arm["in_151"].reindex(rows).fillna(False).astype(bool),
        "DISCOV1": arm["DISCOV1"].reindex(rows).astype("Int64"),
        "AGENCY1": q["AGENCY1"].astype("Int64"),
        "state": q["STATENAME"],
        "size": q["FSUSIZE"].astype(int),
        "qc_benefit": qc_benefit(q),
        "FSBEN": q["FSBEN"],
        "RAWBEN": q["RAWBEN"],
        "AMTERR": q["AMTERR"],
        "filed_value": arm["filed_value"].reindex(rows),
        "verified_value": arm["verified_value"].reindex(rows),
        "utility_tier": tier[rows],
        **{f"verified_{g}": calc_in[a] for g, a in GROUP_AMOUNT.items()},
        **{c: calc_in[c] for c in FIXED_AMOUNTS + PARAMETERS if c != "size"},
    }, index=rows)
    out.index.name = "row"
    return out, log


# --- Ledger -----------------------------------------------------------------------------------------------
def ledger(qc: pd.Series, tier: str) -> list[dict]:
    """Facts of one household from its verified values: one per person x income type with an amount above $0,
    rent, and the utility allowance. Person-level income must add up to FSEARN and FSUNEARN."""
    facts = []
    for group, types, total in (("earned", EARNED_TYPES, "FSEARN"), ("unearned", UNEARNED_TYPES, "FSUNEARN")):
        s = 0.0
        for t in types:
            for p in range(1, 19):
                v = nz(qc.get(f"{t}{p}"))
                s += v
                if v > 0:
                    facts.append({"group": group, "type": t, "person": p, "amount": v})
        assert abs(s - nz(qc[total])) <= 1, f"row {qc.name}: person-level {group} {s} vs {total} {qc[total]}"
    if nz(qc["RENT"]) > 0:
        facts.append({"group": "shelter", "type": "RENT", "person": 1, "amount": nz(qc["RENT"])})
    if nz(qc["UTIL"]) > 0:
        facts.append({"group": "utilities", "type": tier, "person": 1, "amount": nz(qc["UTIL"]),
                      "utility_types": TIER_TYPES[tier]})
    for i, f in enumerate(facts, 1):
        f["fact"] = f"f{i:02d}"
    return facts


# --- Records ----------------------------------------------------------------------------------------------
def _pay_dates(freq: str, offset: int) -> list[dt.date]:
    """Every 2026 pay date of a schedule (weekly and biweekly on Fridays; semi-monthly on the 15th and the last
    day; monthly on the last day)."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(d) for d in range(365)]
    last = [d for d in days if (d + dt.timedelta(1)).day == 1]
    if freq == "Wk":
        return [d for d in days if d.weekday() == 4]
    if freq == "2Wks":
        return [d for d in days if d.weekday() == 4][offset::2]
    if freq == "2x/Mo":
        return sorted([d for d in days if d.day == 15] + last)
    return last


def _in_window(d: dt.date) -> bool:
    return WINDOW[0] <= d <= WINDOW[1]


def _month_days(day: int) -> list[dt.date]:
    return [dt.date(2026, m, day) for m in (9, 10)]


def _record(fact, connector, kind, data, date=None, linkable=True):
    return {"fact": fact["fact"] if fact else None, "group": fact["group"] if fact else None, "linkable": linkable,
            "connector": connector, "kind": kind, "data": data, "date": str(date or "")}


def _txn(fact, date, amount, description):
    return _record(fact, "bank", "transaction",
                   {"Date": str(date), "Amount": round(amount, 2), "Description": description}, date)


def build_case(qc: pd.Series, household: pd.Series, sua: dict) -> dict:
    """Ledger, names and the complete set of records for one household (seeded by its row). `sua` maps the
    allowance tiers of the household's state to their amounts."""
    row = int(qc.name)
    rng = np.random.default_rng([SEED, row])
    facts = ledger(qc, household["utility_tier"])
    n_people = max([int(household["size"])] + [f["person"] for f in facts])
    last = str(rng.choice(LAST))
    first = rng.choice(FIRST, n_people, replace=False)
    people = {p: {"name": f"{first[p - 1]} {last}", "first": str(first[p - 1]),
                  "age": None if pd.isna(qc.get(f"AGE{p}")) else int(qc[f"AGE{p}"])} for p in range(1, n_people + 1)}
    employers = iter(rng.choice(EMPLOYERS, len(EMPLOYERS), replace=False))
    case = {"row": row, "state": household["state"], "household": household, "sua": sua, "facts": facts,
            "people": people,
            "landlord": str(rng.choice(LANDLORDS)),
            "address": f"{rng.integers(100, 9900)} {rng.choice(STREETS)}, Apt {rng.integers(1, 30)}",
            "account": f"****{rng.integers(1000, 9999)}", "records": []}
    rec = case["records"]
    for f in facts:
        who = people[f["person"]]["name"]
        kind = LINK.get(f["type"])
        if f["group"] in ("earned", "unearned") and kind == "payroll":
            f["employer"] = str(next(employers))
            freq = str(rng.choice(list(PAY_FREQUENCY)))
            gross = round(f["amount"] / MONTHLY_FACTOR[freq], 2)
            fed_rate, state_rate = rng.uniform(0.04, 0.10), 0 if case["state"] in NO_STATE_WAGE_TAX else rng.uniform(0.02, 0.045)
            health = float(rng.choice([0.0, 0.0, rng.uniform(8, 40)]))
            schedule = _pay_dates(freq, int(rng.integers(0, 2)))
            for k, d in enumerate(schedule, 1):
                if not _in_window(d):
                    continue
                fed, st, fica, other = (round(gross * fed_rate, 2), round(gross * state_rate, 2),
                                        round(gross * FICA, 2), round(health, 2))
                net = round(gross - fed - st - fica - other, 2)
                rec.append(_record(f, "payroll", "pay_stub", {
                    "Employer": f["employer"], "Employee": who, "Pay date": str(d),
                    "Pay frequency": PAY_FREQUENCY[freq], "Gross pay": gross, "Federal income tax": fed,
                    "State income tax": st, "Social Security and Medicare": fica, "Health insurance": other,
                    "Net pay": net, "Year-to-date gross pay": round(gross * k, 2)}, d))
                rec.append(_txn(f, d, net, f"PAYROLL DEP {f['employer'].upper()}"))
        elif kind == "award_letter":
            rec.append(_record(f, "drive", "award_letter", {
                "Title": "Benefit award letter", "From": PAYER[f["type"]], "Beneficiary": who,
                "Benefit": INCOME_LABEL[f["type"]], "Monthly benefit amount": f["amount"],
                "Date of letter": "2025-12-03"}))
            for d in _month_days(3 if f["type"] == "SOCSEC" else 1):
                rec.append(_txn(f, d, f["amount"], DEPOSIT_LABEL[f["type"]]))
        elif kind == "state_disbursement":
            for d in _month_days(10):
                rec.append(_txn(f, d, f["amount"], CHILD_SUPPORT_LABEL))
        elif kind == "benefit_notice":
            payer = PAYER.get(f["type"], "Paying agency")
            rec.append(_record(f, "drive", "benefit_notice", {
                "Title": "Benefit notice", "From": payer, "Recipient": who, "Benefit": INCOME_LABEL[f["type"]],
                "Monthly amount": f["amount"], "Date of notice": "2026-08-14"}))
            for d in _month_days(5):
                rec.append(_txn(f, d, f["amount"], f"{payer.upper()} PAYMENT"))
        elif kind == "none":
            if f["type"] == "SLFEMP":
                f["work"] = str(rng.choice(WORK))
            rec.append(_record(f, "drive", "income_record", {
                "Title": "Monthly income record", "Person": who, "Income": INCOME_LABEL[f["type"]],
                "Average monthly amount": f["amount"], "Months covered": "September and October 2026"},
                linkable=False))
        elif f["group"] == "shelter":
            rec.append(_record(f, "drive", "lease", {
                "Title": "Lease agreement", "Landlord": case["landlord"], "Tenant": people[1]["name"],
                "Address": case["address"], "Monthly rent": f["amount"], "Lease start": "2026-03-01",
                "Lease term": "12 months"}))
            for d in _month_days(1):
                rec.append(_txn(f, d, -f["amount"], f"RENT PAYMENT {case['landlord'].upper()}"))
        elif f["group"] == "utilities":
            types = f["utility_types"]
            if f["type"] == "actual costs":  # bills add up to the verified amount
                share = rng.dirichlet(np.ones(len(types)))
                amounts = list(np.round(f["amount"] * share, 2))
                amounts[-1] = round(f["amount"] - sum(amounts[:-1]), 2)
            else:  # under an allowance the bill amounts do not enter the benefit
                amounts = [round(rng.uniform(*BILL_RANGE[t]), 2) for t in types]
            for t, a in zip(types, amounts):
                rec.append(_record(f, "drive", "utility_bill", {
                    "Title": "Utility bill", "Company": UTILITY_COMPANY[t], "Account holder": people[1]["name"],
                    "Service": t, "Service address": case["address"], "Billing period": "2026-09-15 to 2026-10-14",
                    "Amount due": float(a)}))
    # Neutral transactions: own-account transfers, groceries, a retail refund. None is income or a shelter cost.
    for _ in range(int(rng.integers(2, 4))):
        d = WINDOW[0] + dt.timedelta(int(rng.integers(0, 61)))
        rec.append(_txn(None, d, -round(rng.uniform(25, 140), 2), f"GROCERY {rng.choice(STORES).upper()}"))
    d = WINDOW[0] + dt.timedelta(int(rng.integers(0, 61)))
    rec.append(_txn(None, d, round(rng.uniform(40, 250), 2), f"TRANSFER FROM SAVINGS {case['account']} (OWN ACCOUNT)"))
    d = WINDOW[0] + dt.timedelta(int(rng.integers(0, 61)))
    rec.append(_txn(None, d, round(rng.uniform(8, 60), 2), f"REFUND {rng.choice(RETAILERS).upper()}"))
    return case


# --- Variants ---------------------------------------------------------------------------------------------
def record_shares(path=None) -> dict:
    """Share of errors found from the case record (DISCOV1 1 or 2) per fact group, from 01 part 5."""
    t = pd.read_csv(path or RESULTS / "01-data-checks" / "5b-discov-frequencies.csv")
    t = t[t["DISCOV1"].astype(str).isin([str(c) for c in FROM_CASE_RECORD])]
    return {g: float(t[g].astype(float).sum()) for g in GROUPS}


CONNECTOR_ORDER = {"payroll": 0, "bank": 1, "drive": 2}
DRIVE_ORDER = {"award_letter": 0, "benefit_notice": 1, "income_record": 2, "lease": 3, "utility_bill": 4}


def _missing_facts(case: dict, variant: str, shares: dict) -> list[str]:
    h, facts = case["household"], case["facts"]
    if variant == "complete":
        return []
    if variant == "faithful_linkability":
        return [r["fact"] for r in case["records"] if r["fact"] and not r["linkable"]]
    if h["kind"] == "error":
        if pd.notna(h["DISCOV1"]) and int(h["DISCOV1"]) in FROM_CASE_RECORD:
            return []
        return [f["fact"] for f in facts if f["group"] == h["group"]]
    # Comparison household: one non-zero group, dropped with probability 1 - share found from the case record.
    rng = np.random.default_rng([SEED, case["row"], 2])
    nonzero = sorted({f["group"] for f in facts}, key=GROUPS.index)
    if not nonzero:
        return []
    g = str(rng.choice(nonzero))
    drop = rng.random() < 1 - shares[g]
    return [f["fact"] for f in facts if f["group"] == g] if drop else []


def make_packet(case: dict, variant: str, shares: dict) -> dict:
    """The packet of one variant: the kept records in a fixed order with document ids doc-01, doc-02, ...
    numbered after removal. `sources`, `missing_facts` and `undocumented_groups` are for scoring, never shown."""
    missing = sorted(set(_missing_facts(case, variant, shares)))
    kept = [r for r in case["records"] if r["fact"] not in missing]
    kept.sort(key=lambda r: (CONNECTOR_ORDER[r["connector"]], DRIVE_ORDER.get(r["kind"], 0),
                             r["data"].get("Employee", ""), r["date"], json.dumps(r["data"], sort_keys=True)))
    docs, sources = [], {}
    for i, r in enumerate(kept, 1):
        doc_id = f"doc-{i:02d}"
        docs.append({"id": doc_id, "connector": r["connector"], "kind": r["kind"], "data": r["data"]})
        sources[doc_id] = r["fact"]
    groups = {f["fact"]: f["group"] for f in case["facts"]}
    return {"row": case["row"], "variant": variant, "state": case["state"], "sua": case["sua"],
            "account": case["account"],
            "window": [str(WINDOW[0]), str(WINDOW[1])], "documents": docs, "sources": sources,
            "missing_facts": missing, "undocumented_groups": sorted({groups[f] for f in missing}, key=GROUPS.index)}


# --- Rendering --------------------------------------------------------------------------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"


def _render_doc(d: dict) -> str:
    data = d["data"]
    if d["kind"] == "transaction":
        sign = "credit" if data["Amount"] >= 0 else "debit"
        return f"[{d['id']}] {data['Date']} | {sign} {money(abs(data['Amount']))} | {data['Description']}"
    title = data.get("Title", "Pay stub")
    lines = [f"[{d['id']}] {title}"]
    for k, v in data.items():
        if k == "Title":
            continue
        lines.append(f"    {k}: {money(v) if isinstance(v, float) else v}")
    return "\n".join(lines)


def render_documents(packet: dict) -> str:
    """The packet as connector results, one block per connector."""
    docs = packet["documents"]
    out = []
    stubs = [d for d in docs if d["connector"] == "payroll"]
    out.append(f"payroll connector returned {len(stubs)} pay stub{'s' if len(stubs) != 1 else ''}"
               f"{':' if stubs else '.'}")
    out += [_render_doc(d) for d in stubs]
    txns = [d for d in docs if d["connector"] == "bank"]
    out.append("")
    out.append(f"bank connector returned {len(txns)} transactions for checking account {packet['account']}, "
               f"{packet['window'][0]} to {packet['window'][1]}:")
    out += [_render_doc(d) for d in txns]
    drive = [d for d in docs if d["connector"] == "drive"]
    out.append("")
    out.append(f"drive search returned {len(drive)} document{'s' if len(drive) != 1 else ''}{':' if drive else '.'}")
    out += [_render_doc(d) for d in drive]
    return "\n".join(out)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def round_50(x: float) -> int:
    """Nearest $50 (at least $50 for an amount above $0)."""
    r = int(math.floor(x / 50 + 0.5)) * 50
    return 50 if r == 0 and x > 0 else r


def note(case: dict, packet: dict, version: str = "A") -> str:
    """The applicant's note. Version A names every fact the household has (people, jobs, benefits, rent,
    utility types) and gives no amounts. Version B adds, for each group the packet leaves undocumented, the
    verified monthly amount of its missing facts rounded to the nearest $50: what an applicant who knows the
    figure roughly would say, no better and no worse."""
    people = case["people"]

    def subj(p):
        return ("I", False) if p == 1 else (people[p]["first"], True)

    def verb(base, third):
        return base + ("es" if base.endswith(("sh", "ch", "o")) else "s") if third else base

    def person(p):
        age = people[p]["age"]
        return people[p]["first"] + (f" ({age})" if age is not None else "")

    lines = [f"Hi, my name is {people[1]['name']}."]
    if len(people) > 1:
        lines.append(f"I live with {_join([person(p) for p in list(people)[1:]])}.")
    lines.append(f"There are {int(case['household']['size'])} of us applying for food assistance.")
    for f in case["facts"]:
        s, third = subj(f["person"])
        t = f["type"]
        if t == "WAGES":
            lines.append(f"{s} {verb('work', third)} at {f['employer']}.")
        elif t == "SLFEMP":
            lines.append(f"{s} {verb('do', third)} self-employed work ({f['work']}).")
        elif f["group"] in ("earned", "unearned"):
            lines.append(f"{s} {verb('get', third)} {INCOME_LABEL[t]}.")
        elif f["group"] == "shelter":
            lines.append(f"We rent our home at {case['address']} from {case['landlord']}.")
        elif f["group"] == "utilities":
            types = [u.lower() for u in f["utility_types"]]
            rest = "" if f["type"] in ("HCSUA", "actual costs") else (
                " Heat is included in the rent." if f["type"] == "LUA" else " Our other utilities are included in the rent.")
            lines.append(f"We pay for {_join(types)} ourselves.{rest}")
    if version == "B":
        by_id = {f["fact"]: f for f in case["facts"]}
        for g in packet["undocumented_groups"]:
            missing = [by_id[i] for i in packet["missing_facts"] if by_id[i]["group"] == g]
            x = round_50(sum(f["amount"] for f in missing))  # verified amount of the missing facts, rounded
            labels = list(dict.fromkeys(INCOME_LABEL.get(f["type"], "") for f in missing))
            taxes = " before taxes" if any(f["type"] == "WAGES" for f in missing) else ""
            lines.append({"earned": f"I'd put the {_join(labels)} at about ${x:,} a month{taxes}.",
                          "unearned": f"I'd put the {_join(labels)} at about ${x:,} a month.",
                          "shelter": f"The rent is about ${x:,} a month.",
                          "utilities": f"Utilities run about ${x:,} a month."}[g])
    return " ".join(lines)


FILER_INSTRUCTIONS = ("You are filling out a SNAP (food assistance) application for the household below. Use the "
                      "applicant's note and the records the connectors returned. Submit the form by calling "
                      "submit_application exactly once.")


def filer_input(packet: dict, note_text: str) -> str:
    """The text block a filer receives: the state, the applicant's note and the pre-fetched connector results."""
    return (f"State: {packet['state']}\n\nApplicant's note:\n{note_text}\n\n"
            f"Connector results (retrieved for this application):\n\n{render_documents(packet)}\n")


# --- Applications and benefit -----------------------------------------------------------------------------
def human_filed(household) -> dict:
    """The application as filed, one monthly amount per group: the rebuilt filed value for the group in error,
    every other group at its verified value."""
    out = {g: float(household[f"verified_{g}"]) for g in GROUPS}
    if household["kind"] == "error":
        out[household["group"]] = float(household["filed_value"])
    return out


def benefit_of(monthly: dict, household) -> float:
    """calc.benefit with the four group amounts from `monthly` ("insufficient_evidence" or missing -> $0) and
    the household's verified fixed amounts and parameters."""
    amt = {GROUP_AMOUNT[g]: 0.0 if monthly.get(g) in (None, ESCAPE) else float(monthly[g]) for g in GROUPS}
    fixed = {c: household[c] for c in FIXED_AMOUNTS + ["std_ded", "homeless_ded", "shelter_cap", "benmax", "min_ben"]}
    return float(benefit(**amt, **fixed, uncapped=bool(household["uncapped"]), size=household["size"]))
