"""The pilot application form, its tool schemas, and the agency-side conversion to monthly amounts.

The form mirrors the income, housing and utility parts of Michigan's application, MDHHS-1171 (Rev. 10-26),
as read in `data/raw/forms/michigan/MDHHS-1171_Assistance_Application_and_Program_Supplements_Rev10-26.txt`:

- lines 335-341, self-employment: who, type of work, monthly income before expenses, monthly expenses;
- lines 342-356, employment: who, employer name, average hours per week, wages/tips before tax,
  "per Hr Wk 2Wks 2x/Mo Mo Yr";
- lines 357-374, additional income: who, type of income, amount received, "per Wk 2Wks 2x/Mo Mo Yr";
- lines 833-851, housing expenses: type of expense, amount, how often paid;
- lines 859-876, utilities paid: heat, cooking fuel, electricity, trash pickup, air conditioning,
  water/sewer, phone (yes or no each; the form asks for no utility amounts).

Three tool schemas for one forced `submit_application` call: "required" (every money field a number),
"escape" (a number or "insufficient_evidence") and "provenance" (as escape, plus a source tag and the
document ids each value rests on).
"""
from __future__ import annotations

import math

import jsonschema
import numpy as np
import pandas as pd

GROUPS = ["earned", "unearned", "shelter", "utilities"]
ESCAPE = "insufficient_evidence"
SOURCES = ["retrieved_record", "applicant_stated", "agent_inferred"]
SCHEMAS = ["required", "escape", "provenance"]

FREQUENCIES = ["Hr", "Wk", "2Wks", "2x/Mo", "Mo", "Yr"]  # employment, form lines 350-353
PAID_FREQUENCIES = ["Wk", "2Wks", "2x/Mo", "Mo", "Yr"]  # additional income, form lines 368-371
# Monthly conversion factors. Weekly 4.3 and every two weeks 2.15 are the factors of 7 CFR 273.10(c)(2)(i)
# (eCFR, version 2026-11-01; quoted in notebook 04 part 2). The rule names no factor for other frequencies: twice a
# month (x 2), monthly (x 1) and yearly (/ 12) are the exact monthly figure of a fixed amount on a fixed schedule, and
# an hourly rate is multiplied by the hours per week and then by the weekly factor (our rules, not the CFR text).
MONTHLY_FACTOR = {"Wk": 4.3, "2Wks": 2.15, "2x/Mo": 2.0, "Mo": 1.0, "Yr": 1 / 12}
UTILITY_TYPES = ["Heat", "Cooking Fuel", "Electricity", "Trash Pickup", "Air Conditioning", "Water/Sewer",
                 "Phone"]  # form lines 875-876
HOUSING_TYPES = ["Rent", "Mortgage", "Land Contract", "Mobile Home Lot Rent", "Homeowner's Insurance",
                 "Property Tax", "Other"]  # form lines 836-844

# --- Utility allowance tiers ------------------------------------------------------------------------------
# The tiers follow the footnotes of Table F.7 of the FY2024 SNAP QC technical documentation: the heating and
# cooling allowance (HCSUA) is "for units with heating and cooling expenses not included in the rent"; the
# limited allowance (LUA) "for units that do not have heating and cooling expenses separate from rent"; the
# telephone allowance (TUA) "for units that have telephone expenses but do not have any other utility expenses".
# Each state's tier amounts are modes of the verified UTIL in the QC file (SUA2 == 1), by SUA1 code: codes 3, 4
# and 8 share the highest common amount in every state (taken as HCSUA), code 5 a middle one (LUA) and code 6 the
# lowest (TUA). This grouping of codes is inferred from the amounts; the SUA1 code labels are not verified.
HCSUA_CODES, LUA_CODES, TUA_CODES = [3, 4, 8], [5], [6]
TIERS = ["HCSUA", "LUA", "TUA"]
HEATING_COOLING = {"Heat", "Air Conditioning"}


def sua_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """State x tier amounts (HCSUA, LUA, TUA) as modes of UTIL by SUA1 code; NaN where no household has the code."""
    d = df[df["SUA2"] == 1]

    def mode(s):
        m = s.mode()
        return float(m.iloc[0]) if len(m) else np.nan

    return pd.DataFrame({
        "HCSUA": d[d["SUA1"].isin(HCSUA_CODES)].groupby("STATENAME")["UTIL"].agg(mode),
        "LUA": d[d["SUA1"].isin(LUA_CODES)].groupby("STATENAME")["UTIL"].agg(mode),
        "TUA": d[d["SUA1"].isin(TUA_CODES)].groupby("STATENAME")["UTIL"].agg(mode),
    })


def tier_for_types(types) -> str | None:
    """Allowance tier for a set of utility types paid: heat or air conditioning -> HCSUA; any other utility
    except the phone -> LUA; the phone alone -> TUA; nothing -> None."""
    types = set(types or [])
    if types & HEATING_COOLING:
        return "HCSUA"
    if types - {"Phone"}:
        return "LUA"
    if types:
        return "TUA"
    return None


def utility_amount(types, sua: dict) -> float:
    """Monthly utility amount for the types paid: the state's tier amount. A tier the state does not have counts
    as $0 (open question: states without an LUA in the file)."""
    tier = tier_for_types(types)
    amount = sua.get(tier) if tier else 0.0
    return 0.0 if amount is None or (isinstance(amount, float) and math.isnan(amount)) else float(amount)


# --- Tool schemas -----------------------------------------------------------------------------------------
def _fact(base: dict, schema: str) -> dict:
    """A field holding a fact: the plain value, or with the escape token, or wrapped with its provenance."""
    if schema == "required":
        return base
    value = {"anyOf": [base, {"type": "string", "enum": [ESCAPE]}]}
    if schema == "escape":
        return value
    return {"type": "object",
            "properties": {"value": value,
                           "source": {"type": "string", "enum": SOURCES},
                           "document_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["value", "source", "document_ids"], "additionalProperties": False}


def _items(props: dict) -> dict:
    return {"type": "array",
            "items": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}


def application_schema(schema: str = "required") -> dict:
    """JSON schema of the application for one of SCHEMAS."""
    money = _fact({"type": "number", "minimum": 0}, schema)
    props = {
        "household_size": {"type": "integer", "minimum": 1},
        "employment": _items({
            "who": {"type": "string"}, "employer": {"type": "string"},
            "avg_hours_per_week": _fact({"type": "number", "minimum": 0}, schema),
            "wages_before_tax": money, "per": {"type": "string", "enum": FREQUENCIES}}),
        "self_employment": _items({
            "who": {"type": "string"}, "type_of_work": {"type": "string"},
            "monthly_income_before_expenses": money, "monthly_expenses": money}),
        "additional_income": _items({
            "who": {"type": "string"}, "type_of_income": {"type": "string"},
            "amount": money, "per": {"type": "string", "enum": PAID_FREQUENCIES}}),
        "housing_expenses": _items({
            "type_of_expense": {"type": "string", "enum": HOUSING_TYPES},
            "amount": money, "how_often": {"type": "string", "enum": PAID_FREQUENCIES}}),
        "utilities_paid": _fact({"type": "array", "items": {"type": "string", "enum": UTILITY_TYPES},
                                 "uniqueItems": True}, schema),
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


TOOL_DESCRIPTION = "Submit the completed SNAP application form."


def tool(schema: str = "required") -> dict:
    """The `submit_application` function tool (Responses API format) for one of SCHEMAS."""
    return {"type": "function", "name": "submit_application", "description": TOOL_DESCRIPTION,
            "parameters": application_schema(schema)}


def validate(application: dict, schema: str = "required") -> list[str]:
    """Schema errors of a parsed tool output (empty list = valid)."""
    validator = jsonschema.Draft202012Validator(application_schema(schema))
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in validator.iter_errors(application)]


# --- Agency-side conversion -------------------------------------------------------------------------------
def _value(field):
    """Plain value of a fact field (unwraps the provenance object); missing or blank -> None."""
    if isinstance(field, dict):
        field = field.get("value")
    return None if field == "" else field


def _num(field):
    v = _value(field)
    return v if v == ESCAPE else (0.0 if v is None else float(v))


def _monthly(amount, per, hours=None):
    if amount == ESCAPE or (per == "Hr" and hours == ESCAPE):  # hours are used only for an hourly rate
        return ESCAPE
    if per == "Hr":
        return amount * (hours or 0.0) * MONTHLY_FACTOR["Wk"]
    return amount * MONTHLY_FACTOR.get(per, 1.0)  # no frequency given: taken as monthly


def _total(parts):
    return ESCAPE if ESCAPE in parts else float(sum(parts))


def to_monthly(application: dict, sua: dict) -> dict:
    """Monthly amount per fact group, as the agency would compute it from the form: per-period amount x factor.
    A group holding any "insufficient_evidence" value stays in that state (not $0); blank or missing counts as $0.
    Utilities: the state's allowance tier for the utility types reported (`sua` maps tier -> amount)."""
    a = application or {}
    earned = [_monthly(_num(j.get("wages_before_tax")), j.get("per"), _num(j.get("avg_hours_per_week")))
              for j in a.get("employment") or []]
    for s in a.get("self_employment") or []:
        income, costs = _num(s.get("monthly_income_before_expenses")), _num(s.get("monthly_expenses"))
        earned.append(ESCAPE if ESCAPE in (income, costs) else max(0.0, income - costs))
    unearned = [_monthly(_num(i.get("amount")), i.get("per")) for i in a.get("additional_income") or []]
    shelter = [_monthly(_num(h.get("amount")), h.get("how_often")) for h in a.get("housing_expenses") or []]
    types = _value(a.get("utilities_paid"))
    return {"earned": _total(earned), "unearned": _total(unearned), "shelter": _total(shelter),
            "utilities": ESCAPE if types == ESCAPE else utility_amount(types, sua)}
