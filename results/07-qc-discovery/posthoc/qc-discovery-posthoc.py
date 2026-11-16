"""Post hoc, no model calls: how often QC found the error from the case record, and how often a budget-gap rule
(rent above filed earned plus unearned income) would have fired, for the 185 omissions of the human arm and for
households with client-caused unreported income.

    PYTHONPATH=src python results/07-qc-discovery/posthoc/qc-discovery-posthoc.py

Reads the FY2024 QC public file (data/raw/fy2024, fetched by data/fetch.sh) and
results/01-data-checks/3d-human-arm.csv. Writes qc-discovery-posthoc.csv beside this script.
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path

import pandas as pd

from snapqc.data import ROOT, load_fy2024

OUT = Path(__file__).resolve().parent
TOL = 56
CLIENT = [1, 2, 3, 4, 7]
CASE_RECORD = [1, 2]  # DISCOV 1 and 2: found from the case record

full = load_fy2024()
arm = pd.read_csv(ROOT / "results" / "01-data-checks" / "3d-human-arm.csv", index_col="row")
omission = arm[arm.group.isin(["earned", "unearned"]) & (arm.filed_value <= 2) & (arm.verified_value > 0)].copy()
states41 = sorted(full.loc[arm.index, "STATENAME"].unique())
assert len(full) == 44891 and len(omission) == 185 and len(states41) == 41

rows = []


def add(measure, k, n):
    rows.append({"measure": measure, "k": int(k), "n": int(n), "share": round(k / n, 4)})


# 1. Budget-gap rule recall on the 185 omissions, split by how QC found the variance.
# The rule fires when RENT exceeds filed earned plus filed unearned income, the omitted group at its filed value.
def fires(row, o):
    earned = 0 if pd.isna(full.at[row, "FSEARN"]) else full.at[row, "FSEARN"]
    unearned = 0 if pd.isna(full.at[row, "FSUNEARN"]) else full.at[row, "FSUNEARN"]
    if o.group == "earned":
        earned = o.filed_value
    else:
        unearned = o.filed_value
    rent = 0 if pd.isna(full.at[row, "RENT"]) else full.at[row, "RENT"]
    return int(rent > earned + unearned)


omission["rule_fires"] = [fires(r, o) for r, o in omission.iterrows()]
omission["found_from_case_record"] = omission.DISCOV1.isin(CASE_RECORD)
hidden = omission[~omission.found_from_case_record]
shown = omission[omission.found_from_case_record]
add("budget-gap rule fires, omissions where the case record did not show the error", hidden.rule_fires.sum(), len(hidden))
add("budget-gap rule fires, omissions found from the case record", shown.rule_fires.sum(), len(shown))
add("budget-gap rule fires, all 185 omissions", omission.rule_fires.sum(), len(omission))

# 2. Households with a client-caused unreported-income variance (client cause, NATURE 35) in an error above $56.
cols = ["ELEMENT", "NATURE", "AGENCY", "AMOUNT", "DISCOV"]
var = pd.concat([
    full[[f"{c}{i}" for c in cols]].set_axis(cols, axis=1).assign(hh=full.index) for i in range(1, 10)
]).dropna(subset=["ELEMENT"])
var = var.join(full[["STATUS", "AMTERR", "FYWGT", "STATENAME"]], on="hh")
var = var[var.STATUS.isin([2, 3]) & (var.AMTERR > TOL)]
var["case_record"] = var.DISCOV.isin(CASE_RECORD)
unrep = var[var.AGENCY.isin(CLIENT) & (var.NATURE == 35)]
first = unrep.groupby("hh", sort=False).head(1).set_index("hh")  # first variance, element order 1 to 9
anyv = unrep.groupby("hh").case_record.any()  # any such variance found from the case record
for label, keep in [("53 jurisdictions", first.index), ("41 states", first.index[first.STATENAME.isin(states41)])]:
    add(f"households with client-caused unreported income found from the case record, first variance, {label}",
        first.loc[keep].case_record.sum(), len(keep))
    add(f"households with client-caused unreported income found from the case record, any variance, {label}",
        anyv.loc[keep].sum(), len(keep))

result = pd.DataFrame(rows)
result.to_csv(OUT / "qc-discovery-posthoc.csv", index=False)
print(result.to_string(index=False))

expected = {
    "budget-gap rule fires, omissions where the case record did not show the error": (54, 169),
    "budget-gap rule fires, omissions found from the case record": (5, 16),
    "households with client-caused unreported income found from the case record, first variance, 53 jurisdictions": (105, 984),
    "households with client-caused unreported income found from the case record, first variance, 41 states": (67, 785),
    "households with client-caused unreported income found from the case record, any variance, 41 states": (70, 785),
}
for m, (k, n) in expected.items():
    r = result[result.measure == m].iloc[0]
    assert (r.k, r.n) == (k, n), (m, int(r.k), int(r.n), k, n)
print("all expected counts reproduced")
