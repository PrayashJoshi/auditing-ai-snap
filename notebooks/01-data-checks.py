# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 Pre-experiment data checks (no LLM calls)
#
# Pre-experiment checks, using data only, on the FY2024 SNAP QC public-use file that fix the human-arm sample, the packets and the interval widths.
# Sections are numbered; letters mark parts of one check. Each section carries its own header (question, expected, method, result, mismatch, feeds, recommendation).
# Expected figures for sections 1 to 3 come from the `sprint-log.md` TABULATION entry in this repo; those for sections 4 to 9 come from a scratch run of the 616-household definition (2026-11-14). A mismatch is reported as a mismatch and never adjusted.
#
# Sections 4 to 9 run on the human arm of 616 households built in part 3d (the decision of 2026-11-14) and read it from `results/01-data-checks/3d-human-arm.csv`.
#
# | § | Test | Parts |
# |---|---|---|
# | 1 | Funnel | a funnel counts, b 41 cases with no variance count, c STATUS 2/3 with error <= $56 |
# | 2 | Weights | a CSV vs old weights, b which file carries corrected weights |
# | 3 | Rebuild | a invert benefit formula, b non-invertible cases, c calculator on rebuilt application, d the 616 |
# | 4 | Fact groups | a ELEMENT -> fact group map, b counts per group, c empty entry vs wrong number, three cells |
# | 5 | Discovery | a DISCOV1 and VERIF1 by fact group, b DISCOV1 frequencies per group for comparison households |
# | 6 | Non-zero groups | a count per household (616 and comparison pool), b by fact group |
# | 7 | Background errors | a error share outside the 616 by cause, weighted error dollars, b comparison sample size |
# | 8 | Action type | a split by ACTNTYPE, b mix difference by fact group, cell and error type |
# | 9 | Intervals | a strata and effective sample size, b recall 0.5 half-widths, c paired filer gap and pooling |
#
# Order of work: 1, 3, 4, then 5 to 9. Section 2 can run any time (it decides which weight column the later weighted results use).

# %% [markdown]
# ## 0. Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from snapqc.data import SINGLE_SUA_STATES, TOLERANCE, load_fy2024, per_variance, results_dir

OUT = results_dir("01-data-checks")
SEED = 20261114
rng = np.random.default_rng(SEED)

df = load_fy2024()
assert df.shape == (44_891, 1_177), df.shape  # expected shape per data/README.md
df.shape

# %%
# Per-variance column blocks (one slot per variance, up to 9).
ELEMENT = per_variance("ELEMENT")
AGENCY = per_variance("AGENCY")
TIMEPER = per_variance("TIMEPER")
df[["STATUS", "AMTERR", "ACTNTYPE", "HWGT", "FYWGT"]].describe().T

# %% [markdown]
# ## 1. Funnel
#
# - **Question:** can the funnel 44,891 -> 2,096 -> 1,098 -> 917 be reproduced, why do 41 error cases have no variance count, and does any case have STATUS 2 or 3 with an error of $56 or less?
# - **Expected:** 44,891 households; 2,096 rebuildable; 1,098 above $56; 917 in the 41 single-row-SUA states.
# - **Method:** apply each filter in turn from the raw file and print the count after each step.
# - **Result:** reproduced exactly: 44,891 -> 2,096 -> 1,098 -> 917. The 41 error cases with no variance have STATUS 2 or 3 and a non-zero AMTERR, but every variance detail field is blank, so they carry no cause code and cannot pass the client-caused step. 11 of them are above $56 (10 in the 41 states). The file does not say why the detail is missing. STATUS alone does not define an error above the tolerance: 12,403 of the 17,836 STATUS 2/3 cases (70%) have an error of $56 or less.
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Households (human-arm sample size).

# %% [markdown]
# ### 1a. Funnel counts
#
# "Rebuildable" means: an error case (STATUS 2 or 3) with exactly one variance, caused by the client (AGENCY 1, 2, 3, 4 or 7), that occurred before or at the agency's most recent action (TIMEPER 1 or 2).

# %%
CLIENT_CAUSES = [1, 2, 3, 4, 7]  # AGENCY codes where the client or a collateral contact caused the variance
AT_OR_BEFORE_ACTION = [1, 2]  # TIMEPER codes; 3 = after the most recent action

n_variances = df[ELEMENT].notna().sum(axis=1)
steps = {
    "all households": pd.Series(True, index=df.index),
    "error case (STATUS 2 or 3)": df["STATUS"].isin([2, 3]),
    "exactly one variance": n_variances == 1,
    "client-caused": df[AGENCY].isin(CLIENT_CAUSES).any(axis=1),
    "at or before the most recent action": df[TIMEPER].isin(AT_OR_BEFORE_ACTION).any(axis=1),
    f"error above ${TOLERANCE}": df["AMTERR"] > TOLERANCE,
    "in the 41 single-row-SUA states": df["STATENAME"].isin(SINGLE_SUA_STATES),
}
keep = pd.DataFrame(steps).cummin(axis=1)  # a household stays only while it passes every step so far
funnel = keep.sum().rename("n").to_frame()
funnel["expected"] = pd.Series({
    "all households": 44_891,
    "at or before the most recent action": 2_096,
    f"error above ${TOLERANCE}": 1_098,
    "in the 41 single-row-SUA states": 917,
}).astype("Int64")
funnel["match"] = (funnel["n"] == funnel["expected"]).where(funnel["expected"].notna())
funnel.to_csv(OUT / "1a-funnel.csv", index_label="step")

h917 = df[keep.iloc[:, -1]]  # the human-arm sample used by the later sections
assert h917["ELEMENT1"].notna().all()  # the single variance always sits in slot 1
funnel

# %% [markdown]
# ### 1b. The 41 error cases with no variance count

# %%
no_variance = df[steps["error case (STATUS 2 or 3)"] & (n_variances == 0)]
detail = [c for p in ["ELEMENT", "NATURE", "AGENCY", "DISCOV", "VERIF", "OCCDATE", "TIMEPER", "E_FINDG"] for c in per_variance(p)]
pd.Series({
    "error cases with no variance": len(no_variance),
    "any variance detail field filled": int(no_variance[detail].notna().any(axis=1).sum()),
    "AMOUNT1..9 all zero": int((no_variance[per_variance("AMOUNT")] == 0).all(axis=1).sum()),
    "overissuance (STATUS 2)": int((no_variance["STATUS"] == 2).sum()),
    "underissuance (STATUS 3)": int((no_variance["STATUS"] == 3).sum()),
    f"error above ${TOLERANCE}": int((no_variance["AMTERR"] > TOLERANCE).sum()),
    f"error above ${TOLERANCE} and in the 41 states": int(((no_variance["AMTERR"] > TOLERANCE) & no_variance["STATENAME"].isin(SINGLE_SUA_STATES)).sum()),
    "states represented": no_variance["STATENAME"].nunique(),
    "median AMTERR": int(no_variance["AMTERR"].median()),
}).to_frame("value")

# %%
no_variance[["STATENAME", "YRMONTH", "STATUS", "AMTERR", "RAWBEN", "FSBEN"]].to_csv(OUT / "1b-no-variance-cases.csv", index_label="row")
no_variance["STATENAME"].value_counts().to_frame("cases").T

# %% [markdown]
# ### 1c. STATUS 2/3 with error of $56 or less

# %%
size = pd.cut(df["AMTERR"], [-1, 0, TOLERANCE, np.inf], labels=["$0", f"$1 to ${TOLERANCE}", f"above ${TOLERANCE}"])
status = df["STATUS"].map({1: "1 correct", 2: "2 overissuance", 3: "3 underissuance"})
status_by_size = pd.crosstab(status, size, margins=True)
status_by_size.to_csv(OUT / "1c-status-by-error-size.csv")
status_by_size

# %% [markdown]
# ## 2. Weights
#
# - **Question:** do the weights in the CSV match the corrected weights (revision of 2026-08-18)?
# - **Expected:** identical, or switch files. The file is stamped 2026-06-26 but the zip is the August revision; the CSV also carries HWGT_OLD and FYWGT_OLD.
# - **Method:** compare HWGT/FYWGT against HWGT_OLD/FYWGT_OLD; check against the technical documentation's totals if stated.
# - **Result:** `HWGT` and `FYWGT` are the corrected weights. They reproduce the August 2026 technical documentation: the ranges (HWGT 52.36 to 77,999.84; FYWGT 4.36 to 6,499.99) and the national weighted totals (40,168,146 individuals; $7,366,534 thousand in benefits). `HWGT_OLD` and `FYWGT_OLD` do not (40,772,270 individuals; largest HWGT 45,716). The two sets differ in 44,838 of 44,891 rows (ratio 0.10 to 5.97) and have the same overall sum. The CSV in use is byte-identical to the one inside the zip from the `2026-08` folder.
# - **Mismatch:** none; no change of file is needed. The 2026-06-26 timestamp on the CSV inside the August zip is not explained by anything in the file.
# - **Feeds:** `rq.md` Statistics (which weights RQ2 uses).

# %% [markdown]
# ### 2a. Current vs `_OLD` weight columns
#
# The August 2026 technical documentation states what the corrected weights look like: the ranges of HWGT and FYWGT (Chapter V codebook) and the national weighted totals of individuals and benefits (Table II.2). Whichever pair of columns reproduces those figures is the corrected pair.

# %%
documented = pd.Series({  # FY2024 technical documentation, August 2026 revision
    "HWGT min": 52.36, "HWGT max": 77_999.84, "FYWGT min": 4.36, "FYWGT max": 6_499.99,
    "individuals (weighted)": 40_168_146, "benefits, $000 (weighted)": 7_366_534,
})


def weight_figures(hwgt, fywgt):
    return pd.Series({
        "HWGT min": hwgt.min(), "HWGT max": hwgt.max(), "FYWGT min": fywgt.min(), "FYWGT max": fywgt.max(),
        "individuals (weighted)": (fywgt * df["FSUSIZE"]).sum(),
        "benefits, $000 (weighted)": (fywgt * df["FSBEN"]).sum() / 1_000,
    })


weights = pd.DataFrame({
    "documented (Aug 2026)": documented,
    "HWGT / FYWGT": weight_figures(df["HWGT"], df["FYWGT"]),
    "HWGT_OLD / FYWGT_OLD": weight_figures(df["HWGT_OLD"], df["FYWGT_OLD"]),
}).round(2)
weights.to_csv(OUT / "2a-weights-vs-documentation.csv", index_label="figure")
weights

# %%
ratio = df["HWGT"] / df["HWGT_OLD"]
pd.Series({
    "rows where HWGT differs from HWGT_OLD": int((~np.isclose(df["HWGT"], df["HWGT_OLD"])).sum()),
    "share of rows that differ": (~np.isclose(df["HWGT"], df["HWGT_OLD"])).mean(),
    "HWGT / HWGT_OLD, min": ratio.min(),
    "HWGT / HWGT_OLD, median": ratio.median(),
    "HWGT / HWGT_OLD, max": ratio.max(),
    "sum of HWGT / sum of HWGT_OLD": df["HWGT"].sum() / df["HWGT_OLD"].sum(),
    "FYWGT equals HWGT / 12 (share of rows)": np.isclose(df["FYWGT"], df["HWGT"] / 12).mean(),
}).round(4).to_frame("value")

# %% [markdown]
# ### 2b. Which file carries the corrected weights
#
# The CSV read above is checked against the copy inside the zip that was downloaded from the `2026-08` folder of snapqcdata.net, and against the checksums recorded in `data/SHA256SUMS`.

# %%
import hashlib
import zipfile

from snapqc.data import RAW, ROOT

zip_path = RAW / "fy2024" / "qcfy2024_csv.zip"
with zipfile.ZipFile(zip_path) as z:
    member = z.getinfo("qc_pub_fy2024.csv")
    sha_in_zip = hashlib.sha256(z.read(member)).hexdigest()
sha_on_disk = hashlib.sha256((RAW / "fy2024" / "qc_pub_fy2024.csv").read_bytes()).hexdigest()
sha_zip = hashlib.sha256(zip_path.read_bytes()).hexdigest()
recorded = dict(line.split()[::-1] for line in (ROOT / "data" / "SHA256SUMS").read_text().splitlines())

pd.Series({
    "CSV timestamp inside the zip": "{}-{:02d}-{:02d}".format(*member.date_time[:3]),
    "CSV on disk is the CSV inside the zip": sha_on_disk == sha_in_zip,
    "CSV matches data/SHA256SUMS": sha_on_disk == recorded["fy2024/qc_pub_fy2024.csv"],
    "zip matches data/SHA256SUMS": sha_zip == recorded["fy2024/qcfy2024_csv.zip"],
}).to_frame("value")

# %%
print("CSV sha256:", sha_on_disk)
print("zip sha256:", sha_zip)

# %% [markdown]
# ## 3. Rebuild the filed value
#
# - **Question:** for each of the 917, can the filed value be recovered by inverting the benefit formula, and does the calculator reproduce the issued benefit on the rebuilt application?
# - **Expected:** calculator reproduces the issued benefit within $1; report how many are dropped.
# - **Method:** invert for the single erroneous variable; count cases with no inverse or no unique inverse (benefit at max or min, shelter cap binding, net income at zero); rerun the calculator.
# - **Result:** 465 of the 917 (50.7%) have a unique filed value, and on those the calculator returns the issued benefit within $1 (462 exactly). The other 452: 81 where the element is not a dollar amount (mostly unit composition); 9 MFIP or SSI-CAP units the calculator does not cover; 72 where `FSBEN` differs from the QC-correct benefit by more than $1; 27 where no value reaches the issued benefit; 263 where a range of values gives the issued benefit (238 issued benefit at the maximum, 5 at the minimum, 14 shelter cap binding, 6 shelter cost below half of income). In 173 of the 238 the amount in error is income and $0 is one of the answers; 151 of those are coded as an unreported source of income. Part 3d builds the human arm decided on 2026-11-14: the 465 plus those 151 with a filed value of $0, 616 households. 21 of the 151 have two or more verified income sources in the group in error, so for them "$0 for the whole group" is an assumption.
# - **Mismatch:** the calculator reproduces the issued benefit within $1 for the 465 only. 452 of the 917 (49.3%) have no identified filed value under this method. The 151 with two or more verified sources: 21, expected 22.
# - **Feeds:** `rq.md` Applications.
# - **Open questions:** how to treat the 21 of the 151 with two or more verified sources (keep at $0 for the whole group, file the largest source only, or drop).

# %% [markdown]
# ### 3a. Invert the benefit formula
#
# What the file holds, per the technical documentation: the income and expense amounts are the values the QC reviewer verified; `RAWBEN` is the benefit the household was issued; `BENFIX` is the issued benefit corrected by the error amount (`RAWBEN` minus `AMTERR` for an overissuance, plus for an underissuance); `FSBEN` is the benefit recomputed from the verified amounts.
#
# The filed value is the value of the one erroneous amount that makes the calculator return the issued benefit when every other amount is left at its verified value. The benefit only moves one way as any single amount rises, so the answers form one unbroken range. The calculator is run on every whole-dollar value from $0 to $30,000 and the range is read off. A standard root-finder (`scipy.optimize.brentq`) would return one answer and hide the width of the range, which is what part b needs.
#
# The calculator is `snapqc.calc.benefit`. First check: does it return `FSBEN` from the verified amounts?

# %%
from snapqc.calc import benefit, inputs

calc_ok = pd.Series(np.abs(benefit(**inputs(df)) - df["FSBEN"]) <= 1, index=df.index)
special_unit = (df["MN_FIP"] == 1) | (df["SSI_CAP"] > 0)  # units whose benefit uses a state formula (MFIP, SSI-CAP)
in_41 = df["STATENAME"].isin(SINGLE_SUA_STATES)
calculator_check = pd.DataFrame({
    "households": [len(df), in_41.sum(), (in_41 & ~special_unit).sum(), len(h917)],
    "calculator within $1 of FSBEN": [calc_ok.sum(), calc_ok[in_41].sum(), calc_ok[in_41 & ~special_unit].sum(), calc_ok[h917.index].sum()],
}, index=["all households", "41 states", "41 states, no MFIP or SSI-CAP units", "the 917"])
calculator_check["share"] = (calculator_check.iloc[:, 1] / calculator_check.iloc[:, 0]).round(4)
calculator_check.to_csv(OUT / "3a-calculator-check.csv", index_label="group")
calculator_check

# %%
# Which application amount each ELEMENT code changes. Codes that are not a dollar amount on the application
# (111 student status, 150 unit composition, 151 disqualification, 160 work programs) have no entry.
ELEMENT_AMOUNT = {
    **dict.fromkeys([311, 312, 314], "earned"),
    **dict.fromkeys([331, 332, 333, 334, 335, 336, 342, 343, 344, 345, 346, 350], "unearned"),
    363: "rent", 364: "util", 323: "depcare", 365: "medical", 366: "child_support",
}
GRID = np.arange(0, 30_001)  # trial dollar values


def solution_range(household, amount, target):
    """Distance to the target benefit at the best trial value, and the lowest and highest value that reach it."""
    gap = np.abs(benefit(**(household.to_dict() | {amount: GRID})) - target)
    best = GRID[gap == gap.min()]
    return gap.min(), best.min(), best.max()


verified = inputs(h917)
rebuild = pd.DataFrame({
    "state": h917["STATENAME"],
    "element": h917["ELEMENT1"].astype(int),
    "amount": h917["ELEMENT1"].map(ELEMENT_AMOUNT),
    "status": h917["STATUS"],
    "issued_benefit": h917["RAWBEN"],
    "qc_benefit": h917["BENFIX"],
})
has_amount = rebuild["amount"].notna()
rebuild.loc[has_amount, "verified_value"] = [verified.at[i, a] for i, a in rebuild.loc[has_amount, "amount"].items()]
rebuild.loc[has_amount, ["gap", "low", "high"]] = [
    solution_range(verified.loc[i], a, rebuild.at[i, "issued_benefit"]) for i, a in rebuild.loc[has_amount, "amount"].items()
]
rebuild["width"] = rebuild["high"] - rebuild["low"]
rebuild["width"].describe().round(0).to_frame().T

# %% [markdown]
# ### 3b. Cases with no inverse or no unique inverse
#
# Each household gets the first outcome that applies, in the order listed. Rounding alone makes the range up to $4 wide (30% of net income is rounded up to a whole dollar), so a range of $5 or less counts as unique.

# %%
UNIQUE_WIDTH = 5
wide = rebuild["width"] > UNIQUE_WIDTH
shelter = rebuild["amount"].isin(["rent", "util"])
outcomes = {
    "dropped: element is not a dollar amount": ~has_amount,
    "dropped: calculator misses FSBEN (MFIP or SSI-CAP unit)": ~calc_ok[h917.index],
    "dropped: FSBEN differs from the QC-correct benefit": (h917["FSBEN"] - h917["BENFIX"]).abs() > 1,
    "no inverse: no value reaches the issued benefit": rebuild["gap"] > 1,
    "not unique: issued benefit at the maximum (net income zero)": wide & (h917["RAWBEN"] >= h917["BENMAX"]),
    "not unique: issued benefit at the minimum": wide & (h917["FSUSIZE"] <= 2) & (h917["RAWBEN"] <= h917["MINIMUM_BEN"]),
    "not unique: shelter cap binding": wide & shelter & (rebuild["high"] == GRID.max()),
    "not unique: shelter cost below half of income": wide & shelter & (rebuild["low"] == 0),
    "not unique: other": wide,
}
rebuild["outcome"] = np.select(list(outcomes.values()), list(outcomes), default="unique inverse")

outcome_counts = rebuild["outcome"].value_counts().reindex([*outcomes, "unique inverse"], fill_value=0).rename("households").to_frame()
outcome_counts["share of 917"] = (outcome_counts["households"] / len(rebuild)).round(3)
outcome_counts.to_csv(OUT / "3b-rebuild-outcomes.csv", index_label="outcome")
outcome_counts

# %%
# The same outcomes by the amount in error.
pd.crosstab(rebuild["outcome"].str.split(":").str[0], rebuild["amount"].fillna("(not a dollar amount)"), margins=True)

# %%
# Not-unique cases at the maximum benefit where $0 is one of the answers (for example an income source left off the application).
at_max = rebuild["outcome"].str.startswith("not unique: issued benefit at the maximum")
pd.Series({
    "not unique, issued benefit at the maximum": int(at_max.sum()),
    "of which the amount in error is income": int((at_max & rebuild["amount"].isin(["earned", "unearned"])).sum()),
    "of which $0 income is an answer": int((at_max & rebuild["amount"].isin(["earned", "unearned"]) & (rebuild["low"] == 0)).sum()),
    "of which coded unreported source of income (NATURE 35)": int((at_max & rebuild["amount"].isin(["earned", "unearned"]) & (rebuild["low"] == 0) & (h917["NATURE1"] == 35)).sum()),
}).to_frame("households")

# %% [markdown]
# ### 3c. Calculator on the rebuilt application
#
# For households with a unique inverse, the filed value is the middle of the range. The calculator is rerun on the rebuilt application (verified amounts, with the filed value in place of the amount in error).

# %%
unique = rebuild["outcome"] == "unique inverse"
rebuild.loc[unique, "filed_value"] = (rebuild["low"] + rebuild["high"]) // 2
rebuilt_application = verified[unique].copy()
for i, a in rebuild.loc[unique, "amount"].items():
    rebuilt_application.at[i, a] = rebuild.at[i, "filed_value"]
rebuild.loc[unique, "rebuilt_benefit"] = benefit(**rebuilt_application)
rebuild.to_csv(OUT / "3-rebuild.csv", index_label="row")

kept = rebuild[unique]
error = kept["rebuilt_benefit"] - benefit(**verified[unique])  # benefit error the rebuilt application implies
signed_amterr = h917.loc[unique, "AMTERR"] * np.where(kept["status"] == 2, 1, -1)
pd.Series({
    "households with a unique inverse": len(kept),
    "the other households": len(rebuild) - len(kept),
    "rebuilt benefit equals issued benefit": int((kept["rebuilt_benefit"] == kept["issued_benefit"]).sum()),
    "rebuilt benefit within $1 of issued benefit": int(((kept["rebuilt_benefit"] - kept["issued_benefit"]).abs() <= 1).sum()),
    "implied error within $1 of AMTERR, same direction": int(((error - signed_amterr).abs() <= 1).sum()),
    f"implied error above ${TOLERANCE}": int((error.abs() > TOLERANCE).sum()),
    "variance amount (AMOUNT1) equals AMTERR": int((h917.loc[unique, "AMOUNT1"] == h917.loc[unique, "AMTERR"]).sum()),
}).to_frame("households")

# %%
# How far the filed value sits from the verified value, by amount (filed minus verified, dollars per month).
(kept["filed_value"] - kept["verified_value"]).groupby([kept["amount"], kept["status"].map({2: "overissuance", 3: "underissuance"})]).describe()[["count", "25%", "50%", "75%"]].round(0)

# %% [markdown]
# ### 3d. The human arm: 616 households
#
# Decision (2026-11-14): the human arm is the 465 households with a unique filed value plus the 151 households at the maximum benefit whose amount in error is income, where $0 is one of the values that reach the issued benefit and the variance is coded as an unreported source of income (NATURE1 = 35). For the 151 the filed value is $0.
#
# One row per household. Fact group: earned, unearned, shelter (ELEMENT 363), utilities (ELEMENT 364), or outside (dependent care, medical, child support paid). Error type: "empty" if the filed value for the group is $2 or less, else "wrong number". Sources: the number of verified person-level income entries (person x income type with an amount above $0, any household member) in the group in error; earned types are the four that make up `FSEARN`, unearned types the 30 that make up `FSUNEARN` (both sums are checked below). Sections 4 to 9 read the arm from this table.

# %%
FACT_GROUP = {"earned": "earned", "unearned": "unearned", "rent": "shelter", "util": "utilities",
              "depcare": "outside", "medical": "outside", "child_support": "outside"}
FOUR_GROUPS = ["earned", "unearned", "shelter", "utilities"]
GROUP_COLUMN = {"earned": "FSEARN", "unearned": "FSUNEARN", "shelter": "RENT", "utilities": "UTIL"}  # verified group totals
EMPTY_AT_MOST = 2  # dollars: a filed value of $2 or less counts as an empty entry

# Person-level income types (persons 1 to 18) that make up FSEARN and FSUNEARN.
EARNED_TYPES = ["WAGES", "SLFEMP", "OTHERN", "RENTMINC"]
UNEARNED_TYPES = ["SOCSEC", "VET", "SSI", "UNEMP", "WCOMP", "OTHGOV", "CONT", "DEEM", "GA", "EDLOAN", "OTHUN", "TANF",
                  "CSUPRT", "ENERGY", "WGESUP", "DIVER", "FOSTER", "INTER", "ANNUITY", "PENSION", "OLDAGE", "SURV",
                  "STRIKE", "RENTINC", "ALIMNY", "GOVROY", "GOVDIV", "GOVINTER", "TRUST", "UNK"]


def person_income(frame, types):
    """(households x persons) array per income type, missing as 0."""
    return {t: frame[[f"{t}{i}" for i in range(1, 19)]].fillna(0).to_numpy() for t in types}


earned_p, unearned_p = person_income(h917, EARNED_TYPES), person_income(h917, UNEARNED_TYPES)
pd.Series({  # the person-level amounts add up to the household totals
    "person-level earned sums to FSEARN (share of 917)": np.isclose(sum(v.sum(1) for v in earned_p.values()), h917["FSEARN"].fillna(0), atol=1).mean(),
    "person-level unearned sums to FSUNEARN (share of 917)": np.isclose(sum(v.sum(1) for v in unearned_p.values()), h917["FSUNEARN"].fillna(0), atol=1).mean(),
}).to_frame("value")

# %%
in_151 = at_max & rebuild["amount"].isin(["earned", "unearned"]) & (rebuild["low"] == 0) & (h917["NATURE1"] == 35)
in_arm = unique | in_151
n_sources = pd.Series({
    "earned": sum((v > 0).sum(1) for v in earned_p.values()),
    "unearned": sum((v > 0).sum(1) for v in unearned_p.values()),
}).apply(lambda a: pd.Series(a, index=h917.index))  # group x household

a = rebuild[in_arm]
src = h917.loc[in_arm]
human_arm = pd.DataFrame({
    "state": a["state"],
    "group": a["amount"].map(FACT_GROUP),
    "element": a["element"],
    "amount": a["amount"],
    "filed_value": np.where(in_151[in_arm], 0, a["filed_value"]),
    "verified_value": a["verified_value"],
    "in_151": in_151[in_arm],
    "NATURE1": src["NATURE1"].astype(int),
    "DISCOV1": src["DISCOV1"].astype(int),
    "VERIF1": src["VERIF1"].astype(int),
    "ACTNTYPE": src["ACTNTYPE"],
    "STATUS": src["STATUS"],
    "AMTERR": src["AMTERR"],
    "FYWGT": src["FYWGT"],
})
human_arm["error_type"] = np.where(human_arm["filed_value"] <= EMPTY_AT_MOST, "empty", "wrong number")
human_arm["n_sources"] = [n_sources.at[g, i] if g in ("earned", "unearned") else np.nan for i, g in human_arm["group"].items()]
human_arm["n_sources"] = human_arm["n_sources"].astype("Int64")
human_arm.to_csv(OUT / "3d-human-arm.csv", index_label="row")

pd.Series({
    "unique filed value (the 465)": int(unique.sum()),
    "at maximum, income, $0 an answer, NATURE 35 (the 151)": int(in_151.sum()),
    "human arm": len(human_arm),
}).to_frame("households")

# %%
# The 151 by number of verified income sources in the group in error. With two or more sources,
# "$0 for the whole group" assumes every source was left off, not just one.
s151 = human_arm.loc[human_arm["in_151"], "n_sources"]
pd.Series({
    "151 with one source": int((s151 == 1).sum()),
    "151 with two or more sources": int((s151 >= 2).sum()),
    "  expected": 22,
    "151 with no source (should be 0)": int((s151 == 0).sum()),
}).to_frame("households")

# %%
pd.crosstab(human_arm.loc[human_arm["in_151"], "n_sources"], human_arm.loc[human_arm["in_151"], "group"], margins=True)

# %% [markdown]
# ## 4. Fact groups
#
# - **Question:** how do the ELEMENT codes of the 616 map to the four fact groups (earned income, unearned income, shelter, utilities) or to "outside"? How many errors are an empty entry for the group, and how many a wrong number?
# - **Expected:** earned 211, shelter 191, unearned 131, utilities 68, outside 15 (dependent care 9, medical 4, child support paid 2). NATURE 35 in 277; group empty on the form in 188 (the 151 plus 37). Three cells: income empty 185, income wrong number 157, housing 259. 95 households coded unreported source whose filed group total is not empty.
# - **Method:** each ELEMENT code is mapped through the amount it changes in the calculator (3a): earned, unearned, shelter (363), utilities (364), outside (dependent care, medical, child support paid). Error type: "empty" if the filed value for the group is $2 or less (all of the 151 have $0), else "wrong number". Three cells: income empty, income wrong number, housing (shelter + utilities); outside stays its own row.
# - **Result:** all expected counts reproduced. Earned 211, shelter 191, unearned 131, utilities 68, outside 15 (dependent care 9, medical 4, child support paid 2). NATURE 35 in 277. Group empty on the form in 188 (the 151 plus 37); 6 of the 188 are not NATURE 35 (3 earned, 3 shelter). Three cells: income empty 185, income wrong number 157, housing 259 (outside 15). 95 households coded NATURE 35 have a filed group total above $2 (44 earned, 51 unearned): the unreported source sat next to other income in the same group. Only 3 of the 259 housing errors are empty entries. Utilities (68) and outside (15) are under 100 errors.
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Terms (fact group, checker); power for RQ1.
# - **Open questions:** merge shelter and utilities in tables; table by error type (empty vs wrong number).

# %% [markdown]
# ### 4a. ELEMENT code -> fact group map
#
# Labels are quoted from the ELEMENT code list of the FY2024 SNAP QC technical documentation (Chapter V codebook). Codes that are not a dollar amount on the application have no fact group and are not in the 616.

# %%
arm = pd.read_csv(OUT / "3d-human-arm.csv", index_col="row")  # the human arm, as saved in 3d
assert len(arm) == 616 and arm["in_151"].sum() == 151

ELEMENT_LABEL = {  # FY2024 SNAP QC technical documentation, ELEMENT codes
    111: "Student status", 150: "Unit composition", 151: "Recipient disqualification",
    160: "Employment and training programs", 311: "Wages and salaries", 312: "Self-employment",
    314: "Other earned income", 323: "Dependent care deduction", 331: "RSDI benefits",
    332: "(label not found in the code list)", 333: "SSI and/or State SSI supplement",
    334: "Unemployment compensation", 335: "(label not found in the code list)", 336: "Other government benefits",
    342: "Contributions", 343: "Deemed income", 344: "TANF, PA, or GA", 346: "Other unearned income",
    350: "Child support payments received from absent parent", 363: "Shelter deduction",
    364: "Standard utility allowance", 365: "Medical expense deductions", 366: "Child support payment deduction",
}
element_map = rebuild["element"].value_counts().sort_index().rename("in the 917").to_frame()
element_map.insert(0, "label", element_map.index.map(ELEMENT_LABEL))
element_map.insert(1, "fact group", element_map.index.map(lambda e: FACT_GROUP.get(ELEMENT_AMOUNT.get(e), "none (not a dollar amount)")))
element_map["in the 616"] = arm["element"].value_counts().reindex(element_map.index, fill_value=0)
element_map.to_csv(OUT / "4a-element-map.csv", index_label="ELEMENT")
element_map

# %% [markdown]
# ### 4b. Errors per fact group

# %%
GROUP_ORDER = ["earned", "shelter", "unearned", "utilities", "outside"]
group_counts = pd.DataFrame({
    "the 616": arm["group"].value_counts().reindex(GROUP_ORDER),
    "of which the 151": arm.loc[arm["in_151"], "group"].value_counts().reindex(GROUP_ORDER, fill_value=0),
    "expected": pd.Series({"earned": 211, "shelter": 191, "unearned": 131, "utilities": 68, "outside": 15}),
})
outside = arm.loc[arm["group"] == "outside", "amount"].value_counts()
outside_rows = pd.DataFrame({
    "the 616": outside.reindex(["depcare", "medical", "child_support"]).to_numpy(),
    "of which the 151": 0,
    "expected": [9, 4, 2],
}, index=["outside: dependent care", "outside: medical", "outside: child support paid"])
group_counts = pd.concat([group_counts, outside_rows])
group_counts["match"] = group_counts["the 616"] == group_counts["expected"]
group_counts.to_csv(OUT / "4b-group-counts.csv", index_label="fact group")
group_counts

# %% [markdown]
# ### 4c. Empty entry versus wrong number
#
# NATURE 35 is "Unreported source of income (do not use for change in employment status)" (FY2024 SNAP QC technical documentation, NATURE codes).

# %%
nature35 = arm["NATURE1"] == 35
empty = arm["error_type"] == "empty"
pd.DataFrame({"households": {
    "coded unreported source of income (NATURE 35)": int(nature35.sum()),
    "group empty on the form (filed value $2 or less)": int(empty.sum()),
    "  of which the 151": int((empty & arm["in_151"]).sum()),
    "  of which among the 465": int((empty & ~arm["in_151"]).sum()),
    "NATURE 35 and group not empty on the form": int((nature35 & ~empty).sum()),
    "group empty on the form and not NATURE 35": int((empty & ~nature35).sum()),
}, "expected": {
    "coded unreported source of income (NATURE 35)": 277,
    "group empty on the form (filed value $2 or less)": 188,
    "  of which the 151": 151, "  of which among the 465": 37,
    "NATURE 35 and group not empty on the form": 95,
}}).astype("Int64")

# %%
error_by_group = pd.crosstab([arm["group"], arm["error_type"]], nature35.map({True: "NATURE 35", False: "other NATURE"}).rename("NATURE1"), margins=True)
error_by_group.to_csv(OUT / "4c-error-type-by-group.csv")
error_by_group

# %%
INCOME = ["earned", "unearned"]
HOUSING = ["shelter", "utilities"]
arm["cell"] = np.select(
    [arm["group"].isin(INCOME) & empty, arm["group"].isin(INCOME), arm["group"].isin(HOUSING)],
    ["income, empty", "income, wrong number", "housing (shelter + utilities)"], default="outside")
CELLS = ["income, empty", "income, wrong number", "housing (shelter + utilities)", "outside"]
three_cells = pd.DataFrame({
    "households": arm["cell"].value_counts().reindex(CELLS),
    "of which the 151": arm.loc[arm["in_151"], "cell"].value_counts().reindex(CELLS, fill_value=0),
    "NATURE 35": arm.loc[nature35, "cell"].value_counts().reindex(CELLS, fill_value=0),
    "expected": pd.Series({"income, empty": 185, "income, wrong number": 157, "housing (shelter + utilities)": 259}),
}).astype("Int64")
three_cells.to_csv(OUT / "4c-three-cells.csv", index_label="cell")
three_cells

# %% [markdown]
# ## 5. How each error was found
#
# - **Question:** how do DISCOV1 and VERIF1 vary by fact group in the 616, and with what frequencies would comparison households copy a discovery source?
# - **Expected:** 38 of the 616 with DISCOV 1 or 2 (65 of the 917).
# - **Method:** crosstabs of DISCOV1 and VERIF1 by fact group, code meanings quoted from the FY2024 SNAP QC technical documentation (Chapter V codebook); per-group shares of DISCOV1 in 5b.
# - **Result:** 38 of the 616 have DISCOV 1 or 2 (case record), 13 of them among the 151; 65 of the 917. The recipient interview is the most common discovery source in every group (387 of 616, 63%). The employer discovers 55 of 211 earned errors (26%); the landlord 46 of 191 shelter errors (24%) and 11 of 68 utility errors (16%). Verification differs from discovery: the employer verifies 123 of 211 earned errors, the landlord 99 of 191 shelter errors; DISCOV equals VERIF in 339 of 616.
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Packets (which document the faithful packet leaves out).

# %% [markdown]
# ### 5a. DISCOV1 and VERIF1 by fact group
#
# Code meanings are quoted from the FY2024 SNAP QC technical documentation (Chapter V codebook). DISCOV: "how the variance was discovered"; VERIF: "how the variance was verified".

# %%
DISCOV_LABEL = {
    1: "Variance clearly identified from case record (not automated match)",
    2: "From case record (automated match)",
    3: "From recipient interview",
    4: "Employer (present or former)",
    5: "Financial institution, insurance company, or other business",
    6: "Landlord",
    7: "Government agency or public records, not automated match",
    8: "Government agency or public records, automated match",
    9: "Other",
}
VERIF_LABEL = {
    1: "From case record (not automated match)",
    2: "From case record (automated match)",
    3: "From information provided by recipient",
    4: "Employer (present or former)",
    5: "Financial institution, insurance company, or other business",
    6: "Landlord",
    7: "Government agency or public records, not automated match",
    8: "Government agency or public records, automated match",
    9: "Other",
}


def by_group(codes, labels):
    table = pd.crosstab(codes, arm["group"], margins=True, margins_name="all").reindex(columns=[*GROUP_ORDER, "all"])
    table.index = [f"{c} {labels[c]}" if c in labels else c for c in table.index]
    return table


discov_by_group = by_group(arm["DISCOV1"], DISCOV_LABEL)
discov_by_group.to_csv(OUT / "5a-discov-by-group.csv", index_label="DISCOV1")
discov_by_group

# %%
verif_by_group = by_group(arm["VERIF1"], VERIF_LABEL)
verif_by_group.to_csv(OUT / "5a-verif-by-group.csv", index_label="VERIF1")
verif_by_group

# %%
case_record = arm["DISCOV1"].isin([1, 2])
pd.DataFrame({"households": {
    "the 616: DISCOV 1 or 2 (case record)": int(case_record.sum()),
    "  of which the 151": int((case_record & arm["in_151"]).sum()),
    "the 917: DISCOV 1 or 2 (case record)": int(h917["DISCOV1"].isin([1, 2]).sum()),
    "the 616: VERIF 1 or 2 (case record)": int(arm["VERIF1"].isin([1, 2]).sum()),
    "the 616: DISCOV equals VERIF": int((arm["DISCOV1"] == arm["VERIF1"]).sum()),
}, "expected": {
    "the 616: DISCOV 1 or 2 (case record)": 38,
    "the 917: DISCOV 1 or 2 (case record)": 65,
}}).astype("Int64")

# %%
discov_by_verif = pd.crosstab(arm["DISCOV1"].rename("DISCOV1"), arm["VERIF1"].rename("VERIF1"), margins=True)
discov_by_verif.to_csv(OUT / "5a-discov-by-verif.csv")
discov_by_verif

# %% [markdown]
# ### 5b. Frequencies per fact group for comparison households
#
# Share of each DISCOV1 code within each fact group of the 616 (columns sum to 1). A comparison household given a planted or background error in a fact group would draw its discovery source from that group's column.

# %%
discov_freq = pd.crosstab(arm["DISCOV1"], arm["group"], normalize="columns").reindex(columns=GROUP_ORDER).round(3)
discov_freq["all 616"] = arm["DISCOV1"].value_counts(normalize=True).sort_index().round(3)
discov_freq.insert(0, "label", discov_freq.index.map(DISCOV_LABEL))
discov_freq.loc["households (n)"] = ["", *arm["group"].value_counts().reindex(GROUP_ORDER), len(arm)]
discov_freq.to_csv(OUT / "5b-discov-frequencies.csv", index_label="DISCOV1")
discov_freq

# %% [markdown]
# ## 6. Non-zero fact groups per household
#
# - **Question:** besides the group in error, how many of the four fact groups are non-zero? Does a faithful packet still have a document left to read? Computed for the 616 and, separately, for the comparison pool (all other households in the 41 states).
# - **Expected:** in the 616, 79 households with no other non-zero group (69 of the 151, 10 of the 465).
# - **Method:** a fact group is non-zero when its verified total (`FSEARN`, `FSUNEARN`, `RENT`, `UTIL`) is above $0. For the 616 the group in error is left out (for the 15 outside errors all four groups count). The comparison pool (35,543 households in the 41 states outside the 616) has no group in error, so all four count.
# - **Result:** 79 of the 616 have no other non-zero group (69 of the 151, 10 of the 465): 52 earned, 27 unearned errors. 460 (74.7%) have two or more other groups non-zero. Comparison pool: 5,358 (15.1%) have no non-zero group, 3,636 (10.2%) one, 26,549 (74.7%) two or more.
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Packets.
# - **Open questions:** keep or drop the households with nothing else on the form.

# %% [markdown]
# ### 6a. Count per household

# %%
nonzero = df[list(GROUP_COLUMN.values())].fillna(0).gt(0)
nonzero.columns = FOUR_GROUPS
own_group = pd.DataFrame({g: arm["group"] == g for g in FOUR_GROUPS})  # outside errors have no own group among the four
arm["other_nonzero"] = (nonzero.loc[arm.index] & ~own_group).sum(axis=1)

pool = in_41 & ~df.index.isin(arm.index)  # comparison pool: every other household in the 41 states
pool_nonzero = nonzero[pool].sum(axis=1)

nonzero_counts = pd.DataFrame({
    "the 616: other groups non-zero": arm["other_nonzero"].value_counts(),
    "  the 151": arm.loc[arm["in_151"], "other_nonzero"].value_counts(),
    "  the 465": arm.loc[~arm["in_151"], "other_nonzero"].value_counts(),
    "comparison pool: groups non-zero (of four)": pool_nonzero.value_counts(),
}).reindex(range(5)).fillna(0).astype(int)
nonzero_counts.loc["households"] = nonzero_counts.sum()
nonzero_counts.to_csv(OUT / "6a-nonzero-groups.csv", index_label="non-zero groups")
nonzero_counts

# %% [markdown]
# ### 6b. By fact group, and how often each group is non-zero

# %%
nonzero_by_group = pd.crosstab(arm["group"], arm["other_nonzero"], margins=True).reindex([*GROUP_ORDER, "All"])
nonzero_by_group.to_csv(OUT / "6b-other-nonzero-by-group.csv")
nonzero_by_group

# %%
group_shares = pd.DataFrame({
    "the 616 (verified values)": nonzero.loc[arm.index].mean(),
    "comparison pool": nonzero[pool].mean(),
}).round(3)
group_shares.loc["households"] = [len(arm), int(pool.sum())]
group_shares.to_csv(OUT / "6b-group-nonzero-shares.csv", index_label="fact group")
group_shares

# %% [markdown]
# ## 7. Background errors outside the 616
#
# - **Question:** among the 41-state households outside the 616, what share has an error above $56, by cause? What share of weighted error dollars do the 616 carry? How many comparison households give 150 or more background errors?
# - **Expected:** the 616 carry 11.9% of weighted error dollars (the 917: 17.5%).
# - **Method:** pool = 41-state households outside the 616 (35,543). Error = STATUS 2 or 3 and AMTERR above $56. Cause from the AGENCY codes of all variances, grouped as in 7a; the 301 are their own row. Weighted error dollars = FYWGT x AMTERR over errors above $56 in all states. Comparison households needed = 150 / share, rounded up.
# - **Result:** 3,608 of the 35,543 (10.2%; 11.6% weighted) have an error above $56: agency 1,951 (5.5%), client outside the 917 700 (2.0%), mixed causes 597 (1.7%), the 301 (0.8%), other (99) 39, not required to be reported (26) 10, no variance detail 10. The 616 carry 11.9% of weighted error dollars (the 917: 17.5%; the 151: 3.8%; the 465: 8.1%). For 150 background errors: 1,478 comparison households from the whole pool; 1,599 without the 301; 2,028 if errors caused by the client alone do not count either.
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Households.

# %% [markdown]
# ### 7a. Error share by cause
#
# Cause categories group the AGENCY codes of the FY2024 SNAP QC technical documentation: client (1 to 4, 7: information not reported, incomplete or incorrect, withheld, incorrect, or inaccurate from a collateral contact); agency (8 and 10 to 25, from "Acted on incorrect federal computer match information" to failed recertification procedures); 26 "Change was not required to be reported by the client or acted upon by the State agency based on time frames and reporting requirements"; 99 "Other". A household with variances in more than one category is "mixed". The 301 client-error households of the 917 that are not in the 616 are their own row.

# %%
CAUSE = {**dict.fromkeys([1, 2, 3, 4, 7], "client"), **dict.fromkeys([8, *range(10, 26)], "agency"),
         26: "not required to be reported (26)", 99: "other (99)"}
cause_sets = df[AGENCY].apply(lambda r: frozenset(CAUSE[int(c)] for c in r.dropna()), axis=1)
cause = cause_sets.map(lambda s: "no variance detail" if not s else (next(iter(s)) if len(s) == 1 else "mixed"))
not_rebuilt = df.index.isin(h917.index) & ~df.index.isin(arm.index)
cause = cause.where(~not_rebuilt, "client, in the 917 but not in the 616 (the 301)")
cause = cause.where(~((cause == "client") & ~df.index.isin(h917.index)), "client, outside the 917")

big_error = df["STATUS"].isin([2, 3]) & (df["AMTERR"] > TOLERANCE)
CAUSE_ORDER = ["client, in the 917 but not in the 616 (the 301)", "client, outside the 917", "agency",
               "not required to be reported (26)", "other (99)", "mixed", "no variance detail"]
fy = df["FYWGT"]
background = pd.DataFrame({
    "households with error above $56": cause[pool & big_error].value_counts().reindex(CAUSE_ORDER, fill_value=0),
})
background.loc["any cause"] = int((pool & big_error).sum())
background["share of pool"] = (background.iloc[:, 0] / pool.sum()).round(4)
background["weighted share of pool (FYWGT)"] = pd.concat([
    fy[pool & big_error].groupby(cause[pool & big_error]).sum().reindex(CAUSE_ORDER, fill_value=0),
    pd.Series({"any cause": fy[pool & big_error].sum()}),
]).div(fy[pool].sum()).round(4)
background.to_csv(OUT / "7a-background-errors-by-cause.csv", index_label="cause")
print("comparison pool (41 states, not in the 616):", int(pool.sum()), "households")
background

# %%
dollars = (fy * df["AMTERR"]).where(big_error, 0)  # weighted error dollars, errors above $56, all states
pd.DataFrame({"share of weighted error dollars": {
    "the 616": dollars[arm.index].sum() / dollars.sum(),
    "the 917": dollars[h917.index].sum() / dollars.sum(),
    "the 151": dollars[arm.index[arm["in_151"]]].sum() / dollars.sum(),
    "the 465": dollars[arm.index[~arm["in_151"]]].sum() / dollars.sum(),
}, "expected": {"the 616": 0.119, "the 917": 0.175}}).round(3)

# %% [markdown]
# ### 7b. Comparison sample size for 150 background errors
#
# Comparison households needed = 150 / (share of the pool with an error above $56), rounded up. Two pools are shown because whether the 301 may be drawn as comparison households is open.

# %%
def needed(households, errors):
    p = (households & errors).sum() / households.sum()
    return pd.Series({"households in pool": int(households.sum()), "with error above $56": int((households & errors).sum()),
                      "share": round(p, 4), "needed for 150 background errors": int(np.ceil(150 / p))})


comparison_size = pd.DataFrame({
    "all 41-state households outside the 616": needed(pool, big_error),
    "the same, without the 301": needed(pool & ~not_rebuilt, big_error),
    "without the 301, errors not caused by the client only": needed(pool & ~not_rebuilt, big_error & ~cause.str.startswith("client")),
}).T.astype({"households in pool": int, "with error above $56": int, "needed for 150 background errors": int})
comparison_size.to_csv(OUT / "7b-comparison-size.csv", index_label="pool")
comparison_size

# %% [markdown]
# ## 8. Certification vs recertification
#
# - **Question:** does the fact-group mix of the 616 differ between certifications (ACTNTYPE 1) and recertifications (ACTNTYPE 2)?
# - **Expected:** 440 of the 616 recertifications. Largest gap about 8.5 points with four groups, about 15 with shelter and utilities merged.
# - **Method:** crosstab of fact group by ACTNTYPE; shares within each action type over all five groups; gap = recertification minus certification, in points.
# - **Result:** 440 of the 616 are recertifications (71.4%). Largest gap with four groups: earned, -8.5 points (unearned -8.4, shelter +7.6, utilities +7.5). With shelter and utilities merged: housing, +15.1 points. Empty entries are 38.6% of certification errors and 27.3% of recertification errors (11.4 points).
# - **Mismatch:** none.
# - **Feeds:** `rq.md` limitations (share of recertifications).
# - **Open questions:** stratify by recertification.

# %% [markdown]
# ### 8a. Split counts

# %%
action = arm["ACTNTYPE"].map({1: "certification", 2: "recertification"})
action_counts = pd.crosstab(arm["group"], action, margins=True).reindex([*GROUP_ORDER, "All"])
action_counts.to_csv(OUT / "8a-group-by-action.csv")
action_counts

# %% [markdown]
# ### 8b. Mix difference
#
# Share of each fact group within certifications and within recertifications (each column over all five groups), and the gap in points (recertification minus certification).

# %%
def mix_gap(groups):
    mix = pd.crosstab(groups, action, normalize="columns") * 100
    mix["gap (points)"] = mix["recertification"] - mix["certification"]
    return mix.round(1)


mix4 = mix_gap(arm["group"]).reindex(GROUP_ORDER)
mix3 = mix_gap(arm["group"].replace({"shelter": "housing", "utilities": "housing"})).reindex(["earned", "unearned", "housing", "outside"])
mix_cells = mix_gap(arm["cell"]).reindex(CELLS)
mix_type = mix_gap(arm["error_type"])
mixes = pd.concat({"four groups": mix4, "shelter + utilities merged": mix3, "three cells": mix_cells, "error type": mix_type})
mixes.to_csv(OUT / "8b-mix-by-action.csv", index_label=["table", "row"])

pd.Series({
    "recertifications": int((arm["ACTNTYPE"] == 2).sum()),
    "largest gap, four groups (points)": mix4.loc[FOUR_GROUPS, "gap (points)"].abs().max(),
    "  in group": mix4.loc[FOUR_GROUPS, "gap (points)"].abs().idxmax(),
    "largest gap, shelter + utilities merged (points)": mix3.loc[["earned", "unearned", "housing"], "gap (points)"].abs().max(),
    "  in group": mix3.loc[["earned", "unearned", "housing"], "gap (points)"].abs().idxmax(),
}).to_frame("value")

# %%
mixes

# %% [markdown]
# ## 9. Interval width
#
# - **Question:** how wide is a 95% interval for a recall of 0.5 on the 616, overall, per fact group and per cell of the three-cell table, unweighted and weighted? How wide for a 10-point paired filer gap? What is the weighted effective sample size?
# - **Expected:** 15 states with under 10 cases (pooled); Kish effective sample size under FYWGT 257.
# - **Method:** simulated reader with recall 0.5; whole households resampled within state (2,000 resamples), the 15 states with under 10 cases pooled into one stratum; subgroups as domains; mean over 20 simulated outcome draws; weighted = FYWGT. Paired gap: recall 0.5 vs 0.6 on the same households, at within-household correlation 0 and 0.5 (an assumption).
# - **Result:** 41 states, 15 with under 10 cases (86 households) pooled, 27 strata. Kish effective sample size under FYWGT 257.4. Half-width for recall 0.5, unweighted / weighted, in points: all 616 3.8 / 6.0; earned 6.5 / 9.6; shelter 6.9 / 10.1; unearned 8.4 / 13.5; utilities 11.6 / 17.8; outside 25.1 / 29.5; income empty 7.0 / 10.5; income wrong number 7.6 / 12.6; housing 5.9 / 8.9. 10-point paired filer gap, all 616: 5.4 / 8.4 with independent outcomes, 3.8 / 5.6 at correlation 0.5. Stratifying by state barely changes the width (3.84 vs 3.93 points unweighted).
# - **Mismatch:** none.
# - **Feeds:** `rq.md` Statistics.

# %% [markdown]
# ### 9a. Strata and effective sample size

# %%
B, R = 2_000, 20  # bootstrap resamples; simulated outcome draws averaged over
MIN_STATE = 10
state_n = arm["state"].value_counts()
thin_states = state_n[state_n < MIN_STATE].index
stratum = arm["state"].where(~arm["state"].isin(thin_states), "pooled").to_numpy()
idx = np.concatenate([rng.choice(np.flatnonzero(stratum == s), size=(B, (stratum == s).sum())) for s in np.unique(stratum)], axis=1)
idx_simple = rng.choice(len(arm), size=(B, len(arm)))  # unstratified, for comparison in 9c
w_fy = arm["FYWGT"].to_numpy()


def kish(w):
    return w.sum() ** 2 / (w ** 2).sum()


pd.Series({
    "states in the 616": arm["state"].nunique(),
    f"states with under {MIN_STATE} cases (pooled into one stratum)": len(thin_states),
    "  expected states": 15,
    "households in the pooled stratum": int(arm["state"].isin(thin_states).sum()),
    "strata": len(np.unique(stratum)),
    "Kish effective sample size, FYWGT": round(kish(w_fy), 1),
    "  expected effective sample size": 257,
}).to_frame("value")

# %% [markdown]
# ### 9b. Half-width for recall 0.5
#
# A reader with recall 0.5 is simulated: each household is caught with probability 0.5. Whole households are resampled within stratum (states with under 10 cases pooled); the half-width is half the distance between the 2.5th and 97.5th percentiles of the recall, in points. Subgroups are domains: the whole stratum is resampled and recall computed on the resampled members of the subgroup. Each figure is the mean over 20 simulated outcome draws. The last column is the simple binomial figure 1.96 x sqrt(0.25 / n), for reference.

# %%
def half_width(y, mask, w, index=idx):
    wm = w * mask
    est = (wm * y)[index].sum(1) / wm[index].sum(1)
    lo, hi = np.nanpercentile(est, [2.5, 97.5])
    return (hi - lo) / 2 * 100


subsets = {"all 616": np.ones(len(arm), bool)}
subsets |= {f"group: {g}": (arm["group"] == g).to_numpy() for g in GROUP_ORDER}
subsets |= {f"cell: {c}": (arm["cell"] == c).to_numpy() for c in CELLS[:3]}
ones = np.ones(len(arm))
draws = rng.random((R, len(arm))) < 0.5
widths = pd.DataFrame({
    "n": {k: int(m.sum()) for k, m in subsets.items()},
    "Kish n (FYWGT)": {k: round(kish(w_fy[m]), 1) for k, m in subsets.items()},
    "half-width, unweighted (points)": {k: np.mean([half_width(y, m, ones) for y in draws]) for k, m in subsets.items()},
    "half-width, weighted (points)": {k: np.mean([half_width(y, m, w_fy) for y in draws]) for k, m in subsets.items()},
    "binomial 1.96 x sqrt(0.25/n) (points)": {k: 196 * np.sqrt(0.25 / m.sum()) for k, m in subsets.items()},
}).round(1)
widths.to_csv(OUT / "9b-recall-half-widths.csv", index_label="subset")
widths

# %% [markdown]
# ### 9c. Paired filer gap, and the effect of pooling
#
# Paired gap: each household is filed twice (human, AI) and read; recall 0.5 on one, 0.6 on the other. The width depends on how correlated the two outcomes are within a household, which nothing in the file tells; two values are shown (0 and 0.5).

# %%
def paired_draws(rho, p1=0.5, p2=0.6):
    both = p1 * p2 + rho * np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))  # P(caught on both)
    u = rng.random((R, len(arm)))
    y1 = u < p1
    y2 = np.where(y1, u < both, (u >= p1) & (u < p1 + p2 - both))
    return y2.astype(float) - y1.astype(float)


gap_rows = {}
for rho in [0.0, 0.5]:
    d = paired_draws(rho)
    for k in ["all 616", *[f"cell: {c}" for c in CELLS[:3]]]:
        m = subsets[k]
        gap_rows[(f"correlation {rho}", k)] = {
            "n": int(m.sum()),
            "mean simulated gap (points)": d[:, m].mean() * 100,
            "half-width, unweighted (points)": np.mean([half_width(y, m, ones) for y in d]),
            "half-width, weighted (points)": np.mean([half_width(y, m, w_fy) for y in d]),
        }
gap_widths = pd.DataFrame(gap_rows).T.round(1)
gap_widths.to_csv(OUT / "9c-paired-gap-half-widths.csv", index_label=["assumption", "subset"])
gap_widths

# %%
# Pooling choice: overall half-width with states stratified (thin states pooled) vs no strata at all.
pd.DataFrame({
    "stratified, thin states pooled": [np.mean([half_width(y, subsets["all 616"], w) for y in draws]) for w in (ones, w_fy)],
    "no strata (simple household bootstrap)": [np.mean([half_width(y, subsets["all 616"], w, idx_simple) for y in draws]) for w in (ones, w_fy)],
}, index=["unweighted", "weighted"]).round(2)

# %% [markdown]
# ## Summary
#
# | § | Headline number | Expected | Match |
# |---|---|---|---|
# | 1 | Funnel 44,891 -> 2,096 -> 1,098 -> 917; 41 error cases have no variance detail; 12,403 STATUS 2/3 cases are at or below $56 | 44,891 -> 2,096 -> 1,098 -> 917 | match |
# | 2 | `HWGT`/`FYWGT` reproduce the August 2026 documented ranges and totals; `_OLD` columns do not | identical to corrected weights | match |
# | 3 | 465 of 917 (50.7%) have a unique filed value; calculator within $1 of the issued benefit on all 465 | within $1; report number dropped | 452 not identified |
# | 3d | Human arm 616 = 465 + 151; 21 of the 151 have two or more verified income sources in the group | 22 | mismatch (21) |
# | 4 | Earned 211, shelter 191, unearned 131, utilities 68, outside 15; NATURE 35 277; empty 188; cells 185 / 157 / 259; 95 NATURE 35 not empty | same | match |
# | 5 | DISCOV 1 or 2: 38 of 616 (65 of 917); recipient interview 63% | 38; 65 | match |
# | 6 | 79 of 616 with no other non-zero group (69 of the 151, 10 of the 465); pool: 15.1% with none | 79 (69, 10) | match |
# | 7 | 616 carry 11.9% of weighted error dollars (917: 17.5%); pool error share 10.2%; 1,478 comparison households for 150 background errors | 11.9%; 17.5% | match |
# | 8 | 440 of 616 recertifications; largest gap 8.5 points (four groups), 15.1 (housing merged) | 440; ~8.5; ~15 | match |
# | 9 | Kish n 257; 15 thin states pooled; recall 0.5 half-width 3.8 points unweighted, 6.0 weighted; paired 10-point gap 5.4 / 8.4 (independent) | 257; 15 | match |
