"""The 19 inputs of the BGL risk model, built from an application through one function.

`bgl_features(application, tables)` is the only place the inputs are computed, for every filer. It reads the
application table and the published program tables, nothing else, so a value that is not on the application
cannot reach the risk score.

It repeats the last steps of BGL's data preparation (github.com/giannella/snap_qc,
`1_data_munging_and_raw_variable_reconstruction_for_using_public_qc_data.R` and `features.R`): amounts are
moved to FY2026 dollars, the deductions are recomputed on the moved amounts, and the two benefit ratios stay
in the review year's dollars. BGL's own reconstruction of pre-review amounts (its steps 6 and 7) is not
repeated: the application already holds the amounts as filed.

An application is one row per household with the columns in APPLICATION_FIELDS, in the review year's dollars.
`application_from_qc` fills it from the QC file (amounts as QC verified them); `file_amounts` then replaces the
amounts of one fact group with what was filed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .review_sim import FEATURES

TARGET_YEAR = 2026  # BGL's modeling_target_year: the dollars the model is trained in
SUA_TOLERANCE = 10  # dollars: a utility amount this close to the state's standard counts as the standard

# Income types and deductions BGL counts round amounts over (count_divisible_by_100), as QC file columns.
EARNED_TYPES = ["FSWAGES", "FSSLFEMP", "FSOTHERN"]
UNEARNED_TYPES = ["FSSSI", "FSTANF", "FSGA", "FSSOCSEC", "FSUNEMP", "FSVET", "FSWCOMP", "FSEDLOAN", "FSCSUPRT",
                  "FSDEEM", "FSCONT", "FSOTHGOV", "FSOTHUN", "FSDIVER", "FSWGESUP", "FSENERGY", "FSEITC",
                  "FSFOSTER"]
INCOME_TYPES = EARNED_TYPES + UNEARNED_TYPES
# ELEMENT code -> the income type it names.
ELEMENT_TYPE = {311: "FSWAGES", 312: "FSSLFEMP", 314: "FSOTHERN", 331: "FSSOCSEC", 332: "FSVET", 333: "FSSSI",
                334: "FSUNEMP", 335: "FSWCOMP", 336: "FSOTHGOV", 342: "FSCONT", 343: "FSDEEM", 344: "FSTANF",
                345: "FSEDLOAN", 346: "FSOTHUN", 350: "FSCSUPRT"}

# Application field -> (QC file source, where an agency holds it at the time of the decision).
APPLICATION_FIELDS = {
    "state": ("STATENAME", "case record"),
    "fiscal_year": ("review year", "case record"),
    "size": ("FSUSIZE", "application: household members"),
    "children": ("FSNKID > 0", "application: household members"),
    "elderly_or_disabled": ("FSNELDER > 0 or FSNDIS > 0", "application: household members"),
    "married": ("any REL1..REL16 == 2 (spouse)", "application: household members"),
    "homeless": ("HOMEDED != 1", "application: living situation"),
    "earned": ("FSEARN", "application: earned income"),
    "unearned": ("FSUNEARN", "application: unearned income"),
    "rent": ("RENT", "application: shelter"),
    "util": ("UTIL", "application: utilities (the state allowance once the household qualifies)"),
    "medical": ("FSMEDDED", "application: medical costs, after the agency's $35 threshold"),
    "depcare": ("FSDEPDED", "application: dependent care"),
    "child_support": ("FSCSEXP when a child-support deduction or exclusion applies", "application: child support paid"),
    "homeless_ded": ("HOMELESS_DED", "agency (homeless households only): homeless shelter deduction granted"),
    **{t: (t, "application: income by type") for t in INCOME_TYPES},
    "n_abawd": ("count of ABWDST1..18 in 2..5", "agency: work-requirement status of each member"),
    "expedited": ("EXPEDSER < 3", "agency: expedited-service decision"),
    "months_since_cert": ("LASTCERT", "agency: date of the last certification"),
    "bbce_state": ("state share of CAT_ELIG >= 1 at least 0.5", "state policy"),
    "benmax": ("BENMAX", "program table: maximum benefit for the size"),
    "min_ben": ("MINIMUM_BEN, size 1 or 2 only", "program table: minimum benefit"),
}
AMOUNTS = ["earned", "unearned", "rent", "util", "medical", "depcare", "child_support"]
# Every QC file column `application_from_qc` reads. None is a total QC derives from the verified amounts
# (FSGRINC, FSNETINC, FSBEN, the earned-income, shelter and total deductions) or a review outcome (RAWBEN,
# AMTERR, STATUS, the variance fields).
QC_COLUMNS_READ = ["STATENAME", "FSUSIZE", "FSNKID", "FSNELDER", "FSNDIS", *[f"REL{i}" for i in range(1, 17)], "HOMEDED",
                   "FSEARN", "FSUNEARN", "RENT", "UTIL", "FSMEDDED", "FSDEPDED", "FSCSDED", "EXCL_FSCSDED", "FSCSEXP",
                   "HOMELESS_DED", *INCOME_TYPES, *[f"ABWDST{i}" for i in range(1, 19)], "EXPEDSER", "LASTCERT",
                   "CAT_ELIG", "BENMAX", "MINIMUM_BEN"]


def load_tables(snap_qc: Path) -> dict:
    """Program tables from BGL's `additional_data/`: CPI and caps by year, state utility and medical standards,
    standard deductions."""
    add = Path(snap_qc) / "additional_data"

    def wide(name):
        t = pd.read_csv(add / name, encoding="utf-8-sig").set_index("state_name")
        t.columns = t.columns.astype(int)
        return t.replace(0, np.nan)  # BGL reads 0 as "no standard"

    std = pd.read_csv(add / "standard_deductions.csv", encoding="utf-8-sig").set_index("year")
    return {
        "year": pd.read_csv(add / "year_data.csv", encoding="utf-8-sig").set_index("year"),
        "sua": wide("state_sua.csv"),
        "smd": wide("standard_medical_deductions.csv"),
        "std": std[[c for c in std.columns if c.lstrip("X").isdigit()]].rename(columns=lambda c: int(c.lstrip("X"))),
        "il_offset": std["IL_OFFSET"],
    }


def _standard_deduction(tables, state, size, year):
    sizes = np.minimum(size.astype(int), tables["std"].columns.max())
    out = np.array([tables["std"].at[y, s] for y, s in zip(np.broadcast_to(year, len(size)), sizes)], float)
    il = (state == "Illinois").to_numpy()
    return out - np.where(il, pd.Series(np.broadcast_to(year, len(size))).map(tables["il_offset"]).to_numpy(), 0)


def _by_state_year(table, state, year):
    return np.array([table.at[s, y] if s in table.index else np.nan
                     for s, y in zip(state, np.broadcast_to(year, len(state)))], float)


def _deductions(a, std_ded, shelter_cap, homeless_ded):
    """BGL's calculate_raw_benefits: earned-income, excess-shelter and total deductions, and net income."""
    earned_ded = np.floor(a["earned"] * 0.2)
    net_before = a["earned"] + a["unearned"] - (earned_ded + a["depcare"] + a["medical"] + a["child_support"] + std_ded)
    shelter_ded = np.minimum(np.maximum(a["rent"] + a["util"] - np.maximum(net_before * 0.5, 0), 0), shelter_cap)
    shelter_ded = np.where(homeless_ded > 0, 0, shelter_ded)
    return earned_ded, shelter_ded, net_before - (shelter_ded + homeless_ded)


def bgl_features(application: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """The 19 model inputs for each application row."""
    app = application
    fy, state = app["fiscal_year"].to_numpy(), app["state"]
    year = tables["year"]
    size = app["size"].to_numpy(float)
    a = {k: app[k].to_numpy(float) for k in AMOUNTS}
    homeless_ded = app["homeless_ded"].to_numpy(float)
    no_cap = app["elderly_or_disabled"].to_numpy(bool)

    # Review-year dollars: the benefit the amounts imply, relative to the maximum.
    std_now = _standard_deduction(tables, state, app["size"], fy)
    cap_now = np.where(no_cap, np.inf, year["max_shelter_deduction"].reindex(fy).to_numpy(float))
    earned_ded_now, shelter_ded_now, net_now = _deductions(a, std_now, cap_now, homeless_ded)
    uncapped = np.floor(app["benmax"] - 0.3 * np.floor(net_now))
    benefit = np.minimum(np.maximum(uncapped, app["min_ben"]), app["benmax"])

    # Round amounts: income types as filed, deductions as the review-year calculation gives them.
    deductions_now = np.column_stack([std_now, np.floor(0.2 * a["earned"] + 0.5), a["depcare"], shelter_ded_now,
                                      a["medical"], a["child_support"], homeless_ded])
    listed = np.column_stack([app[INCOME_TYPES].to_numpy(float), deductions_now])
    round_amounts = ((listed > 0) & (listed % 100 == 0)).sum(1)

    # FY2026 dollars: amounts moved by CPI; the utility and medical standards and the caps are FY2026's.
    factor = year.at[TARGET_YEAR, "cpi"] / year["cpi"].reindex(fy).to_numpy(float)
    b = {k: np.floor(v * factor) for k, v in a.items()}
    sua_now = _by_state_year(tables["sua"], state, fy)
    at_standard = (a["util"] > 0) & (a["util"] + SUA_TOLERANCE >= sua_now)  # NaN standard -> False
    b["util"] = np.where(at_standard, _by_state_year(tables["sua"], state, TARGET_YEAR), b["util"])
    smd = _by_state_year(tables["smd"], state, TARGET_YEAR)
    b["medical"] = np.where((a["medical"] > 0) & (a["medical"] < smd), smd, b["medical"])
    homeless_ded_t = np.where(homeless_ded > 0, year.at[TARGET_YEAR, "homeless_standard"], homeless_ded)
    std_t = _standard_deduction(tables, state, app["size"], TARGET_YEAR)
    cap_t = np.where(no_cap, np.inf, year.at[TARGET_YEAR, "max_shelter_deduction"])
    earned_ded_t, shelter_ded_t, _ = _deductions(b, std_t, cap_t, homeless_ded_t)
    total_ded = b["depcare"] + b["child_support"] + shelter_ded_t + b["medical"] + homeless_ded_t + earned_ded_t + std_t

    out = pd.DataFrame({
        "HH_size_n": size,
        "children_i": app["children"].astype(float),
        "elderly_disabled_i": app["elderly_or_disabled"].astype(float),
        "total_deductions_by_hh_size": total_ded / size,
        "expedited_i": app["expedited"].astype(float),
        "bbce_state_i": app["bbce_state"].astype(float),
        "rawben_rel_max": benefit / app["benmax"],
        "medical_deductions": b["medical"],
        "shelter_expenses_by_hh_size": (b["rent"] + b["util"]) / size,
        "utilities": b["util"],
        "married": app["married"].astype(float),
        "homeless": app["homeless"].astype(float),
        "earned_by_hh_size": b["earned"] / size,
        "unearned_by_hh_size": b["unearned"] / size,
        "gross_by_hh_size": (b["earned"] + b["unearned"]) / size,
        "percent_abawd": app["n_abawd"] / size,
        "unc_rawben_rel_max": uncapped / app["benmax"],
        "months_since_cert_n": app["months_since_cert"].astype(float),
        "count_divisible_by_100": round_amounts.astype(float),
    }, index=app.index)
    return out[FEATURES]


def application_from_qc(df: pd.DataFrame, fiscal_year: int = 2024, bbce: pd.Series | None = None) -> pd.DataFrame:
    """Application table from QC file rows, every amount as QC verified it. `bbce` maps state -> BBCE flag;
    by default it is computed from `df` (which should then hold whole states)."""
    def z(c):  # a column the year's file does not carry (FSEITC in FY2024) counts as $0
        return df[c].fillna(0).astype(float) if c in df else pd.Series(0.0, index=df.index)

    if bbce is None:
        bbce = (df["CAT_ELIG"] >= 1).groupby(df["STATENAME"]).mean() >= 0.5
    excl = df["EXCL_FSCSDED"].fillna(0) if "EXCL_FSCSDED" in df else 0
    app = pd.DataFrame({
        "state": df["STATENAME"],
        "fiscal_year": fiscal_year,
        "size": df["FSUSIZE"],
        "children": df["FSNKID"] > 0,
        "elderly_or_disabled": (df["FSNELDER"] > 0) | (df["FSNDIS"] > 0),
        "married": (df[[f"REL{i}" for i in range(1, 17)]] == 2).any(axis=1),
        "homeless": (df["HOMEDED"] != 1).where(df["HOMEDED"].notna()),
        "earned": z("FSEARN"), "unearned": z("FSUNEARN"), "rent": z("RENT"), "util": z("UTIL"),
        "medical": z("FSMEDDED"), "depcare": z("FSDEPDED"),
        "child_support": np.where(z("FSCSDED") + excl > 0, z("FSCSEXP"), 0),
        "homeless_ded": z("HOMELESS_DED"),
        **{t: z(t) for t in INCOME_TYPES},
        "n_abawd": df[[f"ABWDST{i}" for i in range(1, 19)]].isin([2, 3, 4, 5]).sum(axis=1),
        "expedited": df["EXPEDSER"] < 3,
        "months_since_cert": df["LASTCERT"],
        "bbce_state": df["STATENAME"].map(bbce),
        "benmax": df["BENMAX"],
        "min_ben": np.where(df["FSUSIZE"] < 3, df["MINIMUM_BEN"], 0),
    }, index=df.index)
    return app[list(APPLICATION_FIELDS)]


def file_amounts(application: pd.DataFrame, amount: pd.Series, filed_value: pd.Series, element: pd.Series) -> pd.DataFrame:
    """The application with one amount per household replaced by the filed value.

    `amount` names the field ('earned', 'unearned', 'rent', ...), `filed_value` the dollars filed, `element` the
    ELEMENT code. For income, the income types of that group are filed too: all $0 when the group is filed at
    $0; otherwise the type the element names moves by the same dollars as the group total (not below $0).
    The verified amount of the group is read only to place that difference on a type, never returned.
    """
    app = application.copy()
    for i in app.index:
        field, filed = amount.at[i], float(filed_value.at[i])
        verified = float(app.at[i, field])
        app.at[i, field] = filed
        types = {"earned": EARNED_TYPES, "unearned": UNEARNED_TYPES}.get(field)
        if types is None:
            continue
        if filed == 0:
            app.loc[i, types] = 0.0
            continue
        t = ELEMENT_TYPE[int(element.at[i])]
        if t == "FSTANF" and app.at[i, "FSTANF"] == 0 and app.at[i, "FSGA"] > 0:
            t = "FSGA"  # element 344 covers TANF and general assistance
        app.at[i, t] = max(0.0, float(app.at[i, t]) + filed - verified)
    return app
