"""SNAP benefit calculator, FY2024 rules as written in the SNAP QC technical documentation (Chapter V).

Every argument broadcasts, so the same function scores a whole table or a grid of trial values for one household.
"""
import numpy as np
import pandas as pd


def benefit(earned, unearned, rent, util, depcare, medical, child_support, *,
            std_ded, homeless_ded, shelter_cap, uncapped, benmax, min_ben, size):
    """Monthly benefit in dollars from application amounts (positional) and household parameters (keyword)."""
    earned_ded = np.floor(0.2 * earned + 0.5)  # 20% of earned income, rounded to the nearest dollar
    adjusted = np.maximum(0, earned + unearned - std_ded - earned_ded - depcare - medical - child_support)
    excess = np.maximum(0, rent + util - np.floor(adjusted / 2))  # shelter cost above half of adjusted income
    shelter_ded = np.where(uncapped, excess, np.minimum(excess, shelter_cap))
    shelter_ded = np.where(homeless_ded > 0, 0, shelter_ded)  # homeless deduction replaces the shelter deduction
    net = np.maximum(0, adjusted - shelter_ded - homeless_ded)
    formula = benmax - np.ceil(0.3 * net)  # 30% of net income, rounded up
    return np.where(size <= 2, np.maximum(min_ben, formula), np.maximum(0, formula))


# Calculator argument -> column in the QC file.
AMOUNTS = {"earned": "FSEARN", "unearned": "FSUNEARN", "rent": "RENT", "util": "UTIL",
           "depcare": "FSDEPDED", "medical": "FSMEDDED", "child_support": "FSCSDED"}
PARAMETERS = {"std_ded": "FSSTDDED", "homeless_ded": "HOMELESS_DED", "shelter_cap": "SHELCAP",
              "benmax": "BENMAX", "min_ben": "MINIMUM_BEN", "size": "FSUSIZE"}


def inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Calculator arguments for each household, taken from the QC file (missing amounts count as 0)."""
    out = df[list((AMOUNTS | PARAMETERS).values())].fillna(0)
    out.columns = list(AMOUNTS | PARAMETERS)
    out["uncapped"] = (df["FSELDER"] > 0) | (df["FSDIS"] > 0)  # elderly or disabled member: no shelter cap
    return out
