"""How often the budget-gap rule fires on truthful households, and what that implies for its recall
and precision on known income omissions.

Truthful application = the QC-verified values of a household. The budget-gap rule compares observed
housing cost (lease rent plus utility bill amounts) with reported income (earned + unearned) and, when housing
is larger, flags each income group reported as $0. This script applies that comparison to every household of the
FY2024 SNAP QC public-use file (primary) and of the FY2023 and FY2022 files (exploratory, same computation).

Fixed definitions
- Income: FSEARN + FSUNEARN (missing counts as 0), the two amounts the packet generator reports as the earned
  and unearned groups. FSGRINC is used only in one comparison block (rent only), labelled as such.
- Housing: (a) rent_only = RENT; (b) rent_plus_qc_util = RENT + UTIL ("if the rule used the allowance");
  (c) rent_plus_sim_bills = RENT + simulated utility bill amounts, drawn from the same distributions the packet
  generator uses: actual-cost households get bills that add up to the verified UTIL; allowance households get
  one uniform draw per utility type of their tier from packets.BILL_RANGE; households with no utility fact get 0.
  The draw is vectorised over the whole file with one seed per year ([packets.SEED, year]); it has the
  generator's distribution but not the generator's per-household realised values (those come from a
  per-household stream that also draws names, employers and pay schedules first).
- Variants: "any" = housing > income; "with_zero_group" = housing > income and (earned == 0 or unearned == 0),
  which is when the rule actually flags a group.
- Intervals: Wilson 95%. Weighted shares use FYWGT and the Wilson formula with the Kish effective sample size
  (sum w)^2 / sum w^2.

Recall and precision are computed on the 41 states of the human arm: true positives are the human-arm income
omissions (an earned or unearned group filed at $2 or less with a verified amount above $0; 185 rows) and false
positives are the truthful FY2024 households of the same 41 states on which the rule fires.

Outputs (next to this script, or in --out DIR): rule-a-fpr.csv, rule-a-by-state.csv, rule-a-by-size.csv,
rule-a-recall-precision.csv, rule-a-fpr.svg.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_root(start: Path) -> Path:
    """First directory above `start` holding both pyproject.toml and src/snapqc."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file() and (p / "src" / "snapqc").is_dir():
            return p
    raise RuntimeError(f"no repo root (pyproject.toml and src/snapqc) above {start}")


HERE = Path(__file__).resolve().parent
ROOT = find_root(HERE)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from snapqc import packets  # noqa: E402
from snapqc.forms import TIERS, sua_tiers  # noqa: E402

SEED = packets.SEED  # same seed constant as the packet generator
YEARS = [2024, 2023, 2022]  # 2024 primary; 2023 and 2022 exploratory
PRIMARY = 2024
DEFINITIONS = ["rent_only", "rent_plus_qc_util", "rent_plus_sim_bills"]
DEFINITION_LABEL = {"rent_only": "Rent only",
                    "rent_plus_qc_util": "Rent + QC utility amount\n(if the rule used the allowance)",
                    "rent_plus_sim_bills": "Rent + simulated utility bills"}
VARIANTS = ["any", "with_zero_group"]
VARIANT_LABEL = {"any": "housing > income", "with_zero_group": "housing > income and a $0 income group"}
INCOME_BASIS = "FSEARN + FSUNEARN"
REQUIRED = ["FSEARN", "FSUNEARN", "RENT", "UTIL", "FYWGT", "STATENAME", "SUA1", "SUA2"]
SIZE_COLUMN = "FSUSIZE"
SIZE_CAP = 6
EXPECTED_OMISSIONS = 185
EXPECTED_ARM_STATES = 41
OMISSION_FILED_MAX = 2.0  # a group filed at $2 or less counts as omitted
EXPECTED_N_2024 = 44_891
Z = 1.959963984540054
ARM_PATH = ROOT / "results" / "01-data-checks" / "3d-human-arm.csv"
BILL_RANGE_TEXT = "; ".join(f"{t} ${lo}-${hi}" for t, (lo, hi) in packets.BILL_RANGE.items())
TIER_TYPES_TEXT = "; ".join(f"{t}: {'+'.join(v)}" for t, v in packets.TIER_TYPES.items())


# --- Statistics -------------------------------------------------------------------------------------------
def wilson(p: float, n: float) -> tuple[float, float]:
    """Wilson 95% interval for a share p on n (possibly effective) observations."""
    if not n or n <= 0 or p is None or math.isnan(p):
        return float("nan"), float("nan")
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def kish(w: np.ndarray) -> float:
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else float("nan")


def share_rows(mask, w, **keys) -> list[dict]:
    """Two rows (unweighted, weighted) for a boolean mask and weights."""
    mask = np.asarray(mask, dtype=bool)
    w = np.asarray(w, dtype=float)
    n, k = int(mask.size), int(mask.sum())
    p = k / n if n else float("nan")
    lo, hi = wilson(p, n)
    out = [{**keys, "weighting": "unweighted", "n": n, "k": k, "share": p, "wilson_lo": lo, "wilson_hi": hi,
            "n_eff": float(n), "interval": "Wilson 95%, n = number of households"}]
    sw = float(w.sum())
    pw = float(w[mask].sum()) / sw if sw > 0 else float("nan")
    ne = kish(w)
    lo, hi = wilson(pw, ne)
    out.append({**keys, "weighting": "FYWGT-weighted", "n": n, "k": float("nan"), "share": pw, "wilson_lo": lo,
                "wilson_hi": hi, "n_eff": ne,
                "interval": "Wilson 95% with Kish effective n = (sum w)^2 / sum w^2"})
    return out


# --- Data -------------------------------------------------------------------------------------------------
def load_year(year: int) -> pd.DataFrame:
    """One year's QC public-use file with upper-case column names and the default row index."""
    path = ROOT / "data" / "raw" / f"fy{year}" / f"qc_pub_fy{year}.csv"
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).upper() for c in df.columns]
    return df


def simulated_bills(df: pd.DataFrame, tier: pd.Series, year: int) -> pd.Series:
    """Sum of utility bill amounts per household, drawn as the packet generator draws them (same distributions,
    vectorised): actual costs -> bills add up to the verified UTIL (the Dirichlet split does not change the sum);
    an allowance tier -> one uniform draw per utility type of the tier, each rounded to cents; none -> 0."""
    rng = np.random.default_rng([SEED, year])
    out = pd.Series(0.0, index=df.index)
    actual = tier == "actual costs"
    out[actual] = df.loc[actual, "UTIL"].fillna(0).astype(float)
    for t in TIERS:  # fixed order so the draw is reproducible
        m = (tier == t).to_numpy()
        total = np.zeros(int(m.sum()))
        for utility in packets.TIER_TYPES[t]:
            total += np.round(rng.uniform(*packets.BILL_RANGE[utility], size=total.size), 2)
        out[m] = total
    return out


def prepare(year: int) -> tuple[pd.DataFrame | None, str]:
    """The year's file with the derived columns used below, or (None, reason)."""
    try:
        df = load_year(year)
    except FileNotFoundError as e:
        return None, f"file not found: {e.filename}"
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"columns missing: {', '.join(missing)}"
    df["_earned"] = df["FSEARN"].fillna(0).astype(float)
    df["_unearned"] = df["FSUNEARN"].fillna(0).astype(float)
    df["_income"] = df["_earned"] + df["_unearned"]
    df["_rent"] = df["RENT"].fillna(0).astype(float)
    df["_util"] = df["UTIL"].fillna(0).astype(float)
    df["_w"] = df["FYWGT"].fillna(0).astype(float)
    df["_tier"] = packets.utility_tier(df, sua_tiers(df))  # tiers inferred from this year's own file
    df["_bills"] = simulated_bills(df, df["_tier"], year)
    df["_zero_group"] = (df["_earned"] == 0) | (df["_unearned"] == 0)
    df["_h_rent_only"] = df["_rent"]
    df["_h_rent_plus_qc_util"] = df["_rent"] + df["_util"]
    df["_h_rent_plus_sim_bills"] = df["_rent"] + df["_bills"]
    return df, ""


def fires(df: pd.DataFrame, definition: str, variant: str, income: pd.Series | None = None) -> pd.Series:
    gap = df[f"_h_{definition}"] > (df["_income"] if income is None else income)
    return gap & df["_zero_group"] if variant == "with_zero_group" else gap


# --- Budget-gap rule on truthful files -----------------------------------------------------------------------------
def fpr_table(frames: dict[int, pd.DataFrame], skipped: dict[int, str]) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        role = "primary" if year == PRIMARY else "exploratory"
        if year not in frames:
            rows.append({"year": year, "role": role, "note": f"not computed: {skipped[year]}"})
            continue
        df = frames[year]
        tiers = df["_tier"].value_counts().to_dict()
        for d in DEFINITIONS:
            note = ""
            if d == "rent_plus_sim_bills":
                note = (f"bills drawn with seed [{SEED}, {year}]; BILL_RANGE: {BILL_RANGE_TEXT}; types per tier: "
                        f"{TIER_TYPES_TEXT}; households per tier: {tiers}")
            elif d == "rent_plus_qc_util":
                note = "UTIL is the verified utility amount (the allowance for allowance households)"
            for v in VARIANTS:
                rows += share_rows(fires(df, d, v), df["_w"], year=year, role=role, definition=d, variant=v,
                                   income_basis=INCOME_BASIS, note=note)
        if "FSGRINC" in df.columns:  # comparison block only: the same rent-only test against FSGRINC
            g = df["FSGRINC"].fillna(0).astype(float)
            differ = int(((g - df["_income"]).abs() > 1).sum())
            for v in VARIANTS:
                rows += share_rows(fires(df, "rent_only", v, income=g), df["_w"], year=year, role=role,
                                   definition="rent_only", variant=v, income_basis="FSGRINC (comparison)",
                                   note=f"FSGRINC differs from FSEARN + FSUNEARN by more than $1 in {differ} households")
    cols = ["year", "role", "definition", "variant", "income_basis", "weighting", "n", "k", "share", "wilson_lo",
            "wilson_hi", "n_eff", "interval", "note"]
    return pd.DataFrame(rows).reindex(columns=cols)


def breakdown(df: pd.DataFrame, by: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    for d in ("rent_only", "rent_plus_sim_bills"):
        f = fires(df, d, "with_zero_group")
        for level, idx in df.groupby(by, sort=True).groups.items():
            rows += share_rows(f.loc[idx], df.loc[idx, "_w"], year=PRIMARY, **{name: level}, definition=d,
                               variant="with_zero_group")
    return pd.DataFrame(rows)


# --- Recall and precision on the human-arm omissions ------------------------------------------------------
def recall_precision(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {}
    if not ARM_PATH.is_file():
        return pd.DataFrame([{"metric": "not computed", "note": f"file not found: {ARM_PATH.relative_to(ROOT)}"}]), info
    arm = pd.read_csv(ARM_PATH)
    om = arm[arm["group"].isin(["earned", "unearned"]) & (arm["filed_value"] <= OMISSION_FILED_MAX)
             & (arm["verified_value"] > 0)].copy()
    info["n_omissions"] = len(om)
    assert len(om) == EXPECTED_OMISSIONS, f"omissions: {len(om)}, expected {EXPECTED_OMISSIONS}"
    arm_states = sorted(arm["state"].dropna().unique())
    assert len(arm_states) == EXPECTED_ARM_STATES, f"human-arm states: {len(arm_states)}"
    unknown = sorted(set(arm_states) - set(df["STATENAME"].unique()))
    assert not unknown, f"human-arm states not in STATENAME: {unknown}"
    in_arm = df["STATENAME"].isin(arm_states).to_numpy()
    info["n_arm_state_households"] = int(in_arm.sum())
    om_states = df.loc[om["row"].to_numpy(), "STATENAME"]
    assert om_states.isin(arm_states).all(), "an omission row lies outside the human-arm states"
    count_note = (f"realised {len(om)}, expected {EXPECTED_OMISSIONS}"
                  + ("" if len(om) == EXPECTED_OMISSIONS else " (MISMATCH)"))
    bad_rows = int((~om["row"].isin(df.index)).sum())
    om = om[om["row"].isin(df.index)]
    q = df.loc[om["row"].to_numpy()]
    earned_omitted = (om["group"] == "earned").to_numpy()
    # Filed application: the verified values with the omitted group at its filed amount ($2 or less).
    filed_income = (np.where(earned_omitted, q["_unearned"].to_numpy(), q["_earned"].to_numpy())
                    + om["filed_value"].to_numpy(dtype=float))
    qc_value = np.where(earned_omitted, q["_earned"].to_numpy(), q["_unearned"].to_numpy())
    value_diff = int((np.abs(qc_value - om["verified_value"].to_numpy()) > 1).sum())
    w = q["_w"].to_numpy()
    weight_diff = int((np.abs(w - om["FYWGT"].fillna(0).to_numpy()) > 1e-6).sum())
    data_note = (f"omissions: {count_note}; rows not in the FY2024 file: {bad_rows}; verified_value differs from the "
                 f"QC group total by more than $1 in {value_diff} rows; FYWGT differs from the QC file in {weight_diff} "
                 f"rows; households with more than one omission row: {int(om['row'].duplicated().sum())}")
    undercount = ("the human arm undercounts omissions (it holds only those the QC review found and that could be "
                  "rebuilt), so true positives are understated and this precision is a lower-bound-style figure; "
                  f"false positives = truthful FY2024 households of the {EXPECTED_ARM_STATES} human-arm states "
                  f"({int(in_arm.sum()):,} households) on which the rule fires (with_zero_group), including the "
                  "truthful versions of the omission households")
    rows = []
    for d in ("rent_only", "rent_plus_sim_bills"):
        flagged = q[f"_h_{d}"].to_numpy() > filed_income  # the omitted group is at most $2; counted as flagged
        for r in share_rows(flagged, w, metric="recall", definition=d):
            rows.append({**r, "note": data_note})
        fp = fires(df, d, "with_zero_group").to_numpy() & in_arm
        tp_n, fp_n = int(flagged.sum()), int(fp.sum())
        n = tp_n + fp_n
        p = tp_n / n if n else float("nan")
        lo, hi = wilson(p, n)
        rows.append({"metric": "precision", "definition": d, "weighting": "unweighted", "n": n, "k": tp_n, "share": p,
                     "wilson_lo": lo, "wilson_hi": hi, "n_eff": float(n),
                     "interval": "Wilson 95%, n = TP + FP", "note": f"TP {tp_n}, FP {fp_n}; {undercount}"})
        tp_w, fp_w = float(w[flagged].sum()), float(df["_w"].to_numpy()[fp].sum())
        pw = tp_w / (tp_w + fp_w) if tp_w + fp_w > 0 else float("nan")
        ne = kish(np.concatenate([w[flagged], df["_w"].to_numpy()[fp]]))
        lo, hi = wilson(pw, ne)
        rows.append({"metric": "precision", "definition": d, "weighting": "FYWGT-weighted", "n": n, "k": float("nan"),
                     "share": pw, "wilson_lo": lo, "wilson_hi": hi, "n_eff": ne,
                     "interval": "Wilson 95% with Kish effective n = (sum w)^2 / sum w^2 over flagged cases",
                     "note": f"weighted TP {tp_w:,.0f}, weighted FP {fp_w:,.0f}; {undercount}"})
        info[d] = {"recall": tp_n / len(om) if len(om) else float("nan"), "precision": p}
    cols = ["metric", "definition", "weighting", "n", "k", "share", "wilson_lo", "wilson_hi", "n_eff", "interval", "note"]
    return pd.DataFrame(rows).reindex(columns=cols), info


# --- Figure -----------------------------------------------------------------------------------------------
def figure(fpr: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = fpr[(fpr["year"] == PRIMARY) & (fpr["income_basis"] == INCOME_BASIS)]
    groups = [(d, v) for d in DEFINITIONS for v in VARIANTS]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    height = 0.38
    for j, (weighting, colour) in enumerate([("unweighted", "#4c72b0"), ("FYWGT-weighted", "#dd8452")]):
        s = t[t["weighting"] == weighting].set_index(["definition", "variant"]).reindex(groups)
        y = np.arange(len(groups)) + (j - 0.5) * height
        share = s["share"].to_numpy(dtype=float) * 100
        err = np.vstack([share - s["wilson_lo"].to_numpy(dtype=float) * 100,
                         s["wilson_hi"].to_numpy(dtype=float) * 100 - share])
        ax.barh(y, share, height=height, color=colour, label=weighting, xerr=err, capsize=3,
                error_kw={"elinewidth": 1, "ecolor": "#333333"})
        for yi, si, hi in zip(y, share, s["wilson_hi"].to_numpy(dtype=float) * 100):
            ax.text(hi + 0.3, yi, f"{si:.1f}%", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([f"{DEFINITION_LABEL[d]}\n[{VARIANT_LABEL[v]}]" for d, v in groups], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of households on which the rule's comparison holds (%), Wilson 95% interval")
    ax.set_title("Budget-gap rule firing on truthful QC households, FY2024")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --- Main -------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: next to this script)")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames, skipped = {}, {}
    for year in YEARS:
        df, reason = prepare(year)
        if df is None:
            skipped[year] = reason
            print(f"FY{year}: not computed ({reason})")
        else:
            frames[year] = df
    if PRIMARY not in frames:
        raise SystemExit(f"FY{PRIMARY} file could not be used: {skipped[PRIMARY]}")
    df = frames[PRIMARY]

    fpr = fpr_table(frames, skipped)
    fpr.to_csv(out / "rule-a-fpr.csv", index=False)
    breakdown(df, df["STATENAME"], "state").to_csv(out / "rule-a-by-state.csv", index=False)
    if SIZE_COLUMN in df.columns:
        size = df[SIZE_COLUMN].fillna(0).astype(int).clip(upper=SIZE_CAP)
        size = size.astype(str).where(size < SIZE_CAP, f"{SIZE_CAP}+")
        by_size = breakdown(df, size, "household_size")
        by_size.insert(1, "size_column", SIZE_COLUMN)
    else:
        by_size = pd.DataFrame([{"note": f"not computed: column {SIZE_COLUMN} missing"}])
    by_size.to_csv(out / "rule-a-by-size.csv", index=False)
    rp, info = recall_precision(df)
    rp.to_csv(out / "rule-a-recall-precision.csv", index=False)
    figure(fpr, out / "rule-a-fpr.svg")

    def pick(year, definition, variant, weighting, basis=INCOME_BASIS):
        r = fpr[(fpr["year"] == year) & (fpr["definition"] == definition) & (fpr["variant"] == variant)
                & (fpr["weighting"] == weighting) & (fpr["income_basis"] == basis)]
        return r.iloc[0] if len(r) else None

    def fmt(r):
        return "n/a" if r is None else f"{r['share']:.1%} [{r['wilson_lo']:.1%}, {r['wilson_hi']:.1%}]"

    n = len(df)
    print(f"\nBudget-gap rule on truthful households. Income = {INCOME_BASIS}. Seed {SEED}.")
    print(f"FY{PRIMARY} households: {n:,} (expected {EXPECTED_N_2024:,}{'' if n == EXPECTED_N_2024 else ', MISMATCH'})")
    print(f"BILL_RANGE: {BILL_RANGE_TEXT}")
    for year in frames:
        print(f"\nFY{year}{'' if year == PRIMARY else ' (exploratory)'}")
        for d in DEFINITIONS:
            for v in VARIANTS:
                print(f"  {d:<20} {v:<16} unweighted {fmt(pick(year, d, v, 'unweighted'))}   "
                      f"weighted {fmt(pick(year, d, v, 'FYWGT-weighted'))}")
    print(f"\nFY{PRIMARY} rent only, housing > income: unweighted {fmt(pick(PRIMARY, 'rent_only', 'any', 'unweighted'))}, "
          f"weighted {fmt(pick(PRIMARY, 'rent_only', 'any', 'FYWGT-weighted'))}")
    g = "FSGRINC (comparison)"
    print(f"  same test against FSGRINC: unweighted {fmt(pick(PRIMARY, 'rent_only', 'any', 'unweighted', g))}, "
          f"weighted {fmt(pick(PRIMARY, 'rent_only', 'any', 'FYWGT-weighted', g))} "
          f"(earlier exploratory figures: 6.1% and 7.0%)")
    if "n_omissions" in info:
        k = info["n_omissions"]
        print(f"\nHuman-arm omissions: {k} (expected {EXPECTED_OMISSIONS}); households in the "
              f"{EXPECTED_ARM_STATES} human-arm states: {info['n_arm_state_households']:,}")
    print(rp.drop(columns=["note", "interval"], errors="ignore").to_string(index=False))
    print("Precision is a lower-bound-style figure: the human arm undercounts omissions.")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
