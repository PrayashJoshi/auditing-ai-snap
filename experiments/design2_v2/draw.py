"""Design 2 v2: draw the 12 households and build their ledger and complete packets (no model calls).

Pool (FY2024 SNAP QC public-use file, 44,891 rows): the two filters the pilot draw uses (`packets.pilot_sample`):
`calc.benefit` on the verified amounts within $1 of FSBEN, and a utility tier other than "actual costs"; then
households whose earned income is wages only: at least one person with WAGES above $0 and no self-employment,
odd-job or room-rental income (SLFEMP, OTHERN, RENTMINC $0 for persons 1 to 18). Expected pool: 8,921.
The 20 pilot households are removed from the pool before the draw, so every drawn household is new.
Draw: 12 rows uniformly at random without replacement from the sorted row numbers, numpy default_rng(20261116),
the same idiom `pilot_sample` uses for its comparison households. No strata.

Each drawn household gets the pilot's household record (the columns of `pilot_sample`), its fact ledger and
records (`packets.build_case`), and its complete packet with note A (`packets.make_packet`, `packets.note`,
`packets.filer_input`), written next to this script in the pilot's file formats. Pilot files are not touched.

Checks (all asserted): pool size reported against 8,921; person-level income adds up (inside `packets.ledger`);
every fact group adds up to the verified group value; every pay stub's gross converts back to the verified
monthly wage within $1; the oracle on the complete packet gives the QC-correct benefit within $1.

Run: PYTHONPATH=src python experiments/design2_v2/draw.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snapqc import checker as C
from snapqc import forms as F
from snapqc import packets as P
from snapqc.calc import inputs
from snapqc.data import RESULTS, load_fy2024

HERE = Path(__file__).resolve().parent
SEED = 20261116
N_HOUSEHOLDS = 12
EXPECTED_POOL = 8921
OTHER_EARNED = [t for t in P.EARNED_TYPES if t != "WAGES"]
PERSONS = range(1, 19)
PILOT_HOUSEHOLDS = ROOT / "experiments" / "pilot" / "households.csv"
V1_TARGET_ROWS = [14542, 15847, 28066, 28402, 42362, 42777, 42865, 5744]

OUT = {
    "households": HERE / "households.csv",
    "ledger": HERE / "ledger.csv",
    "packets": HERE / "packets.jsonl",
    "draw_log": HERE / "draw-log.csv",
}


def person_sum(df: pd.DataFrame, t: str) -> pd.Series:
    cols = [f"{t}{p}" for p in PERSONS if f"{t}{p}" in df.columns]
    return df[cols].fillna(0).sum(axis=1)


def wage_facts(df: pd.DataFrame) -> pd.Series:
    cols = [f"WAGES{p}" for p in PERSONS if f"WAGES{p}" in df.columns]
    return (df[cols].fillna(0) > 0).sum(axis=1)


def wage_only_pool(df: pd.DataFrame, tier: pd.Series) -> tuple[pd.Index, dict]:
    calc_ok = (P.qc_benefit(df) - df["FSBEN"]).abs() <= 1
    base = df[calc_ok & (tier != "actual costs")]
    other = sum(person_sum(base, t) for t in OTHER_EARNED)
    n_wage = wage_facts(base)
    pool = base.index[(other == 0) & (n_wage >= 1)]
    log = {
        "QC rows": len(df),
        "calculator within $1 of FSBEN": int(calc_ok.sum()),
        "also utility tier not actual costs": len(base),
        "wage-only pool": len(pool),
        "wage-only pool (expected)": EXPECTED_POOL,
        **{f"wage-only pool: {k} wage fact{'s' if k > 1 else ''}": int(v)
           for k, v in n_wage[pool].value_counts().sort_index().items()},
        "v1 households in the pool": int(pd.Index(V1_TARGET_ROWS).isin(pool).sum()),
    }
    return pool, log


def household_records(df: pd.DataFrame, arm: pd.DataFrame, tier: pd.Series, rows: pd.Index) -> pd.DataFrame:
    """The pilot's household record (columns of `packets.pilot_sample`) for the drawn rows; kind "draw"."""
    q = df.loc[rows]
    calc_in = inputs(q)
    out = pd.DataFrame({
        "kind": "draw",
        "group": arm["group"].reindex(rows).fillna(""),
        "in_151": arm["in_151"].reindex(rows).fillna(False).astype(bool),
        "DISCOV1": arm["DISCOV1"].reindex(rows).astype("Int64"),
        "AGENCY1": q["AGENCY1"].astype("Int64"),
        "state": q["STATENAME"],
        "size": q["FSUSIZE"].astype(int),
        "qc_benefit": P.qc_benefit(q),
        "FSBEN": q["FSBEN"],
        "RAWBEN": q["RAWBEN"],
        "AMTERR": q["AMTERR"],
        "filed_value": arm["filed_value"].reindex(rows),
        "verified_value": arm["verified_value"].reindex(rows),
        "utility_tier": tier[rows],
        **{f"verified_{g}": calc_in[a] for g, a in P.GROUP_AMOUNT.items()},
        **{c: calc_in[c] for c in P.FIXED_AMOUNTS + P.PARAMETERS if c != "size"},
    }, index=rows)
    out.index.name = "row"
    return out


def main() -> None:
    df = load_fy2024()
    arm = pd.read_csv(RESULTS / "01-data-checks" / "3d-human-arm.csv", index_col="row")
    tiers = F.sua_tiers(df)
    tier = P.utility_tier(df, tiers)

    pool, log = wage_only_pool(df, tier)
    pilot_rows = pd.read_csv(PILOT_HOUSEHOLDS)["row"]
    candidates = pool[~pool.isin(pilot_rows)]
    log["pilot households removed"] = int(pool.isin(pilot_rows).sum())
    log["draw candidates"] = len(candidates)
    rows = pd.Index(np.random.default_rng(SEED).choice(np.sort(candidates), N_HOUSEHOLDS, replace=False))
    log["seed"] = SEED
    log["households drawn"] = len(rows)
    log["drawn rows (draw order)"] = " ".join(str(r) for r in rows)
    if len(pool) != EXPECTED_POOL:
        print(f"MISMATCH: wage-only pool {len(pool)}, expected {EXPECTED_POOL}")

    households = household_records(df, arm, tier, rows)
    shares = P.record_shares()
    cases = {}
    for row, h in households.iterrows():
        sua = {k: (None if pd.isna(v) else float(v)) for k, v in tiers.loc[h.state].items()}
        cases[row] = P.build_case(df.loc[row], h, sua)

    # Ledger, in the pilot ledger's columns.
    facts = pd.DataFrame([{"row": r, **f} for r, c in cases.items() for f in c["facts"]])
    facts["link"] = facts["type"].map(P.LINK).fillna(facts["group"].map({"shelter": "lease", "utilities": "bills"}))
    for col in ("utility_types", "employer"):
        if col not in facts:
            facts[col] = None
    facts = facts[["row", "fact", "group", "type", "person", "amount", "utility_types", "link", "employer"]]

    # Every fact group adds up to the verified group value.
    totals = facts.groupby(["row", "group"]).amount.sum().unstack().reindex(columns=F.GROUPS).fillna(0)
    verified = households[[f"verified_{g}" for g in F.GROUPS]].set_axis(F.GROUPS, axis=1)
    assert np.allclose(totals.reindex(verified.index).fillna(0), verified, atol=1), "fact groups != verified"
    assert set(facts.loc[facts.group == "earned", "type"]) == {"WAGES"}, "earned facts other than wages"
    assert (facts[facts.type == "WAGES"].groupby("row").size().reindex(rows).fillna(0) >= 1).all()

    # Every pay stub converts back to its verified monthly wage within $1.
    for r, c in cases.items():
        for f in c["facts"]:
            if f["type"] != "WAGES":
                continue
            stubs = [x["data"] for x in c["records"] if x["fact"] == f["fact"] and x["kind"] == "pay_stub"]
            assert stubs, f"row {r} {f['fact']}: no pay stub"
            monthly = stubs[0]["Gross pay"] * F.MONTHLY_FACTOR[C.FREQUENCY_CODE[stubs[0]["Pay frequency"]]]
            assert abs(monthly - f["amount"]) < 1, f"row {r} {f['fact']}: stub {monthly} vs {f['amount']}"

    # Complete packets, note A (note B equals A on a complete packet), filer inputs.
    pk = {r: P.make_packet(c, "complete", shares) for r, c in cases.items()}
    for r, p in pk.items():
        assert not p["missing_facts"] and not p["undocumented_groups"], f"row {r}: complete packet drops facts"
        oracle = C.oracle(p)
        ben = P.benefit_of(oracle, households.loc[r])
        assert abs(ben - households.at[r, "qc_benefit"]) <= 1, f"row {r}: oracle benefit {ben}"

    households.to_csv(OUT["households"])
    facts.to_csv(OUT["ledger"], index=False)
    with OUT["packets"].open("w") as fh:
        for r in rows:
            p = pk[r]
            notes = {n: P.note(cases[r], p, n) for n in "AB"}
            fh.write(json.dumps({**p, "notes": notes,
                                 "filer_input": {n: P.filer_input(p, notes[n]) for n in "AB"}}) + "\n")
    pd.Series(log).to_frame("value").rename_axis("item").to_csv(OUT["draw_log"])

    for k, v in log.items():
        print(f"{k}: {v}")
    print(f"\nWage facts per drawn household: {facts[facts.type == 'WAGES'].groupby('row').size().reindex(rows).to_dict()}")
    print("ALL DRAW ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
