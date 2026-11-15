"""Repeat-run agreement and per-group flag breakdown for the reader-omission experiment.

Reads experiments/design1/results.jsonl (no model calls) and writes three tables:

  retest-level1.csv        at trace level "none", the omission_zero and truthful_twin inputs of a
                           household are expected to be the same text. Each such pair is two runs of
                           one reader on one input, so flag agreement measures run-to-run stability.
  group-flag-matrix.csv    reader x trace_level x state x group: inputs, inputs flagged, share.
  false-flags-by-group.csv truthful twins: any flag, income-group flag, shelter/utilities-only flag;
                           omission inputs: the same split for flags on groups other than the target.

Usage: python retest_design1.py [--root REPO_ROOT] [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

GROUPS = ["earned", "unearned", "shelter", "utilities"]
INCOME = {"earned", "unearned"}
TRACE_LEVELS = ["none", "budget_gap", "net_deposits", "wage_match"]
STATES = ["omission_zero", "omission_not_provided", "truthful_twin"]
Z = 1.96


def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "snapqc").is_dir():
            return p
    raise SystemExit("repo root not found (need pyproject.toml and src/snapqc); pass --root")


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_cols(prefix: str, k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {
        f"{prefix}_n": k,
        f"{prefix}_share": k / n if n else float("nan"),
        f"{prefix}_lo": lo,
        f"{prefix}_hi": hi,
    }


def load(path: Path) -> pd.DataFrame:
    """One row per job (last record wins), with the flagged groups as a frozenset."""
    latest: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                latest[r["job_id"]] = r
    rows = []
    for r in latest.values():
        flags_list = (r.get("parsed") or {}).get("flags") or []
        flagged = frozenset(f["group"] for f in flags_list if isinstance(f, dict) and "group" in f)
        rows.append(
            {
                "reader": r["reader"],
                "row": r["row"],
                "target_group": r["target_group"],
                "trace_level": r["trace_level"],
                "level_num": r["level_num"],
                "state": r["state"],
                "is_omission": bool(r["is_omission"]),
                "reader_input_sha": r["reader_input_sha"],
                "valid": bool(r.get("valid")),
                "flagged": flagged,
            }
        )
    df = pd.DataFrame(rows)
    n_bad = int((~df["valid"]).sum())
    if n_bad:
        print(f"dropping {n_bad} invalid records")
    df = df[df["valid"]].reset_index(drop=True)
    unknown = sorted({g for s in df["flagged"] for g in s} - set(GROUPS))
    if unknown:
        print(f"note: flags on groups outside {GROUPS}: {unknown}")
    df["trace_level"] = pd.Categorical(df["trace_level"], TRACE_LEVELS, ordered=True)
    df["state"] = pd.Categorical(df["state"], STATES, ordered=True)
    return df


def fmt(s) -> str:
    return "|".join(sorted(s))


def retest_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    l1 = df[df["trace_level"] == "none"]
    a = l1[l1["state"] == "omission_zero"].set_index(["reader", "row"])
    b = l1[l1["state"] == "truthful_twin"].set_index(["reader", "row"])
    pairs = []
    for key in sorted(set(a.index) & set(b.index)):
        ra, rb = a.loc[key], b.loc[key]
        rec = {
            "record": "pair",
            "reader": key[0],
            "row": key[1],
            "target_group": ra["target_group"],
            "sha_a": ra["reader_input_sha"],
            "sha_b": rb["reader_input_sha"],
            "same_input": ra["reader_input_sha"] == rb["reader_input_sha"],
            "flags_a": fmt(ra["flagged"]),
            "flags_b": fmt(rb["flagged"]),
            "identical_flags": ra["flagged"] == rb["flagged"],
        }
        for g in GROUPS:
            rec[f"agree_{g}"] = (g in ra["flagged"]) == (g in rb["flagged"])
        pairs.append(rec)
    pairs = pd.DataFrame(pairs)

    summ = []
    for reader, d in pairs.groupby("reader", sort=True):
        # agreement is only a repeat-run measure where the two inputs are the same text
        same = d[d["same_input"]]
        n = len(same)
        k = int(same["identical_flags"].sum())
        rec = {
            "record": "summary",
            "reader": reader,
            "pairs": len(d),
            "identical_inputs": n,
            "identical_flag_sets": k,
            "either_run_flagged": int(((same["flags_a"] != "") | (same["flags_b"] != "")).sum()),
        }
        rec.update({k_.replace("agree_n", "agree_pairs"): v for k_, v in rate_cols("agree", k, n).items()})
        for g in GROUPS:
            kg = int(same[f"agree_{g}"].sum())
            lo, hi = wilson(kg, n)
            rec[f"agree_{g}_pairs"] = kg
            rec[f"agree_{g}_share"] = kg / n if n else float("nan")
            rec[f"agree_{g}_lo"] = lo
            rec[f"agree_{g}_hi"] = hi
        summ.append(rec)
    return pairs, pd.DataFrame(summ)


def group_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (reader, lvl, st), d in df.groupby(["reader", "trace_level", "state"], sort=True, observed=True):
        for g in GROUPS:
            hit = d["flagged"].map(lambda s, g=g: g in s)
            for label, mask in [
                ("all", pd.Series(True, index=d.index)),
                ("target", d["target_group"] == g),
                ("non_target", d["target_group"] != g),
            ]:
                n = int(mask.sum())
                if n == 0:
                    continue
                k = int(hit[mask].sum())
                lo, hi = wilson(k, n)
                out.append(
                    {
                        "reader": reader,
                        "trace_level": lvl,
                        "state": st,
                        "group": g,
                        "group_is_target": label,
                        "n_inputs": n,
                        "n_flagged": k,
                        "share": k / n,
                        "share_lo": lo,
                        "share_hi": hi,
                    }
                )
    return pd.DataFrame(out)


def false_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # twins: every flag is a false flag. omissions: only flags off the target group count.
    d["extra"] = [
        s if not om else s - {t} for s, om, t in zip(d["flagged"], d["is_omission"], d["target_group"])
    ]
    d["any_flag"] = d["extra"].map(bool)
    d["income_flag"] = d["extra"].map(lambda s: bool(s & INCOME))
    d["expense_only_flag"] = d["any_flag"] & ~d["income_flag"]
    d["scope"] = d["state"].astype(str).map(
        lambda s: "twin_all_flags" if s == "truthful_twin" else f"{s}_non_target_flags"
    )
    pooled = d[d["is_omission"]].assign(scope="omission_pooled_non_target_flags")
    d = pd.concat([d, pooled], ignore_index=True)

    out = []
    levels = [(lv, d["trace_level"] == lv) for lv in TRACE_LEVELS] + [("all", pd.Series(True, index=d.index))]
    for (reader, scope), ds in d.groupby(["reader", "scope"], sort=True):
        for lvl, mask in levels:
            x = ds[mask.loc[ds.index]]
            n = len(x)
            if n == 0:
                continue
            rec = {"reader": reader, "scope": scope, "trace_level": lvl, "n_inputs": n}
            rec.update(rate_cols("any_flag", int(x["any_flag"].sum()), n))
            rec.update(rate_cols("income_flag", int(x["income_flag"].sum()), n))
            rec.update(rate_cols("shelter_utilities_only", int(x["expense_only_flag"].sum()), n))
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve() if args.root else find_root(Path(__file__).resolve().parent)
    out = args.out if args.out else root / "results" / "05-design1-reader-omissions"
    out.mkdir(parents=True, exist_ok=True)

    df = load(root / "experiments" / "design1" / "results.jsonl")
    print(f"{len(df)} valid records, readers: {sorted(df['reader'].unique())}")

    pairs, summ = retest_table(df)
    pd.concat([pairs, summ], ignore_index=True).to_csv(out / "retest-level1.csv", index=False)
    matrix = group_matrix(df)
    matrix.to_csv(out / "group-flag-matrix.csv", index=False)
    ff = false_flags(df)
    ff.to_csv(out / "false-flags-by-group.csv", index=False)

    print("\nRepeat runs at trace level none (omission_zero vs truthful_twin):")
    for _, s in summ.iterrows():
        print(
            f"  {s['reader']}: {s['identical_inputs']}/{s['pairs']} pairs have the same input; "
            f"identical flag sets {s['identical_flag_sets']}/{s['identical_inputs']} "
            f"(95% CI {s['agree_lo']:.2f}-{s['agree_hi']:.2f}); "
            f"{s['either_run_flagged']} pairs had a flag in either run; per group "
            + ", ".join(f"{g} {s[f'agree_{g}_pairs']}/{s['identical_inputs']}" for g in GROUPS)
        )
    diff = pairs[~pairs["identical_flags"]]
    for _, p in diff.iterrows():
        print(f"    differs: {p['reader']} row {p['row']}: [{p['flags_a']}] vs [{p['flags_b']}]")

    print("\nTruthful-twin false flags by level (any / income group / shelter-utilities only, out of n):")
    tw = ff[ff["scope"] == "twin_all_flags"]
    for _, r in tw.iterrows():
        print(
            f"  {r['reader']} {r['trace_level']}: {r['any_flag_n']} / {r['income_flag_n']} / "
            f"{r['shelter_utilities_only_n']} of {r['n_inputs']}"
        )
    print("\nOmission inputs, flags on non-target groups (any / income group / shelter-utilities only, out of n):")
    om = ff[ff["scope"] == "omission_pooled_non_target_flags"]
    for _, r in om.iterrows():
        print(
            f"  {r['reader']} {r['trace_level']}: {r['any_flag_n']} / {r['income_flag_n']} / "
            f"{r['shelter_utilities_only_n']} of {r['n_inputs']}"
        )
    print(f"\nwrote {out}/retest-level1.csv, group-flag-matrix.csv, false-flags-by-group.csv")


if __name__ == "__main__":
    main()
