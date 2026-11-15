"""Scores a reader results file with the frozen Design 1 v2 scorer's functions, for reader and input sets the
frozen scorer does not name. Zero calls. Nothing under `experiments/design1_v2/` is written: the frozen scorer
is loaded as a module and its `main` is never run, because `main` reads the frozen results path and writes
into the frozen directory.

The scoring functions (`reader_flags`, `cell_scores`, `repeat_agreement`, `hypotheses`, `score`) and the
runner's `latest` and `read_jsonl` are the frozen ones, called unchanged. Latest record per job is used, as in
the frozen scorer.

    PYTHONPATH=src python experiments/design1_v2_added/score_results.py \\
        --inputs experiments/design1_v2/inputs.jsonl \\
        --results experiments/design1_v2_added/added-results.jsonl \\
        --validity experiments/design1_v2_added/added-dryrun-validity.csv \\
        --out-dir experiments/design1_v2_added --prefix added-

Checks before anything is scored or written (each exits non-zero with examples when it fails):
  - input ids: every record's `input_id` must be in --inputs (--allow-unknown-ids drops the others);
  - labels: for every kept record, `cls`, `evidence_state`, `target_group` and `row` must equal the input's;
  - completeness: for each reader x repeat x (cls, evidence_state) cell, the number of households with a
    record must equal the number of inputs in that cell. Readers expected are --readers, else the retained
    readers of --validity, else the readers found in the results. Repeats expected are --repeats (default
    "1,2"), with per-reader overrides by --reader-repeats READER=1,2 (repeatable). On a shortfall the script
    prints the shortfall and exits without writing, unless --allow-partial is given: then every output file
    name starts with `partial-` (before --prefix) and hypotheses are not written.

--validity PATH is a dry-run validity CSV with columns `reader` and `retained`; only readers with retained
True are kept, and the retained and dropped readers are printed.

Outputs (in --out-dir, names start with --prefix): `scores.csv` (repeat 1 cells, plus repeat 2 cells when
present), `repeat-agreement.csv` (only when a repeat 2 exists), `flags.csv` (per-household flags, one row per
reader x repeat x household), `hypotheses.csv` (only when the run is complete and every cell the three reader
hypotheses need is present for every reader scored; column `status` is `exploratory`), and
`paired-comparison.csv` (only with --compare-results). The hypothesis 3 summary row counts "k of N" over
whichever readers are passed, which is not necessarily the frozen reader set. The frozen comparators are left
out unless --include-comparators is given; the frozen `scores.csv` is reproduced exactly (same rows, same
order) only with --include-comparators. Cells outside the frozen design (for example a new evidence state) are
scored like any other cell and simply do not feed the hypotheses.

Paired comparison, for an arm that re-runs the same households under a new evidence state:
    --compare-results PATH --compare-inputs PATH --compare-cell CLS/STATE [--compare-repeat N]
The per-household flags of that cell are computed from the comparison files with the frozen functions (latest
record per job, frozen validity rule) at repeat N (default 1) and joined on `row` with this run's repeat 1
flags, per reader, keeping households valid in both. For each flag column (target_flag, income_flag,
any_flag) the file gives n_pairs, k_compare, rate_compare, k_this, rate_this, difference (this minus compare),
b (flagged in the comparison only), c (flagged in this run only) and the exact two-sided McNemar p-value (a
binomial test of b against c at p = 0.5).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # loading the frozen scorer must not leave bytecode files behind

import argparse
import importlib.util
import math
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FROZEN_DIR = HERE.parent / "design1_v2"
_spec = importlib.util.spec_from_file_location("score_design1_v2", FROZEN_DIR / "score_design1_v2.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)  # module-level code only; main() is not called
runner = scorer.runner
run_design1 = scorer.run_design1

LABEL_KEYS = ("cls", "evidence_state", "target_group", "row")
FLAG_COLS = ("target_flag", "income_flag", "any_flag")


def hypothesis_cells_missing(scores: pd.DataFrame, readers: list[str]) -> list[tuple]:
    """Cells (reader, class, state) that the hypotheses read and that are absent or have no valid output."""
    s1 = scores[scores["repeat"] == 1]
    have = {(r.reader, r.cls, r.evidence_state): r.n for r in s1.itertuples()}
    need = [scorer.HYP1_CLS, scorer.HYP1_NEG, scorer.HYP2_CELL, scorer.HYP3_CELL]
    return [(r, *c) for r in readers for c in need if have.get((r, *c), 0) == 0]


def reader_order(m: str) -> int:
    return runner.MODELS.index(m) if m in runner.MODELS else 99


def parse_repeats(text: str) -> list[int]:
    return sorted({int(t) for t in text.split(",") if t.strip()})


def label_mismatches(records: list[dict], by_id: dict[str, dict]) -> list[tuple]:
    bad = []
    for r in records:
        x = by_id[r["input_id"]]
        for k in LABEL_KEYS:
            if r.get(k) != x[k]:
                bad.append((r["input_id"], r["reader"], r.get("repeat"), k, r.get(k), x[k]))
    return bad


def completeness_shortfall(records: list[dict], inputs: list[dict], readers: list[str],
                           repeats_for: dict[str, list[int]]) -> list[dict]:
    """Per reader x repeat x (cls, evidence_state) cell: households with a record against inputs in the cell."""
    want = Counter((x["cls"], x["evidence_state"]) for x in inputs)
    cell_of = {x["input_id"]: (x["cls"], x["evidence_state"]) for x in inputs}
    have: dict[tuple, set] = {}
    for r in records:
        have.setdefault((r["reader"], int(r["repeat"]), *cell_of[r["input_id"]]), set()).add(r["input_id"])
    short = []
    for m in readers:
        for rep in repeats_for[m]:
            for (cls, st), n in sorted(want.items()):
                k = len(have.get((m, rep, cls, st), ()))
                if k != n:
                    short.append({"reader": m, "repeat": rep, "cls": cls, "evidence_state": st,
                                  "expected": n, "have": k, "missing": n - k})
    return short


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p: binomial test of b against c discordant pairs at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
        return min(1.0, 2 * tail)


def paired_comparison(this_flags: pd.DataFrame, readers: list[str], cmp_inputs_path: Path, cmp_results_path: Path,
                      cell: tuple[str, str], cmp_repeat: int) -> pd.DataFrame:
    cmp_inputs = [x for x in run_design1.read_jsonl(cmp_inputs_path)
                  if (x["cls"], x["evidence_state"]) == cell]
    if not cmp_inputs:
        raise SystemExit(f"No inputs in {cmp_inputs_path.name} for cell {cell[0]}/{cell[1]}.")
    ids = {x["input_id"] for x in cmp_inputs}
    cmp_records = [r for r in run_design1.latest(run_design1.read_jsonl(cmp_results_path)).values()
                   if r["input_id"] in ids and int(r["repeat"]) == cmp_repeat and r["reader"] in readers]
    cflags = scorer.reader_flags(cmp_inputs, cmp_records) if cmp_records else pd.DataFrame(columns=this_flags.columns)
    rows = []
    for m in readers:
        t = this_flags[(this_flags["reader"] == m) & (this_flags["repeat"] == 1)]
        if t["row"].duplicated().any():
            raise SystemExit(f"Reader {m}: this run has more than one repeat 1 record per household row, so the "
                             f"paired comparison cannot be joined on row.")
        c = cflags[cflags["reader"] == m]
        if c["row"].duplicated().any():
            raise SystemExit(f"Reader {m}: the comparison has more than one record per household row.")
        both = t.merge(c, on="row", suffixes=("_this", "_cmp"))
        both = both[both["valid_this"] & both["valid_cmp"]]
        n = len(both)
        for col in FLAG_COLS:
            kc, kt = int(both[f"{col}_cmp"].sum()), int(both[f"{col}_this"].sum())
            b = int(((both[f"{col}_cmp"] == 1) & (both[f"{col}_this"] == 0)).sum())
            cc = int(((both[f"{col}_cmp"] == 0) & (both[f"{col}_this"] == 1)).sum())
            rc, rt = (kc / n, kt / n) if n else (float("nan"), float("nan"))
            rows.append({"reader": m, "flag": col, "compare_cell": f"{cell[0]}/{cell[1]}", "compare_repeat": cmp_repeat,
                         "n_pairs": n, "k_compare": kc, "rate_compare": rc, "k_this": kt, "rate_this": rt,
                         "difference": rt - rc, "b": b, "c": cc,
                         "mcnemar_p": mcnemar_exact_p(b, cc) if n else float("nan")})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--readers", default=None, help="comma-separated; default: every reader in the results file")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--include-comparators", action="store_true",
                    help="also score the frozen deterministic comparators on these inputs")
    ap.add_argument("--validity", type=Path, default=None,
                    help="dry-run validity CSV (columns reader, retained): keep only readers with retained True")
    ap.add_argument("--repeats", default="1,2", help="repeats expected for every reader (default 1,2)")
    ap.add_argument("--reader-repeats", action="append", default=[], metavar="READER=1,2",
                    help="repeats expected for one reader, overriding --repeats; repeatable")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write partial- files when records are missing; hypotheses are never written then")
    ap.add_argument("--allow-unknown-ids", action="store_true",
                    help="drop records whose input_id is not in --inputs instead of failing")
    ap.add_argument("--compare-results", type=Path, default=None)
    ap.add_argument("--compare-inputs", type=Path, default=None)
    ap.add_argument("--compare-cell", default=None, help="CLS/STATE of the comparison cell")
    ap.add_argument("--compare-repeat", type=int, default=1)
    args = ap.parse_args()

    cmp_args = (args.compare_results, args.compare_inputs, args.compare_cell)
    if any(cmp_args) and not all(cmp_args):
        ap.error("--compare-results, --compare-inputs and --compare-cell are needed together")
    cmp_cell = tuple(args.compare_cell.split("/", 1)) if args.compare_cell else None
    if cmp_cell is not None and len(cmp_cell) != 2:
        ap.error("--compare-cell must look like CLS/STATE")

    inputs = run_design1.read_jsonl(args.inputs)
    by_id = {x["input_id"]: x for x in inputs}
    records = list(run_design1.latest(run_design1.read_jsonl(args.results)).values())

    off = [r for r in records if r["input_id"] not in by_id]
    if off:
        examples = ", ".join(sorted({r["input_id"] for r in off})[:5])
        if not args.allow_unknown_ids:
            print(f"ERROR: {len(off)} records have an input_id that is not in {args.inputs.name}, "
                  f"for example: {examples}. Nothing written. (--allow-unknown-ids drops them.)")
            return 2
        print(f"Dropped {len(off)} records whose input is not in {args.inputs.name}, for example: {examples}.")
        records = [r for r in records if r["input_id"] in by_id]

    want = [m.strip() for m in args.readers.split(",") if m.strip()] if args.readers else None
    if want:
        records = [r for r in records if r["reader"] in want]

    retained = None
    if args.validity is not None:
        v = pd.read_csv(args.validity).set_index("reader")
        retained = [m for m in v.index if bool(v.loc[m, "retained"])]
        dropped = [m for m in v.index if m not in retained]
        print("Retained readers:", ", ".join(retained) or "none")
        print("Dropped readers:", ", ".join(dropped) or "none")
        records = [r for r in records if r["reader"] in retained]

    if not records:
        print("No reader results for these inputs and readers: nothing to score.")
        return 1

    if want is not None:
        expected = [m for m in want if retained is None or m in retained]
    elif retained is not None:
        expected = list(retained)
    else:
        expected = sorted({r["reader"] for r in records}, key=reader_order)
    expected = sorted(dict.fromkeys(expected), key=reader_order)
    readers = [m for m in expected if any(r["reader"] == m for r in records)]
    absent = [m for m in expected if m not in readers]
    if absent:
        print("Readers with no records:", ", ".join(absent))

    bad = label_mismatches(records, by_id)
    if bad:
        print(f"ERROR: {len(bad)} label mismatches between records and inputs (input_id, reader, repeat, field, "
              f"record value, input value). Nothing written. First {min(5, len(bad))}:")
        for b in bad[:5]:
            print("   ", b)
        return 2

    repeats_for = {m: parse_repeats(args.repeats) for m in expected}
    for item in args.reader_repeats:
        name, _, reps = item.partition("=")
        if name not in repeats_for or not reps:
            ap.error(f"--reader-repeats {item!r}: expected READER=1,2 for a reader that is expected")
        repeats_for[name] = parse_repeats(reps)
    short = completeness_shortfall(records, inputs, expected, repeats_for)
    partial = bool(short)
    if partial:
        total = sum(s["missing"] for s in short)
        print("=" * 100)
        print(f"INCOMPLETE: {total} household records missing across {len(short)} reader x repeat x cell groups "
              f"(expected readers: {', '.join(expected)}).")
        print(pd.DataFrame(short).to_string(index=False))
        print("=" * 100)
        if not args.allow_partial:
            print("Nothing written. Re-run when the run is complete, or pass --allow-partial for partial- files.")
            return 3
        print("--allow-partial: writing partial- files; hypotheses are not written.")

    saved = scorer.comparator_flags
    if not args.include_comparators:
        # the frozen `score` then gets a comparator frame with the flag columns and no rows
        def no_comparators(inputs: list[dict], hh: dict) -> pd.DataFrame:
            row = scorer._flag_row(inputs[0], "none", 1, True, set())
            return pd.DataFrame([row]).iloc[0:0]
        scorer.comparator_flags = no_comparators
    try:
        # readers=[] keeps the frozen `score` from computing hypotheses, which raise on a missing cell
        out = scorer.score(inputs, records, scorer.households(), readers=[])
    finally:
        scorer.comparator_flags = saved
    scores, agreement, flags = out["scores"], out["agreement"], out["flags"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = ("partial-" if partial else "") + args.prefix
    scores.to_csv(args.out_dir / f"{p}scores.csv", index=False)
    flags.to_csv(args.out_dir / f"{p}flags.csv", index=False)
    cols = ["reader", "repeat", "cls", "evidence_state", "metric", "n", "invalid", "target_k", "target_rate",
            "target_lo", "target_hi", "income_k", "any_k"]
    print(scores[cols].to_string(index=False))
    if (scores["repeat"] == 2).any() and not agreement.empty:
        agreement.to_csv(args.out_dir / f"{p}repeat-agreement.csv", index=False)
        print(agreement.to_string(index=False))
    else:
        print("No repeat 2 records: repeat agreement not written.")

    if cmp_cell is not None:
        paired = paired_comparison(flags, readers, args.compare_inputs, args.compare_results, cmp_cell,
                                   args.compare_repeat)
        paired.to_csv(args.out_dir / f"{p}paired-comparison.csv", index=False)
        print(paired.to_string(index=False))

    if partial:
        print("Hypotheses not written: the run is incomplete.")
        return 0
    gaps = hypothesis_cells_missing(scores, readers)
    if gaps:
        print(f"Hypotheses not written: {len(gaps)} needed (reader, class, state) cells are absent or empty, "
              f"for example {gaps[0]}.")
    else:
        h = scorer.hypotheses(scores, readers)
        h["status"] = "exploratory"
        h.to_csv(args.out_dir / f"{p}hypotheses.csv", index=False)
        print(h.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
