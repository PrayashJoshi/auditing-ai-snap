"""Amount-only arm for the Design 1 v2 reader experiment: inputs. Zero calls.

    PYTHONPATH=src python experiments/design1_v2_amount/build_amount.py

The 48 frozen `stray_ambiguous` inputs (households whose target income group is $0, QC-verified, whose records
show one small credit labelled "ACH CREDIT ELECTRONIC DEPOSIT") are copied with one change: the amount of that
credit. The new amounts come from the 24 omission households' verified monthly target amounts
(`verified_monthly[target_group]` in the frozen omission inputs, $300 to $3,581). Assignment, fixed before any
call: sort the 24 omission households by row and take each one's amount; sort the 48 stray inputs by row; stray
input i gets amount[i mod 24], so each amount is used twice. The original credit's cents are kept
(e.g. $145.02 with $600 becomes $600.02), so the credit's size changes and its roundness does not.

Each new input keeps cls = zero_stray, target_group and every other field; evidence_state becomes
`stray_amount` and input_id `{row}|zero_stray|stray_amount`. The packet's credit amount is changed and the reader
text is re-rendered with the frozen renderer, so `reader_input` and `reader_input_sha` change with it. The hard
check asserts that each new reader text differs from its frozen counterpart in exactly one line, the credit line,
and only in the amount on it.

The frozen builder is imported read-only and its `build()` is run in memory (its `main` is never run); the
rebuilt inputs must equal `inputs.jsonl` byte for byte, the check the frozen runner makes. The in-memory packets
are used because the stored ones were serialized with sorted keys, which reorders document fields and so cannot
be re-rendered to the frozen text. Nothing in `experiments/design1_v2/` is written by this file. Writes amount-inputs.jsonl and amount-map.csv next to it.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing the frozen modules must not leave bytecode files behind

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from snapqc import llm
from snapqc import evidence_states as E

HERE = Path(__file__).resolve().parent
FROZEN_DIR = HERE.parent / "design1_v2"
_spec = importlib.util.spec_from_file_location("build_inputs", FROZEN_DIR / "build_inputs.py")
build_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_inputs)  # module-level code only; build() and main() are not called
run_design1 = build_inputs.run_design1

AMOUNT_INPUTS_PATH = HERE / "amount-inputs.jsonl"
AMOUNT_MAP_PATH = HERE / "amount-map.csv"
SOURCE_STATE = "stray_ambiguous"
NEW_STATE = "stray_amount"
EXPECT_INPUTS_SHA_PREFIX = "9b84d771"
N_STRAY, N_OMISSION = 48, 24


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money(v: float) -> str:
    return f"${v:,.2f}"


def stray_doc(x: dict) -> dict:
    docs = [d for d in x["packet"]["documents"]
            if d["kind"] == "transaction" and d["data"]["Description"] == E.AMBIGUOUS_LABEL]
    assert len(docs) == 1, (x["input_id"], len(docs))
    return docs[0]


def omission_amounts(inputs: list[dict]) -> list[tuple[int, str, float]]:
    """(row, target group, verified monthly target amount) for the 24 omission households, by row. The amount is
    the same in every evidence state of a household; that is asserted."""
    by_row: dict[int, set] = {}
    for x in inputs:
        if x["cls"] == "omission":
            by_row.setdefault(x["row"], set()).add((x["target_group"], x["verified_monthly"][x["target_group"]]))
    assert len(by_row) == N_OMISSION, len(by_row)
    out = []
    for row in sorted(by_row):
        assert len(by_row[row]) == 1, (row, by_row[row])
        tg, amt = next(iter(by_row[row]))
        assert amt > 0, row
        out.append((row, tg, float(amt)))
    return out


def line_diff(a: str, b: str) -> list[tuple[int, str, str]]:
    la, lb = a.splitlines(), b.splitlines()
    assert len(la) == len(lb), "line counts differ"
    return [(i, u, v) for i, (u, v) in enumerate(zip(la, lb)) if u != v]


def build(inputs: list[dict]) -> tuple[list[dict], pd.DataFrame]:
    stray = sorted([x for x in inputs if x["cls"] == "zero_stray" and x["evidence_state"] == SOURCE_STATE],
                   key=lambda x: x["row"])
    assert len(stray) == N_STRAY and len({x["row"] for x in stray}) == N_STRAY
    amounts = omission_amounts(inputs)
    out, rows = [], []
    for i, x in enumerate(stray):
        src_row, src_tg, amt = amounts[i % N_OMISSION]
        # The frozen renderer reproduces the frozen text from the in-memory packet: the re-render is like for like.
        assert run_design1.render_reader_input(x["app_monthly"], x["packet"], x["target_group"], "none") == x["reader_input"], x["input_id"]
        y = copy.deepcopy(x)
        d = stray_doc(y)
        old = float(d["data"]["Amount"])
        cents = round(old - int(old), 2)
        new_amt = round(amt + cents, 2)
        d["data"]["Amount"] = new_amt
        y["evidence_state"] = NEW_STATE
        y["input_id"] = f"{x['row']}|{x['cls']}|{NEW_STATE}"
        y["reader_input"] = run_design1.render_reader_input(y["app_monthly"], y["packet"], y["target_group"], "none")
        y["reader_input_sha"] = llm.sha(y["reader_input"])
        out.append(y)
        rows.append({"stray_row": x["row"], "stray_input_id": x["input_id"], "new_input_id": y["input_id"],
                     "stray_target_group": x["target_group"], "doc_id": d["id"], "original_amount": old,
                     "new_amount": new_amt, "omission_monthly_amount": amt, "source_omission_row": src_row,
                     "source_omission_target_group": src_tg})
    return out, pd.DataFrame(rows)


def check(new: list[dict], inputs: list[dict], amap: pd.DataFrame) -> list[tuple[int, str, str]]:
    """Hard check: each new input differs from its frozen counterpart only in the stray credit's amount.
    Returns the first input's diff as an example."""
    frozen_by_row = {x["row"]: x for x in inputs if x["evidence_state"] == SOURCE_STATE}
    frozen_ids = {x["input_id"] for x in inputs}
    assert len({y["input_id"] for y in new}) == N_STRAY and not frozen_ids & {y["input_id"] for y in new}
    allowed = {"input_id", "evidence_state", "packet", "reader_input", "reader_input_sha"}
    example = None
    for y, m in zip(new, amap.itertuples()):
        x = frozen_by_row[y["row"]]
        # Record fields: only the id, the state and what carries the amount differ.
        assert set(y) == set(x), y["input_id"]
        changed = {k for k in x if x[k] != y[k]}
        assert changed == allowed, (y["input_id"], changed)
        assert y["cls"] == "zero_stray" and y["target_group"] == x["target_group"]
        # Packet: one document changed, and in that document only the amount.
        dx, dy = x["packet"]["documents"], y["packet"]["documents"]
        assert {k: v for k, v in x["packet"].items() if k != "documents"} == \
               {k: v for k, v in y["packet"].items() if k != "documents"}
        ddiff = [(u, v) for u, v in zip(dx, dy) if u != v]
        assert len(dx) == len(dy) and len(ddiff) == 1, y["input_id"]
        u, v = ddiff[0]
        assert u["id"] == v["id"] == m.doc_id and u["kind"] == v["kind"] == "transaction"
        assert {k: w for k, w in u["data"].items() if k != "Amount"} == {k: w for k, w in v["data"].items() if k != "Amount"}
        assert v["data"]["Description"] == E.AMBIGUOUS_LABEL and v["data"]["Amount"] == m.new_amount
        # Reader text: exactly one line differs, the credit line, and only in its amount.
        diff = line_diff(x["reader_input"], y["reader_input"])
        if len(diff) != 1:
            print(f"STOP {y['input_id']}: {len(diff)} reader lines change:")
            for i, a, b in diff:
                print(f"  line {i}: {a!r} -> {b!r}")
            raise AssertionError("reader text differs in more than the credit line")
        _, a, b = diff[0]
        assert a.startswith(f"[{m.doc_id}] ") and a.endswith(E.AMBIGUOUS_LABEL), a
        assert money(m.original_amount) in a and money(m.new_amount) in b, (a, b)
        assert a.replace(money(m.original_amount), money(m.new_amount), 1) == b, (a, b)
        # Same leak scans as the frozen preflight.
        hits = [h.group(0) for h in run_design1.LEAK_RE.finditer(y["reader_input"])] + \
               [h.group(0) for h in build_inputs.EXTRA_LEAK_RE.finditer(y["reader_input"])]
        assert not hits, (y["input_id"], hits)
        assert y["reader_input_sha"] == llm.sha(y["reader_input"])
        if example is None:
            example = diff
    # Assignment rule: each omission amount used exactly twice, at positions i and i + 24, with the original cents.
    assert amap["omission_monthly_amount"].value_counts().ge(2).all()
    cents = (amap["original_amount"] - amap["original_amount"].astype(int)).round(2)
    assert ((amap["new_amount"] - amap["omission_monthly_amount"]).round(2) == cents).all()
    assert (amap["omission_monthly_amount"] % 1 == 0).all()
    assert list(amap["source_omission_row"][:N_OMISSION]) == list(amap["source_omission_row"][N_OMISSION:])
    assert amap["source_omission_row"].value_counts().eq(2).all() and amap["source_omission_row"].nunique() == N_OMISSION
    return example


def serialize(inputs: list[dict]) -> str:
    return build_inputs.serialize(inputs)


def main() -> int:
    sha_frozen = file_sha256(build_inputs.INPUTS_PATH)
    assert sha_frozen.startswith(EXPECT_INPUTS_SHA_PREFIX), f"inputs.jsonl sha256 {sha_frozen} differs from the frozen inputs"
    inputs, _, _ = build_inputs.build()
    sha_built = hashlib.sha256(serialize(inputs).encode()).hexdigest()
    assert sha_built == sha_frozen, f"inputs.jsonl ({sha_frozen}) differs from a fresh build ({sha_built})"
    on_disk = run_design1.read_jsonl(build_inputs.INPUTS_PATH)
    assert [x["input_id"] for x in on_disk] == [x["input_id"] for x in inputs] and len(inputs) == build_inputs.N_INPUTS
    new, amap = build(inputs)
    example = check(new, inputs, amap)
    new2, amap2 = build(inputs)
    text = serialize(new)
    assert text == serialize(new2) and amap.equals(amap2), "build is not deterministic"
    AMOUNT_INPUTS_PATH.write_text(text)
    amap.to_csv(AMOUNT_MAP_PATH, index=False)
    assert file_sha256(build_inputs.INPUTS_PATH) == sha_frozen, "frozen inputs changed"

    print(f"frozen inputs.jsonl sha256 {sha_frozen}")
    print(f"amount inputs: {len(new)}, target groups {amap['stray_target_group'].value_counts().to_dict()}")
    print(f"original amounts ${amap['original_amount'].min():,.2f} to ${amap['original_amount'].max():,.2f}; "
          f"new amounts ${amap['new_amount'].min():,.2f} to ${amap['new_amount'].max():,.2f} "
          f"(omission monthly amounts plus the original cents; {int((amap['new_amount'] % 1 != 0).sum())} of "
          f"{len(amap)} with cents)")
    print("hard check passed: every input differs from its stray_ambiguous counterpart in exactly one reader line, "
          "the credit line, and only in its amount")
    i, a, b = example[0]
    print(f"example ({new[0]['input_id']}), line {i}:\n  - {a}\n  + {b}")
    print(f"amount-inputs.jsonl sha256 {file_sha256(AMOUNT_INPUTS_PATH)} (two builds equal)")
    print(f"wrote {AMOUNT_INPUTS_PATH.name}, {AMOUNT_MAP_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
