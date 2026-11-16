"""Clear-label twin of the amount-only arm, Design 1 v2 reader experiment: inputs. Zero calls.

    PYTHONPATH=src python experiments/design1_v2_amount_clear/build_clear.py

Completes a 2 x 2 on the 48 `zero_stray` households (target income group $0, QC-verified, one small
non-recurring credit): label (clear / ambiguous) x amount (small / omission-sized). Three cells exist: frozen
`stray_clear` (clear, small), frozen `stray_ambiguous` (ambiguous, small) and the amount arm's `stray_amount`
(ambiguous, omission-sized). This file builds the fourth, clear label with omission-sized amount.

The 48 frozen `stray_clear` inputs are copied with one change: the amount of the stray credit (the one
transaction whose label starts "TRANSFER FROM SAVINGS"). The new amount is the amount arm's `new_amount` for the
same row in `amount-map.csv` (read only, SHA-256 asserted), so each twin carries the same amount as its
`stray_amount` counterpart. Each new input keeps cls = zero_stray, target_group and every other field;
evidence_state becomes `stray_clear_amount` and input_id `{row}|zero_stray|stray_clear_amount`. The reader text
is re-rendered with the frozen renderer, so `reader_input` and `reader_input_sha` change with it.

Hard checks: each twin reader text differs (a) from its frozen `stray_clear` counterpart in exactly one line,
only in the amount, and (b) from its `stray_amount` counterpart in exactly one line, only in the label. The same
48 rows are in all four cells.

The frozen builder and the amount arm's builder are imported read-only and their `build()` functions run in
memory (no `main` is run). The rebuilt frozen inputs must equal `inputs.jsonl` byte for byte and the rebuilt
amount inputs must equal `amount-inputs.jsonl` byte for byte. Nothing outside this directory is written.
Writes clear-inputs.jsonl next to this file.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing the frozen modules must not leave bytecode files behind

import copy
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from snapqc import llm
from snapqc import evidence_states as E

HERE = Path(__file__).resolve().parent
AMOUNT_DIR = HERE.parent / "design1_v2_amount"
_spec = importlib.util.spec_from_file_location("build_amount", AMOUNT_DIR / "build_amount.py")
build_amount = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_amount)  # module-level code only; build() is called in memory, main() is not
build_inputs = build_amount.build_inputs
run_design1 = build_amount.run_design1

CLEAR_INPUTS_PATH = HERE / "clear-inputs.jsonl"
AMOUNT_MAP_PATH = AMOUNT_DIR / "amount-map.csv"
AMOUNT_INPUTS_PATH = AMOUNT_DIR / "amount-inputs.jsonl"
SOURCE_STATE = "stray_clear"
NEW_STATE = "stray_clear_amount"
EXPECT_INPUTS_SHA_PREFIX = "9b84d771"
EXPECT_AMOUNT_MAP_SHA = "212f603759d749fb4f9f144af9851cdfeeaf8185f5ceb500df09a770f84eb46b"
EXPECT_AMOUNT_INPUTS_SHA = "e0dac1fc12643c2da8acd9fe05a7b765974682b831080175ab1b219f24841f7e"
N_STRAY = 48

file_sha256 = build_amount.file_sha256
money = build_amount.money
line_diff = build_amount.line_diff


def clear_doc(x: dict, doc_id: str) -> dict:
    docs = [d for d in x["packet"]["documents"]
            if d["kind"] == "transaction" and d["data"]["Description"].startswith(E.SAVINGS_PREFIX)]
    assert len(docs) == 1, (x["input_id"], len(docs))
    assert docs[0]["id"] == doc_id, (x["input_id"], docs[0]["id"], doc_id)
    return docs[0]


def build(inputs: list[dict], amap: pd.DataFrame) -> list[dict]:
    clear = sorted([x for x in inputs if x["cls"] == "zero_stray" and x["evidence_state"] == SOURCE_STATE],
                   key=lambda x: x["row"])
    assert len(clear) == N_STRAY and len({x["row"] for x in clear}) == N_STRAY
    by_row = {int(m.stray_row): m for m in amap.itertuples()}
    out = []
    for x in clear:
        m = by_row[x["row"]]
        # The frozen renderer reproduces the frozen text from the in-memory packet: the re-render is like for like.
        assert run_design1.render_reader_input(x["app_monthly"], x["packet"], x["target_group"], "none") == x["reader_input"], x["input_id"]
        y = copy.deepcopy(x)
        d = clear_doc(y, m.doc_id)
        assert float(d["data"]["Amount"]) == float(m.original_amount), (x["input_id"], d["data"]["Amount"])
        d["data"]["Amount"] = float(m.new_amount)
        y["evidence_state"] = NEW_STATE
        y["input_id"] = f"{x['row']}|{x['cls']}|{NEW_STATE}"
        y["reader_input"] = run_design1.render_reader_input(y["app_monthly"], y["packet"], y["target_group"], "none")
        y["reader_input_sha"] = llm.sha(y["reader_input"])
        out.append(y)
    return out


def one_line(a_text: str, b_text: str, input_id: str) -> tuple[int, str, str]:
    diff = line_diff(a_text, b_text)
    if len(diff) != 1:
        print(f"STOP {input_id}: {len(diff)} reader lines change:")
        for i, a, b in diff:
            print(f"  line {i}: {a!r} -> {b!r}")
        raise AssertionError("reader text differs in more than one line")
    return diff[0]


def check(new: list[dict], inputs: list[dict], amount_new: list[dict], amap: pd.DataFrame):
    """Hard checks (a) and (b). Returns one example diff for each."""
    clear_by_row = {x["row"]: x for x in inputs if x["cls"] == "zero_stray" and x["evidence_state"] == SOURCE_STATE}
    amb_rows = {x["row"] for x in inputs if x["cls"] == "zero_stray" and x["evidence_state"] == "stray_ambiguous"}
    amount_by_row = {x["row"]: x for x in amount_new}
    new_rows = {y["row"] for y in new}
    # Same 48 households in all four cells.
    assert len(new_rows) == N_STRAY
    assert new_rows == set(clear_by_row) == amb_rows == set(amount_by_row) == set(amap["stray_row"]), "rows differ across cells"
    ids = {y["input_id"] for y in new}
    assert len(ids) == N_STRAY
    assert not ids & {x["input_id"] for x in inputs} and not ids & {x["input_id"] for x in amount_new}
    allowed = {"input_id", "evidence_state", "packet", "reader_input", "reader_input_sha"}
    by_row = {int(m.stray_row): m for m in amap.itertuples()}
    ex_a = ex_b = None
    for y in new:
        x, z, m = clear_by_row[y["row"]], amount_by_row[y["row"]], by_row[y["row"]]
        assert set(y) == set(x) == set(z), y["input_id"]
        assert {k for k in x if x[k] != y[k]} == allowed, y["input_id"]
        assert y["cls"] == "zero_stray" and y["target_group"] == x["target_group"] == z["target_group"]
        # Packet against stray_clear: one document changed, only its amount.
        assert {k: v for k, v in x["packet"].items() if k != "documents"} == \
               {k: v for k, v in y["packet"].items() if k != "documents"}
        dx, dy, dz = x["packet"]["documents"], y["packet"]["documents"], z["packet"]["documents"]
        assert len(dx) == len(dy) == len(dz), y["input_id"]
        dd = [(u, v) for u, v in zip(dx, dy) if u != v]
        assert len(dd) == 1 and dd[0][0]["id"] == dd[0][1]["id"] == m.doc_id, y["input_id"]
        u, v = dd[0]
        assert {k: w for k, w in u["data"].items() if k != "Amount"} == {k: w for k, w in v["data"].items() if k != "Amount"}
        # Packet against stray_amount: one document changed, only its label.
        dd = [(w, v) for w, v in zip(dz, dy) if w != v]
        assert len(dd) == 1 and dd[0][0]["id"] == dd[0][1]["id"] == m.doc_id, y["input_id"]
        w, v = dd[0]
        assert {k: q for k, q in w["data"].items() if k != "Description"} == {k: q for k, q in v["data"].items() if k != "Description"}
        assert w["data"]["Description"] == E.AMBIGUOUS_LABEL and v["data"]["Description"].startswith(E.SAVINGS_PREFIX)
        assert v["data"]["Amount"] == w["data"]["Amount"] == float(m.new_amount)
        label = v["data"]["Description"]

        # (a) Reader text against stray_clear: exactly one line, only the amount.
        i, a, b = one_line(x["reader_input"], y["reader_input"], y["input_id"])
        assert a.startswith(f"[{m.doc_id}] ") and a.endswith(label), a
        assert money(m.original_amount) in a and money(m.new_amount) in b, (a, b)
        assert a.replace(money(m.original_amount), money(m.new_amount), 1) == b, (a, b)
        ex_a = ex_a or (y["input_id"], i, a, b)
        # (b) Reader text against stray_amount: exactly one line, only the label.
        i, a, b = one_line(z["reader_input"], y["reader_input"], y["input_id"])
        assert a.startswith(f"[{m.doc_id}] ") and a.endswith(E.AMBIGUOUS_LABEL) and b.endswith(label), (a, b)
        assert money(m.new_amount) in a and money(m.new_amount) in b, (a, b)
        assert a[: -len(E.AMBIGUOUS_LABEL)] + label == b, (a, b)
        ex_b = ex_b or (y["input_id"], i, a, b)

        # Same leak scans as the frozen preflight.
        hits = [h.group(0) for h in run_design1.LEAK_RE.finditer(y["reader_input"])] + \
               [h.group(0) for h in build_inputs.EXTRA_LEAK_RE.finditer(y["reader_input"])]
        assert not hits, (y["input_id"], hits)
        assert y["reader_input_sha"] == llm.sha(y["reader_input"])
    return ex_a, ex_b


def main() -> int:
    sha_frozen = file_sha256(build_inputs.INPUTS_PATH)
    assert sha_frozen.startswith(EXPECT_INPUTS_SHA_PREFIX), f"inputs.jsonl sha256 {sha_frozen} differs from the frozen inputs"
    sha_map = file_sha256(AMOUNT_MAP_PATH)
    assert sha_map == EXPECT_AMOUNT_MAP_SHA, f"amount-map.csv sha256 {sha_map} differs from the committed map"
    sha_amount = file_sha256(AMOUNT_INPUTS_PATH)
    assert sha_amount == EXPECT_AMOUNT_INPUTS_SHA, f"amount-inputs.jsonl sha256 {sha_amount} differs from the committed file"

    inputs, _, _ = build_inputs.build()
    sha_built = hashlib.sha256(build_inputs.serialize(inputs).encode()).hexdigest()
    assert sha_built == sha_frozen, f"inputs.jsonl ({sha_frozen}) differs from a fresh build ({sha_built})"
    on_disk = run_design1.read_jsonl(build_inputs.INPUTS_PATH)
    assert [x["input_id"] for x in on_disk] == [x["input_id"] for x in inputs] and len(inputs) == build_inputs.N_INPUTS

    # The amount arm rebuilt in memory must equal the committed amount inputs and map.
    amount_new, amap_built = build_amount.build(inputs)
    assert hashlib.sha256(build_amount.serialize(amount_new).encode()).hexdigest() == sha_amount, "amount arm rebuild differs"
    amap = pd.read_csv(AMOUNT_MAP_PATH)
    assert len(amap) == N_STRAY
    assert list(amap["stray_row"]) == list(amap_built["stray_row"])
    assert (amap["new_amount"].round(2) == amap_built["new_amount"].round(2)).all()

    new = build(inputs, amap)
    ex_a, ex_b = check(new, inputs, amount_new, amap)
    text = build_amount.serialize(new)
    assert text == build_amount.serialize(build(inputs, amap)), "build is not deterministic"
    CLEAR_INPUTS_PATH.write_text(text)
    assert file_sha256(build_inputs.INPUTS_PATH) == sha_frozen, "frozen inputs changed"
    assert file_sha256(AMOUNT_MAP_PATH) == sha_map and file_sha256(AMOUNT_INPUTS_PATH) == sha_amount, "amount arm files changed"

    print(f"frozen inputs.jsonl sha256 {sha_frozen}")
    print(f"amount-map.csv sha256 {sha_map}; amount-inputs.jsonl sha256 {sha_amount} (rebuild equal)")
    print(f"twin inputs: {len(new)}, target groups {pd.Series([y['target_group'] for y in new]).value_counts().to_dict()}")
    print(f"new amounts ${amap['new_amount'].min():,.2f} to ${amap['new_amount'].max():,.2f}")
    print("check (a) passed: every twin differs from its stray_clear counterpart in exactly one reader line, only in the amount")
    print(f"  example ({ex_a[0]}), line {ex_a[1]}:\n  - {ex_a[2]}\n  + {ex_a[3]}")
    print("check (b) passed: every twin differs from its stray_amount counterpart in exactly one reader line, only in the label")
    print(f"  example ({ex_b[0]}), line {ex_b[1]}:\n  - {ex_b[2]}\n  + {ex_b[3]}")
    print("same 48 rows in stray_clear, stray_ambiguous, stray_amount and stray_clear_amount")
    print(f"clear-inputs.jsonl sha256 {file_sha256(CLEAR_INPUTS_PATH)} (two builds equal)")
    print(f"wrote {CLEAR_INPUTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
