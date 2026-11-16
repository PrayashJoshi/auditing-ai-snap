"""Design 1 v2 inputs: household draws, evidence states, reader inputs and preflight assertions. Zero calls.

24 omission households (12 earned, 12 unearned target) x 4 evidence states, 24 truthful documented households,
48 true-zero households with rent, and 48 true-zero households with one stray credit shown under a clear and
an ambiguous label: 96 + 24 + 48 + 96 = 264 inputs. Seed 20261115.

Run from the repo root:
    PYTHONPATH=src python experiments/design1_v2/build_inputs.py
Writes inputs.jsonl, households.csv, draw-log.csv and samples/ next to this file.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from snapqc import checker, forms, llm
from snapqc import evidence_states as E
from snapqc import packets as P
from snapqc.data import load_fy2024

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS_PATH = HERE / "inputs.jsonl"
HOUSEHOLDS_PATH = HERE / "households.csv"
DRAW_LOG_PATH = HERE / "draw-log.csv"
SAMPLES = HERE / "samples"

# The first reader run's prompt text, rendering and baselines are reused unchanged.
_spec = importlib.util.spec_from_file_location("run_design1", ROOT / "experiments" / "design1" / "run_design1.py")
run_design1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_design1)

N_INPUTS = 264
EXPECTED_CELLS = {("omission", s): 24 for s in E.CLASS_STATES["omission"]} | {
    ("truthful_documented", "documented"): 24, ("zero_rent", "none"): 48,
    ("zero_stray", "stray_clear"): 48, ("zero_stray", "stray_ambiguous"): 48}
# Words that would tell a reader the answer or the design. The first run's scan plus design vocabulary.
EXTRA_LEAK_RE = re.compile(r"\b(omi\w*|stray|truthful|twin|synthetic|ground|scor\w*|target|class|evidence|ambig\w*|"
                           r"clear|zero_\w+|deposits_\w+|cls|seed|qc)\b", re.IGNORECASE)


def money_equal(a, b) -> bool:
    return abs(float(a) - float(b)) < 0.005


def build(seed: int = E.SEED) -> tuple[list[dict], pd.DataFrame, dict]:
    df = load_fy2024()
    arm = E.human_arm()
    pool, log = E.pools(df, arm)
    tier = log.pop("tier")
    drawn = E.draw(pool, seed)
    hh = E.household_frame(df, arm, drawn, tier)
    tiers = forms.sua_tiers(df)
    inputs = []
    extra = []
    for row, h in hh.iterrows():
        sua = {k: (None if pd.isna(v) else float(v)) for k, v in tiers.loc[h["state"]].items()}
        case = P.build_case(df.loc[row], h, sua)
        cls, target = h["cls"], h["target_group"]
        verified = {g: float(h[f"verified_{g}"]) for g in forms.GROUPS}
        app = E.application(h, cls)
        impact = abs(P.benefit_of({**verified, target: 0.0}, h) - P.benefit_of(verified, h))
        housing = verified["shelter"] + verified["utilities"]
        gap = housing - (app["earned"] + app["unearned"])
        extra.append({"row": row, "budget_gap": round(gap, 2), "target_impact": round(impact, 2),
                      "target_types": ";".join(sorted({f["type"] for f in case["facts"] if f["group"] == target})),
                      "target_links": ";".join(sorted({P.LINK[f["type"]] for f in case["facts"] if f["group"] == target}))})
        for state in E.CLASS_STATES[cls]:
            packet, meta = E.evidence_packet(case, cls, target, state)
            text = run_design1.render_reader_input(app, packet, target, "wage_match" if state == "lead" else "none")
            inputs.append({
                "input_id": f"{row}|{cls}|{state}",
                "row": int(row), "cls": cls, "target_group": target, "evidence_state": state,
                "is_omission": cls == "omission",
                "app_monthly": app, "verified_monthly": verified,
                "budget_gap": round(gap, 2), "target_impact": round(impact, 2),
                "meta": meta, "packet": packet,
                "reader_input": text, "reader_input_sha": llm.sha(text),
            })
    hh = hh.join(pd.DataFrame(extra).set_index("row"))
    return inputs, hh, log


def serialize(inputs: list[dict]) -> str:
    return "".join(json.dumps(x, sort_keys=True) + "\n" for x in inputs)


def preflight(inputs: list[dict], hh: pd.DataFrame) -> dict:
    """Assertions before any call. Returns counts worth reporting."""
    assert len(inputs) == N_INPUTS, len(inputs)
    assert len({x["input_id"] for x in inputs}) == N_INPUTS
    cells = Counter((x["cls"], x["evidence_state"]) for x in inputs)
    assert dict(cells) == EXPECTED_CELLS, cells
    for (cls, state), n in EXPECTED_CELLS.items():
        t = Counter(x["target_group"] for x in inputs if x["cls"] == cls and x["evidence_state"] == state)
        assert t["earned"] == t["unearned"] == n // 2, (cls, state, t)
    assert len(hh) == 144 and hh.index.is_unique, "households must be distinct across classes"
    assert Counter(hh["cls"]) == {"omission": 24, "truthful_documented": 24, "zero_rent": 48, "zero_stray": 48}

    out = Counter()
    for x in inputs:
        cls, st, tg, app, ver, p, text = (x["cls"], x["evidence_state"], x["target_group"], x["app_monthly"],
                                           x["verified_monthly"], x["packet"], x["reader_input"])
        iid = x["input_id"]
        docs = p["documents"]
        groups = x["meta"]["doc_groups"]
        documented = checker.documented_groups(p)
        orc = checker.oracle(p)
        # Packet carries no scoring metadata; ids are sequential.
        assert set(p) == {"row", "state", "sua", "account", "window", "documents"}, iid
        assert [d["id"] for d in docs] == [f"doc-{i:02d}" for i in range(1, len(docs) + 1)], iid
        assert all(set(d) == {"id", "connector", "kind", "data"} for d in docs), iid
        # Shelter and utilities at verified values, with their documents, in every input.
        for g in ("shelter", "utilities"):
            assert money_equal(app[g], ver[g]), (iid, g)
            if ver[g] > 0:
                assert g in documented and money_equal(orc[g], ver[g]), (iid, g, orc[g], ver[g])
            else:
                assert g not in documented, (iid, g)
        # The other income group at its verified value, documented when above $0.
        other = "unearned" if tg == "earned" else "earned"
        assert money_equal(app[other], ver[other]), iid
        if ver[other] > 0:
            assert other in documented and abs(orc[other] - ver[other]) <= 1, (iid, orc[other], ver[other])
        # Target group by class and state.
        n_target_docs = sum(g == tg for g in groups)
        target_kinds = {d["kind"] for d, g in zip(docs, groups) if g == tg}
        n_amb = sum(d["kind"] == "transaction" and d["data"]["Description"] == E.AMBIGUOUS_LABEL for d in docs)
        lead_lines = [l for l in text.splitlines() if l.startswith("Agency cross-match lead")]
        if cls == "omission":
            assert ver[tg] > 0 and app[tg] == 0.0, iid
            if st in ("none", "lead"):
                assert n_target_docs == 0, iid
            else:
                assert n_target_docs >= 2 and target_kinds == {"transaction"}, (iid, target_kinds)
                assert all(d["data"]["Amount"] > 0 for d, g in zip(docs, groups) if g == tg), iid
            if st == "deposits_ambiguous":
                assert n_amb == n_target_docs, iid
                assert all(d["data"]["Description"] == E.AMBIGUOUS_LABEL for d, g in zip(docs, groups) if g == tg)
            else:
                assert n_amb == 0, iid
            assert len(lead_lines) == (1 if st == "lead" else 0), iid
            if st == "lead":
                assert lead_lines[0] == E.LEAD_TEXT[tg], iid
        else:
            assert not lead_lines and app[tg] == ver[tg], iid
            if cls == "truthful_documented":
                assert ver[tg] > 0 and tg in documented and abs(orc[tg] - ver[tg]) <= 1, (iid, orc[tg], ver[tg])
            else:
                assert ver[tg] == 0 and n_target_docs == 0 and tg not in documented, iid
            assert n_amb == (1 if st == "stray_ambiguous" else 0), iid
            if cls == "zero_rent":
                assert ver["shelter"] > 0, iid
        # Exactly one own-account savings transfer (or its relabel) in every file.
        n_sav = sum(d["kind"] == "transaction" and d["data"]["Description"].startswith(E.SAVINGS_PREFIX) for d in docs)
        assert n_sav + (st == "stray_ambiguous") == 1, (iid, n_sav)
        # No ground truth or design vocabulary in what the reader sees.
        hits = [m.group(0) for m in run_design1.LEAK_RE.finditer(text)] + [m.group(0) for m in EXTRA_LEAK_RE.finditer(text)]
        assert not hits, (iid, hits)
        fields = " ".join(sorted({k for d in docs for k in [*d["data"], d["kind"], d["connector"]]}))
        assert not run_design1.LEAK_RE.search(fields) and not EXTRA_LEAK_RE.search(fields), iid
        # Application lines: fixed labels, whole dollars, never "not provided" in this design.
        app_lines = text.split("\n\nDocuments:")[0].splitlines()[1:]
        assert [l.split(": ")[0] for l in app_lines] == [checker.GROUP_LABEL[g] for g in forms.GROUPS], iid
        assert all(re.match(r"^\$\d{1,3}(,\d{3})*$", l.split(": ", 1)[1]) for l in app_lines), iid
        assert x["reader_input_sha"] == llm.sha(text)
        out[f"impact > $56 ({cls})"] += x["target_impact"] > 56 and st == E.CLASS_STATES[cls][0]
        out[f"budget gap > 0 ({cls})"] += x["budget_gap"] > 0 and st == E.CLASS_STATES[cls][0]

    # Within-household contrasts change only what they are meant to change.
    by = {(x["row"], x["evidence_state"]): x for x in inputs}
    for row in hh.index[hh["cls"] == "zero_stray"]:
        a = by[(row, "stray_clear")]["reader_input"].splitlines()
        b = by[(row, "stray_ambiguous")]["reader_input"].splitlines()
        diff = [(u, v) for u, v in zip(a, b) if u != v]
        assert len(a) == len(b) and len(diff) == 1 and diff[0][1].endswith(E.AMBIGUOUS_LABEL), row
    for row in hh.index[hh["cls"] == "omission"]:
        c, a = by[(row, "deposits_clear")], by[(row, "deposits_ambiguous")]
        dc, da = c["packet"]["documents"], a["packet"]["documents"]
        assert len(dc) == len(da)
        for u, v in zip(dc, da):
            if u != v:
                assert u["kind"] == "transaction" and v["data"]["Description"] == E.AMBIGUOUS_LABEL
                assert {k: w for k, w in u["data"].items() if k != "Description"} == \
                       {k: w for k, w in v["data"].items() if k != "Description"}
        n_none = by[(row, "none")]["reader_input"]
        n_lead = by[(row, "lead")]["reader_input"]
        assert n_lead.replace("\n" + E.LEAD_TEXT[c["target_group"]] + "\n", "", 1) == n_none, row
        assert c["app_monthly"] == by[(row, "none")]["app_monthly"] == by[(row, "lead")]["app_monthly"]
    return dict(out)


def sample_inputs(inputs: list[dict]) -> list[dict]:
    """One per class and state: the first household drawn in the cell, alternating earned and unearned target."""
    out = []
    for i, (cls, st) in enumerate(EXPECTED_CELLS):
        tg = E.TARGETS[i % 2]
        out.append(next(x for x in inputs if x["cls"] == cls and x["evidence_state"] == st and x["target_group"] == tg))
    return out


def main() -> int:
    inputs, hh, log = build()
    counts = preflight(inputs, hh)
    text = serialize(inputs)
    sha1 = hashlib.sha256(text.encode()).hexdigest()
    inputs2, hh2, _ = build()
    sha2 = hashlib.sha256(serialize(inputs2).encode()).hexdigest()
    assert sha1 == sha2 and hh.equals(hh2), "build is not deterministic"
    INPUTS_PATH.write_text(text)
    assert hashlib.sha256(INPUTS_PATH.read_bytes()).hexdigest() == sha1
    hh.to_csv(HOUSEHOLDS_PATH)
    pd.Series(log, name="value").rename_axis("item").to_frame().to_csv(DRAW_LOG_PATH)
    SAMPLES.mkdir(exist_ok=True)
    for x in sample_inputs(inputs):
        name = f"{x['cls']}__{x['evidence_state']}__{x['target_group']}__{x['row']}.txt"
        (SAMPLES / name).write_text(x["reader_input"])

    print("draw log:", json.dumps(log, indent=1, default=int))
    print("cells:", dict(Counter((x["cls"], x["evidence_state"]) for x in inputs)))
    print("households per class and target:", hh.groupby(["cls", "target_group"]).size().to_dict())
    print("per-household counts:", counts)
    print("target links (omission):", hh[hh.cls == "omission"]["target_links"].value_counts().to_dict())
    print("input chars: median", int(pd.Series([len(x["reader_input"]) for x in inputs]).median()),
          "max", max(len(x["reader_input"]) for x in inputs))
    print(f"inputs.jsonl sha256 {sha1} (two builds equal)")
    print("ALL PREFLIGHT ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
