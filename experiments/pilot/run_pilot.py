"""Pilot runner: AI filers and readers on the 20 pilot households, one JSONL line per call.

Phases (run in this order; each appends to its own file and skips jobs that already have an answered record):
  dryrun            4 calls (three filers, one per schema, and one reader) -> dryrun.jsonl
  core-filers       required and escape schemas x {complete, faithful} x note A, plus escape x faithful x note B,
                    both filer models -> filers-core.jsonl
  core-readers      human-filed applications x {complete, faithful}, read once by each reader -> readers-core.jsonl
  proposed-filers   provenance schema x {complete, faithful} x note A, provenance x faithful x note B,
                    provenance x faithful_linkability x note A, both filer models -> filers-proposed.jsonl
  proposed-readers  AI-filed provenance note-A applications (escape note-A ones if the provenance switch is off),
                    both packets, both filer models, read once by each reader; then the re-reads: 10 seeded
                    applications (5 human-filed, 5 AI-filed) read twice more by each reader -> readers-proposed.jsonl
  rereads           the re-reads alone (also part of proposed-readers)
  report            per-file summary of the records so far (no calls)

Proposed switches (on by default): --no-provenance (provenance_schema), --no-linkability (linkability_packet),
--no-confidence (reader_confidence: the reader's flag items carry a 0-1 confidence).

Every reader input sent is saved once per application id in reader-inputs.jsonl, with the application's monthly
values, so scoring can re-derive everything without new calls. AI-filed applications that cannot be parsed or
converted to monthly amounts are not read; each skip is logged in reader-skips.jsonl. Before conversion, a field
sent as a JSON-encoded string is decoded (flagged filer_repaired); a utilities_paid that is still not a list is a skip.

Run: uv run python experiments/pilot/run_pilot.py PHASE [--no-provenance] [--no-linkability] [--no-confidence]
     [--max-output-tokens N]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from snapqc import checker, forms, llm, packets

HERE = Path(__file__).resolve().parent
MODELS = ["deepseek-v4-flash", "qwen3.5-397b"]  # filer and reader models
MAX_OUTPUT_TOKENS = 8192  # 2048 truncated 2 of 3 filer calls in the dry run (reasoning used the cap)
CONCURRENCY = 8
SEED = 20261114
N_REREAD_HUMAN, N_REREAD_AI = 5, 5
REREADS = [2, 3]
CORE_VARIANTS = ["complete", "faithful"]

FILES = {"dryrun": "dryrun.jsonl", "core-filers": "filers-core.jsonl", "core-readers": "readers-core.jsonl",
         "proposed-filers": "filers-proposed.jsonl", "proposed-readers": "readers-proposed.jsonl",
         "rereads": "readers-proposed.jsonl"}
READER_INPUTS = HERE / "reader-inputs.jsonl"
READER_SKIPS = HERE / "reader-skips.jsonl"

# --- Filer --------------------------------------------------------------------------------------------------
ESCAPE_SENTENCE = (f'If the records and the note do not support a value, enter "{forms.ESCAPE}" for it '
                   "instead of a number.")
PROVENANCE_SENTENCES = (f'If the records and the note do not support a value, enter "{forms.ESCAPE}" as its value '
                        "instead of a number. For every value, set source to retrieved_record, applicant_stated or "
                        "agent_inferred, and list in document_ids the ids of the documents you used (empty list if "
                        "none).")
FILER_SUFFIX = {"required": "", "escape": ESCAPE_SENTENCE, "provenance": PROVENANCE_SENTENCES}


def filer_instructions(schema: str) -> str:
    return " ".join(s for s in (packets.FILER_INSTRUCTIONS, FILER_SUFFIX[schema]) if s)


TOOLS = {s: json.loads((HERE / f"tool-{s}.json").read_text()) for s in forms.SCHEMAS}

# --- Reader -------------------------------------------------------------------------------------------------
READER_INSTRUCTIONS = """You review SNAP applications for a state agency. You see the application's monthly total per fact group (whole dollars; "not provided" means the filer gave no amount) and the household's records.
Flag every fact group whose amount the records do not support: a reported amount with no record, or a recorded amount that, converted to monthly gross (weekly x 4.3, biweekly x 2.15, twice a month x 2, yearly / 12), differs from the filed amount enough to change the monthly benefit by more than $56. As a guide, the benefit moves about $0.24 per $1 of earned income, $0.30 per $1 of unearned income and up to $0.30 per $1 of shelter or utility cost. Utilities are the state's standard allowance for the utility types the records show.
Call report_flags exactly once, {per_flag}. Use an empty list if nothing is wrong."""
PER_FLAG = {True: "giving for each flag your confidence from 0 to 1 that the amount is wrong and a one-line reason",
            False: "giving a one-line reason for each flag"}


def reader_instructions(confidence: bool) -> str:
    return READER_INSTRUCTIONS.format(per_flag=PER_FLAG[confidence])


def reader_tool(confidence: bool) -> dict:
    item = {"group": {"type": "string", "enum": forms.GROUPS}}
    if confidence:
        item["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    item["reason"] = {"type": "string"}
    return {"type": "function", "name": "report_flags",
            "description": "Report the fact groups whose amounts the records do not support.",
            "parameters": {"type": "object",
                           "properties": {"flags": {"type": "array", "items": {
                               "type": "object", "properties": item, "required": list(item),
                               "additionalProperties": False}}},
                           "required": ["flags"], "additionalProperties": False}}


SUA_LABEL = [("heating/cooling", "HCSUA"), ("limited", "LUA"), ("telephone", "TUA")]


def sua_line(sua: dict) -> str:
    """The state's allowance amounts, one line, the same for every filer of a household."""
    def amount(v):
        return "none" if v is None or (isinstance(v, float) and math.isnan(v)) else f"${float(v):,.0f}"
    return "State utility allowances: " + ", ".join(f"{label} {amount(sua.get(t))}" for label, t in SUA_LABEL)


def reader_input(monthly: dict, packet: dict) -> str:
    return checker.reader_view(monthly, packet) + "\n" + sua_line(packet["sua"]) + "\n"


# --- Data ---------------------------------------------------------------------------------------------------
def load_packets() -> dict:
    out = {}
    for line in (HERE / "packets.jsonl").read_text().splitlines():
        p = json.loads(line)
        out[(p["row"], p["variant"])] = p
    return out


def load_households() -> dict:
    df = pd.read_csv(HERE / "households.csv")
    return {int(r["row"]): r for r in df.to_dict("records")}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def latest(records: list[dict]) -> dict:
    """Last record per job id, answered records preferred."""
    out = {}
    for r in records:
        if r.get("ok_http") or r["job_id"] not in out:
            out[r["job_id"]] = r
    return out


# --- Jobs ---------------------------------------------------------------------------------------------------
def filer_job(phase, schema, variant, note, row, model, pk, switches) -> dict:
    packet = pk[(row, variant)]
    return dict(job_id=f"filer|{schema}|{variant}|{note}|{row}|{model}", task="filer", phase=phase, schema=schema,
                variant=variant, note=note, row=row, filer=model, reader=None, read=None,
                app_id=f"{model}|{schema}|{variant}|{note}|{row}", switches=switches,
                _model=model, _instructions=filer_instructions(schema), _input=packet["filer_input"][note],
                _tool=TOOLS[schema])


def filer_jobs(phase, pk, switches) -> list[dict]:
    rows = list(dict.fromkeys(r for r, _ in pk))
    combos = []
    if phase == "core-filers":
        combos += [(s, v, "A") for s in ["required", "escape"] for v in CORE_VARIANTS]
        combos += [("escape", "faithful", "B")]
    elif phase == "proposed-filers" and switches["provenance_schema"]:
        combos += [("provenance", v, "A") for v in CORE_VARIANTS]
        combos += [("provenance", "faithful", "B")]
        if switches["linkability_packet"]:
            combos += [("provenance", "faithful_linkability", "A")]
    return [filer_job(phase, s, v, n, row, m, pk, switches) for s, v, n in combos for row in rows for m in MODELS]


def application(app_id, filer, schema, variant, note, row, monthly, packet, extra=None) -> dict:
    text = reader_input(monthly, packet)
    return dict(app_id=app_id, filer=filer, schema=schema, variant=variant, note=note, row=row,
                app_monthly=monthly, reader_input=text, reader_input_sha=llm.sha(text), **(extra or {}))


def human_apps(pk, hh) -> list[dict]:
    return [application(f"human|{v}|{row}", "human", None, v, None, row, packets.human_filed(hh[row]), pk[(row, v)])
            for v in CORE_VARIANTS for row in hh]


def decode_strings(x):
    """Decode string values that hold a JSON object or array (a model sometimes sends a field as an encoded string;
    trailing characters after the first JSON value are ignored). Returns (value, whether anything was decoded)."""
    if isinstance(x, dict):
        out = {k: decode_strings(v) for k, v in x.items()}
        return {k: v for k, (v, _) in out.items()}, any(r for _, r in out.values())
    if isinstance(x, list):
        out = [decode_strings(v) for v in x]
        return [v for v, _ in out], any(r for _, r in out)
    if isinstance(x, str) and x.lstrip()[:1] in ("{", "["):
        try:
            return decode_strings(json.JSONDecoder().raw_decode(x.lstrip())[0])[0], True
        except json.JSONDecodeError:
            return x, False
    return x, False


def ai_apps(pk, switches) -> list[dict]:
    """AI-filed note-A applications read in proposed-readers (provenance schema, or escape if that switch is off)."""
    schema, src = ("provenance", "proposed-filers") if switches["provenance_schema"] else ("escape", "core-filers")
    recs = latest(read_jsonl(HERE / FILES[src]))
    apps, skips = [], []
    for v in CORE_VARIANTS:
        for row in dict.fromkeys(r for r, _ in pk):
            for m in MODELS:
                jid = f"filer|{schema}|{v}|A|{row}|{m}"
                rec = recs.get(jid)
                reason, monthly = None, None
                if rec is None:
                    reason = "no filer record"
                elif not isinstance(rec.get("parsed"), dict):
                    reason = f"filer output not parsed ({rec.get('invalid_reason')})"
                else:
                    app, repaired = decode_strings(rec["parsed"])
                    types = app.get("utilities_paid")
                    types = types.get("value") if isinstance(types, dict) else types
                    if not (types is None or types == forms.ESCAPE or isinstance(types, list)):
                        reason = f"utilities_paid not a list: {str(types)[:80]}"
                    else:
                        try:
                            monthly = forms.to_monthly(app, pk[(row, v)]["sua"])
                        except Exception as e:  # malformed but decoded output: cannot be converted
                            reason = f"to_monthly failed: {type(e).__name__}: {str(e)[:120]}"
                if reason:
                    skips.append(dict(filer_job_id=jid, app_id=f"{m}|{schema}|{v}|A|{row}", reason=reason))
                    continue
                apps.append(application(f"{m}|{schema}|{v}|A|{row}", m, schema, v, "A", row, monthly, pk[(row, v)],
                                        dict(filer_job_id=jid, filer_valid=rec["valid"],
                                             filer_invalid_reason=rec["invalid_reason"], filer_repaired=repaired)))
    return apps, skips


def reader_job(phase, app, reader, read, switches) -> dict:
    conf = switches["reader_confidence"]
    meta = {k: app[k] for k in ("app_id", "filer", "schema", "variant", "note", "row", "reader_input_sha",
                                "app_monthly")}
    meta.update({k: app[k] for k in ("filer_job_id", "filer_valid", "filer_invalid_reason", "filer_repaired")
                 if k in app})
    return dict(job_id=f"reader|{app['app_id']}|{reader}|{read}", task="reader", phase=phase, reader=reader,
                read=read, switches=switches, **meta, _model=reader, _instructions=reader_instructions(conf),
                _input=app["reader_input"], _tool=reader_tool(conf))


def reread_apps(pk, hh, switches) -> list[dict]:
    rng = random.Random(SEED)
    human = human_apps(pk, hh)
    ai, _ = ai_apps(pk, switches)

    def pick(apps, n_complete, n_faithful):
        out = []
        for v, n in (("complete", n_complete), ("faithful", n_faithful)):
            pool = [a for a in apps if a["variant"] == v]
            out += rng.sample(pool, min(n, len(pool)))
        return out
    return pick(human, 3, N_REREAD_HUMAN - 3) + pick(ai, 2, N_REREAD_AI - 2)


def save_reader_inputs(apps: list[dict]) -> None:
    have = {r["app_id"]: r["reader_input_sha"] for r in read_jsonl(READER_INPUTS)}
    with READER_INPUTS.open("a") as fh:
        for a in apps:
            if a["app_id"] in have:
                if have[a["app_id"]] != a["reader_input_sha"]:
                    print(f"WARNING reader input changed for {a['app_id']}", flush=True)
                continue
            fh.write(json.dumps(a, default=str) + "\n")
            have[a["app_id"]] = a["reader_input_sha"]


def log_skips(skips: list[dict], phase: str) -> None:
    if not skips:
        return
    have = {(s["app_id"], s["phase"]) for s in read_jsonl(READER_SKIPS)}
    with READER_SKIPS.open("a") as fh:
        for s in skips:
            if (s["app_id"], phase) not in have:
                fh.write(json.dumps({**s, "phase": phase}) + "\n")
            print(f"skip {s['app_id']}: {s['reason']}", flush=True)


# --- Running ------------------------------------------------------------------------------------------------
def run(jobs: list[dict], out: Path) -> list[dict]:
    done = {j for j, r in latest(read_jsonl(out)).items() if r.get("ok_http")}
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"{out.name}: {len(jobs)} jobs, {len(jobs) - len(todo)} already answered, {len(todo)} to run", flush=True)
    if not todo:
        return []
    cl = llm.client()
    lock = threading.Lock()
    n = [0]
    with out.open("a") as fh, ThreadPoolExecutor(CONCURRENCY) as pool:
        def work(job):
            meta = {k: v for k, v in job.items() if not k.startswith("_")}
            rec = llm.call(cl, job["_model"], job["_instructions"], job["_input"], job["_tool"], MAX_OUTPUT_TOKENS,
                           meta)
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                n[0] += 1
                if n[0] % 20 == 0 or n[0] == len(todo):
                    print(f"  {n[0]}/{len(todo)} done", flush=True)
            return rec
        recs = list(pool.map(work, todo))
    return recs


def snapshot(switches) -> None:
    if not (HERE / "models.json").exists():
        (HERE / "models.json").write_text(json.dumps(llm.hosted_models(llm.client()), indent=1, default=str) + "\n")
    cfg_path = HERE / "run-config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"instructions": {}, "tools": {}}
    cfg.update(models=MODELS, max_output_tokens=MAX_OUTPUT_TOKENS, concurrency=CONCURRENCY,
               temperature=llm.TEMPERATURE, seed=SEED)
    texts = [filer_instructions(s) for s in forms.SCHEMAS] + [reader_instructions(c) for c in (True, False)]
    cfg["instructions"].update({llm.sha(t): t for t in texts})
    tools = list(TOOLS.values()) + [reader_tool(c) for c in (True, False)]
    cfg["tools"].update({llm.tool_sha(t): t for t in tools})
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")


def report(paths: list[Path]) -> None:
    for path in paths:
        recs = read_jsonl(path)
        if not recs:
            continue
        for phase in dict.fromkeys(r["phase"] for r in recs):
            rs = [r for r in recs if r["phase"] == phase]
            trunc = sum(r.get("status") == "incomplete" or
                        (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens" for r in rs)
            lat = [r["latency_s"] for r in rs if r.get("ok_http")]
            tok = {k: sum(r.get(k) or 0 for r in rs) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
            wall = max(r["t_end"] for r in rs) - min(r["t_start"] for r in rs)
            print(f"\n{path.name} / {phase}: {len(rs)} calls, {sum(r['valid'] for r in rs)} valid, "
                  f"{trunc} truncated, {sum(r['retries'] for r in rs)} retries, wall {wall / 60:.1f} min, "
                  f"credits {sum(r.get('cost') or 0 for r in rs):.4f}, median latency "
                  f"{statistics.median(lat) if lat else float('nan'):.1f}s, tokens {tok}")
            by = Counter((r.get("filer") if r["task"] == "filer" else r.get("reader"), r["valid"]) for r in rs)
            print("  valid by model:", dict(sorted(by.items(), key=str)))
            inv = Counter((r["invalid_reason"] or "")[:90] for r in rs if not r["valid"])
            for k, v in inv.most_common():
                print(f"  invalid {v}: {k}")


def main() -> None:
    global MAX_OUTPUT_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(FILES) + ["report"])
    ap.add_argument("--no-provenance", action="store_true")
    ap.add_argument("--no-linkability", action="store_true")
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    a = ap.parse_args()
    MAX_OUTPUT_TOKENS = a.max_output_tokens
    switches = dict(provenance_schema=not a.no_provenance, linkability_packet=not a.no_linkability,
                    reader_confidence=not a.no_confidence)
    if a.phase == "report":
        report([HERE / f for f in dict.fromkeys(FILES.values())])
        return
    pk, hh = load_packets(), load_households()
    snapshot(switches)
    out = HERE / FILES[a.phase]
    if a.phase == "dryrun":
        row = 15847
        jobs = [filer_job("dryrun", s, "faithful", "A", row, m, pk, switches)
                for s, m in [("required", MODELS[0]), ("escape", MODELS[1]), ("provenance", MODELS[0])]]
        app = [x for x in human_apps(pk, hh) if x["app_id"] == f"human|faithful|{row}"][0]
        jobs.append(reader_job("dryrun", app, MODELS[1], 1, switches))
        for j in jobs:  # dry runs are repeated at different output caps
            j["job_id"] += f"|cap{MAX_OUTPUT_TOKENS}"
        run(jobs, out)
    elif a.phase in ("core-filers", "proposed-filers"):
        run(filer_jobs(a.phase, pk, switches), out)
    elif a.phase == "core-readers":
        apps = human_apps(pk, hh)
        save_reader_inputs(apps)
        run([reader_job(a.phase, x, r, 1, switches) for x in apps for r in MODELS], out)
    if a.phase == "proposed-readers":
        apps, skips = ai_apps(pk, switches)
        log_skips(skips, a.phase)
        save_reader_inputs(apps)
        run([reader_job(a.phase, x, r, 1, switches) for x in apps for r in MODELS], out)
    if a.phase in ("proposed-readers", "rereads"):
        apps = reread_apps(pk, hh, switches)
        save_reader_inputs(apps)
        print("re-read applications:", [x["app_id"] for x in apps], flush=True)
        run([reader_job("rereads", x, r, k, switches) for k in REREADS for x in apps for r in MODELS], out)
    report([out])


if __name__ == "__main__":
    sys.exit(main())
