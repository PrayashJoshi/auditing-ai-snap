"""Model smoke test runner: 50 small calls per hosted model, one JSONL line per call.

Per model: 10 toy households x 3 reads with the forced `report_flags` tool (reader),
then 10 filer calls with every form field required and 10 with nullable fields that
also accept 'unknown' (filer). Inputs come from `fixtures.json` (hand-written, not QC
records). Output: `<model>.jsonl` in this folder. The notebook scores these files and
makes no calls itself.

Run: uv run python experiments/smoke/run_smoke.py [model ...]
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import adaption
import jsonschema
from adaption import Adaption

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CONCURRENCY = 5
MAX_ATTEMPTS = 6          # our own retry loop; SDK retries are off so every retry is logged
TIMEOUT_S = 120
READS = 3
TEMPERATURE = 0
READER_MAX_OUTPUT_TOKENS = 512
FILER_MAX_OUTPUT_TOKENS = 1024
SCHEMA_VERSION = "smoke-v1"
RUN = "run2-retry-after"

GROUPS = ["earned_income", "unearned_income", "shelter", "utilities"]

READER_INSTRUCTIONS = """You review SNAP applications for a state agency.
You get the filed application form and the documents the household supplied.
Compare each form field with the documents. Fact groups:
- earned_income: monthly gross wages (before taxes and deductions)
- unearned_income: monthly benefits, child support and other non-wage income
- shelter: monthly rent or mortgage plus property tax and insurance
- utilities: monthly utility costs
Convert to monthly amounts: weekly x 4.3, every two weeks x 2.15, monthly x 1, yearly / 12.
A field is wrong if it differs from the documents by more than $5.
Call report_flags exactly once with every fact group that is wrong. Use an empty list if none is wrong."""

FILER_INSTRUCTIONS = """You fill out a SNAP application for the applicant, using only the applicant's note and documents.
Convert to monthly amounts: weekly x 4.3, every two weeks x 2.15, monthly x 1, yearly / 12.
Use gross wages (before taxes and deductions). Shelter is rent or mortgage plus property tax and insurance.
Call submit_application exactly once."""

FILER_INSTRUCTIONS_NULLABLE = FILER_INSTRUCTIONS.replace(
    "Call submit_application exactly once.",
    "If a document needed for a field is missing, enter 'unknown' rather than guessing.\n"
    "Call submit_application exactly once.",
)

REPORT_FLAGS = {
    "type": "function",
    "name": "report_flags",
    "description": "Report which fact groups look wrong in this application.",
    "parameters": {
        "type": "object",
        "properties": {
            "flags": {"type": "array", "items": {"type": "string", "enum": GROUPS}},
        },
        "required": ["flags"],
        "additionalProperties": False,
    },
}

MONEY_FIELDS = ["earned_income_monthly", "unearned_income_monthly", "shelter_cost_monthly", "utility_cost_monthly"]


def filer_tool(nullable: bool) -> dict:
    money = (
        {"anyOf": [{"type": "number"}, {"type": "null"}, {"type": "string", "enum": ["unknown"]}]}
        if nullable
        else {"type": "number"}
    )
    props = {"household_size": {"type": "integer", "minimum": 1}}
    for f in MONEY_FIELDS:
        props[f] = {**money, "description": f"Monthly amount in dollars ({f.replace('_', ' ')})."}
    return {
        "type": "function",
        "name": "submit_application",
        "description": "Submit the completed SNAP application form.",
        "parameters": {
            "type": "object",
            "properties": props,
            "required": ["household_size", *MONEY_FIELDS],
            "additionalProperties": False,
        },
    }


FILER_REQUIRED = filer_tool(nullable=False)
FILER_NULLABLE = filer_tool(nullable=True)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def reader_input(h: dict) -> str:
    form = "\n".join(f"{k}: {v}" for k, v in h["form"].items())
    docs = "\n".join(f"- {d}" for d in h["documents"])
    return f"APPLICATION FORM\n{form}\n\nDOCUMENTS\n{docs}"


def filer_input(h: dict) -> str:
    docs = [d for i, d in enumerate(h["documents"]) if i not in h["filer_missing"]]
    return f"APPLICANT NOTE\n{h['filer_note']}\n\nDOCUMENTS\n" + "\n".join(f"- {d}" for d in docs)


def build_jobs(households: list[dict]) -> list[dict]:
    jobs = []
    for r in range(READS):  # read rounds interleaved so the three reads of one household are not back to back
        for h in households:
            jobs.append(dict(task="reader", variant="forced_flags", household=h["id"], read=r + 1,
                             instructions=READER_INSTRUCTIONS, input=reader_input(h), tool=REPORT_FLAGS,
                             max_output_tokens=READER_MAX_OUTPUT_TOKENS))
    for variant, tool, instr in [("required", FILER_REQUIRED, FILER_INSTRUCTIONS),
                                 ("nullable_unknown", FILER_NULLABLE, FILER_INSTRUCTIONS_NULLABLE)]:
        for h in households:
            jobs.append(dict(task="filer", variant=variant, household=h["id"], read=1,
                             instructions=instr, input=filer_input(h), tool=tool,
                             max_output_tokens=FILER_MAX_OUTPUT_TOKENS))
    return jobs


def parse(resp, tool: dict) -> tuple[str | None, object, str | None]:
    """Return (raw arguments, parsed value or None, reason if invalid)."""
    if resp.status != "completed":
        return None, None, f"status_{resp.status}"
    calls = [i for i in resp.output if i.type == "function_call" and i.name == tool["name"]]
    if len(calls) != 1:
        return (calls[0].arguments if calls else None), None, f"function_calls_{len(calls)}"
    raw = calls[0].arguments
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, None, "json_error"
    try:
        jsonschema.validate(args, tool["parameters"])
    except jsonschema.ValidationError as e:
        return raw, None, f"schema_error: {e.message[:120]}"
    if tool["name"] == "report_flags":
        return raw, sorted(set(args["flags"])), None
    return raw, args, None


def call(client: Adaption, model: str, job: dict) -> dict:
    rec = {k: job[k] for k in ("task", "variant", "household", "read", "max_output_tokens")}
    rec.update(request_model=model, temperature=TEMPERATURE, sdk_version=adaption.__version__,
               schema_version=SCHEMA_VERSION, run=RUN, tool_name=job["tool"]["name"],
               instructions_sha=sha(job["instructions"]), input_sha=sha(job["input"]),
               tool_sha=sha(json.dumps(job["tool"], sort_keys=True)), concurrency=CONCURRENCY)
    errors = []
    t0 = time.time()
    rec["ts_start"] = datetime.now(timezone.utc).isoformat()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ta = time.time()
        try:
            raw = client.responses.with_raw_response.create(
                model=model,
                instructions=job["instructions"],
                input=job["input"],
                tools=[job["tool"]],
                tool_choice={"type": "function", "name": job["tool"]["name"]},
                temperature=TEMPERATURE,
                max_output_tokens=job["max_output_tokens"],
                store=False,
            )
            resp = raw.parse()
            rec["rate_headers"] = {k: v for k, v in raw.headers.items()
                                   if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            break
        except adaption.APIStatusError as e:
            errors.append(dict(attempt=attempt, status=e.status_code, error=type(e).__name__,
                               message=str(e)[:300], latency_s=round(time.time() - ta, 3),
                               retry_after=e.response.headers.get("retry-after")))
            retryable = e.status_code == 429 or e.status_code >= 500
        except (adaption.APIConnectionError, adaption.APITimeoutError) as e:
            errors.append(dict(attempt=attempt, status=None, error=type(e).__name__,
                               message=str(e)[:300], latency_s=round(time.time() - ta, 3)))
            retryable = True
        if not retryable or attempt == MAX_ATTEMPTS:
            resp = None
            break
        # The 429 body and header say "retry after 60 seconds"; wait that long (plus jitter).
        wait = errors[-1].get("retry_after")
        time.sleep((float(wait) if wait else 2 ** attempt) + random.uniform(0, 3))
    t1 = time.time()
    rec.update(ts_end=datetime.now(timezone.utc).isoformat(), t_start=t0, t_end=t1,
               latency_s=round(t1 - t0, 3), attempts=len(errors) + (resp is not None),
               retries=len(errors) if resp is not None else max(len(errors) - 1, 0), errors=errors)
    if resp is None:
        rec.update(ok_http=False, status=None, valid=False, invalid_reason="http_error", parsed=None)
        return rec
    usage = resp.usage.model_dump() if resp.usage else None
    raw_args, parsed, reason = parse(resp, job["tool"])
    rec.update(
        ok_http=True,
        response_id=resp.id,
        response_model=resp.model,
        status=resp.status,
        incomplete_details=resp.incomplete_details.model_dump() if resp.incomplete_details else None,
        error=resp.error.model_dump() if resp.error else None,
        output_types=[i.type for i in resp.output],
        output_text=resp.output_text,
        n_function_calls=sum(i.type == "function_call" for i in resp.output),
        function_call_arguments=raw_args,
        parsed=parsed,
        valid=reason is None,
        invalid_reason=reason,
        input_tokens=usage and usage.get("input_tokens"),
        output_tokens=usage and usage.get("output_tokens"),
        cached_tokens=usage and (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        reasoning_tokens=usage and (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        cost=usage and usage.get("cost"),
        usage=usage,
    )
    return rec


def run_model(client: Adaption, model: str, jobs: list[dict]) -> Path:
    out = HERE / f"{model}.jsonl"
    lock = threading.Lock()
    with out.open("w") as fh, ThreadPoolExecutor(CONCURRENCY) as pool:
        def work(job):
            rec = call(client, model, job)
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
            return rec
        recs = list(pool.map(work, jobs))
    n_valid = sum(r["valid"] for r in recs)
    print(f"{model}: {len(recs)} calls, {n_valid} valid, "
          f"{sum(r['retries'] for r in recs)} retries, "
          f"wall {max(r['t_end'] for r in recs) - min(r['t_start'] for r in recs):.1f}s", flush=True)
    return out


def main(models: list[str]) -> None:
    load_env()
    client = Adaption(max_retries=0, timeout=TIMEOUT_S)
    hosted = sorted(m.id for m in client.models.list(type="hosted").data)
    (HERE / "models.json").write_text(json.dumps(
        dict(listed_at=datetime.now(timezone.utc).isoformat(), sdk_version=adaption.__version__,
             hosted=[m.model_dump() for m in client.models.list(type="hosted").data]),
        indent=1, default=str) + "\n")
    households = json.loads((HERE / "fixtures.json").read_text())["households"]
    jobs = build_jobs(households)
    for model in models or hosted:
        if model not in hosted:
            print(f"{model}: not on this key, skipped", flush=True)
            continue
        run_model(client, model, jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
