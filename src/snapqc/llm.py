"""Calling hosted models through the Adaption Responses API: one forced tool call per request, every attempt logged.

`call` sends one job (instructions, input, one function tool forced with `tool_choice`) at temperature 0 with
`store=False` and the SDK's own retries off. HTTP 429 and 5xx answers, timeouts and connection errors are retried
here, waiting as long as the 429's `retry-after` says (plus up to 3 s of jitter), so every attempt is in the record.
The account has one rate limit (50 calls per minute) shared by all models and keys.

A record holds the job metadata, the request (model, hashes of instructions, input and tool, output-token cap),
the response (model id, response id, status, incomplete details, raw function-call arguments, the decoded
arguments, whether they pass the tool's JSON schema and why not), token counts and `usage.cost`.
`parsed` holds the decoded arguments even when they fail the schema (None only when there is no call or the
arguments are not JSON), so a looser parse rule can be scored later without new calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import adaption
import jsonschema
from adaption import Adaption

ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE = 0
MAX_ATTEMPTS = 6
TIMEOUT_S = 180
MAX_SCHEMA_ERRORS = 10


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def tool_sha(tool: dict) -> str:
    return sha(json.dumps(tool, sort_keys=True))


def load_env() -> None:
    """Read KEY=value lines from the repo's .env into the environment (existing variables win)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def client(timeout: float = TIMEOUT_S) -> Adaption:
    load_env()
    return Adaption(max_retries=0, timeout=timeout)


def parse(resp, tool: dict) -> tuple[str | None, object, str | None, list[str]]:
    """(raw arguments, decoded arguments or None, reason if invalid, schema errors).
    Valid: status completed, exactly one call to the named tool, JSON arguments that pass the tool's schema."""
    calls = [i for i in resp.output if i.type == "function_call" and i.name == tool["name"]]
    raw = calls[0].arguments if calls else None
    try:
        args = json.loads(raw) if raw is not None else None
    except (json.JSONDecodeError, TypeError):
        args = None
    if resp.status != "completed":
        return raw, args, f"status_{resp.status}", []
    if len(calls) != 1:
        return raw, args, f"function_calls_{len(calls)}", []
    if args is None:
        return raw, None, "json_error", []
    errors = [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
              for e in jsonschema.Draft202012Validator(tool["parameters"]).iter_errors(args)]
    if errors:
        return raw, args, f"schema_error: {errors[0][:160]}", errors[:MAX_SCHEMA_ERRORS]
    return raw, args, None, []


def call(cl: Adaption, model: str, instructions: str, input: str, tool: dict, max_output_tokens: int,
         meta: dict | None = None) -> dict:
    """One forced tool call with logged retries. Returns the record (meta fields first)."""
    rec = dict(meta or {})
    rec.update(request_model=model, temperature=TEMPERATURE, max_output_tokens=max_output_tokens,
               sdk_version=adaption.__version__, tool_name=tool["name"], instructions_sha=sha(instructions),
               input_sha=sha(input), tool_sha=tool_sha(tool))
    errors = []
    resp = None
    t0 = time.time()
    rec["ts_start"] = datetime.now(timezone.utc).isoformat()
    rec["date"] = rec["ts_start"][:10]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ta = time.time()
        try:
            raw = cl.responses.with_raw_response.create(
                model=model, instructions=instructions, input=input, tools=[tool],
                tool_choice={"type": "function", "name": tool["name"]},
                temperature=TEMPERATURE, max_output_tokens=max_output_tokens, store=False)
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
            break
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
    raw_args, parsed, reason, schema_errors = parse(resp, tool)
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
        schema_errors=schema_errors,
        input_tokens=usage and usage.get("input_tokens"),
        output_tokens=usage and usage.get("output_tokens"),
        cached_tokens=usage and (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        reasoning_tokens=usage and (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        cost=usage and usage.get("cost"),
        usage=usage,
    )
    return rec


def hosted_models(cl: Adaption) -> dict:
    """Snapshot of the hosted model list on this key."""
    return dict(listed_at=datetime.now(timezone.utc).isoformat(), sdk_version=adaption.__version__,
                hosted=[m.model_dump() for m in cl.models.list(type="hosted").data])
