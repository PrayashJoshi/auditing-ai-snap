# 03 Pre-experiment model checks: note

Notebook: `notebooks/03-model-checks.ipynb`. Only section 1 has been run. Raw calls: `experiments/smoke/*.jsonl` (main run), `experiments/smoke/run1-short-backoff/` (first pass, kept as evidence of the rate limit), and `experiments/smoke/rate-probe/probe.jsonl` (two-key rate-limit check). Runners: `experiments/smoke/run_smoke.py` and `experiments/smoke/probe_keys.py`. Inputs: 10 hand-written toy households (`experiments/smoke/fixtures.json`, about 200 input tokens each). They are not QC records.

## 1. Model smoke test

- **Models on the key** (`models.list(type="hosted")`, 2026-11-14, SDK 0.19.0): `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `qwen3.8-27b`, `qwen3.5-397b`, `glm-5.3-flash`, `glm-5.2`, `deepseek-v4-flash`. All seven answered all 50 calls. The largest are `qwen3.5-397b` and `glm-5.2`. No closed frontier model is listed.
- **Valid output** (50 calls per model: 30 reader calls with a forced `report_flags` call, 20 filer calls):

| Model | All 50 | Reader (30) | Filer, required (10) | Filer, nullable + 'unknown' (10) | Hit token cap |
|---|---:|---:|---:|---:|---:|
| gemma-4-26b-a4b-it | 100% | 100% | 100% | 100% | 0 |
| gemma-4-31b-it | 100% | 100% | 100% | 100% | 0 |
| qwen3.8-27b | 96% | 100% | 100% | 80% | 2 |
| qwen3.5-397b | 98% | 96.7% | 100% | 100% | 1 |
| glm-5.3-flash | 90% | 100% | 90% | 60% | 2 |
| glm-5.2 | 86% | 80% | 90% | 100% | 0 |
| deepseek-v4-flash | 98% | 100% | 100% | 90% | 1 |

- **Below the 95% target:** `glm-5.3-flash` and `glm-5.2`. `glm-5.2` sends empty arguments (`{}`) instead of `{"flags": []}` on 6 of its 9 reads of the three clean households, and once on a filer call. `glm-5.3-flash` writes 'unknown' with its quotes (`"\"unknown\""`) on 3 filer calls. Both are counted invalid by the parse rule. The raw arguments are kept, so a different parse rule can be scored without new calls.
- **Forced tool call:** `tool_choice={"type": "function", "name": ...}` on a custom function is honoured by all seven models. Every answered call that finished contains exactly one call to the named tool.
- **Reasoning and truncation:** every model except the two Gemmas returns a `reasoning` item before the tool call, with median reader output of 170 to 360 tokens (Gemma: 18 tokens). 6 of 350 calls stopped at the output-token cap (`status` `incomplete`, reason `max_output_tokens`): 5 filer calls at 1,024 tokens and 1 reader call at 512. In those calls the reasoning used up to all 1,024 tokens. Once, `gemma-4-31b-it` wrote a text message before the tool call.
- **Calls per hour:** the account has one inference rate limit per user, shared by all models and keys. At most 50 calls were answered in any 60 seconds. After that the API returns HTTP 429 ("Inference rate limit reached for this user; retry after 60 seconds", `retry-after: 60`). Successful responses carry no rate-limit headers. In a two-key probe (395 short calls across two keys, `1c-key-probe.csv`), both keys together hit the same 50-call ceiling (when `key1` took 50 of 80 calls in a joint burst, `key2` got HTTP 429 on all 80; when `key2` took 50 calls first, `key1` got HTTP 429 on all 5 immediate follow-up calls). The full run of 350 calls took 9.2 minutes, 2,283 calls per hour including the 60-second waits. The limit alone allows about 3,000 per hour. In the first pass, retries waited only 2 to 8 seconds, and 42 of 350 calls failed after 4 attempts. The main run waits as long as the 429 says: it retried 30 times and every call was answered. Median time per call without a retry: 0.6 s (`gemma-4-26b-a4b-it`) to 4.0 s (`qwen3.5-397b`).
- **Credits:** the main run cost 0.200 credits (183,082 input tokens, 70,528 of them cached; 79,381 output tokens, 63,209 of them reasoning). The first pass cost another 0.186 credits, and the two-key probe 0.0007 credits. Per reader call: 0.00002 (`gemma-4-26b-a4b-it`) to 0.0016 (`qwen3.5-397b`) credits. `usage.cost` is returned on every call but is not in the docs. The token-price formula matches it exactly for five models. For `gemma-4-26b-a4b-it` the formula is 1.3% higher and for `deepseek-v4-flash` 4.2% higher, so cached input on those two is priced differently from the models page. The account balance was not read.
- **Three reads at temperature 0:** exact agreement of the flag set across the three reads of each household.

| Model | Three valid reads | Agree | Majority equals planted flags (toy check) |
|---|---:|---:|---:|
| gemma-4-26b-a4b-it | 10 | 9 | 5 of 10 |
| gemma-4-31b-it | 10 | 9 | 5 of 10 |
| qwen3.8-27b | 10 | 10 | 10 of 10 |
| qwen3.5-397b | 9 | 9 | 10 of 10 |
| glm-5.3-flash | 10 | 10 | 10 of 10 |
| glm-5.2 | 7 | 7 | 8 of 10 |
| deepseek-v4-flash | 10 | 10 | 10 of 10 |

  Both Gemmas flag `earned_income` on households where the wages are right. Toy fixtures do not measure reader accuracy, but the pattern is consistent across both Gemmas.
- **Filer schema variants:** the nullable variant (each money field a number, null or 'unknown', using JSON schema `anyOf`) is accepted by all seven models, and all seven used null or 'unknown' at least once. Nullable validity is lower than required validity for `glm-5.3-flash` (60% vs 90%), `qwen3.8-27b` (80% vs 100%) and `deepseek-v4-flash` (90% vs 100%). The causes are truncation and the quoted 'unknown'.
- **Mismatches with the documentation:** the models page names `glm-53-flash`, but the key lists `glm-5.3-flash`. The docs give no rate limit, but the key has one. The docs do not mention `usage.cost`.
- **How computed:** valid means status `completed`, exactly one call to the named tool, and JSON arguments that pass the tool's JSON schema, enum values included. A call with no HTTP answer counts as invalid. Calls run at concurrency 5 with temperature 0 and `store=False`. SDK retries are off; the runner retries 429 and 5xx up to 6 times and logs every attempt. Tables: `1a-models.csv`, `1b-valid-output.csv`, `1b-invalid-reasons.csv`, `1c-rate-limit.csv`, `1c-key-probe.csv`, `1c-calls-per-minute.csv` and `.svg`, `1c-latency.csv`, `1c-credits.csv`, `1d-agreement.csv`, `1e-filer-validity.csv`.
- **Options:**
  - Reader shortlist: by valid output and agreement, the candidates are `deepseek-v4-flash`, `qwen3.8-27b` and `qwen3.5-397b`, plus `gemma-4-31b-it` as a fast non-reasoning model that over-flags here.
  - `glm-5.2` is reachable if `{}` is parsed as no flags.
  - Filer: the same models.
  - Output-token caps: raise them for reasoning models (for example 2,048 for the filer).
  - Run time: plan around the shared limit of 50 calls per minute.
- **Feeds:** `rq.md` RQ1 (reader and filer shortlist, run time).

Summarized by Claude Code (Opus 5.5).
