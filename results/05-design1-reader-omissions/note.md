# Design 1 — Reader Omission Detection Diagnostic

- **Test ID:** Design 1 (`05-design1-reader-omissions`)
- **Date:** 2026-11-14
- **Branch / Files:** `exp/design1` — [`experiments/design1/run_design1.py`](../../experiments/design1/run_design1.py), [`experiments/design1/inputs.jsonl`](../../experiments/design1/inputs.jsonl), [`experiments/design1/results.jsonl`](../../experiments/design1/results.jsonl), [`experiments/design1/summary.csv`](../../experiments/design1/summary.csv), [`notebooks/05-design1-reader-omissions.py`](../../notebooks/05-design1-reader-omissions.py), [`results/05-design1-reader-omissions/`](./)

## 1. Question & Setup

Can goal-oriented LLM readers (`deepseek-v4-flash`, `qwen3.5-397b`) catch silent income omissions (`$0` vs `"not provided"`) across four progressive file-trace levels without raising false flags on paired truthful twins, and how do they compare against `frozen_checker` and `checker_plus_2_rules`?

- **Households (`n = 12`):** 5 `earned` targets (`5744`, `14542`, `28402`, `42865`, `42777`) and 7 `unearned` targets (`28066`, `30966`, `21739`, `43456`, `41978`, `34791`, `30269`) from [`experiments/pilot/households.csv`](../../experiments/pilot/households.csv). Every selected household has verified shelter costs exceeding remaining non-target income, bank deposit transactions for the target income group, and omission benefit impact exceeding `$56` (`$126`–`$748`).
- **Grid (`144` inputs):** `12` households x `4` trace levels (`none`, `budget_gap`, `net_deposits`, `wage_match`) x `3` states (`omission_zero` with target = `$0`, `omission_not_provided` with target = `not provided`, `truthful_twin`).
- **Level 3 (`net_deposits`) split:** `6` clear deposits (`5744`, `14542`, `28402`, `28066`, `30966`, `21739`) vs `6` ambiguous (`42865`, `42777`, `43456`, `41978`, `34791`, `30269`), where omission target deposits are relabeled `"ACH CREDIT ELECTRONIC DEPOSIT"` and truthful twins' single non-recurring `TRANSFER FROM SAVINGS` is relabeled `"ACH CREDIT ELECTRONIC TRANSFER"`.
- **LLM run (`288` calls):** `temperature = 0`, `max_output_tokens = 8192`, `concurrency = 8`, forced tool `report_flags(confidence=True)`.
  - `deepseek-v4-flash`: `144/144` valid, `0` truncated, `0` retries, `168,718` input tokens, `128,837` output tokens (`119,133` reasoning), median `705.0` output tokens, median latency `5.583s`, `0.0210` credits (`0.020962` exact).
  - `qwen3.5-397b`: `144/144` valid, `0` truncated, `6` retries, `170,742` input tokens, `181,751` output tokens (`173,341` reasoning), median `1,031.0` output tokens, median latency `13.117s`, `0.7300` credits (`0.730037` exact).

## 2. How Numbers Were Computed

- **Exact target hit (`exact_target_hit`):** fraction of omissions where `target_group` appears in `flagged` groups (`n = 12` per level/marker; `n = 6` per L3 subset).
- **Any-income hit (`any_income_hit`):** fraction of omissions where `"earned"` or `"unearned"` appears in `flagged`.
- **Truthful-twin FPR:** `twin_any_fpr` is the fraction of paired truthful twins (`n = 12` per level) with any flag (`len(flagged) > 0`); `twin_income_fpr` restricts to income flags (`"earned"` or `"unearned"`).
- **Net separation:** `exact_target_hit - twin_any_fpr` (primary) and `any_income_hit - twin_any_fpr`, plus paired household differences with 95% percentile bootstrap CIs (`5,000` resamples, `seed = 20261114`).

## 3. Main Results (`experiments/design1/summary.csv`)

### Hit Rate, Twin FPR, and Net Separation by Reader x Trace Level x Marker

| Reader | Trace level (subset) | Marker | Exact hit | Any-inc hit | Twin any FPR | Twin inc FPR | Net sep (exact - any FPR) |
|---|---|---|---:|---:|---:|---:|---:|
| `frozen_checker` | all 4 levels | `$0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `frozen_checker` | all 4 levels | `not_provided` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `checker_plus_2_rules` | `none` | `$0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `checker_plus_2_rules` | `none` | `not_provided` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `checker_plus_2_rules` | `budget_gap`, `net_deposits`, `wage_match` | both | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `deepseek-v4-flash` | `none` | `$0` / `not_provided` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `deepseek-v4-flash` | `budget_gap` | `$0` / `not_provided` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `deepseek-v4-flash` | `net_deposits` (all, n=12) | `$0` / `not_provided` | 0.9167 | 1.0000 | 0.3333 | 0.3333 | 0.5833 |
| `deepseek-v4-flash` | `net_deposits` (clear, n=6) | `$0` / `not_provided` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `deepseek-v4-flash` | `net_deposits` (ambig, n=6) | `$0` / `not_provided` | 0.8333 | 1.0000 | 0.6667 | 0.6667 | 0.1667 |
| `deepseek-v4-flash` | `wage_match` | `$0` / `not_provided` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `qwen3.5-397b` | `none` | `$0` / `not_provided` | 0.0000 | 0.0000 | 0.0833 | 0.0833 | -0.0833 |
| `qwen3.5-397b` | `budget_gap` | `$0` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `qwen3.5-397b` | `budget_gap` | `not_provided` | 0.1667 | 0.2500 | 0.0000 | 0.0000 | 0.1667 |
| `qwen3.5-397b` | `net_deposits` (all, n=12) | `$0` / `not_provided` | 0.8333 | 1.0000 | 0.3333 | 0.3333 | 0.5000 |
| `qwen3.5-397b` | `net_deposits` (clear, n=6) | `$0` / `not_provided` | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| `qwen3.5-397b` | `net_deposits` (ambig, n=6) | `$0` / `not_provided` | 0.6667 | 1.0000 | 0.6667 | 0.6667 | 0.0000 |
| `qwen3.5-397b` | `wage_match` | `$0` / `not_provided` | 1.0000 | 1.0000 | 0.3333 | 0.0000 | 0.6667 |

### Paired Household Comparisons (`4a-paired-reader-diffs.csv`, `4b-paired-marker-diffs.csv`, Pooled Levels 2–4, `n = 12`)

- **`deepseek-v4-flash - frozen_checker`:** `$0` exact hit `+0.6389` `[0.5833, 0.6667]` (`12W/0L/0T`), net sep `+0.5278` `[0.3889, 0.6389]`; `not_provided` exact hit `-0.3611` `[-0.4167, -0.3333]` (`0W/12L/0T`), net sep `-0.4722` `[-0.5556, -0.3889]`; `twin_any_fpr` `+0.1111` `[0.0278, 0.1944]`.
- **`deepseek-v4-flash - checker_plus_2_rules`:** `$0` exact hit `-0.3611` `[-0.4167, -0.3333]`, net sep `-0.4722` `[-0.6111, -0.3611]`; `not_provided` exact hit `-0.3611`, net sep `-0.4722` `[-0.5556, -0.3889]`.
- **`qwen3.5-397b - frozen_checker`:** `$0` exact hit `+0.6111` `[0.5278, 0.6667]` (`12W/0L/0T`), net sep `+0.3889` `[0.2222, 0.5556]`; `not_provided` exact hit `-0.3333` `[-0.4444, -0.2222]`, net sep `-0.5556` `[-0.7500, -0.3611]`; `twin_any_fpr` `+0.2222` `[0.1111, 0.3333]`, `twin_income_fpr` `+0.1111` `[0.0278, 0.1944]`.
- **`qwen3.5-397b - checker_plus_2_rules`:** `$0` exact hit `-0.3889` `[-0.4722, -0.3333]`, net sep `-0.6111` `[-0.7778, -0.4444]`; `not_provided` exact hit `-0.3333`, net sep `-0.5556` `[-0.7500, -0.3611]`.
- **Marker shift (`not_provided - $0`, pooled L2–4):** `deepseek-v4-flash` `0.0000` `[-0.0833, 0.0833]` exact (`0.0000` any-income); `qwen3.5-397b` `+0.0556` `[0.0000, 0.1389]` exact (`+0.0833` `[0.0000, 0.1667]` any-income).

## 4. Key Findings & Mismatches

1. **Budget-gap blind spot (Level 2):** Despite explicit instruction to flag income omissions when housing/utility outflows exceed reported income, `deepseek-v4-flash` scored `0/24` across both `$0` and `not_provided`. `qwen3.5-397b` scored `0/12` on `$0` and `2/12` exact (`3/12` any-income) on `not_provided` — inspecting those 3 cases shows only `1` (`30269`) explicitly used the `$1,267` housing/utility budget gap on the omitted group (`14542` cited living expenses but flagged `$0` `unearned` instead of `not provided` `earned`; `28066` flagged `unearned` at `0.35` confidence due to a `$222.20` savings transfer). Meanwhile `checker_plus_2_rules` reached `12/12` (`0%` twin FPR).
2. **Abstention (`"not provided"`) under-use:** On Level 1 (`none`), both LLMs scored `0/12` on `not_provided`, treating `"not provided"` nearly identically to `"$0"` unless other evidence existed (`checker_plus_2_rules` and `frozen_checker` both caught `12/12` via missing-documentation scope rules).
3. **Clear vs. ambiguous bank deposits (Level 3):** With clear descriptions (`6/6` clear subset), both LLMs achieved `100%` hit rate and `0%` twin FPR (`+1.0` net separation). With generic `"ACH CREDIT ELECTRONIC DEPOSIT"` / `"ACH CREDIT ELECTRONIC TRANSFER"` descriptions (`6/6` ambiguous subset), both caught `6/6` (`100%`) at the `any_income` level (missing exact `unearned` vs `earned` on `$0`-both cases: `deepseek-v4-flash` on `41978` `$0` and `34791` `not_provided`; `qwen3.5-397b` on `43456` and `34791` for both markers), **but falsely flagged `4/6` (`66.7%`) truthful twins** even though those twins contained only a single non-recurring transfer (`$51.07`–`$243.14`).
4. **Spurious `$0` shelter/utilities flags on `qwen3.5-397b`:** Despite explicit instruction *not* to flag `$0` solely for absent records when nothing conflicts with `$0`, `qwen3.5-397b` flagged `['shelter', 'utilities']` as "unusual" on `4/12` Level 4 twins (`14542`, `42865`, `42777`, `34791`, giving `33.3%` twin FPR) and `1/12` Level 1 twins (`28066`, where it also miscalculated biweekly `$177.67` pay as inconsistent with `$382`). `deepseek-v4-flash` kept `0/12` twin FPR across Levels 1, 2, and 4.
5. **Confidence separation (`6c-confidence.csv`):** Target-group flags averaged `0.8548` (`[0.60, 0.98]`) for `deepseek-v4-flash` and `0.8750` (`[0.35, 0.95]`) for `qwen3.5-397b`, compared to twin false-flag means of `0.5750` (`[0.50, 0.60]`, `deepseek-v4-flash`) and `0.6000` (`twin_income`) / `0.6250` (`twin_non_income`) for `qwen3.5-397b`.

## 5. Which Part of `rq.md` It Feeds

- **RQ1 (reader accuracy & failure modes on silent omissions and abstention):** Quantifies exact sensitivity/FPR across trace strength (`none` -> `budget_gap` -> `net_deposits` -> `wage_match`) and shows where goal-oriented LLM prompting succeeds (explicit wage-match leads, clear bank deposits) vs. breaks (arithmetic budget gaps, single ambiguous ACH transfers, `"not provided"` handling, spurious `$0` expense skepticism).
- **RQ2 (deterministic rules + checker vs. LLM readers):** Demonstrates that two simple deterministic rules (`checker_plus_2_rules`) strictly dominate both frontier/flash goal-oriented LLMs on net separation (`+1.0` vs `+0.5833`/`+0.5000` on L3 and `+1.0` vs `0.0000`/`+0.1667` on L2) while maintaining `0%` twin FPR.

## 6. Recommendation — NOT DECIDED (for Prayash)

- **NOT DECIDED (for Prayash):** Whether Design 2 / full evaluation should (a) adopt `checker_plus_2_rules` (or require explicit arithmetic pre-checks for lease + utility bills vs. reported income rather than relying on the LLM reader for Level 2 budget gaps), (b) require `>= 2` recurring ACH deposits or add an ACH confidence/escalation threshold (`~0.65`) before acting on ambiguous generic bank transfers, and (c) enforce strict `"not provided"` abstention handling via deterministic pre-check rather than prompt wording.

## 7. LLM Use

- Experiment harness, pre-flight assertions, scoring scripts, notebook (`05-design1-reader-omissions`), and report drafted and executed with Antigravity (Gemini); reader inference runs executed via `deepseek-v4-flash` and `qwen3.5-397b`.
