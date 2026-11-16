# Design 1 — Reader Omission Detection Diagnostic

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

1. **Budget-gap blind spot (Level 2):** Despite explicit instruction to flag income omissions when housing/utility outflows exceed reported income, `deepseek-v4-flash` scored `0/24` across both `$0` and `not_provided`. `qwen3.5-397b` scored `0/12` on `$0` and `2/12` exact (`3/12` any-income) on `not_provided` — inspecting those 3 cases shows only `1` (`30269`) explicitly used the `$1,267` housing/utility budget gap on the omitted group (`14542` cited living expenses but flagged `$0` `unearned` instead of `not provided` `earned`; `28066` flagged `unearned` at `0.35` confidence due to a `$222.20` savings transfer). `checker_plus_2_rules` reached `12/12` (`0%` twin FPR), but by construction: the 12 households were selected because housing exceeds the income left after dropping the target group, the input checks assert that condition on every level-2 omission and its negation on every level-2 twin, and level-2 twins report their income, so the budget-gap rule cannot fire on them.
2. **Abstention (`"not provided"`) under-use:** On Level 1 (`none`), both LLMs scored `0/12` on `not_provided`, treating `"not provided"` nearly identically to `"$0"` unless other evidence existed (`checker_plus_2_rules` and `frozen_checker` both caught `12/12` via missing-documentation scope rules).
3. **Clear vs. ambiguous bank deposits (Level 3):** With clear descriptions (`6/6` clear subset), both LLMs achieved `100%` hit rate and `0%` twin FPR (`+1.0` net separation). With generic `"ACH CREDIT ELECTRONIC DEPOSIT"` / `"ACH CREDIT ELECTRONIC TRANSFER"` descriptions (`6/6` ambiguous subset), both caught `6/6` (`100%`) at the `any_income` level (missing exact `unearned` vs `earned` on `$0`-both cases: `deepseek-v4-flash` on `41978` `$0` and `34791` `not_provided`; `qwen3.5-397b` on `43456` and `34791` for both markers), **but falsely flagged `4/6` (`66.7%`) truthful twins** even though those twins contained only a single non-recurring transfer (`$51.07`–`$243.14`).
4. **Spurious `$0` shelter/utilities flags on `qwen3.5-397b`:** Despite explicit instruction *not* to flag `$0` solely for absent records when nothing conflicts with `$0`, `qwen3.5-397b` flagged `['shelter', 'utilities']` as "unusual" on `4/12` Level 4 twins (`14542`, `42865`, `42777`, `34791`, giving `33.3%` twin FPR) and `1/12` Level 1 twins (`28066`, where it also miscalculated biweekly `$177.67` pay as inconsistent with `$382`). At Level 4 the input itself reports shelter and utilities as `$0` (they are reported at their verified values only at Level 2), so these flags are partly an input artefact. At Level 1 the twin input is byte-identical to the `omission_zero` input for all 12 households (same input hash), so the Level 1 twin is a repeat run, not a negative (part 4a). `deepseek-v4-flash` kept `0/12` twin FPR across Levels 1, 2, and 4.
5. **Confidence separation (`6c-confidence.csv`):** Target-group flags averaged `0.8548` (`[0.60, 0.98]`) for `deepseek-v4-flash` and `0.8750` (`[0.35, 0.95]`) for `qwen3.5-397b`, compared to twin false-flag means of `0.5750` (`[0.50, 0.60]`, `deepseek-v4-flash`) and `0.6000` (`twin_income`) / `0.6250` (`twin_non_income`) for `qwen3.5-397b`.

### 4a. Repeat runs and flags by group (zero model calls)
Computed by `experiments/design1/retest_design1.py` from `results.jsonl`; tables [retest-level1.csv](../../results/05-design1-reader-omissions/retest-level1.csv), [group-flag-matrix.csv](../../results/05-design1-reader-omissions/group-flag-matrix.csv), [false-flags-by-group.csv](../../results/05-design1-reader-omissions/false-flags-by-group.csv).
- **Repeat runs at Level 1.** The 12 twin inputs at Level 1 have the same input hash as the 12 `omission_zero` inputs, so each pair is one input read twice at temperature 0. `deepseek-v4-flash` gave identical flag sets on 12 of 12 pairs (95% Wilson 0.76 to 1.00; no flag in either run). `qwen3.5-397b` gave identical flag sets on 11 of 12 (0.65 to 0.99); row 28066 had no flag in one run and `earned`, `shelter`, `utilities` in the other. The `qwen3.5-397b` Level 1 net separation of -0.0833 is this run-to-run difference, not a false flag on a truthful file.
- **Twin false flags by group.** `deepseek-v4-flash`: 4 of 48 twins flagged, all at Level 3 and all on an income group (`unearned`). `qwen3.5-397b`: 9 of 48. Of these, 4 are income-group flags at Level 3, 1 is the Level 1 repeat-run difference (row 28066, which includes `earned`), and 4 are shelter or utilities only at Level 4.
- **Non-target flags on omissions.** `deepseek-v4-flash` 8 of 96, all at Level 3 and all on the other income group. `qwen3.5-397b` 14 of 96: 8 on an income group, 6 on shelter or utilities only.
- The "twin" is a different kind of negative at each level. At Level 1 it is identical to the omission. At Levels 2 and 4 it reports verified income with documents. At Level 3 it reports $0 for the target group with one non-recurring credit, which is the only genuine hard negative. False-flag rates are therefore not comparable across levels.

## 5. Which Part of `rq.md` It Feeds

- **RQ1 (reader accuracy on silent omissions and abstention):** Measures hit and false-flag rates across the four file-trace levels. It shows where the readers succeed (an explicit wage-match lead, clear bank deposits) and where they fail (the level-2 budget gap, ambiguous single credits, `"not provided"`). The twin is not a consistent negative across levels (part 4a), and n = 12 households.
- **RQ2 (where existing review rules fall short):** The design cannot rank the two deterministic rules against the readers. The 12 households were chosen so that the budget-gap rule fires on every omission and cannot fire on any twin. On truthful FY2024 QC households the same rule fires on 5.8% (rent only) to 11.9% (rent plus simulated bills) of households that have a zero income group. It catches 34% to 44% of the 174 real income omissions in the human arm (see the budget-gap rule base-rate note). The unmatched-deposit rule fires on 0 of 1,000 truthful generated packets, so its false-positive rate cannot be measured outside the generator.

## 6. Open Questions

- How the prompt should state the handling of `"not provided"`. The prompt exempts `"$0"` from the "flag only with a concrete reason" clause but says nothing about `"not provided"`, so the readers' 0 of 12 at Level 1 is partly a wording effect.
- Whether a rebuilt run should use one negative taxonomy at every level: truthful and documented; true zero with rent; true zero with a stray credit. That would also report shelter and utilities at their verified values at every level and draw households at random from QC strata.
- Whether the clear-versus-ambiguous contrast at Level 3 should be run within each household. At present the split is fixed per household (6 and 6).

Summarized by Claude Code (Opus 5.5).
