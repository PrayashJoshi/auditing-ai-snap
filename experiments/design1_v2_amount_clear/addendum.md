# Design 1 v2 addendum: clear-label twin of the amount-only arm

To be committed before the first twin call. `freeze.md`, the frozen run, the added-reader arm and the amount-only arm are unchanged.

## What is added and why

The 48 `zero_stray` households have a target income group of $0 (QC-verified) and one small non-recurring credit ($40.90 to $219.38). The frozen run shows them in two states that differ only in that credit's label: `stray_clear` ("TRANSFER FROM SAVINGS ... (OWN ACCOUNT)") and `stray_ambiguous` ("ACH CREDIT ELECTRONIC DEPOSIT"). The amount-only arm (`experiments/design1_v2_amount/`) took the `stray_ambiguous` inputs and changed only the credit's amount, to omission-sized amounts ($300.89 to $3,581.46).

That gives three cells of a 2 x 2 on the same 48 households, label (clear / ambiguous) x amount (small / omission-sized). This arm adds the fourth: clear label, omission-sized amount. It asks whether readers flag an omission-sized credit even when its label explains it as an own-account transfer, or whether the frozen pattern (the ambiguous label alone turns flags on) also holds at omission-sized amounts.

| | small amount | omission-sized amount |
|---|---|---|
| clear label | frozen `stray_clear` | this arm, `stray_clear_amount` |
| ambiguous label | frozen `stray_ambiguous` | amount arm, `stray_amount` |

This arm was designed after the amount arm's results were seen and before any twin call. Those results: flags on either income group rose from the frozen `stray_ambiguous` rate to 47, 39 and 48 of 48 (`deepseek-v4-flash`, `qwen3.5-397b`, `gemma-4-31b-it`), paired household by household. In the frozen run, `stray_clear` drew 0 to 1 flags of 48 per reader and `stray_ambiguous` 34, 32 and 38.

The arm is exploratory. It is outside reader hypotheses 1 to 3 and is never pooled into hypothesis 3. Results are reported as flags on QC-verified zeros.

## Inputs

- The 48 frozen `stray_clear` inputs are copied with one change: the stray credit's amount is set to the amount arm's `new_amount` for the same row in `amount-map.csv`. Each twin therefore carries the same amount as its `stray_amount` counterpart and the same label as its `stray_clear` counterpart. No new assignment is made.
- Each new input keeps `cls = zero_stray`, `target_group` and every other field. `evidence_state` is `stray_clear_amount` and `input_id` is `{row}|zero_stray|stray_clear_amount`, so no id collides with an earlier one. The reader text is re-rendered with the frozen renderer, so `reader_input` and `reader_input_sha` change with it.
- Checks in `build_clear.py`: every twin reader text differs (a) from its `stray_clear` counterpart in exactly one line, only in the amount, and (b) from its `stray_amount` counterpart in exactly one line, only in the label. The same 48 rows are in all four cells. For row 3173:
  - (a) `[doc-04] 2026-09-26 | credit $145.02 | TRANSFER FROM SAVINGS ****3445 (OWN ACCOUNT)` becomes `credit $600.02` on the same line.
  - (b) `[doc-04] 2026-09-26 | credit $600.02 | ACH CREDIT ELECTRONIC DEPOSIT` becomes `... | TRANSFER FROM SAVINGS ****3445 (OWN ACCOUNT)`.

## Comparisons

Two comparisons, fixed before any twin call, both within each reader and household by household. Flags are read under the rules of `freeze.md` section 7.

1. Twin against the frozen repeat 1 `stray_clear` records: the amount effect under a clear label.
2. Twin against the amount arm's `stray_amount` records: the label effect at omission-sized amounts.

- Main rate: a flag on either income group, the hypothesis 3 definition in `freeze.md` section 8. Every income group of these households is filed at its verified value, so a flag on either is false. Each comparison counts households that change from unflagged to flagged and from flagged to unflagged.
- Reported beside it: the target-group rate.

## Readers and calls

| Reader | Inputs | Repeats | Calls |
|---|---:|---|---:|
| `deepseek-v4-flash` | 48 | 1 | 48 |
| `qwen3.5-397b` | 48 | 1 | 48 |
| `gemma-4-31b-it` | 48 | 1 | 48 |

144 calls, job ids tagged `design1v2amountclear`. There is no dry run, because these readers were validated on prompts that are identical except for one credit amount. Reader instructions `faa22e9fc6f7b58e`, tool `report_flags` with confidence `d7bf54860554aac4`, token cap 8192, temperature 0, concurrency 8, forced tool call, and the same call function and retry code. The runner stops if any of these differ.

## Files

- Frozen inputs: `experiments/design1_v2/inputs.jsonl`, SHA-256 `9b84d77162ba9e75d1e49e969e2d1d656b6142f54dd816388ac55c54b0737839` (read only).
- Amount map reused: `experiments/design1_v2_amount/amount-map.csv`, SHA-256 `212f603759d749fb4f9f144af9851cdfeeaf8185f5ceb500df09a770f84eb46b` (read only). The builder also rebuilds the amount arm's inputs in memory and checks them against `amount-inputs.jsonl`, SHA-256 `e0dac1fc12643c2da8acd9fe05a7b765974682b831080175ab1b219f24841f7e`.
- Twin inputs: `experiments/design1_v2_amount_clear/clear-inputs.jsonl`, SHA-256 `054983c3ef7ce938c67a4b81edc502600dc88001ee266355888c76c6f9b6ee34`.
- Builder: `experiments/design1_v2_amount_clear/build_clear.py`, SHA-256 `d2eff921ffb07324e6cc4576957d6900f6f0df773e7481b6a7904cafc163400c`.
- Runner: `experiments/design1_v2_amount_clear/run_clear.py`, SHA-256 `4e63f4a103cc1dde5a54165bf21e7a0e4b2e099035d1542224dd1ba39775b941`.
- Results go to `clear-results.jsonl` in the frozen record format and are not mixed into any earlier results file. Run settings are written to `clear-run-config.json` before any call.
- Scoring, with the existing wrapper `experiments/design1_v2_added/score_results.py`:

  ```
  PYTHONPATH=src python experiments/design1_v2_added/score_results.py \
    --inputs experiments/design1_v2_amount_clear/clear-inputs.jsonl \
    --results experiments/design1_v2_amount_clear/clear-results.jsonl \
    --out-dir experiments/design1_v2_amount_clear --repeats 1 --prefix clear-vs-frozen- \
    --compare-results experiments/design1_v2/results.jsonl \
    --compare-inputs experiments/design1_v2/inputs.jsonl \
    --compare-cell zero_stray/stray_clear

  PYTHONPATH=src python experiments/design1_v2_added/score_results.py \
    --inputs experiments/design1_v2_amount_clear/clear-inputs.jsonl \
    --results experiments/design1_v2_amount_clear/clear-results.jsonl \
    --out-dir experiments/design1_v2_amount_clear --repeats 1 --prefix clear-vs-amount- \
    --compare-results experiments/design1_v2_amount/amount-results.jsonl \
    --compare-inputs experiments/design1_v2_amount/amount-inputs.jsonl \
    --compare-cell zero_stray/stray_amount
  ```
- Estimated cost: about 0.28 credits, from the frozen run's mean cost per call per reader. This is probably slightly high, because the twin inputs are shorter than the frozen average.

Committed before the first call.
