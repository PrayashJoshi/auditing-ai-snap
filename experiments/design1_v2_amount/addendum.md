# Design 1 v2 addendum: amount-only arm

To be committed before the first amount-arm call. `freeze.md`, the frozen run and the added-reader arm are unchanged.

## What is added and why

In the frozen inputs, the 48 `stray_ambiguous` inputs are households whose target income group is $0 (QC-verified) and whose records show one small credit under the label "ACH CREDIT ELECTRONIC DEPOSIT" ($40.90 to $219.38). The 24 omission households have verified monthly target amounts of $300 to $3,581. A reader that flags a `stray_ambiguous` input may be responding to the credit's amount rather than to the evidence. This arm keeps the 48 `stray_ambiguous` inputs as they are and changes only the amount of that one credit, to amounts in the omission range.

The arm tests one large non-recurring credit, not omission-like deposits. In 42 of the 48 new inputs the changed credit is the largest transaction in the feed, and in 40 it exceeds the household's reported monthly income. The frozen omission deposit states carry 2 to 9 recurring credits instead.

The arm is exploratory. It is outside reader hypotheses 1 to 3 and is never pooled into hypothesis 3. Results are reported as flags on QC-verified zeros.

## Inputs

- Assignment rule, fixed before any call: sort the 24 omission households by row and take each one's verified monthly target amount (`verified_monthly[target_group]`, the amount every omission input of that household carries). Sort the 48 `stray_ambiguous` inputs by row. Stray input i gets amount[i mod 24], so each amount is used twice.
- The original credit's cents are kept: $145.02 assigned $600 becomes $600.02. The omission amounts are whole dollars and 47 of the 48 original credits have cents, so without this the arm would change the credit's size and its roundness together. The new amounts run from $300.89 to $3,581.46.
- The mapping is in `amount-map.csv`: stray row, original amount, new amount, omission monthly amount, source omission row. In 18 of the 48 the source omission household's target group differs from the stray household's (for example an unearned amount goes to an earned-target household). The rule does not match on group.
- Each new input keeps `cls = zero_stray`, `target_group` and every other field. `evidence_state` is `stray_amount` and `input_id` is `{row}|zero_stray|stray_amount`, so no id collides with a frozen one. The packet's credit amount is changed and the reader text is re-rendered with the frozen renderer, so `reader_input` and `reader_input_sha` change with it.
- Check in `build_amount.py`: every new reader text differs from its `stray_ambiguous` counterpart in exactly one line, the credit line, and only in the amount on it. No total, budget-gap figure or other derived line in the reader text depends on that credit. For example, row 3173: `credit $145.02` becomes `credit $600.02`.

## Comparison

Results are compared within each reader, household by household, with that reader's frozen repeat 1 `stray_ambiguous` record (48 per reader in `results.jsonl`). Flags are read under the rules of `freeze.md` section 7.

- Main rate: a flag on either income group, the hypothesis 3 definition in `freeze.md` section 8. Every income group of these households is filed at its verified value, so a flag on either is false. The comparison counts households that change from unflagged to flagged and from flagged to unflagged.
- Reported beside it: the target-group rate, and which group (earned or unearned) the flag names. In the frozen run, readers attached the small credit to unearned income in most flagged households, whichever group was the target. The target-group rate therefore partly reflects group attribution, which a larger credit could shift.

## Readers and calls

| Reader | Inputs | Repeats | Calls |
|---|---:|---|---:|
| `deepseek-v4-flash` | 48 | 1 | 48 |
| `qwen3.5-397b` | 48 | 1 | 48 |
| `gemma-4-31b-it` | 48 | 1 | 48 |

144 calls, job ids tagged `design1v2amount`. There is no dry run, because these readers were validated on prompts that are identical except for one credit amount. Reader instructions `faa22e9fc6f7b58e`, tool `report_flags` with confidence `d7bf54860554aac4`, token cap 8192, temperature 0, concurrency 8, forced tool call, and the same call function and retry code. The runner stops if any of these differ.

## Files

- Frozen inputs: `experiments/design1_v2/inputs.jsonl`, SHA-256 `9b84d77162ba9e75d1e49e969e2d1d656b6142f54dd816388ac55c54b0737839` (read only).
- Amount inputs: `experiments/design1_v2_amount/amount-inputs.jsonl`, SHA-256 `e0dac1fc12643c2da8acd9fe05a7b765974682b831080175ab1b219f24841f7e`.
- Amount map: `experiments/design1_v2_amount/amount-map.csv`, SHA-256 `212f603759d749fb4f9f144af9851cdfeeaf8185f5ceb500df09a770f84eb46b`.
- Builder: `experiments/design1_v2_amount/build_amount.py`, SHA-256 `85486a40e9aefd5dd48601da620be55a062c09b3aaf86c5ae35cb76efdb44d51`.
- Runner: `experiments/design1_v2_amount/run_amount.py`, SHA-256 `d78c9ada79a1c99efe9097f84063bfe4266005db2ac23cb9c4e3528807ff14d1`.
- Results go to `amount-results.jsonl` in the frozen record format and are not mixed into the frozen `results.jsonl`. Run settings are written to `amount-run-config.json` before any call.
- Estimated cost: about 0.28 credits, from the frozen run's mean cost per call per reader. This is probably slightly high, because the amount inputs are shorter than the frozen average.
