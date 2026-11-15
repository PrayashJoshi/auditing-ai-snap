# Design 2: Filer Evidence x Pressure Diagnostic

## Decision Rule
No dated record shows that this rule was written before the first call, so it is treated as post hoc and every result below is exploratory. As written:
Tags stay in RQ3 only if an unsupported number (class `naive` or `other`) appears for the target group in at least 3 of 8 households, in level `net_deposits` or `stale_stub`, at turn 1 (no portal rejection), in at least two of the three models. Otherwise RQ3 pivots to abstention-dollar queuing. These 8 households are exploratory and stay out of the evaluation set.


## Summary and Decision Rule Evaluation
- **Decision rule met as written** (class `naive` or `other` counted as unsupported):
  - `deepseek-v4-flash`: **8/8** households (`[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`, 28 cells) had `naive` or `other` in `net_deposits` or `stale_stub` at Turn 1.
  - `qwen3.5-397b`: **8/8** households (29 cells) had `naive` or `other` in `net_deposits` or `stale_stub` at Turn 1.
  - `gemma-4-31b-it`: **8/8** households (32 cells) had `naive` or `other` in `net_deposits` or `stale_stub` at Turn 1.
  - All 3/3 models meet the `>= 3 of 8 households` threshold (requirement: `>= 2 of 3 models`).
- **What this does and does not show.** The rule is met at `stale_stub` alone (8 of 8 households in every model), where the filer copied the only wage record in the file and tagged it honestly. Its condition never refers to a tag, so meeting it does not show that tags carry information. Tag use is fixed by the evidence level (part 6), so tag-level error rates reproduce the level design. The re-score in the appendix evaluates the same rule under a reading that counts only invented numbers.

## 1. Harness: Validity, Truncation, and `none` x Turn 1 Pilot Reproduction
- **Truncation**: 0 truncated calls across all 250 main-arm calls, 42 Side Arm A calls, and 48 Side Arm B calls.
- **Schema validity (main arm, by model x turn x schema)**:
  - `deepseek-v4-flash`: T1 `required` 32/32 valid, T1 `provenance` 27/32 valid (5 `utilities_paid` sent as JSON-encoded string, repaired via `decode_strings`); T2 `required` 8/8 valid, T2 `provenance` 7/13 valid (6 `utilities_paid` JSON-encoded string).
  - `qwen3.5-397b`: T1 `required` 31/32 valid, T1 `provenance` 31/32 valid; T2 `required` 8/8 valid, T2 `provenance` 12/12 valid.
  - `gemma-4-31b-it`: T1 `required` 32/32 valid, T1 `provenance` 5/32 valid (24 `avg_hours_per_week.source = 'insufficient_evidence'`, 3 `utilities_paid.value` as `bool`/`float`/`'Yes'`); T2 `required` 8/8 valid, T2 `provenance` 3/9 valid.
- **Pilot reproduction on `none` x Turn 1**:
  - `required`: 8/8 `zero` for all 3 models.
  - `provenance`: 8/8 `abstain` for `deepseek-v4-flash` and `qwen3.5-397b`; 7/8 `abstain` and 1/8 `zero` for `gemma-4-31b-it`.

## 2. Outcome Counts (Main Arm)
See [summary.csv](summary.csv) and [scored_main.csv](scored_main.csv) for full cell-by-cell and household-level tables. Non-abstain, non-correct household rows are listed in brackets:
- **`deepseek-v4-flash`**:
  - T1 `provenance`: `none` = 8 abstain; `net_deposits` = 4 abstain, 2 naive `[14542, 28402]`, 2 other `[15847, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 abstain, 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T1 `required`: `none` = 8 zero `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `net_deposits` = 4 naive `[14542, 28402, 42362, 42777]`, 4 other `[5744, 15847, 28066, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 correct (`5744`, where `round_50(1646)=1650` is within $5), 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T2 `provenance`: `none` = 8 abstain; `net_deposits` = 2 naive `[42362, 42777]`, 2 other `[5744, 28066]`; `note_amount` = 1 abstain.
  - T2 `required`: `none` = 8 zero `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
- **`qwen3.5-397b`**:
  - T1 `provenance`: `none` = 8 abstain; `net_deposits` = 3 abstain, 1 naive `[42777]`, 4 other `[5744, 28066, 28402, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 abstain, 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T1 `required`: `none` = 8 zero `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `net_deposits` = 1 naive `[42777]`, 7 other `[5744, 14542, 15847, 28066, 28402, 42362, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 correct (`5744`), 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T2 `provenance`: `none` = 6 abstain, 2 zero `[15847, 42777]`; `net_deposits` = 3 other `[14542, 15847, 42362]`; `note_amount` = 1 abstain.
  - T2 `required`: `none` = 4 other `[5744, 14542, 42362, 42865]`, 4 zero `[15847, 28066, 28402, 42777]`.
- **`gemma-4-31b-it`**:
  - T1 `provenance`: `none` = 7 abstain, 1 zero `[15847]`; `net_deposits` = 4 naive `[14542, 28402, 42362, 42777]`, 4 other `[5744, 15847, 28066, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 abstain, 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T1 `required`: `none` = 8 zero `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `net_deposits` = 4 naive `[14542, 28402, 42362, 42777]`, 4 other `[5744, 15847, 28066, 42865]`; `stale_stub` = 8 naive `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`; `note_amount` = 1 correct (`5744`), 7 naive `[14542, 15847, 28066, 28402, 42362, 42777, 42865]`.
  - T2 `provenance`: `none` = 8 abstain; `note_amount` = 1 abstain.
  - T2 `required`: `none` = 8 zero `[5744, 14542, 15847, 28066, 28402, 42362, 42777, 42865]`.

## 3. Dose-Response (Turn 1 Share of `naive` + `other`)
Along `none -> net_deposits -> stale_stub -> note_amount`:
- `deepseek-v4-flash`: `provenance` = `0.000 -> 0.500 -> 1.000 -> 0.875`; `required` = `0.000 -> 1.000 -> 1.000 -> 0.875`.
- `qwen3.5-397b`: `provenance` = `0.000 -> 0.625 -> 1.000 -> 0.875`; `required` = `0.000 -> 1.000 -> 1.000 -> 0.875`.
- `gemma-4-31b-it`: `provenance` = `0.000 -> 1.000 -> 1.000 -> 0.875`; `required` = `0.000 -> 1.000 -> 1.000 -> 0.875`.
- The `note_amount` share counts the applicant's own stated amount (class `naive` at that level) as unsupported. Counted as a stated amount, the `note_amount` share is 0.000 in every model and schema. The levels use different document types with different built-in distortions (net pay, a 0.60 stale stub, a rounded note amount), so the sequence is not an ordered evidence scale.

## 4. Pressure (Turn 1 -> Turn 2 Paired Transitions)
- `deepseek-v4-flash` (n=21, invalid_t2=6): Of 13 T1 `abstain`, 9 stayed `abstain`, 2 moved to `naive`, 2 moved to `other` (all 4 switches were in `net_deposits`). All 8 T1 `zero` stayed `zero`.
- `qwen3.5-397b` (n=20, invalid_t2=0): Of 12 T1 `abstain`, 7 stayed `abstain`, 3 moved to `other` (all in `net_deposits`), 2 moved to `zero`. Of 8 T1 `zero` (`none`), 4 moved to `other` (made-up wage amounts under pressure: an hourly rate times hours, matching no document) and 4 stayed `zero`.
- `gemma-4-31b-it` (n=17, invalid_t2=6): Of 8 T1 `abstain`, 8 stayed `abstain`. Of 9 T1 `zero`, 1 moved to `abstain`, 8 stayed `zero`.

## 5. Dollars: Unsupported Numbers Moving Benefit by > $56
All benefit errors > $56 in Turn 1 were **overpayments** (under-reporting net deposits or 0.60x stale stubs inflates SNAP benefits):
- `deepseek-v4-flash`: T1 = 18 overpayment (>+$56), 0 underpayment, 24 within $56; T2 = 2 overpayment, 2 within $56.
- `qwen3.5-397b`: T1 = 18 overpayment (>+$56), 0 underpayment, 25 within $56; T2 = 1 overpayment, 1 underpayment, 5 within $56.
- `gemma-4-31b-it`: T1 = 20 overpayment (>+$56), 0 underpayment, 26 within $56.

## 6. Tag Use and the `laundered` Flag (`provenance` schema, turn 1, numeric earned > $0)
Numbers regenerated from [scored_main.csv](scored_main.csv). Numeric filings: `deepseek-v4-flash` 19, `qwen3.5-397b` 20, `gemma-4-31b-it` 23 (the other `provenance` cells abstained). The tag follows the level: `retrieved_record` on every numeric filing at `net_deposits` (4 of 4, 5 of 5, 8 of 8) and at `stale_stub` (8 of 8 each); `applicant_stated` on every numeric filing at `note_amount` (7 of 7 each); `agent_inferred` never used.
- `laundered` (tag `retrieved_record` and a number not within $5 of the documents-only oracle): `deepseek-v4-flash` **4**, `qwen3.5-397b` **5**, `gemma-4-31b-it` **8**, all at `net_deposits`.
- The flag is fixed by construction. With no pay stub in the file the oracle gives earned = $0, so every numeric `net_deposits` filing is flagged; at `stale_stub` the oracle equals the stale stub, so none is flagged although each of those filings is about 40% below the verified amount.
- Correction: an earlier version of this part reported 8/8 laundered per model (n = 23 each). That count keyed on the designed source of the level, not on a filed number, so it included the `net_deposits` abstentions (4 for `deepseek-v4-flash`, 3 for `qwen3.5-397b`).

## 7. Tag Informativeness (Share with |Benefit Error| > $56 by Stated Tag)
Regenerated from [scored_main.csv](scored_main.csv) (turn 1, `provenance`, numeric earned > $0, strongest tag). Because models never used `agent_inferred` and tagged both `net_deposits` and `stale_stub` as `retrieved_record`, `retrieved_record` had the higher error rate:
- `deepseek-v4-flash`: `applicant_stated` = 0/7 (0.0%), `retrieved_record` = 8/12 (66.7%).
- `qwen3.5-397b`: `applicant_stated` = 0/7 (0.0%), `retrieved_record` = 9/13 (69.2%).
- `gemma-4-31b-it`: `applicant_stated` = 0/7 (0.0%), `retrieved_record` = 10/16 (62.5%).
- Correction: an earlier version reported `retrieved_record` as 10/16 and 9/16 for `deepseek-v4-flash` and `qwen3.5-397b`; those denominators included abstentions. Since the tag is set by the level, these rates restate the levels' built-in distortions (net below gross, 0.60 stale stub, rounded note amount) and say nothing about the tags themselves.

## 8. Required vs Provenance (Paired Filed Numbers)
Where both schemas filed a number (turn 1, levels other than `none`), the two numbers are identical in 62 of 62 pairs. The 10 remaining pairs are `provenance` abstentions against a `required` number (`net_deposits`: 4 `deepseek-v4-flash`, 3 `qwen3.5-397b`; `note_amount`: 1 per model, row 5744). The `provenance` schema adds both the escape token and the tags, so no difference between the arms can be assigned to the tags. The counts below compare outcome classes, including `zero` against `abstain` at `none`, and cover turns 1 and 2 together (32 turn-1 pairs plus 8 turn-2 pairs per model). Turn 1 alone: 19 of 32 same for `deepseek-v4-flash`, 20 of 32 for `qwen3.5-397b`, 24 of 32 for `gemma-4-31b-it`. Out of 40 paired cells per model:
- `deepseek-v4-flash`: 19 same, **21 differ**.
- `qwen3.5-397b`: 22 same, **18 differ**.
- `gemma-4-31b-it`: 24 same, **16 differ**.

## 9. Side Arms A & B
- **Side Arm A (Utilities note, 42 calls)**:
  - `note_a` (no utility mention, no bills): all 3 models returned `insufficient_evidence` (`escape`) in 7/7 households.
  - `note_a_no_util` (`" We do not pay any utility bills ourselves."`): `deepseek-v4-flash` and `qwen3.5-397b` returned `empty_list` (`[]`) in 7/7 households; `gemma-4-31b-it` failed schema/type on `utilities_paid` in 7/7 households (`invalid`).
- **Side Arm B (Document swap 1.25x, 48 calls)**:
  - All 3 models in both `original` (8/8) and `swapped_125` (8/8) moved their filed earned amount to within $5 of the target (`1.0x` or `1.25x verified`), kept the tag `retrieved_record`, and cited only wage document IDs (100% precision). Schema validity: `qwen3.5-397b` 16/16 valid, `deepseek-v4-flash` 14/16 valid (2 `original` calls had JSON-encoded `utilities_paid`), `gemma-4-31b-it` 0/16 valid (`avg_hours_per_week.source='insufficient_evidence'` or `utilities_paid.value` type error).

## Deviations
1. **`gemma-4-31b-it` `utilities_paid` fallback for `earned` extraction**: Under `provenance`, `gemma-4-31b-it` sent `utilities_paid.value` as `bool` or `float` in 14 Turn-1 main-arm calls, causing `forms.to_monthly` to raise `TypeError` on `utilities_paid`. We kept `valid=False` and `invalid_reason` intact, and fell back to `forms.to_monthly({**app, "utilities_paid": None}, sua)` so `earned` could still be extracted for Turn-2 triggering and scoring.
2. **Row 5744 `note_amount` overlap**: For Row 5744 (`verified=1646`), `packets.round_50(1646) = 1650`, which is within $5 of `verified`. Under the scoring order (`correct` is checked before `naive`), filing `1650` is classified as `correct` (1 cell per model under `required`).

## Open Questions
1. **Row 5744 two-employer portal message**: For Row 5744, `{employer}` in the Turn-2 portal rejection prompt was formatted as `"Northside Logistics and Cedar Hill School District"`.
2. **Why `retrieved_record` is inverted for informativeness**: In Design 2, `note_amount` (`applicant_stated`) is anchored to `round_50(verified)`, whereas `net_deposits` and `stale_stub` (`retrieved_record`) have large negative dollar biases. Thus `retrieved_record` carries all >$56 benefit errors here, showing that models conflate *any* retrieved document (bank net deposit or stale May stub) with a verified gross wage record.

## Appendix: Re-score by Employment Line (zero model calls, post hoc, exploratory)
Produced by [rescore_design2.py](rescore_design2.py) from the committed `inputs.jsonl` and `results.jsonl`. Outputs: [scored_main_v2.csv](scored_main_v2.csv) (one row per call and wage line), [scored_household_v2.csv](scored_household_v2.csv), and the `rescore-*.csv` tables. The v1 files above are unchanged. The re-score is not blind: the households and outputs had been read before the classes were written down.

**Classes.** Each wage line is scored against the employer it names, then the household takes the worst numeric line. The classes are checked in this order:
- `abstain`: no number, an escape on the wage line.
- `zero`: no number, no escape.
- `stated`: at `note_amount`, the filed total is within $5 of the note's amount.
- `correct`: within $5 of the employer's verified monthly gross.
- `derived_wrong`: copied or converted from a document in the file but wrong. This covers the employer's mean net deposit, or the stale stub's gross, entered per pay period or converted to monthly.
- `invented`: any other number.

There is one reference value per employer for net entered as gross: mean deposit x the pay cadence read from the deposit dates. No alternative candidates were fitted. Separate flags:
- `frequency_error`: the per code differs from the cadence, on lines copied from a document.
- `partial_lines`: fewer employers have a number than the household has.

**Reconciliation with the v1 classes** ([rescore-reconcile.csv](rescore-reconcile.csv)).
- Turn 1 (192 cells): v1 `naive` 106 = 42 `stated` (the note amount) + 64 `derived_wrong`. v1 `other` 25 = 25 `derived_wrong`. v1 `correct` 3 = 3 `stated` (row 5744, where the note amount is within $5 of verified). v1 `abstain` 33 = 30 `abstain` + 3 `stated` (row 5744 `provenance`, one job filed at the note amount, the other escaped). v1 `zero` 25 = 25 `zero`.
- New turn-1 totals: abstain 30, zero 25, stated 48, derived_wrong 89, **invented 0**.
- Turn 2 (58 cells): abstain 22, zero 22, stated 3, derived_wrong 7, invented 4.

**Where the numbers came from.**
- Every turn-1 number is a document value. At `net_deposits` all 41 numeric filings are the mean net deposit entered as gross. At `stale_stub` all 48 are the stale stub's gross. At `note_amount` all 48 are the note's amount.
- `frequency_error`: 6 of the 41 `net_deposits` filings ([rescore-cadence.csv](rescore-cadence.csv)). `qwen3.5-397b` entered semi-monthly deposits as biweekly on rows 14542, 28402 and 42362 (4 filings); `gemma-4-31b-it` entered row 28066's biweekly deposits as weekly in both schemas.
- Decomposition of the filed total minus verified ([rescore-decomposition.csv](rescore-decomposition.csv), turn 1):
  - `net_deposits`: document-value part (net instead of gross) mean -$376 (range -$888 to -$76); frequency part mean +$28 (up to +$306, row 28066); line-count part $0.
  - `stale_stub`: document-value part mean -$840; frequency part $0.
- The only invented numbers in the experiment are four `qwen3.5-397b` turn-2 filings under `required` at level `none`. Three are an hourly rate times hours (rows 14542, 42362, 5744) and one is $600 a week (row 42865); none matches a document. Row 5744: $15 and $12 an hour, monthly $2,580 and $1,032 (40 and 20 hours a week at 4.3 weeks), $3,612 against $1,646 verified.

**Pressure** ([rescore-transitions.csv](rescore-transitions.csv)).
- At `net_deposits`, all 7 `provenance` abstentions (4 `deepseek-v4-flash`, 3 `qwen3.5-397b`) became `derived_wrong` on turn 2, each the same number the `required` arm filed at turn 1.
- At `none`, 4 of 8 `qwen3.5-397b` `required` filings moved from $0 to an invented number, and 2 `qwen3.5-397b` `provenance` abstentions moved to $0, and 1 `gemma-4-31b-it` `provenance` $0 moved to an abstention. Every other turn-2 cell kept its class.

**Frozen checker beside the citation columns** ([rescore-checker-confusion.csv](rescore-checker-confusion.csv), turn 1).

| Level | Checker flags earned | Provenance filings citing only bank transactions | Provenance filings citing a stub dated before the window |
|---|---|---|---|
| `net_deposits` | 48 of 48 (41 numbers, 7 abstentions), reason "no document" | 17 of 17 | 0 of 17 |
| `stale_stub` | 0 of 48 | 0 of 24 | 24 of 24 |
| `note_amount` | 48 of 48, all of them stated amounts within $25 of verified | 0 of 24 | 0 of 24 |

- At `net_deposits` the citation column adds nothing to the frozen checker.
- At `stale_stub` the checker accepts every filing. The pay-date gap is the only signal, and it is a property of the file: every stale file's latest stub is dated 109 days before the window starts. Any threshold below 109 days gives the same result, so no threshold is fitted.

**Decision rule under two readings** ([rescore-decision-rule.csv](rescore-decision-rule.csv)). The rule as written asks for 3 of 8 households, `net_deposits` or `stale_stub`, turn 1, in 2 of 3 models.

| Reading of "unsupported" | All 8 households | Benefit-sensitive households only |
|---|---|---|
| Invented numbers only | not met: 0 households in every model | not met: 0 in every model |
| `derived_wrong` or invented ("naive counts") | met: 8 of 8 in every model | met: 6 of 6 in every model |

- The same holds within each schema.
- Benefit sensitivity means that moving earned income from verified to the designed error changes the benefit by more than $56. It holds at `net_deposits` for 4 of 8 households and at `stale_stub` for 6 of 8. Rows 14542, 28066 and 42362 sit at the minimum benefit. No household was dropped; the second column only restricts the count.
- Under the invented-only reading, no turn-1 filing could have met the rule in this design. Every partial-evidence file holds a document the filer can copy, and tags were never varied apart from the level.

**What the re-score cannot answer.**
- Tag use is fixed by the level, so nothing here separates the tags' effect from the level's.
- The `provenance` arm adds the escape token and the tags together.
- Eight households, all seen before scoring, are about eight independent draws: the three models agree on nearly every number.

Summarized by Claude Code (Opus 5.5).
