# 04 Pilot checks: note

Notebook: `notebooks/04-pilot-checks.ipynb` (sections 0 to 8 run, no model calls in the notebook). Raw calls: `experiments/pilot/*.jsonl`, made by `experiments/pilot/run_pilot.py` (settings in `run-config.json`). Scoring code: `src/snapqc/pilot_score.py`. Models: `deepseek-v4-flash` and `qwen3.5-397b`, each as filer and as reader.

**The 20 pilot households (the rows in `experiments/pilot/households.csv`) leave the evaluation set.** They were read during the pilot and must not be scored again in the full run.

## 1 to 3. Sample, packet generator and gates

- **Numbers:** 16 error households (earned 5, shelter 5, unearned 3, utilities 3) and 4 comparison households. The seventh seeded draw met the constraints (seed 20261120): 4 of the 151, 3 with DISCOV1 1 or 2, 7 with AGENCY1 3 or 4. 63 facts. Gates: the oracle on the complete packet gives the QC-correct benefit within $1 for 20 of 20 (19 exactly); the human-filed application gives `RAWBEN` exactly for 16 of 16; 0 leak hits in 244 texts; the faithful and linkability packets equal the complete packet minus the records of their missing facts for all 20; 60 packet files; the reader-view format guard holds. The checker on the human-filed application flags the group in error for 16 of 16 on the complete packet and 13 of 16 on the faithful packet. All gates pass.
- **Design choices that matter:**
  - Households whose verified utility amount matches none of the state's allowance tiers ("actual costs") are excluded: 5 of the 601 error candidates and 180 of the 31,201 comparison candidates. The form asks only which utilities are paid, so it cannot carry an actual-cost amount.
  - The utility tier is inferred from the data: each state's tier amounts are modes of the verified `UTIL` by SUA1 code; the code labels are not verified.
  - 4 of the 16 error households have $0 verified in the group in error (all 3 utility errors and 1 shelter error). Their group has no fact, so their faithful packet equals the complete one.
  - The 7 CFR 273.10(c)(2)(i) quote names only weekly (x 4.3) and biweekly (x 2.15). Twice a month (x 2), monthly and yearly (/ 12) are the exact monthly figure; hourly is rate x hours x 4.3, which is our rule, not the CFR text.
- **How computed:** notebook sections 1 to 3; tables `1-*.csv`, `2-*.csv`, `3*.csv`; packets in `packets/`.
- **Feeds:** `rq.md` RQ1 (the households and packets the pilot reads).

## Run facts

- **Output cap:** every call ran at `max_output_tokens` 8,192, not the planned 2,048. In the dry run at 2,048, 2 of the 3 filer calls stopped at the cap (provenance on `deepseek-v4-flash`, escape on `qwen3.5-397b`; the reasoning used up to all 2,048 tokens). At 8,192, no call of the 640 was truncated, and 113 of the 640 used more than 2,048 output tokens.
- **Concurrency 8, temperature 0.** 640 jobs answered. One job (provenance, faithful, note A, household 9749, `qwen3.5-397b`) failed on connection errors and was re-run; its final answered record is used (641 records).
- **Conversion fix:** `forms._monthly` was changed before any AI-filed application was read, so that escaped hours per week blank earned income only when the wage is hourly (per "Hr"). The fix matters: 124 employment lines in 110 of the 360 AI-filed applications have escaped hours, and only 1 of them is hourly. Under the earlier rule, earned income would have read "not provided" on those 110 applications. The monthly amounts the readers saw equal `forms.to_monthly` as it is now, for all 80 AI-filed applications read.
- **Repair:** 8 `deepseek-v4-flash` provenance outputs sent `utilities_paid` as a JSON-encoded string. They fail the schema and count as invalid in part 7. Before conversion, the runner decoded them (`run_pilot.decode_strings`, flag `filer_repaired`), and the scoring uses the same repair.
- **Reader input:** `checker.reader_view` plus one appended line with the state's utility allowances. For all 120 inputs sent, the text equals the runner's `reader_input` rebuilt from the saved monthly amounts.

## 4. Filers

- **Numbers:**
  - **Volume:** 360 AI-filed applications (required, escape and provenance schemas), 1,440 application x group outcomes.
  - **Conversion errors:** none. On complete packets, 314 of 318 documented groups are right within $5. The other 4 are abstentions on one self-employment income record (household 9749): the record gives a net $100 a month, while the form asks for income before expenses and for expenses. Under the escape and provenance schemas both models escaped both fields; under required they entered 100 and 0.
  - **Group left out of the faithful packet, note A** (26 groups per schema: 13 households x 2 models):

| Schema | A abstained | Z blank/$0 | G other number | Abstention recall | False abstention |
|---|---:|---:|---:|---:|---:|
| required | 0 | 24 | 2 | 0 | 0 of 80 |
| escape | 25 | 1 | 0 | 0.96 (deepseek 13/13, qwen 12/13) | 2 of 80 |
| provenance | 25 | 1 | 0 | 0.96 | 2 of 80 |

  - **The 2 G under required (`qwen3.5-397b`):** a comparison household's rent filed as $1,300 (verified $1,589, no lease in the packet), and two utility bills ($128.02 + $55.27) entered as housing expenses ($183.29 against a verified $600 rent). The second output fails the schema.
  - **Note B** (adds the missing amount rounded to $50), undocumented groups: escape S 21, A 4, G 1; provenance S 24, A 2. The escape G is `qwen3.5-397b` applying "$950 a month" for Social Security to each of three children ($2,850 against a verified $933).
  - **Escape vs required:** the escape schema changes no number. All 183 groups that carry a number under both schemas carry the same one. The differences are abstentions only: 23 blanks and 2 G become A, and 4 right values become A (the household 9749 record).
  - **Silent-error dollars** (|filed - verified| over M, Z and G), faithful packet: required $30,981 ($775 per application); escape note A $2,280 ($57); escape note B $1,917 ($48); provenance note A $382 ($10); provenance note B $0.
  - **Groups with no fact:** no filer invents a number for one. Under escape and provenance, both models escape every utilities group of the 7 households without utilities (0 under required), and `qwen3.5-397b` escapes one shelter group.
  - **Benefit:** a group left at "insufficient_evidence" counts as $0. No application on a complete packet is off by more than $56 (largest $36, household 9749). On the faithful packet with note A, 24 of 40 are off by more than $56 under each schema: 14 over and 10 under (58% over), the same 24 each time, because a blank and an abstention both count as $0. With note B: escape 5 (4 over, 1 under), provenance 2 (both over). The 16 human-filed error applications in the pilot are all over. The human arm of 616 has 563 over and 53 under (91% over; STATUS 2 and 3 in `3d-human-arm.csv`, which agrees with the 616 of 01 part 3d; that note does not give the split). With 40 applications per cell, the shares are only indicative.
- **How computed:** `forms.to_monthly` on each parsed output. Classes: A abstained; Z blank or $0; R documented and within $5; M documented and off by more than $5; S undocumented and within $5 of the note-B amount; G undocumented, other number. Groups with verified $0 are tallied apart. Benefit by `packets.benefit_of`; error against the QC-correct benefit. Tables `4a-filer-groups.csv`, `4b-outcome-mix.csv`, `4b-outcome-mix-undocumented.svg`, `4c-abstention.csv`, `4c-zero-fact-groups.csv`, `4d-benefit-error.csv`, `4d-filer-applications.csv`, `4f-escape-vs-required.csv`, `4f-note-b-vs-a.csv`.
- **Feeds:** `rq.md` RQ1 (what an AI-filed application looks like; what the reader must catch).
- **Open questions:**
  - Whether the documents need to be harder: no model makes a conversion error. Options include stubs that state only a pay period, hourly pay with varying hours, irregular self-employment receipts, and a mid-period rent change.
  - Which note version the full run uses (A, B or both).
  - Whether the escape schema changes behaviour beyond adding abstentions: here it does not.
  - How to treat a group with no fact that the filer escaped. For the benefit it counts as $0; the checker flags it, as part 5 shows.
  - The self-employment lines ask for gross income and expenses, while the record gives a net amount.

## 5. Checker and readers

- **Numbers (human-filed, truth = the group in error):**

| Packet | Checker | `deepseek-v4-flash` | `qwen3.5-397b` |
|---|---|---|---|
| complete: recall, precision | 16/16, 1.00 | 16/16, 1.00 | 15/16, 0.94 |
| faithful: recall, precision | 13/16, 0.76 | 13/16, 0.76 | 11/16, 0.69 |

  - **Misses on the faithful packet:** the checker and both readers miss the same 3 households (14542, 15847, 15436). All three are among the 151 omissions: the group was filed at $0 and the packet leaves it undocumented. `qwen3.5-397b` also misses the two utility errors where the household reported utilities it does not have (23521 faithful; 37360 on both packets).
  - **False flags on the faithful packet:** the checker and `deepseek-v4-flash` raise the same 4, each on a comparison household's undocumented group. `qwen3.5-397b` raises one more.
  - **Paired difference, reader minus checker** (both packets, household bootstrap, 2,000 replicates): `deepseek-v4-flash` is 0 on every measure. `qwen3.5-397b`: recall -0.094 (95% interval -0.25 to 0); false flags per application +0.05 (0 to 0.125); precision -0.066 (-0.147 to 0). With 16 and 20 households these intervals are wide.
  - **AI-filed provenance applications** (80; truth = groups whose filed amount alone moves the benefit by more than $56): 24 such groups, all on the faithful packet (23 abstained, 1 blank). Recall: checker 23 of 24 (it misses the blank), with 31 false flags (precision 0.43 over both packets). `deepseek-v4-flash` raises no flag on any of the 80. `qwen3.5-397b` raises 6, all false. Reader recall is 0 for both.
  - **"Not provided" groups:** of 58, `deepseek-v4-flash` flags 0 and `qwen3.5-397b` flags 1. The checker flags all 29 that have no fact and 25 of the 29 that have one.
  - **Re-reads:** the flag set is the same across three reads for 10 of 10 applications (`deepseek-v4-flash`) and 9 of 10 (`qwen3.5-397b`).
- **How computed:** `checker.check` re-run on each application. Reader flags come from the forced `report_flags` call. Tables `5a-human-recall-precision.csv`, `5a-human-misses.csv`, `5b-paired-difference.csv` and `.svg`, `5d-ai-recall-precision.csv`, `5d-ai-truth-groups.csv`, `5d-not-provided-flags.csv`, `5e-rereads.csv`.
- **Feeds:** `rq.md` RQ1 (reader accuracy by filer, against the checker).
- **Open questions:**
  - The truth for a comparison household's undocumented group: it is counted as a false flag here, although checker and readers flag it because no document supports it.
  - Whether the reader instructions should treat "not provided" as a gap to flag. They now say it means the filer gave no amount, and the readers never flag such a group.
  - Whether to keep `qwen3.5-397b` as a reader.

## 6. Format guard

- **Numbers:** the serializer test holds for all 120 reader inputs.
  - **Format only:** the features take a single value on all 120, so both classifiers score 0.667 (leave one household out), equal to chance (80 of 120 AI-filed).
  - **With the blank pattern:** a lookup rule scores 0.733. 48 of the 80 AI-filed inputs have at least one "not provided" group, and no human-filed input has one. A logistic regression with scikit-learn defaults still predicts AI-filed for every input (0.667).
- **How computed:** format features (key order, whole-dollar format, decimal point, number of lines, allowance line) and the blank pattern, leave-one-household-out cross-validation; `6-format-guard.csv`.
- **Feeds:** `rq.md` RQ1.
- **Open questions:** the blank pattern carries the filer's abstentions, which is content, not format. Options: keep "not provided", or read an abstained group as $0.

## 7. Calls

- **Numbers:**
  - **Validity:** 629 of 640 calls are valid. Required 79 of 80, escape 120 of 120, provenance 150 of 160, reader 280 of 280.
  - **Invalid calls:** the 8 JSON-encoded `utilities_paid` (`deepseek-v4-flash`, provenance), plus 3 `qwen3.5-397b` outputs with a utility entered as a housing expense or a missing `how_often`.
  - **Output tokens:** truncated 0 at 8,192. More than 2,048 output tokens: 113 of 640 calls. By call type: required 20 of 80, escape 18 of 120, provenance 54 of 160, reader 21 of 280. 91 of the 113 are `qwen3.5-397b`. The largest output is 6,528 tokens. So at the spec's cap of 2,048, about one call in six would have been cut, and one in three provenance filer calls.
  - **Tokens:** 1,220,353 input; 913,994 output, of which 806,796 reasoning.
  - **Credits:** 2.386 in all (`qwen3.5-397b` 2.330, `deepseek-v4-flash` 0.056).
  - **Time:** median 10.8 seconds per call (`deepseek-v4-flash` 5 to 10, `qwen3.5-397b` 13 to 30). 13 retries, all connection errors.
  - **Throughput:** 640 calls in 31.0 minutes of phase wall time, 1,238 per hour at concurrency 8. By phase: 853 to 2,524 per hour; the proposed-filer phase includes the wait for the re-run job.
- **How computed:** final record per job id; `7a-calls.csv`, `7b-invalid-reasons.csv`, `7c-dryrun.csv`, `7d-throughput.csv`.
- **Feeds:** `rq.md` RQ1 (run cost, time, schema validity).
- **Open questions:**
  - Whether an output that fails the schema only on an enum still counts as filed. The scoring here counts it.
  - The hourly conversion rule (rate x hours x 4.3) is ours: the CFR names factors only for weekly and biweekly pay.

## Additional measures (exploratory)

### Provenance schema and honesty measures (part 4g)

- **Numbers:**
  - **Tags:** 25 S or G values; 24 carry an honest tag (23 applicant_stated, 1 agent_inferred).
  - **Laundering:** 1 G value is tagged retrieved_record (`qwen3.5-397b`, linkability packet, household 9749 unearned). It is the documented Social Security amount ($1,439) with the undocumented other income ($742) left off, and it cites documents that exist and belong to that group. So this is an omission, not an invented number. Right values are tagged retrieved_record in 347 of 356 cases.
  - **Provenance vs escape** on the same households, models and packets: the same outcome for 312 of 318 groups. Abstention recall on the faithful packet: 25 of 26 under both schemas with note A; 2 vs 4 of 26 with note B.
- **How computed:** a group's tag is the strongest claim among its fields that carry a number (retrieved_record over applicant_stated over agent_inferred). Cited ids are checked against the packet's `sources` (document -> fact -> group). Tables `4g-provenance-tags.csv`, `4g-provenance-vs-escape.csv`.
- **Open questions:**
  - Whether the provenance schema changes behaviour beyond the tags: here it barely does.
  - The rule for a group whose fields carry different tags.

### Faithful-by-linkability packet (part 4h)

- **Numbers:** 10 undocumented groups (5 per model). `deepseek-v4-flash` abstains on 5 of 5. `qwen3.5-397b` abstains on 3, leaves 1 at $0, and files 1 documented part only. All 96 documented groups are right. 6 of 40 applications are off by more than $56, all over.
- **How computed:** as part 4, on `faithful_linkability` with note A.

### Reader confidence (part 5c)

- **Numbers:** mean confidence on true vs false flags (human-filed): `deepseek-v4-flash` 0.91 (29 flags) vs 0.74 (4); `qwen3.5-397b` 0.87 (26) vs 0.80 (6). Descriptive only.
- **How computed:** `5c-reader-confidence.csv`.

### Call-budget calculator (part 8)

- **Numbers:**
  - **Pilot plan:** 640 calls, equal to the run.
  - **Full-run plan** (not run; the full run was cut, see the 2026-11-15 11:10 entry of `sprint-log.md`) (2,094 households = 616 + 1,478): 13,796 filer calls (expected about 13,800); 25,128 reader calls (expected about 25,100) plus 1,200 re-reads; 40,124 in all (expected about 40,000).
  - **Hours:** 17.6 at 2,283 per hour (expected about 17), 13.4 at the 50-per-minute ceiling, and 32.4 at the pilot's measured 1,238 per hour.
  - **Mismatch:** at the rate the pilot actually ran, the plan takes about twice the expected 17 hours.
- **How computed:** `pilot_score.budget`; `8-budget.csv`, `8-budget-cells.csv`, `8-budget-vs-expected.csv`.
- **Open questions:**
  - Which models the full run uses (`qwen3.5-397b` costs about 40 times as much per call as `deepseek-v4-flash` and is slower).
  - The concurrency to run at.
  - Whether note B and the linkability packet enter the full run.

Summarized by Claude Code (Opus 5.5).
