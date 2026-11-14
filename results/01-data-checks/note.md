# 01 Pre-experiment data checks: note

Source: FY2024 SNAP QC public-use file (`qc_pub_fy2024.csv`, 44,891 households). Notebook: `notebooks/01-data-checks.ipynb`. Sections 4 to 9 run on the human arm of 616 households built in part 3d.

## 1. Funnel

- **Numbers:** 44,891 households -> 17,836 error cases (STATUS 2 or 3) -> 13,585 with exactly one variance -> 3,289 client-caused -> 2,096 at or before the most recent action -> 1,098 above $56 -> 917 in the 41 single-row-SUA states. All four expected figures match.
- **41 error cases with no variance:** STATUS 2 or 3 and AMTERR above zero, but every variance detail field (element, nature, cause, discovery, verification, date, time period) is blank. With no cause code they cannot pass the client-caused step. 29 overissuances, 12 underissuances, 17 states, median error $24; 11 are above $56, 10 of those in the 41 states. The file does not say why the detail is missing.
- **STATUS and the tolerance:** 12,403 of the 17,836 STATUS 2/3 cases (70%) have an error of $56 or less. STATUS 1 always has AMTERR of 0. So "error" needs AMTERR above $56, not STATUS alone.
- **How computed:** filters applied in order to the raw file; counts after each step (`1a-funnel.csv`, `1b-no-variance-cases.csv`, `1c-status-by-error-size.csv`).
- **Feeds:** `rq.md` Households (human-arm sample size).

## 2. Weights

- **Numbers:** `HWGT`/`FYWGT` reproduce the August 2026 technical documentation: HWGT 52.36 to 77,999.84, FYWGT 4.36 to 6,499.99 (Chapter V codebook); 40,168,146 individuals and $7,366,534 thousand in benefits, weighted (Table II.2). `HWGT_OLD`/`FYWGT_OLD` give 40,772,270 individuals, $7,393,081 thousand, and a largest HWGT of 45,716. The two sets differ in 44,838 of 44,891 rows (ratio 0.10 to 5.97, median 1.00) and sum to the same total.
- **File:** the CSV in use is byte-identical to the CSV inside `qcfy2024_csv.zip` from the `2026-08` folder, and both match `data/SHA256SUMS`. The CSV inside the zip is timestamped 2026-06-26; nothing in the file explains that date.
- **How computed:** ranges and weighted totals from each pair of columns against the documented figures (`2a-weights-vs-documentation.csv`); SHA-256 of the zip, the CSV inside it and the CSV on disk.
- **Feeds:** `rq.md` Statistics (which weights RQ2 uses).
- **Note:** The check supports using `HWGT`/`FYWGT` as they are.

## 3. Rebuild the filed value

- **Calculator:** `snapqc.calc.benefit` returns `FSBEN` within $1 from the verified amounts for 35,394 of 35,394 households in the 41 states once MFIP and SSI-CAP units are set aside (44,296 of 44,891 over the whole file; 908 of the 917).
- **Numbers:** 465 of the 917 (50.7%) have a unique filed value. On those the rebuilt application returns the issued benefit within $1 (462 exactly), and the implied error is within $1 of AMTERR in the same direction for all 465.
- **The other 452:**

| Outcome | Households |
|---|---|
| Element is not a dollar amount (unit composition 71, student status 8, others 2) | 81 |
| Calculator misses FSBEN (MFIP or SSI-CAP unit) | 9 |
| FSBEN differs from the QC-correct benefit (BENFIX) by more than $1 | 72 |
| No value of the amount reaches the issued benefit | 27 |
| Not unique: issued benefit at the maximum (net income zero) | 238 |
| Not unique: issued benefit at the minimum | 5 |
| Not unique: shelter cap binding | 14 |
| Not unique: shelter cost below half of income | 6 |

- **Inside the 238:** in 173 the amount in error is income and $0 is one of the answers; 151 of those are coded as an unreported source of income (NATURE 35).
- **Also seen:** among the 465, the variance amount (AMOUNT1) equals AMTERR in 387; in the other 78 the one recorded variance is not the whole recorded error.
- **How computed:** for each household, the calculator is run with the amount in error set to every whole-dollar value from $0 to $30,000 and every other amount at its verified value; the values that return the issued benefit (`RAWBEN`) form one range. A range of $5 or less counts as unique (rounding alone gives up to $4) and the filed value is its middle. Outcomes are assigned in the order of the table. Per-household results: `3-rebuild.csv`; counts: `3b-rebuild-outcomes.csv`; calculator check: `3a-calculator-check.csv`.
- **Limits:** "benefit at the maximum" and "net income at zero" are the same condition, so they are one row. The rebuild recovers the total for the amount in error (for example total earned income), not a per-person or per-source figure. It assumes the one variance explains the whole gap between the issued and the QC-correct benefit.
- **Feeds:** `rq.md` Applications.
- **Open questions:** Choices the numbers leave open: keep only the 465; also keep the 173 income cases at the maximum benefit by taking $0 as the filed value (638 in total); rebuild unit-composition errors by a different method or leave them out.
- **3d. Human arm (decision of 2026-11-14):** 616 households = the 465 with a unique filed value + the 151 at the maximum benefit whose amount in error is income, where $0 is one of the answers and the variance is coded NATURE 35 (unreported source of income); for the 151 the filed value is $0. One row per household in `3d-human-arm.csv`: row, state, fact group, ELEMENT, filed and verified value, in the 151, NATURE1, DISCOV1, VERIF1, ACTNTYPE, STATUS, AMTERR, FYWGT, error type ("empty" if the filed value is $2 or less, else "wrong number"), and the number of verified person-level income sources in the group in error (person x income type with an amount above $0; the person-level amounts add up to `FSEARN` and `FSUNEARN` for all 917).
- **The 151 by number of sources:** 130 have one source, 21 have two or more (18 two, 3 three). Expected 22: **mismatch (21)**. For the 21, "$0 for the whole group" assumes every source was left off the form.
- **Open questions (3d):** treatment of the 21 (keep at $0 for the whole group, file only the source in error, or drop them).

## 4. Fact groups

- **Numbers:** earned 211, shelter 191, unearned 131, utilities 68, outside 15 (dependent care 9, medical 4, child support paid 2). NATURE 35 in 277. Group empty on the form (filed value $2 or less) in 188: the 151 plus 37 of the 465; 6 of the 188 are not NATURE 35 (3 earned, 3 shelter). Three cells: income empty 185, income wrong number 157, housing (shelter + utilities) 259, plus outside 15. 95 households coded NATURE 35 have a filed group total above $2 (44 earned, 51 unearned): the unreported source sat next to other reported income in the same group. Only 3 of the 259 housing errors are empty entries. Utilities (68) and outside (15) have fewer than 100 errors. All expected figures match.
- **How computed:** each ELEMENT code mapped through the amount it changes in the calculator (`4a-element-map.csv`, labels quoted from the FY2024 SNAP QC technical documentation; ELEMENT 332 and 335 have no label in the code list read); counts per group (`4b-group-counts.csv`); error type by group and NATURE (`4c-error-type-by-group.csv`); three cells (`4c-three-cells.csv`).
- **Feeds:** `rq.md` Terms (fact group, checker); power for RQ1.
- **Open questions:** merge shelter and utilities in tables (utilities has 68 errors); table by error type (empty vs wrong number).

## 5. How each error was found

- **Numbers:** 38 of the 616 have DISCOV1 1 or 2 (from the case record), 13 of them among the 151; 65 of the 917. Both match. The recipient interview (DISCOV 3) is the most common source in every group: 387 of 616 (63%). The employer (4) discovers 55 of 211 earned errors (26%); the landlord (6) 46 of 191 shelter errors (24%) and 11 of 68 utility errors (16%). Verification differs from discovery: the employer verifies 123 of 211 earned errors, the landlord 99 of 191 shelter errors, information from the recipient 163 of 616; DISCOV1 equals VERIF1 in 339 of 616.
- **Code meanings (quoted, FY2024 SNAP QC technical documentation, Chapter V codebook):** DISCOV 1 "Variance clearly identified from case record (not automated match)"; 2 "From case record (automated match)"; 3 "From recipient interview"; 4 "Employer (present or former)"; 5 "Financial institution, insurance company, or other business"; 6 "Landlord"; 7 "Government agency or public records, not automated match"; 8 "Government agency or public records, automated match"; 9 "Other". VERIF uses the same list except 1 "From case record (not automated match)" and 3 "From information provided by recipient".
- **How computed:** crosstabs of DISCOV1 and VERIF1 by fact group (`5a-discov-by-group.csv`, `5a-verif-by-group.csv`, `5a-discov-by-verif.csv`); share of each DISCOV1 code within each group, the frequencies comparison households would copy (`5b-discov-frequencies.csv`).
- **Feeds:** `rq.md` Packets (which document the faithful packet leaves out).

## 6. Non-zero fact groups per household

- **Numbers:** in the 616, 79 households have none of the other fact groups non-zero (69 of the 151, 10 of the 465; 52 earned and 27 unearned errors): match. 77 have one, 369 two, 87 three; 460 (74.7%) have two or more. Comparison pool (35,543 households in the 41 states outside the 616), out of all four groups: 5,358 (15.1%) none, 3,636 (10.2%) one, 26,549 (74.7%) two or more.
- **How computed:** a group is non-zero when its verified total (`FSEARN`, `FSUNEARN`, `RENT`, `UTIL`) is above $0. For the 616 the group in error is left out (for the 15 outside errors all four groups count); comparison households have no group in error, so all four count (`6a-nonzero-groups.csv`; by group `6b-other-nonzero-by-group.csv`; share non-zero per group `6b-group-nonzero-shares.csv`).
- **Feeds:** `rq.md` Packets.
- **Open questions:** keep or drop the 79 households with nothing else on the form (69 of them among the 151, whose form is then empty in all four groups).

## 7. Background errors outside the 616

- **Numbers:** of the 35,543 households in the 41 states outside the 616, 3,608 (10.2%; 11.6% weighted by FYWGT) have an error above $56. By cause: agency 1,951 (5.5%); client, outside the 917, 700 (2.0%); mixed causes 597 (1.7%); the 301 client-error households of the 917 not in the 616, 301 (0.8%); other (AGENCY 99) 39; change not required to be reported (AGENCY 26) 10; no variance detail 10. The 616 carry 11.9% of weighted error dollars (FYWGT x AMTERR, errors above $56, all states); the 917 carry 17.5%; the 151 3.8% and the 465 8.1%. Both expected figures match. Comparison households needed for 150 or more background errors: 1,478 from the whole pool; 1,599 if the 301 are left out; 2,028 if errors caused by the client alone also do not count.
- **How computed:** error = STATUS 2 or 3 and AMTERR above $56. Cause from the AGENCY codes of every variance: client 1 to 4 and 7; agency 8 and 10 to 25; 26; 99; "mixed" when a household has variances in more than one category (`7a-background-errors-by-cause.csv`). Needed = 150 / share, rounded up (`7b-comparison-size.csv`).
- **Feeds:** `rq.md` Households.
- **Open questions:** whether the 301 may be drawn as comparison households, and which causes count as background errors.

## 8. Certification vs recertification

- **Numbers:** 440 of the 616 are recertifications (71.4%): match. Largest gap in the fact-group mix with four groups: earned, 8.5 points more common among certifications (unearned 8.4, shelter 7.6 and utilities 7.5 the other way): match. With shelter and utilities merged: housing, 15.1 points more common among recertifications: match. Empty entries are 38.6% of certification errors and 27.3% of recertification errors (11.4 points).
- **How computed:** crosstab of fact group by ACTNTYPE (`8a-group-by-action.csv`); share of each group within each action type over all five groups, gap = recertification minus certification, also for the three cells and the error type (`8b-mix-by-action.csv`).
- **Feeds:** `rq.md` limitations (share of recertifications).
- **Open questions:** stratify by recertification.

## 9. Interval width

- **Numbers:** 15 of the 41 states have under 10 cases (86 households), pooled into one stratum: 27 strata. Kish effective sample size under FYWGT: 257.4. Both match. Half-width of a 95% interval for recall 0.5, unweighted / weighted, in points: all 616 3.8 / 6.0; earned 6.5 / 9.6; shelter 6.9 / 10.1; unearned 8.4 / 13.5; utilities 11.6 / 17.8; outside 25.1 / 29.5; income empty 7.0 / 10.5; income wrong number 7.6 / 12.6; housing 5.9 / 8.9. A 10-point paired filer gap (recall 0.5 vs 0.6 on the same households), all 616: 5.4 / 8.4 if the two outcomes are independent within a household, 3.8 / 5.6 at a correlation of 0.5. Stratifying by state barely changes the width (3.84 vs 3.93 points unweighted, 6.00 vs 6.08 weighted).
- **How computed:** each household caught with probability 0.5 (20 simulated draws, averaged); whole households resampled within stratum, 2,000 resamples, seed fixed in the notebook; subgroups as domains (the whole stratum resampled, recall computed on the subgroup's resampled members); half-width = half the 2.5th to 97.5th percentile range; weighted = FYWGT (`9b-recall-half-widths.csv`, `9c-paired-gap-half-widths.csv`). The within-household correlation for the paired gap is an assumption; nothing in the file gives it.
- **Feeds:** `rq.md` Statistics.

Summarized by Claude Code (Opus 5.5).
