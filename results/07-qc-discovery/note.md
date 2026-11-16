# How QC found each error, the risk score on a matched face, and the false-flag rate 80% precision needs

Notebook: `notebooks/07-qc-discovery.py` (no model calls). Feeds `rq.md` RQ2 and RQ3.

## 1. Where QC found each kind of error

FY2024 QC public file, all 53 jurisdictions, one row per variance in error households (STATUS 2 or 3, AMTERR above $56). Dollars = FYWGT x AMOUNT. Cause classes from the AGENCY codes of the FY2024 SNAP QC technical documentation: client 1, 2, 3, 4, 7; agency 8 and 10 to 25; 26; 99. Client-caused unreported income = client cause with NATURE 35 (unreported source of income). "Found from the case record" = DISCOV 1 or 2, i.e. QC found the variance from the case record; DISCOV records how the QC reviewer found it, not what the eligibility worker held.

| figure | expected | this run | |
|---|---|---|---|
| client-caused unreported income, share of weighted error dollars | 22.1% | 22.35% | mismatch (small) |
| ... share of those dollars QC found from the case record | 8.5% | 8.50% | matched |
| agency errors | 59.7% | 50.07% of dollars; 59.62% of agency dollars found from the case record | 59.7% matches the case-record share of agency dollars, not the dollar share |
| client-caused unreported-income households | about 808 | 984 (785 in the 41 states) | mismatch |
| 185 omissions found from the case record | 8.6% (16) | 8.65% (16) | matched |
| 185 omissions found from the recipient interview | 117 | 117 | matched |
| other household cut | 10.0% | 984 households, unweighted: 10.98% | not matched to a defined cut |
| other household cut | 6.1% | no cut tried gives 6.1% (range 8.3% to 11.7%) | not matched to a defined cut |

Cuts used (`1c-case-record-share-by-cut.csv`): the 185 omissions (households 8.65%, FYWGT 8.83%, FYWGT x AMTERR 11.73%); the 984 unreported-income households in 53 jurisdictions (10.98%, 9.37%, 9.70%); the 785 of them in the 41 states (8.92%, 8.27%, 9.00%). The expected 10.0% and 6.1% could not be matched to any of these. Tables: `1a-dollars-by-cause.csv`, `1b-discov-by-household.csv`.

## 2. Is the AUC of 0.451 a selection effect?

The 151 omissions filed at $0 (the part of the 185 that notebook 02 scores), scored from the filed application with notebook 02's risk score (`snapqc.review_sim`, `snapqc.features`). Comparison: error-free households (no error above $56) in the same 41 states with a score; same face = target group $0 and net income $0, pairs matched on target group (103 earned, 48 unearned). Bootstrap 95% interval, 2,000 draws, seed 20261115.

- All error-free households: AUC 0.451 (0.410 to 0.492), reproduces notebook 02.
- Same face, matched by target group: **0.628 (0.586 to 0.670)**. Earned only 0.601 (0.550 to 0.653); unearned only 0.698 (0.629 to 0.765). (`2b-matched-auc.csv`)
- Pool (`2a-comparison-pool.csv`): 31,935 error-free households in the 41 states; net income $0 12,449 (39.0%); earned $0 and net $0 10,618 (33.2%); unearned $0 and net $0 8,563 (26.8%). Expected about 11,080 (34.7%): mismatch, no face definition tried gives it; the base of 31,935 agrees with 11,080 / 0.347.

Against the thresholds: the matched AUC of 0.628 is below 0.65 and its interval excludes 0.5, so neither case applies as stated. It is above chance on a matched face; the "below chance" reading of 0.451 does not hold once the comparison households show the same face.

## 3. False-flag rate for 80% precision

f = r p / (4 (1 - p)). At 80% recall: 0.092% at 0.46% prevalence (expected 0.09%, matched) and 0.507% at 2.47% (expected 0.5%, matched). Full grid, prevalence 0.46%, 0.9625%, 1.465%, 1.9675%, 2.47% by recall 0.5 to 1.0: `3a-false-flag-rate-for-80pct-precision.csv`. Wilson 95% upper bound with 0 flags: 13.8% of 24, 7.41% of 48 (expected 7.4%, matched), 3.85% of 96 (`3b-wilson-upper-bound-zero-flags.csv`). Even 96 clean negatives cannot show a false-flag rate below the 0.09% to 0.63% the grid requires.

## Open questions

- Which household cuts the expected 10.0% and 6.1% refer to, and which definition gives about 808 households.
- Which face definition gives 11,080.
- How to report a matched AUC of 0.628 that falls between the two thresholds.

Summarized by Claude Code (Opus 5.5).
