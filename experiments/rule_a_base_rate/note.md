# Budget-gap rule on truthful QC households

## Question
How often does the budget-gap rule fire on truthful applications, and how many real income omissions does it catch?

The budget-gap rule flags a $0 or "not provided" income group when lease rent plus utility-bill amounts exceed the reported monthly income.

## Numbers
All figures are from the CSVs in this folder, produced by `rule_a_base_rate.py` (zero model calls). "Truthful" means the application reports the QC verified amounts (FSEARN, FSUNEARN, RENT, UTIL).

FY2024, n = 44,891 households (expected 44,891, match). Share of households on which housing > income ([rule-a-fpr.csv](rule-a-fpr.csv)):

| Housing definition | All households, unweighted | Weighted (FYWGT) | With a zero income group, unweighted | Weighted |
|---|---|---|---|---|
| Rent only | 6.1% [5.9, 6.3] | 7.0% [6.6, 7.4] | 5.8% [5.5, 6.0] | 6.6% [6.2, 7.0] |
| Rent + simulated bills | 12.6% [12.3, 12.9] | 14.5% [14.0, 15.1] | 11.9% [11.6, 12.2] | 13.8% [13.3, 14.3] |
| Rent + QC UTIL (if the rule used the allowance) | 24.6% [24.2, 25.0] | 28.5% [27.9, 29.2] | 23.3% [22.9, 23.7] | 27.0% [26.4, 27.7] |

- The rule flags only when a zero income group exists, so the "with a zero income group" columns are its false-positive rate on truthful files (all jurisdictions).
- The expected rent-only figures (6.1% unweighted, 7.0% weighted) match. The same test against FSGRINC gives the same shares.
- The simulated-bill row depends on the generator's BILL_RANGE (Heat $35-$140, Electricity $55-$160, Water/Sewer $30-$80, Phone $25-$70); actual-cost households get bills that add up to the verified amount.
- FY2023 and FY2022 (exploratory): rent only 6.5% and 7.1% unweighted; simulated bills 13.5% and 14.4%.
- By household size, the rent-only rate with a zero group runs from 4.5% (one person) to 8.7% (three people) ([rule-a-by-size.csv](rule-a-by-size.csv)). By state it runs from 1.4% (Virgin Islands, n = 278) to 10.4% (Colorado, n = 856) ([rule-a-by-state.csv](rule-a-by-state.csv)).

Recall and precision on real omissions ([rule-a-recall-precision.csv](rule-a-recall-precision.csv)). An omission is an earned or unearned group filed at $2 or less with a verified amount above $0: 185 rows of the human arm (`results/01-data-checks/3d-human-arm.csv`), expected 185, match. Both terms of precision come from the 41 states of the human arm (36,159 FY2024 households).

| Housing definition | Recall, unweighted | Recall, weighted | Precision, unweighted | Precision, weighted |
|---|---|---|---|---|
| Rent only | 59 of 185 = 31.9% [25.6, 38.9] | 31.5% [22.5, 42.2] | 59 of 2,179 = 2.7% [2.1, 3.5] | 2.7% [1.8, 4.1] |
| Rent + simulated bills | 77 of 185 = 41.6% [34.8, 48.8] | 46.8% [36.4, 57.6] | 77 of 4,092 = 1.9% [1.5, 2.3] | 2.2% [1.5, 3.1] |

- False positives are the truthful households of the 41 states on which the rule fires (2,120 rent only; 4,015 with simulated bills).
- Rent-only precision is 2.7%, the expected value (about 2.7%). It is well below 10%.
- The human arm undercounts omissions in the caseload, so precision here is a lower-bound-style figure.
- For an omission filed at $1 or $2, the filed income includes that amount, and the group is counted as flagged when housing exceeds it.

## How it was computed
- Income = FSEARN + FSUNEARN (missing counted as 0), as the packet generator reports the earned and unearned groups. Rent = RENT, the generator's lease amount.
- Simulated bills follow the generator's draw, with one fixed seed per year (packets.SEED combined with the year).
- Weighted intervals use the Wilson formula with the Kish effective sample size.
- For each omission row, the filed application is the verified one with the omitted group at its filed amount.

## What it feeds
This feeds `rq.md` RQ2, on where existing review rules fall short. A rule that looks perfect on households chosen for its trigger has a 6% to 14% false-positive rate on truthful QC households and catches 32% to 42% of the real omissions in the human arm. It does not test the readers.

## Open questions
- Which utility definition the rebuilt reader experiment should treat as the rule's real input: the generator's bills (the rule as run) or a definition not tied to BILL_RANGE.
- Whether the folder and the CSV/SVG file names (`rule_a_base_rate`, `rule-a-*`) should be renamed to describe the budget-gap rule.

Summarized by Claude Code (Opus 5.5).
