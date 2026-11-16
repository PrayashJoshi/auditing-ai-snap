# 02 Rule checks: note

Notebook: `notebooks/02-rule-checks.ipynb`. Sections 1 to 5.

## 1. Baseline rules

Paper: Chohlas-Wood, A. (2026). *Reducing SNAP Payment Errors at Lower Cost With AI: A Simulation Study*. arXiv 2611.03213. Single author (NYU).

- **Claim tested:** combined review (all AI flags first, flagged cases in risk order, then category checks on unflagged cases in risk order) beats risk-based review alone. Abstract: "risk-based review alone costs about seven times as much as combining AI flags with risk scores." Section 3, p. 9: "Combined review brings 26 of these 27 jurisdictions below the threshold, compared with 25 for risk-based review and 19 for AI-based review. [...] Among the 25 jurisdictions where both risk-based and combined review reach the target, combined review requires less spending on reviews in 22."
- **The public code is not the simulation.** The paper's reference [9] (github.com/giannella/snap_qc, cited "n.d.") is the Better Government Lab risk-model code in R, which the paper says it "adapt[s]" (p. 4). The repository (commit `c9b2e7f`, 2026-11-12, 568 commits) contains no flag simulation, no review queues and no combined rule, and never names the paper. Its current case model also differs from the paper's (one pooled model with state terms, trained on FY2017-19 and FY2022-24). So the rules were rebuilt from the paper's text in `src/snapqc/review_sim.py`. From the repository we used its reconstructed frame (`additional_data/reg_model_data.rds`, sha256 `08e0e825...`), its household-size strata, its 19 inputs, its xgboost settings and its tolerances by year. R and xgboost could not run here, so the risk score is scikit-learn gradient-boosted trees at the same settings, without row subsampling.
- **Numbers (FY2024, 48 states and DC, `HWGT`):**

| Figure | Paper | This run |
|---|---|---|
| Jurisdictions at or above 6% PER | 27 (p. 9) | 27, the same names as Figure 2 |
| Reached: combined / risk only / AI only | 26 / 25 / 19 (p. 9) | 26 / 25 / 19 |
| Reached: combined, risk order throughout (App. D.9) | 27 (p. 28) | 27 |
| Not reached: risk only | DC, RI (Fig. 2) | DC, RI |
| Not reached: AI only | CT, DC, IL, MA, NM, OK, OR, RI (Fig. 2) | the same 8 |
| Not reached: combined | DC (Fig. 2) | DC |
| Combined cheaper, of those both reach | 22 of 25 (p. 9) | **25 of 25** |
| Median cost ratio, risk only / combined | about 7 (p. 9) | **8.98** (8.97-9.15 across draw seeds) |
| Ratio risk only / combined: California, Massachusetts | 2-11 (p. 9) | **14.97**, 1.73 |
| California: risk only cost / AI only cost | about 10 (p. 9) | **14.22** |
| AI only cheaper than combined | DE, MS, NV, AL, AZ (p. 10) | **AZ, DE, MS, ND, TX** |
| DC: error $ to remove / combined removes at most | about 57% / 55% (p. 10) | 56.9% / 54.9% |
| Median within-state AUC of the risk score | 0.77 (App. D.7) | **0.752** |
| Households outside the simulation sample | 9.5% (p. 4) | **6.6%** |

Bold marks a mismatch. The rebuilt Figure 1 (`1d-figure1-rebuilt.svg`) matches the paper's California and Massachusetts curves as read by eye from its plot. AI-only plateaus are 5.06% and 9.36% (paper about 5.1 and 9.4). At 1,000 hours per 1,000 households, risk only reaches 4.52% and 5.66% and combined 3.56% and 5.34% (paper about 4.5, 5.7, 3.6 and 5.3).

- **D.9 beside the main rule (1c):** D.9 reaches all 27 jurisdictions, including DC at $1.99m a year. It costs more than the main combined rule in all 26 jurisdictions both reach, a median 5.7 times as much. The paper says D.9 "typically requires more review spending" (p. 28). D.9 still costs less than risk only in all 25 jurisdictions risk only reaches.
- **Budgets (1d):** Figure 1's axis runs from 0 to 1,000 annual review hours per 1,000 scored households, up to 60 minutes per household. The combined rule reaches 6% at a median of 13.6 hours per 1,000 (about 0.8 minutes per household; range 0.6 to 549). At 0.5, 1 and 2 minutes per household, combined has a lower PER than risk only in all 27 jurisdictions. It removes a median 14%, 25% and 35% of error dollars; risk only removes 3%, 5% and 10% (`1d-capture-at-budgets.csv`).
- **D.9 against the main rule at each budget (1d):** the share of the 27 jurisdictions where the D.9 rule has a lower PER than the main combined rule is 0 of 27 at 0.5, 1, 2 and 10 minutes per household and 18 of 27 at 60 minutes. Where it has a lower PER than risk only: 24, 26, 27 and 27 of 27 at 0.5, 1, 2 and 10 minutes and 27 of 27 at 60 minutes.
- **Weights (1e):** with `HWGT` the 27 jurisdictions at or above 6% are exactly the paper's 27. With `HWGT_OLD`, Indiana replaces Nevada. So the paper used the revised weights. BGL's frame carries its own `hwgt`, which equals `HWGT` in only 62.8% of FY2024 rows; the run uses the file's `HWGT`.
- **Table B1 of the paper (1f):** 54.8% of FY2024 sample error dollars (weighted) have a cause the paper assumes AI can detect. The largest undetectable cause is 1, information not reported (25.7% of error dollars). Missing causes count as detectable (3.1%).
- **Review model (1g):** coded as stated: category check 10 minutes, success 0.5; flag check 5 minutes, success 0.8; recall 0.8; false flags Poisson, set per state and category to 80% precision; 200 draws; $30 an hour. The realised share of expected flags that are real is 0.8000; there are 0.298 expected flags per scored household. Choices the paper does not state are listed in `1g-review-model.csv`. They are: how a household's error is split across its variances, the cost weight (`FYWGT`, one review per household a year), category-ranking ties, and whether D.9 checks flags outside the scope.
- **Choices the paper does not state (1g, 1h):** `1h-unstated-choices.csv` runs two choices on their own and together: (i) variances in households at or below the tolerance draw no flag; (ii) the category ranking counts above-tolerance dollars only. Reach counts are unchanged under all four variants (risk only 25, AI only 19, combined 26, D.9 27; combined cheaper in 25 of 25).

| Variant | Median cost ratio, risk only / combined | Expected flags per scored household |
|---|---|---|
| As run in section 1 | 8.98 | 0.298 |
| No flag for households at or below the tolerance | 19.97 | 0.12 |
| Category ranking from above-tolerance dollars only | 8.98 | 0.298 |
| Both changes | 19.97 | 0.12 |

  `1g-review-model.csv` gives the share of flaggable variances in households at or below the tolerance as 0.624. The two other orders of Appendix D.8 (historical return, random) are not coded: see the row "AI-only flag order" in `1g-review-model.csv`.
- **How computed:** `snapqc.review_sim`; per-state costs in `1b-reach-cost-by-state.csv`, paper-versus-run table in `1b-paper-vs-run.csv`, rebuilt Figure 2 in `1b-figure2-rebuilt.svg`, stability in `1h-stability.csv`.
- **Mismatch causes:** not known. The mismatches are in cost ratios and in which small states favour AI only, not in reach. The two unstated choices tested in 1h do not close them: letting only above-tolerance households draw flags moves the median ratio from 8.98 to 19.97, further from the paper, and changing the category ranking leaves it at 8.98. One unverified guess: the paper reads costs off a grid of budgets. In its Figure 2 the risk-only and combined points coincide for Alabama and Mississippi, and risk only is cheaper in Nevada, while here combined is several times cheaper in all three. The lower AUC may come from the engine change (no xgboost, no row subsampling) or from the frame version.
- **Feeds:** `rq.md` RQ2 (the baseline rule, and whether "combined beats risk score only" holds on FY2024) and RQ3 (what our rule is compared against).
- **Outcome:** `snapqc.review_sim` is the baseline rule, described as "rebuilt from the paper". It reproduces the paper's reach counts, not-reached sets and Figure 1. The authors' simulation code was not requested (2026-11-14 15:10 entry of `sprint-log.md`), and no OpenMP runtime or R was installed, so the risk score does not use xgboost itself; that may explain the AUC gap.

The budget result in 1d bears on which budgets to test.

## 2. Risk score by error source

- **Question:** how well does the risk score rank the human arm's client errors, compared with agency errors?
- **Numbers:** AUC against the 29,898 scored comparison households of the 41 states (no expected figure was set).

| Group | Score | Households (scored) | AUC | 95% interval | Top 20% of state risk order |
|---|---|---|---|---|---|
| Human arm | filed application | 616 (616) | 0.694 | 0.678 to 0.714 | 33.9% |
| Unique filed value | filed application | 465 (465) | 0.773 | 0.758 to 0.789 | 42.8% |
| Filed at $0 | filed application | 151 (151) | 0.451 | 0.412 to 0.490 | 6.6% |
| Human arm | section 1 score | 616 (614) | 0.716 | 0.697 to 0.734 | 36.5% |
| Human arm | QC-verified application | 616 (616) | 0.615 | 0.593 to 0.637 | 28.2% |
| Not rebuilt (301) | section 1 score | 301 (221) | 0.704 | 0.674 to 0.735 | 34.4% |
| Agency errors only | section 1 score | 2,060 (1,916) | 0.755 | 0.745 to 0.765 | 43.9% |
| Agency errors only | QC-verified application | 2,060 (2,060) | 0.680 | 0.668 to 0.690 | 35.2% |

  The score ranks client errors filed as a wrong number (465) about as well as agency errors, and it ranks the errors filed at $0 (151) below chance. In the top 20% of their state's risk order: 43% of the 465, 7% of the 151 and 44% of agency-error households. Within-state and weighted figures tell the same story.
- **How computed:** the 616 households of `results/01-data-checks/3d-human-arm.csv` are scored from their filed application through `snapqc.features.bgl_features` and the models of section 1. Agency-error households and the 301 that could not be rebuilt use section 1's score. AUC against households in the same 41 states with no error above $56, pooled and within state. Files: `2b-filed-by-group.csv`, `2d-not-rebuilt.csv`, `2e-risk-score-by-error-source.csv`, `2e-roc-by-error-source.svg`.
- **Mismatch:** none, because no expected figure was set. Three things to read with the result. (1) The score depends on its inputs: for the 616, 0.694 from the filed application, 0.716 from BGL's frame (section 1's score) and 0.615 from QC-verified amounts; for agency errors 0.755 from BGL's frame and 0.680 from QC-verified amounts. (2) The two scores agree on the 465 (0.773 both) and differ on the 151 (0.451 filed, 0.541 section 1), where BGL's reconstructed amount is within $5 of the $0 filed value in 98 of 151 households (part 3d). (3) Not every household has a section 1 score: 614 of the 616, 221 of the 301 and 1,916 of the 2,060 agency-error households.
- **Feeds:** `rq.md` RQ2 (whether "risk score only" is a fair comparison on client errors).
- **Open questions:**
  - (a) whether "risk score only" is reported for all client errors together or with the errors filed at $0 shown apart, since it ranks them below chance;
  - (b) which inputs comparison and agency-error households are scored on in the caseload simulation: BGL's frame, as in section 1, or the feature function on QC-verified amounts;
  - (c) whether the 301 enter RQ2 with section 1's score or are left out.

## 3. Feature leakage

- **Question:** does every risk-score feature come from the filed application, through one function, for both filers?
- **Numbers:** pass (no expected figure was set). One function, `snapqc.features.bgl_features`, builds the 19 inputs from an application table. On the 24,908 FY2024 households that BGL's reconstruction left unchanged it equals BGL's frame exactly for 18 features and for 99.67% of households on the 19th (`count_divisible_by_100`; 82 households differ by 1). Changing the QC-verified amount in the error group moves no filed-application feature for any of the 616; the same change moves the QC-verified features of 614. The builder reads 75 QC columns, none of them a total QC derives from the verified amounts or a review outcome. All 151 households filed at $0 have a $0 income feature for the group in error, $0 for each of its income types and an implied benefit at the maximum. Of the 19 features, 5 come from the form alone, 10 from the form with published tables and the benefit formula, and 4 need an agency or state record (`expedited_i`, `months_since_cert_n`, `percent_abawd`, `bbce_state_i`).
- **How computed:** `snapqc.features`; files `3a-feature-sources.csv`, `3b-function-vs-frame.csv`, `3b-leak-check.csv`, `3c-filed-at-zero.csv`, `3d-not-on-the-form.csv`, `3d-training-inputs.csv`, `3d-bgl-amount-vs-filed-value.csv`, `3d-filed-vs-trained-inputs.csv`, `3d-score-by-input-version.csv`.
- **Mismatch:** none against an expected figure. One difference between training and scoring: the model is trained neither on filed applications nor on QC-verified amounts but on BGL's frame, where the amounts of 98.5% of FY2022-23 error rows (8,345 of 8,469) and of 29.2% of rows without an error were moved by BGL's reconstruction, which uses the QC finding. `count_divisible_by_100` is counted over QC-verified amounts in training. For the human arm, BGL's reconstructed amount is within $5 of our filed value in 530 of 614 households (432 of 463 with a unique filed value, 98 of the 151); all 19 features agree in 171 of 614, and the two scores correlate 0.931.
- **Feeds:** `rq.md` RQ2 and RQ3 (leakage guard).
- **Open questions:**
  - (a) for AI-filed applications, whether the four agency-record features are copied from the household's QC row (the same for both filers) or dropped and the model refitted;
  - (b) whether to accept that the model is trained on BGL's reconstruction and scored on filed applications, or to refit it on inputs built by this function (FY2022-23 filed values would have to be rebuilt first);
  - (c) how an AI filer's form maps to the 21 income types that `count_divisible_by_100` counts.

## 4. Toy simulator

- **Question:** on a toy caseload with planted errors and synthetic 80/80 flags, does the oracle score 1.0 and random order score about the budget share?
- **Numbers:** the sanity values hold. The oracle's capture is 1.0 and no rule exceeds it (highest 0.948). Random order corrects 1.24%, 2.56%, 5.07% and 25.66% of its everything-checked dollars at budget shares of 1.28%, 2.56%, 5.13% and 25.64%. Capture, flag-check price for an undocumented-group flag:

| Rule | 0.5 min | 1 min | 2 min | 10 min |
|---|---|---|---|---|
| Random order | 0.017 | 0.027 | 0.051 | 0.257 |
| Risk score only | 0.049 | 0.070 | 0.117 | 0.367 |
| Flag checks only | 0.630 | 0.868 | 0.863 | 0.863 |
| Gold standard (combined) | 0.668 | 0.877 | 0.887 | 0.948 |
| Combined, risk order (D.9) | 0.086 | 0.116 | 0.190 | 0.567 |
| Dollars per minute, fact-group rates | 0.691 | 0.878 | 0.881 | 0.920 |
| Dollars per minute, rates by risk score | 0.698 | 0.880 | 0.881 | 0.941 |
| Oracle | 1.0 | 1.0 | 1.0 | 1.0 |

  With an undocumented-group flag at the category-check price (10 minutes, success 0.5) the gold standard falls to 0.468, 0.631, 0.817 and 0.920. Flag-led rules stay far above risk score only, and ranks change only among the gold standard and the two queues, whose captures are within 0.03 of each other at 0.5 to 2 minutes under either price. Capture and dollars remaining order the rules identically at a given budget and price. They move in opposite directions in 4 of 32 cells when the price changes (random order and the D.9 rule) and for flag checks only between 1 and 2 minutes (capture 0.868 to 0.863 while dollars remaining fall).
- **How computed:** `snapqc.queue_sim`: 400 synthetic households by four fact groups, redrawn 400 times; every rule is a queue of checks and one engine spends the budget down it. Budgets 0.5, 1, 2 and 10 reviewer minutes per application. Files: `4a-toy-caseload.csv`, `4b-rules-run.csv`, `4c-sanity.csv`, `4d-capture-by-budget.csv`, `4d-capture-by-budget.svg`, `4d-capture-by-error-share.csv`, `4d-queue-vs-gold-standard.csv`, `4d-what-a-budget-buys.csv`, `4e-undocumented-flag-price.csv`, `4f-capture-and-dollars-remaining.csv`, `4f-opposite-directions.csv`.
- **Mismatch:** (1) "Random order about the budget share" holds for random order's share of its own everything-checked dollars. Its capture as defined (divided by the oracle) is 0.017, 0.027, 0.051 and 0.257, which equals the budget share only once the oracle has run out of errors (2 and 10 minutes). (2) The three budgets do not all differ: the gold standard's capture is 0.668 at 0.5 minutes but 0.877 and 0.887 at 1 and 2, because at a 20% error share every flag is checked by 1.0 minute. At a 30% share (close to section 1's 0.298 flags per household) the three are 0.565, 0.761 and 0.882; at 10% they are 0.857, 0.878 and 0.898. (3) The dollars-per-minute queue does not beat the gold standard at every budget, even with true rates: with fact-group rates the paired difference is +0.023 at 0.5 minutes, +0.001 at 1, -0.007 at 2 and -0.028 at 10; with rates by risk score +0.030, +0.003, -0.007 and -0.007. At the category-check price it is not ahead at any budget.
- **Feeds:** `rq.md` RQ2 and RQ3 (the simulator and the capture metric).
- **Options:**
  - (a) the price of a flag on an undocumented fact group in the main run: category check (10 minutes, 0.5) or flag check (5 minutes, 0.8); the order of rules is the same either way in the toy, and the capture of flag-led rules is up to 0.27 lower at the category price;
  - (b) report error dollars remaining per 1,000 applications beside capture, since capture is not comparable across prices or budgets;
  - (c) what the queue's expectation uses: fact-group rates only, or the risk score as well (which closes most of the gap at 10 minutes), and whether it should skip the other fact groups of a flagged household as the gold standard does;
  - (d) whether budgets are fixed at 0.5, 1 and 2 minutes or set relative to the minutes needed to check every flag (5 minutes times flags per application).

## 5. Standard errors for RQ3

- **Question:** how many random category checks (n) are needed before filer-specific flag precision and miss rate are known well enough to help?
- **Numbers:** the earlier rough figure holds for checks that always find the error: 497 checks give +/- 1.5 points at a 3% rate (expected about 500). Checks that find the error half the time need 1,009. Base rates with one comparison household per error household and 80/80 flags: an error in 17.1% (earned), 15.5% (shelter), 10.6% (unearned) and 5.5% (utilities) of applications, so miss rates of 4.1%, 3.7%, 2.4% and 1.2%, and precision 0.8. With n spread over eight cells (`5d-grid-of-n.csv`, earned and utilities):

| n | Reviewer hours | SE of miss rate / rate (earned, utilities) | SE of precision, points (earned, utilities) |
|---|---|---|---|
| 50 | 8.3 | 3.03, 5.36 | 94.7, 166.8 |
| 800 | 133.3 | 0.76, 1.34 | 23.7, 41.7 |
| 3,200 | 533.3 | 0.38, 0.67 | 11.8, 20.9 |
| 12,800 | 2,133.3 | 0.19, 0.33 | 5.9, 10.4 |

  Precision is known far worse because few random checks land on a flag. Filer-specific against pooled rates (`5c-crossing-n.csv`; filer-specific at least 5% better than pooled from n):

| True gap between filers | Precision | Miss rate |
|---|---|---|
| None | above 12,800 | above 12,800 |
| Precision -10 points, miss rate x1.5 | above 12,800 | 12,800 |
| Precision -20 points, miss rate x2 | 6,400 | 3,200 |
| Precision -40 points, miss rate x3 | 800 | 800 |

  With no gap, pooled always wins. The shrunk estimate is never more than 3.5% worse than pooled (worst ratio 1.035).
- **How computed:** `snapqc.calibration`: standard error of flag precision and miss rate from n checks that find an error half the time; base rates from the 616; then filer-specific against pooled rates with shrinkage, by simulation. Files: `5a-base-rates.csv`, `5b-se-by-n.csv`, `5b-se-by-n.svg`, `5b-se-formula-vs-simulation.csv`, `5c-crossing-n.csv`, `5c-rmse-by-n.csv`, `5c-rmse-by-n.svg`, `5d-grid-of-n.csv`.
- **Mismatch:** none against the rough figure. Two results bear on the grid. (1) Its largest value is not clearly enough: at n = 3,200 a cell's precision still has a standard error of 12 to 21 points, and filer-specific rates beat pooled rates only if the true gap is about 20 precision points and a doubled miss rate, or more. (2) The base rates depend on which errors can be flagged. Under Table B1 of the paper only 6% (earned), 5% (unearned), 34% (shelter) and 43% (utilities) of the 616's errors have a cause assumed detectable; then flags are rare and miss rates are 16.5%, 10.3%, 11.9% and 3.7%. At the 41-state caseload mix the 616's errors are in under 1% of applications and the earned miss rate is 0.12%, where 6,440 checks on unflagged groups in one cell give a standard error of half the rate.
- **Feeds:** `rq.md` RQ3 (the grid of n for the capture curve).
- **Options:**
  - (a) the grid 50, 100, 200, 400, 800, 1,600, 3,200: the smallest values are too noisy to help, and the largest is enough only for a large gap;
  - (b) add 6,400 and 12,800 (1,067 and 2,133 reviewer hours of category checks);
  - (c) aim half of each cell's checks at flagged fact groups: at n = 3,200 the precision standard error falls from 12 to 21 points to 6.9, and the earned miss rate's relative standard error rises from 0.38 to 0.49;
  - (d) estimate precision from the outcomes of the flag checks a rule makes anyway (not computed here);
  - (e) use the shrunk estimate wherever filer-specific rates are used.

Summarized by Claude Code (Opus 5.5).
