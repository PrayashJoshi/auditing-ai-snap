# Literature Review
Written By: Prayash
Organized, Quote Blocks, Formatted: Gemini 4 Hydrogen
Notes, excerpts, and critiques for the research questions in [rq.md](rq.md). Local PDF copies live under [literature/](literature/).


## Index

| Reference | Year | Focus / Data | Role in `rq.md` | Local copy |
|---|---|---|---|---|
| [Chohlas-Wood (2026)](#1-chohlas-wood-2026) | 2026 | Preauthorization SNAP review simulation (FY2022–2024 public QC) | Baseline review-allocation rules (RQ2, RQ3); cost/time model | [Chohlas-Wood_2026_Reducing_SNAP_Payment_Errors_AI_Simulation.pdf](literature/Chohlas-Wood_2026_Reducing_SNAP_Payment_Errors_AI_Simulation.pdf) |

---

## 1. Chohlas-Wood (2026)

- **Paper:** *Reducing SNAP Payment Errors at Lower Cost With AI: A Simulation Study*
- **Author / Citation:** Alex Chohlas-Wood, Department of Applied Statistics, Social Science, and Humanities, New York University. arXiv:2611.03213v1 [cs.CY], 2 November 2026. https://arxiv.org/abs/2611.03213
- **Local PDF:** [literature/Chohlas-Wood_2026_Reducing_SNAP_Payment_Errors_AI_Simulation.pdf](literature/Chohlas-Wood_2026_Reducing_SNAP_Payment_Errors_AI_Simulation.pdf)

### Overview

- No AI system built, no LLM prompted, and no AI tested on case documents.
- Counterfactual simulation on public USDA SNAP QC data:
  - *Def:* replaying a real historical "what-if" scenario to estimate how the outcome changes when that scenario did not actually happen.
- **Goal:** Estimate how much staff time (and money) a state would need to spend on pre-authorization case reviews to push its payment error rate below 6%.
  - *Why 6%?* Legal cutoff in the 2025 federal budget law (H.R. 1) where states stop paying $0 for SNAP benefits and start owing a share of total benefit costs.
  - *FY2024 baseline:* 44 states were above 6%, with a 10.93% national average (USDA FNS official FY2024 release). Crossing this error threshold means millions to billions a year in budget obligations for states.
  - *Simulation sample baseline:* Because the public-use QC file is lower than official PERs, only **27 jurisdictions** start at or above 6% PER in the paper's FY2024 sample (48 contiguous states + D.C).
  - Makes sense to establish a challenging threshold—it encourages better training, review, and overflagging. Overflagging creates unnecessary delays for people who desperately need benefits, plus extra paperwork all around.

> **Why this study is important:**  
> "Under the new law, states will for the first time be required to pay a portion of SNAP benefit costs if their payment error rate is at or above 6%. Their required contribution starts at 5% of their total SNAP benefit spending, with higher error rates triggering larger contributions... starting in October 2027..."  
> — *Introduction, p. 2*

### Key Results

- **Jurisdictions reaching <6% PER (Section 3, p. 9):** Out of the 27 jurisdictions starting at or above 6%, **Combined** brings **26** below 6% (missing D.C.), **Risk-only** brings **25**, and **AI-only** brings **19**.
- **Cost comparison (Section 3, pp. 9–10):** Among the 25 jurisdictions where both reach 6%, Combined is cheaper in **22**, with Risk-only costing a median of **about 7x** as much as Combined. (In CA and MA, Risk-only costs ~2–11x as much; in 5 states—DE, MS, NV, AL, AZ—AI-only costs less than Combined.)
- **Scale vs. penalties (Figure 2, p. 11):** Annual review costs to reach 6% are far smaller than even the lowest 5% benefit cost-sharing penalty.

> **Headline result (Abstract, p. 1):**   
> "Under our main assumptions, risk-based review alone costs about seven times as much as combining AI flags with risk scores. This cost advantage persists across a wide range of plausible assumptions."

### Assumptions & Critiques

1. **Public SNAP QC file excludes ineligible households who received benefits in error.**
   - The starting PER is lower than the official USDA PER, making the 6% threshold easier to reach.
   - Public-file PER is lower than official PER in 52 of 53 jurisdictions ($r = 0.79$ in included jurisdictions, $0.63$ overall).
     > **Quote (Appendix A, p. 15):** "Public-file rates are lower in 52 of the 53 jurisdictions shown. The public file thus broadly tracks official PER rates, but does not reproduce the official rates exactly."

   - The simulation also excludes **9.5%** of public-use households from error correction (special benefit formulas or incomplete/inconsistent records) before adding them back into totals.
     > **Quote (Sec. 1, p. 4):** "When simulating payment error correction, we additionally exclude 9.5% of households included in the public use data because they are either subject to special benefit calculations or have incomplete or inconsistent records."


2. **The risk model trains pooled across all states for FY2022–2023, split by household size (1, 2–3, 4+), and is tested on FY2024.**
   - Predicts only whether total error exceeds the $56 tolerance.
   - *Why `$56?`* Under USDA rules, small mistakes of `$56` or less per month are ignored when calculating a state's official Payment Error Rate (PER).
   - *PJ:* Would be interesting to see if an adaptive threshold is better than a static threshold.
   - Because the XGBoost model predicts binary `> $56` error rather than expected error dollars, a `$57 error` and `$600` error look identical.

3. **The paper assumes AI spots detectable errors like a coin flip.**
   - Assumes the LLM flags 80 out of 100 detectable errors, and for every 100 flags it raises, 80 are actual errors (80/80 recall and precision, with Poisson false flags).
   - Not proven; purely hypothetical.
   - AI is *not* assumed to catch all error types—only 15 primary-cause codes (`AGENCYn`) plus missing causes visible in the case record (e.g., data entry, policy misapplication, disregarded info, incomplete info). Unreported/withheld client info and collateral-contact errors (codes 1, 3, 4, 7, 8, 26, 99) are assumed undetectable.
     > **Quote (Sec. 2.2, pp. 6–7):** "Notably, the system (at least as envisioned above) would only be able to flag errors visible in the information already reported by the applicant. To represent this limitation in our simulation, we use the 'primary cause of variance' to define which errors we assume AI can detect."
     >
     > "Specifically, we assume each detectable error has a 80% probability of generating a flag, independently of other errors (see Appendix D.2 for sensitivity to this probability). Within each state and category, we set the expected number of false flags so that 80% of flags correspond to actual errors."


4. **Given a time budget, the simulation buys review checks under three designs:** 
   - **Risk-only:** category checks cost more time (10 min) and have a lower shot at fixing the errors in that category (50%).
   - **AI-only:** checks only AI flags (5 min each, 80% fix rate), ordered by each state's historical error-dollar category totals (with no risk scores).
   - **Combined:** checks AI flags first in descending case-risk order (5 min each, 80% fix rate), then runs ordinary category checks on unflagged cases in risk order.
   - **No fallback on flagged cases:** In the main Combined design, cases that receive an AI flag get *only* flag checks and no subsequent category search—which leaves D.C. at 55% of error dollars removed vs. 57% needed (p. 10). (Appendix D.9 tests a risk-order-throughout variant that reaches all 27 jurisdictions, at higher cost elsewhere.)
     > **Quote (Sec. 2.3, p. 8):** "In addition, we assume that cases with flags receive only flag checks; reviewers do not subsequently search those cases for unflagged errors."

   - **Hindsight scope selection:** At every budget, Risk-only and Combined report the *best-performing review scope* (1–12 categories) in hindsight—an optimistic bound a state could not know in advance.
     > **Quote (Sec. 3, p. 9):** "At each review cost, we report the largest reduction in PER across the review scopes we test, selecting the best-performing scope with the benefit of hindsight. This gives an optimistic estimate of what agencies could achieve in practice, since an agency would have to choose a single review scope for preauthorization review without knowing in advance which would perform best."


5. **The simulation prices labor as recurring cost only:**
   - Reviewer time is priced at `$30/hr` net to the state (`$40/hr` total after the 25% federal admin match starting FY2027).
   - Does not include the cost of AI, API integration, or building the system out safely and accurately.
   - **Unpriced applicant burden:** False AI flags and preauthorization review could slow approvals or trigger burdensome document requests, yet the simulation assigns zero cost to applicant delay or erroneous denials.
     > **Quote (Sec. 4, p. 12):** "But preauthorization review also runs the risk of slowing approval or could lead to burdensome requests for additional documents, especially when AI flags are incorrect or difficult to resolve. Our simulations do not measure these effects, nor do they establish whether such tools would reduce erroneous denials or improve participation among eligible households."


6. **Simulation code is not public:**
   - Reference `[9]` (`github.com/giannella/snap_qc`) contains only the BGL risk model in R, not the flag perturbation or queue simulation. Our repo rebuilds the paper's four review rules from the text in [src/snapqc/review_sim.py](src/snapqc/review_sim.py) (checked in [results/02-rule-checks/note.md](results/02-rule-checks/note.md)).

### Relevance and Gaps for This Project ([rq.md](rq.md))

- **Baseline (RQ2, RQ3):** Supplies the Risk-only, AI-only, and Combined baselines and review-cost parameters (`5 min / 0.8` flag check, `10 min / 0.5` category check, `$30/hr`).
- **Gap 1 — Assumed vs. measured LLM triage (RQ1):** No LLM actually reads applications or documents; 80/80 accuracy on the paper's Table B1 causes is assumed.

  > **Quote (Sec. 4, p. 12):** "Whether AI can identify SNAP errors reliably enough to deliver these gains remains an open question. We simulate under a set of assumptions that seem plausible to us about detection and correction, but performance may be different in practice."

- **Gap 2 — Human-only vs. AI-filed applications (RQ1, RQ2):** Models only historical human/caseworker QC errors, not applications filled out by AI agents or whether reader accuracy shifts by filer.
- **Gap 3 — Filer-aware review allocation (RQ3):** Treats all applications under one uniform rule rather than conditioning review priorities on who filed the application.
