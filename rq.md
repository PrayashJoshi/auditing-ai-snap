# Research Questions & Limitations (final)
Written By: Prayash
Drafted with: Claude

## Question

#### When an AI agent files applications for Supplemental Nutrition Assistance Program (SNAP), what can an agency's AI reader catch?

We are interested in two sides of the complete application pipeline: 
- **Applicant's AI Filer**: We want to understand its capabilities and limitations in filing documents 
- **Agency's AI Reader**: We want to understand what the reader flags.
- Both studies are done on the Quality Control household case files synthetically reconstructed from real FY2024 SNAP Quality Control (QC) households.
    - Across 44,891 audited households in 53 jurisdictions, the FY2024 public-use QC file records audited dollar totals and error codes, with no case documents (Leftin et al., 2026). Our benefit calculator reproduces issued benefits (FSBEN) within $1 on 44,296 of those 44,891 households.  
  

RQ1: What does an AI filer write when the evidence is missing, and does the form decide whether the gap stays visible?
RQ2: What does an AI reader flag, and how sure is it?


## Why it matters

- Starting in October 2027, H.R. 1 requires a state with a SNAP payment error rate at or above 6% to pay a share of benefit costs. The share starts at 5% and could reach hundreds of millions or billions of dollars a year for some states (Public Law 119-21, 2025; Chohlas-Wood, 2026).
- In FY2024, 44 states were at or above 6%, and the national payment error rate was 10.93% (USDA FNS, 2025).
- Michigan's SNAP agency has already deployed an AI case reader, built on Google Vertex AI (Davidson, 2026).
- AI agents can already attempt SNAP applications (Propel, 2026). The number of applicants that use one is unknown.
    - Over 10 million Americans already have access to commercial browser agents. In Propel's exploratory tests on live SNAP portals in California and Florida, ChatGPT Agent and Claude Cowork navigated forms (ChatGPT Agent also uploaded a document)—yet when information was missing, they defaulted to guessing instead of handing off to the user.
- Recent work by Chohlas-Wood (2026) looks at simulation estimates. Results showed that, under its main assumptions, risk-based review alone costs about seven times as much as combining AI flags with risk scores. The cases that AI flags are simply assumed, not measured yet.
    - Assumes AI flags errors visible in the case record with fixed 80% recall (of detectable errors) and 80% precision.
    - The simulation assumes that AI cannot detect unreported income, because case documents would be unlikely to show it is missing. In our computation from the FY2024 QC data, client-withheld income is 22.35% of weighted SNAP error dollars ($105.8M across 984 households), and auditors found 8.50% of those error dollars from the case record, compared with 59.62% for agency-caused errors. Across 185 human-arm omissions, auditors found 16 (8.65%) from the case record and relied primarily on recipient interviews (117 of 185, 63.2%).
- We found no study that runs form-schema abstention and evidence-completeness tests on QC-grounded synthetic SNAP case files. 


## RQ1. What does an AI filer write when the evidence is missing, and does the form decide whether the gap stays visible?
- Usman (2026) showed that when an input lacks evidence and a structured schema requires a field with no escape option, language models fabricate answers instead of abstaining
    - No follow-up turn is described, and the tests used categorical enums and text strings where no neutral value exists.
    - Reported 100% fabrication in 10 of 13 tested models under required JSON fields without an escape slot, and 60–100% fabrication across all 9 tested open-weight models even when an "insufficient evidence" escape option was provided. That directly anchors why RQ1 tests open-weight models.
    - On a numeric SNAP income field, $0 is a legitimate entry for zero-wage households, so alongside fabricating outright, models can fail through silent omission.
- The filer is a large language model. In our case, open-weight. It consumes the applicant's note and the records a connector retrieved (pay stubs, bank transactions, lease, utility bills) and submits the form by calling it once.
- The evidence for the wage line has four levels: missing (the note names the employer; no pay stub and no payroll deposit), partial (payroll deposits of net pay only), stale (one pay stub dated 2026-05-15, months before the September to October bank window) and a note that states a rounded amount.
  - This is a convenient simplification of how a personal agent harness would orchestrate tool use to gather all relevant information and have a ledger to go off of. The expectations are that personal AI agents' tools will only get smarter and more compatible. We are not interested in researching if tool calls are accurate and they work.
- A filing for the wage line is a $0, an "unknown", a figure copied or derived from a document, or a number that appears in no document (an invented number).
- RQ1 compares two forms. 
  - Strictly numeric schema: requires a numeric value in every financial field.
  - Abstention-and-provenance schema: allows an explicit "unknown" (insufficient_evidence) entry and requires a source classification and supporting document IDs for each filled field    
- Downstream Visibility: a deterministic consistency rule check for applications against the packet's authoritative documents. Flags "unknown" or non-zero without a supporting document.

### Limitations:
- **Synthetic case files [assumption].** Households, amounts and error structure come from FY2024 QC. Every document (pay stub, bank line, employer name, note) is generated by our code. Real records are scanned, misspelled, split across payers and inconsistent in format, so a rate here describes this testbed, not real documents.
- **No applicant follow-up [limit].** The filer submits the form by calling it once and gets one scripted rejection. Propel found that agents guess instead of handing off to the user, and a handoff is not available here. The results show what a filer enters when it cannot ask.
- **One income line, 12 households [limit].** RQ1 tests the wage line on 12 wage-only households with one wage source each, drawn at random from the 8,921 that qualify. Each rate rests on a small number of filings per model, so intervals are wide. Self-employment, unearned income and households with several jobs are outside the test, and the results say nothing about them.
- **Three open-weight models, one fixed instruction per form [limit].** Models run at temperature 0 with one instruction text for each form. Repeats are identical requests, so they measure run-to-run variation, not sensitivity to wording. The filers are not the commercial browser agents (ChatGPT Agent, Claude Cowork) that Propel tested, so the results describe these three models, not those agents.
- **The two forms differ in more than one feature [limit].** The abstention-and-provenance form allows "unknown" and also requires a source and document IDs. The RQ1 comparison is between two bundles, so a difference between forms is not attributed to either feature alone. A form that allows "unknown" without asking for sources was run only in an exploratory pilot of 8 households, not in the main run.
- **Twelve households do not separate the models [limit].** The pre-specified filer hypotheses are one-sided thresholds (for example, invented numbers at most 5% of numeric filings). With 12 households the intervals around each rate are wide, so differences between models are not resolved.
- **The second turn is descriptive [limit].** The rejection turn has no pre-specified hypothesis. What a filer does after the rejection (holds an "unknown" or a $0, or replaces it with a number) is reported as it occurs and is not tested against a threshold. Filings that do not parse into a number are classed invalid and stay in the denominators.
- **QC-verified amounts are treated as true [assumption].** The wage a filing is compared with is the QC reviewer's verified amount. The pool keeps only households whose benefit our calculator reproduces within $1 (44,296 of 44,891). Any error in a verified amount carries into how a filing is scored.
- **Evidence levels are constructed [assumption].** "Stale" is one stub dated 2026-05-15 with its amounts scaled to 0.60 of the original. "Partial" is net-pay deposits only. The note's amount is always the true wage rounded to the nearest $50, so the note never misreports. The results do not show how a filer handles a note that misstates income.
- **No adversarial applicant [scope].** The note is cooperative, and the applicant does not tell the agent to hide or shade income. The results describe a filer working for an honest applicant.
- **No estimate of how often or how much [scope].** RQ1 reports what a filer enters under stated conditions. It gives no estimate of how many applicants use a filer, how often evidence is missing in a real case, or the effect on a state's error rate or cost share.
- **Text-only documents [scope].** The filer reads text, not images, PDFs or scans, so the results do not cover documents submitted as images.


## RQ2. What does an AI reader flag, and how sure is it?
- An omitted income source is income that the household has but did not report.
- The reader is an open-weight LLM given the household's monthly application (whole dollars per group) and case-file documents. It outputs group-level flags, each with a 0–1 confidence score and reason, or an empty list if nothing looks unsupported.
    - The reader never sees an "unknown" token; every unentered group appears as $0, mirroring strictly numeric intake forms from RQ1 so we can measure what AI review catches after missing evidence is flattened to $0.
- A true zero is an income line that the household correctly reported as $0.
- The hit rate (true-positive rate) is the share of omitted income sources that the reader flags. The false-flag rate (false-positive rate) is the share of true zeros that it flags. A flag counts only if it names the income group in question.
    - In the 41-state FY2024 QC data, known omissions make up 0.46% to 2.47% of $0 income filings, so a $0 is overwhelmingly a true zero. However, the real-world mix of evidence traces and AI-filed cases is unknown. Rather than computing precision at an artificial experimental balance, we measure hit rate and false-flag rate within controlled strata, and check the false-flag rate against the 0.09%–0.51% ceiling required for 80% precision at 80% recall.
- For an omitted income source, RQ2 compares four levels of evidence in the file: none; bank deposits with a clear label (such as a payroll deposit from the employer); the same deposits with an ambiguous label ("ACH credit electronic deposit"); and a lead from the agency's cross-check of external wage or benefit records (acting as a positive control)
- For negative controls, RQ2 tests three cases: 
    - Truthful documented households (full records matching reported income)
    - True zeros with rent (an income group correctly reported as $0, with the household's rent shown in the file)
        - Across all truthful FY2024 QC households (n = 44,891), 5.8% unweighted (6.6% weighted) have both a $0 income group and rent alone exceeding reported income, and that share rises to 11.9% unweighted (13.8% weighted) with simulated utility bills. A deterministic rent-only rule yields only 2.7% precision (59 of 2,179) at 31.9% recall (59 of 185).
    - True zeros with one stray credit (a single non-recurring savings transfer shown under a clear own-account label versus the same "ACH CREDIT ELECTRONIC DEPOSIT" label).
        - Because the stray transfer is a single small credit rather than the 2–9 recurring deposits ($73–$2,500 each; $300–$3,581 monthly verified target income) seen in omission files, comparing the clear and ambiguous labels isolates how label ambiguity alone shifts false flags.

### Limitations & Assumptions:
- **The files are not filer output [limit].** No application written by the RQ1 filer is given to a reader. Reader inputs are built from QC households, and the link to RQ1 is the $0 that the required form produces when evidence is missing. The results show what a reader catches in that $0 input, not in applications a filer wrote.
- **Omissions are those QC found [assumption].** An omitted income source is a group filed at $2 or less whose QC-verified amount is above $0. It records what the reviewer found, not intent and not omissions nobody found. Hit rates apply to omissions of that kind.
- **True zeros are lines QC verified as $0 [assumption].** Known omissions are 0.46% to 2.47% of $0 filings in the FY2024 data, so some drawn true zeros may hide income nobody found. A false-flag rate may therefore include flags on undetected omissions.
- **Small samples, reused households [limit].** There are 24 omission households (12 earned, 12 unearned target), 24 truthful documented, 48 true zeros with rent and 48 with a stray credit. The same households appear at every evidence level, and the 24 omissions are spread over many income types (wages are 11 of them). Intervals are wide, per-income-type results are not estimable, and the rates describe these households.
- **A hit at "none" may reflect the budget gap [limit].** The file always holds shelter and utility amounts next to reported income. A rule that flags whenever shelter plus utilities exceed reported income flags about half of the omission files with no evidence at all, and some true zeros. A hit at "none" may therefore come from that gap and not from detecting the omission.
- **A stray credit stands in for an ambiguous deposit [assumption].** The stray credit is the generator's own savings transfer, $41 to $219, one time. The contrast with the clear label isolates label ambiguity, and the false-flag rates describe this credit type, not all ambiguous deposits.
- **Readers never see "unknown" [scope].** Every group is shown as a whole-dollar amount, mirroring the required form. RQ2 therefore measures what a reader catches after missing evidence has been turned into $0, not how it handles an explicit "unknown".
- **Evidence states are removals from a complete file [assumption].** "Clear deposits" and "ambiguous deposits" keep only the target group's bank credits, with generator labels such as `PAYROLL DEP <EMPLOYER>`. Real bank records mix many labels, amounts and dates, so the rates describe these simplified files.
- **The lead is a positive control, not a cross-match [assumption].** The lead is one added line saying records elsewhere show wages or benefits. It checks that a reader acts on stated evidence. It does not test how a reader uses an agency's actual cross-check.
- **Three open-weight readers, one instruction text [limit].** Readers run at temperature 0 on instructions reused unchanged from an earlier exploratory run, with two identical repeats per input. The instructions define "not provided", a value that never appears in these files. The results describe these three readers under this instruction text.
- **Invalid outputs are left out of the rates [limit].** A call without a valid flag report is excluded from the denominator and counted beside each rate. A reader that fails more often is scored on fewer files, so rates for different readers rest on different numbers of files.
- **A flag is not a decision [scope].** The reader does not adjudicate, and no caseworker acts on a flag. RQ2 shows what a reader flags and gives no estimate of the effect on error rates, review cost or a state's cost share.
- **Text-only documents [scope].** The reader sees text, not scans or images, so the results do not cover documents submitted as images.


## Sources

- Afane, M., Robitschek, E., Ouyang, D., and Ho, D. E. (2026). Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication. arXiv:2604.19895. `literature/Afane_et_al_2026_Learning_When_Not_to_Decide.pdf`
- Chohlas-Wood, A. (2026). Reducing SNAP Payment Errors at Lower Cost With AI: A Simulation Study. arXiv:2611.03213. `literature/Chohlas-Wood_2026_Reducing_SNAP_Payment_Errors_AI_Simulation.pdf`
- Davidson, K. (2026). Michigan's use of AI to process SNAP applications draws concerns about past automation failures. Michigan Advance, republished by Route Fifty. https://www.route-fifty.com/artificial-intelligence/2026/03/michigans-use-ai-process-snap-applications-draws-concerns-about-past-automation-failures/412395/ `literature/Davidson_2026_Michigan_AI_SNAP_Applications_Route_Fifty.md`
- Leftin et al. (2026). FY2024 SNAP Quality Control technical documentation (August 2026). `literature/Leftin_et_al_2026_FY2024_SNAP_QC_Technical_Documentation_Aug2026.pdf`
- Propel: Solomon, J., and Williams, A. (2026). Can AI Agents Help People Apply for SNAP? (Part 1). `literature/Propel_2026_Can_AI_Agents_Help_People_Apply_for_SNAP_Part_1.md`
- Public Law 119-21 (2025). H.R. 1, section on the state cost share of SNAP benefits. `literature/Public_Law_119-21_HR1_2025_enacted_text.txt`
- Usman (2026). When the Form Demands an Answer, Language Models Invent One. arXiv:2607.20492 (v2). `literature/Usman_2026_When_the_Form_Demands_an_Answer_PhantomFill.pdf`
- USDA Food and Nutrition Service (2025). Press release: FY2024 SNAP payment error rates. `literature/USDA_FNS_2025_Press_Release_FY2024_SNAP_Payment_Error_Rates.md`
