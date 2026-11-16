# Design 1 v2, post hoc analyses

All five CSVs are post hoc: computed after every result was seen, with zero model calls, from the committed records of the frozen run, the added-readers run, the amount-only arm and the clear-label twin. Validity is the frozen `reader_output_valid`, the latest record per job is used, and everything is repeat 1. Readers: the three frozen and the three added.

- `flags-by-rule-state.csv`: at evidence state "none" (omission and zero_rent pooled, and each alone), per reader, target flag k/n when the observable budget-gap rule (lease rent plus bills above reported income, as `checker_plus_budget_gap_rule` defines it) fires vs when it is silent, with the exact Fisher p.
- `r1-lease-matched.csv`: hit rate at omission/none minus target false-flag rate at zero_rent/none with the frozen scorer's Newcombe interval, on all households and on the omission households with rent.
- `twoxtwo-by-reported-income.csv`: the clear/ambiguous label x small/large amount cells on the 48 stray-credit true zeros, frozen readers, split by whether the application reports any income, with the per-household interaction sign test and the count of households without a lease.
- `any-group-false-flags.csv`: any-group flag rate on zero_rent/none and the groups flagged, per reader.
- `qwen38-recurrence.csv`: qwen3.8-27b either-income flags at omission/deposits_ambiguous and zero_stray/stray_ambiguous, validity share, and how often it flags any group in the stray cell (with the counts for non-income and non-target groups).

Run from the repository root (the script asserts the known counts):

    PYTHONPATH=src python experiments/design1_v2_posthoc/posthoc.py
