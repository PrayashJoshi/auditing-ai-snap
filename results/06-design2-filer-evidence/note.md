# 06 Design 2 filer evidence: note

Notebook: `notebooks/06-design2-filer-evidence.py` (first run, no new model calls). The outputs of the filer experiments are kept next to their runners under `experiments/`, not here.

- **First run (exploratory):** 8 pilot households, 4 evidence levels, 2 schemas, 3 models, a portal-rejection turn and two side arms. Scoring, tables and the full discussion are in `experiments/design2/note.md`; the wage-line re-score is in `experiments/design2/rescore_design2.py`.
- **Replication (pre-specified):** 12 households drawn at random from the 8,921-household wage-only pool (seed 20261116), both schemas, three models, two repeats. The design, draw and the three filer hypotheses are in `experiments/design2_v2/freeze.md`, committed before the first call (`334e943`). Outcomes: `experiments/design2_v2/hypotheses.csv`; household-level and call-level scores: `scored_household_v2.csv`, `scored_main_v2.csv`; re-score tables: `rescore-*.csv`; raw calls: `results.jsonl`; run logs: `turn1.log`, `turn2.log`.
- **How to read the replication:** all nine model-by-hypothesis cells pass at turn 1, so 12 households do not separate the models. The informative results are the turn 2 invented wages and the schema-invalid filings, both in the re-score tables. Turn 2 carries no hypothesis in the freeze file and is descriptive.

Summarized by Claude Code (Opus 5.5).
