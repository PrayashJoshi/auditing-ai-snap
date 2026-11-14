# Rules for agents working in this repo

This repo holds the experiments for the Apart Research AI x Epistemics sprint (Nov 13-15, 2026): data handling, notebooks, results. Nothing else. Git history is the evidence of work done during the sprint.

`rq.md` (research question) and `sprint-log.md` (decisions) are this repo's own copies. Do not edit them unless Prayash asks; report any conflict between them and your results instead.

`literature-review.md` (literature review) is written and maintained by Prayash himself. Do not edit it unless Prayash asks.

## Workflow
- One test ID = one notebook pair: `notebooks/<ID>-<slug>.py` (jupytext percent format; the file you edit) and the `.ipynb`, executed headless with outputs saved.
- Outputs go to `results/<ID>/`: tables (csv), plain matplotlib or graphviz figures (svg), and `note.md` with: the number(s); how they were computed; which part of `rq.md` it feeds; a recommendation marked "NOT DECIDED (for Prayash)"; and a short LLM-use line (which agent/model did the work).
- Each notebook starts with a header cell: test ID, question, expected number, method, result, mismatch (if any).
- Commit message starts with the test ID ("A1: reproduce funnel 44,891 -> 917"). Commit per test. No attribution trailers. Never touch history (no amend, rebase, force push).

## Numbers
- Numbers that do not match the expected figures are reported as mismatches. Never adjust them to match.
- Never invent numbers, sources or citations.

## Order
A1, A3, A4, then A5-A9, then B1-B5 (plus the "added after review" rows). C tests wait until Prayash says go (they need decisions, an API key and a packet generator).

## Secrets and LLM output
- Secrets live in `.env` only (git-ignored).
- LLM outputs, when they exist, are committed as JSONL under `experiments/`.
