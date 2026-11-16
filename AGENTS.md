# Rules for agents working in this repo

This repo holds the experiments for the Apart Research AI x Epistemics sprint (Nov 13-15, 2026): data handling, notebooks, results. Git history is the record of work done during the sprint.

`rq.md` (research questions) and `sprint-log.md` (decisions) are this repo's own copies. Do not edit them unless the author asks; report any conflict between them and your results instead.

`literature-review.md` is written and maintained by the author. Do not edit it unless the author asks.

## Workflow
- One numbered check = one notebook pair: `notebooks/NN-slug.py` (jupytext percent format; the file you edit) and the `.ipynb`, executed headless with outputs saved.
- Outputs go to `results/NN-slug/`: tables (csv), plain matplotlib or graphviz figures (svg), and `note.md` with the number(s), how they were computed, which part of `rq.md` they feed, and open questions or options (never a recommendation). Experiments that call models keep their runners, inputs, raw JSONL calls and notes under `experiments/`.
- Each notebook starts with a header cell: question, expected number, method, result, mismatch (if any), feeds.
- Commit messages say what the commit contains ("Design 1 v2: dry run of 72 calls, all three readers valid 24 of 24"), one commit per piece of work, no attribution trailers. Never touch history (no amend, rebase, force push).

## Numbers
- Numbers that do not match the expected figures are reported as mismatches. Never adjust them to match.
- Never invent numbers, sources or citations.

## Secrets and model output
- Secrets live in `.env` only (git-ignored).
- Raw model calls are committed as JSONL under `experiments/`.

## Submission
- `submission/` holds the sprint report (`draft.md`, `main.tex`, `references.bib`, `figures/`, `main.pdf`) in ACM `sigconf` (`nonacm`) layout; nothing beyond 4–6 pages excluding references and appendices (hard cap 6 because this format has 2 columns).
- `submission/draft.md` is the author's own Markdown draft, written solely by the author. Never edit, rewrite, or expand it unless explicitly asked.
- Never write prose or edit `submission/main.tex` on your own. Only when the author explicitly asks to update `main.tex` from `draft.md` (or assist with LaTeX mechanics), faithfully port the `draft.md` content into `sigconf` LaTeX without altering its wording, claims, or meaning, alongside formatting, tables, figures, equations, and build/layout fixes. Flag gaps or mismatches instead of filling them.
- Keep `submission/` clean for presentation: LaTeX build artifacts, `main-backup.tex` and `figures/sampleteaser.pdf` are git-ignored; never force-add them or alter `main-backup.tex`.
- Keep emails and Discord handles out of `submission/`, and verify any formatted table values or BibTeX entries against `results/` and verified sources.
