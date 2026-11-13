# auditing-ai-snap

Experiment repo for an Apart Research AI x Epistemics sprint submission (Nov 13-15, 2026). It holds only the research work: data handling for the SNAP Quality Control public database, notebooks for a set of pre-experiment tests, and their results. Commit history documents what was done during the sprint. (A project site will be built separately later.)

| Path | Contents |
|---|---|
| `AGENTS.md` | Rules for agents working here |
| `rq.md` | Overarching idea and the three research questions |
| `sprint-log.md` | Running log of decisions, newest first (copy) |
| `data/` | `README.md` (sources), `fetch.sh`, `SHA256SUMS`, `raw/` (not committed), `processed/` (small derived tables) |
| `src/snapqc/` | Check scripts for the QC funnel |
| `notebooks/` | One notebook pair (`.py` jupytext + executed `.ipynb`) per test ID |
| `results/` | `results/<ID>/` outputs and `note.md` |
| `experiments/` | Later LLM runs (JSONL) |

## Setup
```
uv sync
bash data/fetch.sh        # or symlink existing files into data/raw/
uv run python -c "import snapqc"
```

## Preexisting and third-party
- SNAP QC public-use database (USDA FNA, prepared by Mathematica): third-party data, not committed; see `data/README.md`.
- `src/snapqc/*.py` were written on 2026-11-13, during the sprint, and imported here unchanged.
- Python dependencies are listed in `pyproject.toml`.
