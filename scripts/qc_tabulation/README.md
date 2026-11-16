# QC tabulation scripts

One-off command-line checks on the FY2024 SNAP QC public file, run during data exploration on 2026-11-13. They are not imported by the notebooks or by `src/snapqc`. Each takes the path of the QC CSV as its first argument, for example:

```
uv run python scripts/qc_tabulation/qc_checks.py data/raw/fy2024/qc_pub_fy2024.csv
```

| Script | What it prints |
|---|---|
| `qc_checks.py` | Row and column counts, key-column presence, and the filter that selects rebuildable client-error cases |
| `recompute.py` | The benefit recomputed from QC inputs against the issued benefit |
| `fullchain.py` | The same recomputation from application-level inputs and state parameters |
| `diag2.py` | Minimum-benefit values and the rows the calculator cannot match |
| `state_table.py` | Per-state benefit parameters |
| `sua.py` | Standard utility allowance by state |
| `formcheck.py` | Text and fillable fields of an application form PDF (needs `pypdf`) |

The notebooks in `notebooks/` supersede these scripts; they are kept so the first tabulation can be reproduced.
