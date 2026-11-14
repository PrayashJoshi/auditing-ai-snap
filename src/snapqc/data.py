"""Shared loaders and constants for the notebooks."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

TOLERANCE = 56  # dollars: AMTERR above this counts as an error (per rq.md)
N_ELEMENTS = 9  # ELEMENT1..ELEMENT9 and the parallel per-variance columns

# The 41 states (40 states and DC) with a single utility-allowance row in the FY2024 SNAP QC technical
# documentation, Appendix Table F.7. The other 12 have several rows (by household size, region or date).
SINGLE_SUA_STATES = [
    "Alabama", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia",
    "Florida", "Georgia", "Idaho", "Illinois", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "South Carolina", "South Dakota", "Texas", "Utah", "Vermont", "Washington", "West Virginia", "Wisconsin",
    "Wyoming",
]


def per_variance(prefix: str) -> list[str]:
    """Column names such as ELEMENT1..ELEMENT9 for a per-variance prefix."""
    return [f"{prefix}{i}" for i in range(1, N_ELEMENTS + 1)]


def load_fy2024() -> pd.DataFrame:
    """FY2024 SNAP QC public-use file, as published (no filtering)."""
    return pd.read_csv(RAW / "fy2024" / "qc_pub_fy2024.csv", low_memory=False)


def results_dir(name: str) -> Path:
    """Create and return results/<name>/."""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
