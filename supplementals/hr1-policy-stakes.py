# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Supplemental Policy Analysis: State Fiscal Exposure Under Public Law 119-21 (H.R. 1)
#
# **Research motivation ("Why SNAP Right Now"):** Public Law 119-21 (enacted July 4, 2025) fundamentally reshapes SNAP fiscal incentives by shifting federal benefit and administrative costs onto state budgets through two statutory mechanisms:
#
# 1. **Sec. 10105 — Error-Rate-Tiered Benefit Cost-Sharing (FY2028 onward):** Beginning in FY2028, every state with a Quality Control Payment Error Rate (PER) at or above **6.00%** must fund an increasing share of SNAP benefit allotments out of state budgets:
#    - **PER < 6.00%:** `0%` state benefit share
#    - **6.00% <= PER < 8.00%:** `5%` state benefit share
#    - **8.00% <= PER < 10.00%:** `10%` state benefit share
#    - **PER >= 10.00%:** `15%` state benefit share
#    - **FY2028 Lookback & Delay Rule:** For FY2028, a state uses either its **FY2025** or **FY2026** PER (before reverting to a standard `t - 3` fiscal-year lag in FY2029+). States where `1.5 × PER >= 20.0%` (**PER >= 13.333%**) in FY2025 (or FY2026) receive a **1-year delayed implementation** to FY2029 (or FY2030).
# 2. **Sec. 10106 — Universal Administrative Match Reduction (FY2027 onward):** Reduces the federal reimbursement share of State agency SNAP administrative costs from **50% to 25%** starting in FY2027, doubling the statutory state match (`50% -> 75%`, a `+25 percentage point` shift) across **all** 53 SNAP jurisdictions — including states whose PER falls below `6.00%`.
#
# ## Primary Data Sources (All Verified on Disk under `data/raw/policy/`)
# - **Official USDA FNS / FNA Payment Error Rates (FY2022–FY2025):**
#   - `snap-qcfy25-per.pdf` (Dated June 24, 2026; U.S. national PER `10.62%` = `9.28%` over + `1.33%` under; tolerance `$57`)
#   - `snap-fy24QC-PER.pdf` (Dated June 30, 2025; U.S. national PER `10.93%` = `9.26%` + `1.67%`; tolerance `$56`)
#   - `snap-fy23-qc-payment-error-rate.pdf` (Dated June 28, 2024; U.S. national PER `11.68%`; tolerance `$54`)
#   - `snap-fy22-qc-payment-error-rate.pdf` (Dated June 30, 2023; U.S. national PER `11.54%`; tolerance `$48`)
# - **Official USDA State Monthly & Annual SNAP Benefits (`snap-zip-fy69tocurrent-10.zip`, Data as of October 9, 2026):** Complete 12-month closed FY2022–FY2025 state issuance (`FY25` total = `$95,759,225,827` across 53 jurisdictions), reconciled to `$0` against `US Summary` and `snap-annualsummary-10.xlsx`.
# - **Official USDA State Agency Administrative Costs (`snap-sar-fy23.pdf`, Table 8):** Latest published state-by-state breakdown of State vs. Federal administrative match excluding temporary COVID/ARPA funds (`FY2023` non-ARPA Federal share = `$5,203,372,299`, implying a `+25 pp` match shift equal to `50%` of the federal share = `$2.602B/yr`; national USDA "All Other Costs" in `snap-annualsummary-10.xlsx` grew `+12.65%` from FY2023 `$6,053.82M` to FY2025 `$6,819.81M`).
# - **U.S. Census Bureau State Population (`NST-EST2025-ALLDATA.csv`, Vintage 2025):** `POPESTIMATE2025` for all 50 states + D.C.
# - **U.S. Census Bureau State Government Finances (`GS00STATEFIN01` FY2024 & `GS00LOCALFIN` FY2024 for D.C.):** FY2024 Total General Expenditure (`SF0133` for 50 states; `LF0103` local/district general expenditure for D.C.).
#
# ## Methodological Framing
# Rather than extrapolating volatile post-pandemic OLS slopes (which project unphysical negative or `>100%` error rates in unwinding states like New Jersey or Maryland), we evaluate:
# - **Primary Baseline (`FY2025 Persistence`):** Full-year closed FY2025 benefits multiplied by the Sec. 10105 statutory tier implied by official FY2025 PER (released June 24, 2026 — one of the two statutory election years for FY2028), paired with FY2023 Table 8 Sec. 10106 admin cost shift (`0.5 × Federal Share`).
# - **Sensitivity Benchmarks:** 3-year mean PER (`FY2023–FY2025`), FY2024 PER persistence, and immediate FY2028 vs. 1-year delayed FY2029 timing (`PER >= 13.333%`).
# - **Size-Normalized Fiscal Strain:** Per-capita annual burden (`$/resident`) alongside `% of FY2024 State General Expenditure`, so populous states (`CA`, `NY`, `FL`, `TX`) do not obscure acute per-resident budget stress in smaller states (`DC`, `HI`, `OR`, `NM`, `RI`, `AK`).

# %%
from __future__ import annotations

import csv
import io
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reproducible SVG/PDF text & house visual styling
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
mpl.rcParams["axes.edgecolor"] = "#1F1F1F"
mpl.rcParams["axes.linewidth"] = 0.8

# Okabe-Ito-grounded sequential + dual-mechanism palette (WCAG & grayscale verified)
PALETTE = {
    "ink": "#1F1F1F",
    "muted": "#5C5C5C",
    "context": "#BDBDBD",
    "rule": "#E3E3E3",
    "paper": "#FFFFFF",
    "tier0": "#F1F1F1",        # 0% benefit share (<6% PER)      | gray ~94.5%
    "tier5": "#F6D277",        # 5% benefit share (6-<8%)       | gray ~83.5%
    "tier10": "#E69F00",       # 10% benefit share (8-<10%)     | gray ~63.6%
    "tier15": "#C44D00",       # 15% benefit share (>=10%, FY28)| gray ~41.0%
    "tier15_delay": "#7A2503", # 15% deferred FY29 (>=13.33%)   | gray ~23.0%
    "bar_benefit": "#A63603",  # Uniform stacked benefit bar     | gray ~32.0%
    "admin_blue": "#56B4E9",   # Sec. 10106 universal admin cut  | gray ~62.0%
    "ref_gray": "#D6D6D6",     # 3-yr sensitivity ref bar        | gray ~84.0%
    "up_vermilion": "#C44D00",
    "down_blue": "#0072B2",
}

def find_repo_root() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists() and (p / "data").exists():
            return p
    raise FileNotFoundError("Could not locate repository root containing pyproject.toml and data/")

ROOT = find_repo_root()
RAW_DIR = ROOT / "data" / "raw" / "policy"
PROCESSED_PATH = ROOT / "data" / "processed" / "hr1_state_snap_exposure.csv"
FIG_DIR = ROOT / "supplementals" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Guam": "GU", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
    "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virgin Islands": "VI", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

print("ROOT:", ROOT)
print("RAW_DIR files:", sorted(p.name for p in RAW_DIR.iterdir()))

# %% [markdown]
# ## 1. Parse and Verify Primary Raw Sources
#
# Every input dataset is parsed deterministically from files cached under `data/raw/policy/` and validated with exact reconciliation assertions against official national totals.

# %%
def parse_per_pdfs(raw_dir: Path) -> tuple[dict[int, dict[str, tuple[float, float, float]]], dict[int, tuple[float, float, float]]]:
    """Extract FY2022-FY2025 state PER tables from official USDA PDFs via macOS PDFKit (or cached text)."""
    txt_cache = raw_dir / "per-all-4yrs.txt"
    pdf_files = [
        raw_dir / "snap-fy22QC-PER.pdf",
        raw_dir / "snap-fy23QC-PER.pdf",
        raw_dir / "snap-fy24QC-PER.pdf",
        raw_dir / "snap-qcfy25-per.pdf",
    ]
    if all(p.exists() for p in pdf_files):
        swift_code = """
import Foundation
import PDFKit
for idx in 1..<CommandLine.arguments.count {
    let url = URL(fileURLWithPath: CommandLine.arguments[idx])
    print("=== FILE:", url.lastPathComponent)
    if let doc = PDFDocument(url: url) {
        print(doc.page(at: 0)?.string ?? "")
    }
}
"""
        res = subprocess.run(["swift", "-", *[str(p) for p in pdf_files]], input=swift_code, capture_output=True, text=True, check=True)
        raw_text = res.stdout
        txt_cache.write_text(raw_text)
    elif txt_cache.exists():
        raw_text = txt_cache.read_text()
    else:
        raise FileNotFoundError("Missing official PER PDF files in data/raw/policy/")

    per_by_fy: dict[int, dict[str, tuple[float, float, float]]] = {}
    nat_by_fy: dict[int, tuple[float, float, float]] = {}

    for block in raw_text.split("=== FILE:")[1:]:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        fy = 2000 + int(re.search(r"fy(\d{2})", lines[0], re.I).group(1))
        per_by_fy[fy] = {}
        for ln in lines[1:]:
            m = re.match(r"^([A-Z .]+?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)$", ln)
            if not m:
                continue
            name = m.group(1).strip().title().replace("Of ", "of ")
            vals = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            if name in ("United States", "National"):
                nat_by_fy[fy] = vals
            else:
                per_by_fy[fy][name] = vals
        assert len(per_by_fy[fy]) == 53, f"Expected 53 jurisdictions in FY{fy}, got {len(per_by_fy[fy])}"

    assert nat_by_fy[2022] == (9.84, 1.70, 11.54)
    assert nat_by_fy[2023] == (10.03, 1.64, 11.68)
    assert nat_by_fy[2024] == (9.26, 1.67, 10.93)
    assert nat_by_fy[2025] == (9.28, 1.33, 10.62)
    return per_by_fy, nat_by_fy


def _read_xlsx_bytes(data: bytes) -> dict[str, list[list[str | None]]]:
    """Pure-stdlib XLSX reader using zipfile + ElementTree (no external spreadsheet dependency)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        ss: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root:
                ss.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
        sheets = sorted(
            [n for n in zf.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")],
            key=lambda s: int(re.search(r"sheet(\d+)", s).group(1)),
        )
        out: dict[str, list[list[str | None]]] = {}
        for sh in sheets:
            sroot = ET.fromstring(zf.read(sh))
            srows: list[list[str | None]] = []
            for row in sroot.iter():
                if row.tag.endswith("}row"):
                    vals: list[str | None] = []
                    for c in row:
                        t = c.attrib.get("t", "")
                        v = None
                        for ch in c:
                            if ch.tag.endswith("}v"):
                                v = ch.text
                            elif ch.tag.endswith("}is"):
                                v = "".join(x.text or "" for x in ch.iter() if x.tag.endswith("}t"))
                        if v is not None and t == "s":
                            v = ss[int(v)]
                        vals.append(v)
                    if any(x is not None for x in vals):
                        srows.append(vals)
            out[sh] = srows
        return out


def parse_snap_zip_benefits(raw_dir: Path) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, float]]:
    """Parse complete state monthly summary workbooks (FY22-FY25) from official USDA ZIP."""
    exclude = {
        "NERO", "MARO", "SERO", "MWRO", "SWRO", "MPRO", "WRO", "US Summary",
        "American Samoa", "Commonwealth of Northern Mariana Islands", "Puerto Rico",
    }
    ben_by_fy: dict[int, dict[str, dict[str, float]]] = {}
    us_cost_by_fy: dict[int, float] = {}

    def _flt(v: str | None) -> float | None:
        if v is None:
            return None
        s = str(v).strip()
        return None if s in ("", "--", "NA", "-") else float(s)

    with zipfile.ZipFile(raw_dir / "snap-zip-fy69tocurrent-10.zip") as zf:
        for fy in (2022, 2023, 2024, 2025):
            sheets = _read_xlsx_bytes(zf.read(f"FY{fy - 2000}.xlsx"))
            ben_by_fy[fy] = {}
            for rows in sheets.values():
                cur: str | None = None
                for r in rows:
                    c0 = str(r[0]).strip() if r and r[0] is not None else ""
                    if (
                        len(r) >= 2
                        and r[0]
                        and r[1] is None
                        and not any(
                            k in c0
                            for k in ("National Data Bank", "SNAP Monthly", "Fiscal Year", "Footnotes", "ALL DATA", "1.", "2.", "Total")
                        )
                    ):
                        cur = c0
                    elif c0 == "Total" and cur and len(r) >= 4:
                        hh, prs, cost = _flt(r[1]), _flt(r[2]), _flt(r[3])
                        if cur == "US Summary":
                            assert cost is not None
                            us_cost_by_fy[fy] = cost
                        elif cur not in exclude:
                            assert hh is not None and prs is not None and cost is not None
                            ben_by_fy[fy][cur] = {"households": hh, "persons": prs, "cost_usd": cost}

            sum_53 = sum(v["cost_usd"] for v in ben_by_fy[fy].values())
            assert len(ben_by_fy[fy]) == 53, f"Expected 53 SNAP jurisdictions in FY{fy}"
            assert abs(sum_53 - us_cost_by_fy[fy]) < 1.0, f"FY{fy} checksum mismatch: {sum_53} vs {us_cost_by_fy[fy]}"

    return ben_by_fy, us_cost_by_fy


def parse_sar_fy23_admin(raw_dir: Path) -> dict[str, dict[str, int]]:
    """Parse Table 8 (State vs. Federal Admin Costs, excl. ARPA/CAA) from FY23 State Activity Report."""
    txt_cache = raw_dir / "sar-fy23-text.txt"
    if (raw_dir / "snap-sar-fy23.pdf").exists() and not txt_cache.exists():
        swift_code = """
import Foundation
import PDFKit
let url = URL(fileURLWithPath: CommandLine.arguments[1])
if let doc = PDFDocument(url: url) {
    for i in 0..<doc.pageCount {
        print("--- PAGE \\(i+1) ---")
        print(doc.page(at: i)?.string ?? "")
    }
}
"""
        res = subprocess.run(["swift", "-", str(raw_dir / "snap-sar-fy23.pdf")], input=swift_code, capture_output=True, text=True, check=True)
        txt_cache.write_text(res.stdout)

    p14 = txt_cache.read_text().split("--- PAGE ")[14]
    toks: list[str] = []
    for ln in p14.splitlines():
        s = ln.strip()
        if not s or s.startswith(("14 ---", "TABLE 8", "STATE", "Administrative", "Costs", "ARPA Funding", "12", "U.S.", "Federal Share of")):
            continue
        toks.extend(re.findall(r"\$[\d,]+|[A-Za-z][A-Za-z .]+", s))

    sar: dict[str, dict[str, int]] = {}
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i].startswith("$") and not toks[i + 1].startswith("$"):
            s1, s2 = toks[i].strip(), toks[i + 1].strip()
            vals = [int(toks[i + 2 + k].replace("$", "").replace(",", "")) for k in range(8)]
            sar[s1] = {"state_share_usd": vals[0], "fed_share_usd": vals[2], "arpa_usd": vals[4], "reported_tot_usd": vals[6]}
            sar[s2] = {"state_share_usd": vals[1], "fed_share_usd": vals[3], "arpa_usd": vals[5], "reported_tot_usd": vals[7]}
            i += 10
        else:
            st = toks[i].strip()
            vals = [int(toks[i + 1 + k].replace("$", "").replace(",", "")) for k in range(4)]
            sar[st] = {"state_share_usd": vals[0], "fed_share_usd": vals[1], "arpa_usd": vals[2], "reported_tot_usd": vals[3]}
            i += 5

    assert len(sar) == 53
    assert sum(v["state_share_usd"] for v in sar.values()) == 5_392_601_443
    assert sum(v["fed_share_usd"] for v in sar.values()) == 5_203_372_299
    assert sum(v["arpa_usd"] for v in sar.values()) == 526_404_622
    return sar


def load_census_benchmarks(raw_dir: Path) -> tuple[dict[str, int], dict[str, float]]:
    """Load Census Vintage 2025 population and FY2024 General Expenditure (SF0133 states, LF0103 D.C.)."""
    pop25: dict[str, int] = {}
    with open(raw_dir / "NST-EST2025-ALLDATA.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["SUMLEV"] == "040" and r["NAME"] != "Puerto Rico":
                pop25[r["NAME"]] = int(r["POPESTIMATE2025"])

    gen_exp24: dict[str, float] = {}
    with open(raw_dir / "census_statefin_2022_2024.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["YEAR"] == "2024" and r["AGG_DESC"] == "SF0133":
                gen_exp24[r["NAME"]] = float(r["AMOUNT_THOUSANDS"]) * 1_000.0
    with open(raw_dir / "census_dc_localfin_2022_2024.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["YEAR"] == "2024" and r["GOVTYPE_LABEL"] == "Local" and r["AGG_DESC"] == "LF0103":
                gen_exp24["District of Columbia"] = float(r["AMOUNT_THOUSANDS"]) * 1_000.0

    assert len(pop25) == 51 and len(gen_exp24) == 51
    return pop25, gen_exp24


def sec10105_rate(per_pct: float) -> float:
    if per_pct < 6.0:
        return 0.00
    if per_pct < 8.0:
        return 0.05
    if per_pct < 10.0:
        return 0.10
    return 0.15


def is_delayed_fy29(per_pct: float) -> bool:
    """P.L. 119-21 Sec. 10105(a)(2)(B)(iii): delayed to FY2029 if 1.5 * FY2025 PER >= 20%."""
    return 1.5 * per_pct >= 20.0


per_by_fy, nat_per = parse_per_pdfs(RAW_DIR)
ben_by_fy, us_cost = parse_snap_zip_benefits(RAW_DIR)
sar_admin = parse_sar_fy23_admin(RAW_DIR)
pop_2025, gen_exp_2024 = load_census_benchmarks(RAW_DIR)

print(f"Verified FY2025 SNAP Benefits across 53 jurisdictions: ${us_cost[2025]:,.0f}")
print(f"Verified FY2025 National PER: {nat_per[2025][2]:.2f}% (over={nat_per[2025][0]:.2f}%, under={nat_per[2025][1]:.2f}%)")
print(f"Verified FY2023 Non-ARPA Federal Admin Share: ${sum(v['fed_share_usd'] for v in sar_admin.values()):,.0f}")

# %% [markdown]
# ## 2. Build Master State Exposure Dataset & Reconcile Scenarios

# %%
records = []
for state in sorted(per_by_fy[2025]):
    abbr = STATE_ABBR[state]
    is_state_or_dc = state not in ("Guam", "Virgin Islands")

    over25, under25, per25 = per_by_fy[2025][state]
    per24 = per_by_fy[2024][state][2]
    per23 = per_by_fy[2023][state][2]
    per22 = per_by_fy[2022][state][2]
    per_3yr = (per23 + per24 + per25) / 3.0

    b25 = ben_by_fy[2025][state]["cost_usd"]
    b24 = ben_by_fy[2024][state]["cost_usd"]
    prs25 = ben_by_fy[2025][state]["persons"]
    hh25 = ben_by_fy[2025][state]["households"]

    r25 = sec10105_rate(per25)
    r24 = sec10105_rate(per24)
    r_3yr = sec10105_rate(per_3yr)
    delayed25 = is_delayed_fy29(per25)
    delayed24 = is_delayed_fy29(per24)

    liab_ben_25 = b25 * r25
    liab_ben_3yr = b25 * r_3yr
    liab_ben_24rate = b25 * r24

    # Sec. 10106: federal admin share cut from 50% to 25% -> state absorbs 50% of current Fed match
    fed_adm_23 = sar_admin[state]["fed_share_usd"]
    state_adm_23 = sar_admin[state]["state_share_usd"]
    adm_shift = 0.5 * fed_adm_23

    tot_combined = liab_ben_25 + adm_shift
    pop = pop_2025.get(state)
    gexp = gen_exp_2024.get(state)

    records.append({
        "state": state,
        "abbr": abbr,
        "is_50_plus_dc": is_state_or_dc,
        "per_fy22": per22,
        "per_fy23": per23,
        "per_fy24": per24,
        "per_fy25": per25,
        "per_fy25_over": over25,
        "per_fy25_under": under25,
        "per_3yr_mean_23_25": round(per_3yr, 2),
        "per_delta_24_25_pp": round(per25 - per24, 2),
        "snap_persons_fy25": round(prs25, 1),
        "snap_households_fy25": round(hh25, 1),
        "snap_benefits_fy24_usd": b24,
        "snap_benefits_fy25_usd": b25,
        "sec10105_rate_fy25": r25,
        "sec10105_rate_fy24": r24,
        "sec10105_rate_3yr": r_3yr,
        "delayed_fy29_flag": delayed25,
        "delayed_fy24_flag": delayed24,
        "sec10105_start_fy": 0 if r25 == 0 else (2029 if delayed25 else 2028),
        "sec10105_benefit_liability_usd": round(liab_ben_25, 2),
        "sec10105_benefit_liability_3yr_usd": round(liab_ben_3yr, 2),
        "sec10105_benefit_liability_fy24rate_usd": round(liab_ben_24rate, 2),
        "sar_fy23_state_admin_usd": state_adm_23,
        "sar_fy23_fed_admin_usd": fed_adm_23,
        "sec10106_admin_shift_usd": round(adm_shift, 2),
        "combined_annual_exposure_usd": round(tot_combined, 2),
        "pop_2025": pop,
        "census_gen_exp_fy24_usd": gexp,
        "benefit_liab_per_capita_usd": round(liab_ben_25 / pop, 2) if pop else None,
        "combined_per_capita_usd": round(tot_combined / pop, 2) if pop else None,
        "benefit_liab_per_participant_usd": round(liab_ben_25 / prs25, 2) if prs25 else None,
        "benefit_liab_pct_gen_exp": round(100.0 * liab_ben_25 / gexp, 4) if gexp else None,
        "combined_pct_gen_exp": round(100.0 * tot_combined / gexp, 4) if gexp else None,
    })

df = pd.DataFrame(records).sort_values("combined_annual_exposure_usd", ascending=False).reset_index(drop=True)
df.to_csv(PROCESSED_PATH, index=False)
df.to_csv(FIG_DIR / "hr1-state-summary.csv", index=False)

# Scenario totals table
df51 = df[df["is_50_plus_dc"]]
scenarios = pd.DataFrame([
    {
        "scenario": "Primary baseline: FY2025 PER persistence (Sec. 10105 benefit only)",
        "jurisdictions_affected_53": int((df["sec10105_rate_fy25"] > 0).sum()),
        "states_dc_affected_51": int((df51["sec10105_rate_fy25"] > 0).sum()),
        "annual_cost_53_usd_b": round(df["sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51["sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "notes": "Evaluated on closed FY2025 issuance ($95.76B total) & June 2026 PER release.",
    },
    {
        "scenario": "Immediate FY2028 wave (6% <= PER < 13.33%)",
        "jurisdictions_affected_53": int(((df["sec10105_rate_fy25"] > 0) & (~df["delayed_fy29_flag"])).sum()),
        "states_dc_affected_51": int(((df51["sec10105_rate_fy25"] > 0) & (~df51["delayed_fy29_flag"])).sum()),
        "annual_cost_53_usd_b": round(df.loc[~df["delayed_fy29_flag"], "sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51.loc[~df51["delayed_fy29_flag"], "sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "notes": "Begins Oct 1, 2027 (FY2028) assuming FY2025 PER is elected or persists.",
    },
    {
        "scenario": "Deferred FY2029 wave (1.5 * PER >= 20%, i.e. PER >= 13.33%)",
        "jurisdictions_affected_53": int(df["delayed_fy29_flag"].sum()),
        "states_dc_affected_51": int(df51["delayed_fy29_flag"].sum()),
        "annual_cost_53_usd_b": round(df.loc[df["delayed_fy29_flag"], "sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51.loc[df51["delayed_fy29_flag"], "sec10105_benefit_liability_usd"].sum() / 1e9, 3),
        "notes": "AK, DC, NM, DE, GA, IL, OR deferred 1 year under Sec. 10105(a)(2)(B)(iii).",
    },
    {
        "scenario": "Sec. 10106 Admin Match Cut (FY2027+, 50% -> 25% Fed share)",
        "jurisdictions_affected_53": 53,
        "states_dc_affected_51": 51,
        "annual_cost_53_usd_b": round(df["sec10106_admin_shift_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51["sec10106_admin_shift_usd"].sum() / 1e9, 3),
        "notes": "Hits all 53 jurisdictions (+25 pp admin match) using FY2023 non-ARPA Table 8 baseline.",
    },
    {
        "scenario": "Combined steady-state annual shift (Sec. 10105 FY25 + Sec. 10106)",
        "jurisdictions_affected_53": 53,
        "states_dc_affected_51": 51,
        "annual_cost_53_usd_b": round(df["combined_annual_exposure_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51["combined_annual_exposure_usd"].sum() / 1e9, 3),
        "notes": "Full phased-in annual budget burden once FY2029 deferrals expire.",
    },
    {
        "scenario": "Sensitivity: 3-year mean PER (FY2023-FY2025) benefit share only",
        "jurisdictions_affected_53": int((df["sec10105_rate_3yr"] > 0).sum()),
        "states_dc_affected_51": int((df51["sec10105_rate_3yr"] > 0).sum()),
        "annual_cost_53_usd_b": round(df["sec10105_benefit_liability_3yr_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51["sec10105_benefit_liability_3yr_usd"].sum() / 1e9, 3),
        "notes": "Confirms FY2025 baseline ($11.11B) is conservative vs. 3-year average ($11.94B).",
    },
    {
        "scenario": "Sensitivity: FY2024 PER applied to FY2025 benefits (benefit only)",
        "jurisdictions_affected_53": int((df["sec10105_rate_fy24"] > 0).sum()),
        "states_dc_affected_51": int((df51["sec10105_rate_fy24"] > 0).sum()),
        "annual_cost_53_usd_b": round(df["sec10105_benefit_liability_fy24rate_usd"].sum() / 1e9, 3),
        "annual_cost_51_usd_b": round(df51["sec10105_benefit_liability_fy24rate_usd"].sum() / 1e9, 3),
        "notes": "Shows 1-year prior error distribution would yield $11.80B benefit liability.",
    },
])
scenarios.to_csv(FIG_DIR / "hr1-national-scenarios.csv", index=False)
scenarios

# %% [markdown]
# ## 3. Figure 1: Composite Policy-Stakes Architecture (`hr1-policy-stakes-map`)
#
# Three synchronized panels designed at 1:1 print dimensions (`11.8 in x 6.4 in`, minimum label `7.0 pt`):
# - **Panel `a` (Tile Grid Map):** Equal-area U.S. state tile grid (`50 states + D.C.`, plus `GU` / `VI` SNAP territory callouts) color-coded by Sec. 10105 statutory benefit tier (`0%`, `5%`, `10%`, `15%`), with darkest fill + diagonal corner hatch + `†` marker isolating the 7 jurisdictions deferred to FY2029 (`PER >= 13.33%`). Each tile directly prints combined steady-state annual liability (`Sec. 10105 + Sec. 10106`) and FY2025 PER.
# - **Panel `b` (Size-Normalized Strain):** Top 12 jurisdictions ranked by per-resident combined annual burden (`$/resident/yr`), decomposing Sec. 10105 benefit cost-sharing (tier vermilion) from Sec. 10106 administrative match shift (`#0072B2` blue), with absolute `$M/$B` totals printed directly alongside each bar.
# - **Panel `c` (Cliff Volatility):** FY2024 → FY2025 error-rate trajectories for notable threshold crossers overlaid on statutory tier bands, illustrating how a 1–3 percentage-point improvement or regression swings state liabilities by hundreds of millions per year.

# %%
TILE_GRID = {
    "AK": (0, 0), "ME": (11, 0),
    "WI": (6, 1), "VT": (10, 1), "NH": (11, 1),
    "WA": (1, 2), "ID": (2, 2), "MT": (3, 2), "ND": (4, 2), "MN": (5, 2),
    "IL": (6, 2), "MI": (8, 2), "NY": (9, 2), "MA": (10, 2),
    "OR": (1, 3), "NV": (2, 3), "WY": (3, 3), "SD": (4, 3), "IA": (5, 3),
    "IN": (6, 3), "OH": (7, 3), "PA": (8, 3), "NJ": (9, 3), "CT": (10, 3), "RI": (11, 3),
    "CA": (1, 4), "UT": (2, 4), "CO": (3, 4), "NE": (4, 4), "MO": (5, 4),
    "KY": (6, 4), "WV": (7, 4), "VA": (8, 4), "MD": (9, 4), "DE": (10, 4),
    "AZ": (2, 5), "NM": (3, 5), "KS": (4, 5), "AR": (5, 5), "TN": (6, 5),
    "NC": (7, 5), "SC": (8, 5), "DC": (9, 5),
    "OK": (4, 6), "LA": (5, 6), "MS": (6, 6), "AL": (7, 6), "GA": (8, 6),
    "HI": (0, 7), "TX": (4, 7), "FL": (9, 7),
}


def fmt_dollars(usd: float, zero_benefit: bool = False) -> str:
    m = usd / 1e6
    prefix = "+" if zero_benefit else ""
    if abs(m) >= 1000:
        return f"{prefix}${m / 1000:.2f}B"
    return f"{prefix}${m:.0f}M"


def tile_style(rate: float, delayed: bool) -> tuple[str, str, str, float]:
    """Return (facecolor, textcolor, edgecolor, linewidth) preserving grayscale monotonicity."""
    if rate == 0:
        return PALETTE["tier0"], PALETTE["ink"], "#B8B8B8", 0.7
    if rate == 0.05:
        return PALETTE["tier5"], PALETTE["ink"], "#8C6D1F", 0.8
    if rate == 0.10:
        return PALETTE["tier10"], PALETTE["ink"], "#7D5400", 0.85
    if delayed:
        return PALETTE["tier15_delay"], "#FFFFFF", PALETTE["ink"], 1.5
    return PALETTE["tier15"], "#FFFFFF", "#7A2D00", 0.9


def plot_figure1(df: pd.DataFrame, save_stem: Path) -> None:
    fig = plt.figure(figsize=(12.2, 6.5), dpi=300, facecolor=PALETTE["paper"])
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.28, 1.0],
        height_ratios=[1.18, 1.0],
        left=0.02, right=0.985, top=0.92, bottom=0.085,
        wspace=0.22, hspace=0.38,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_cliff = fig.add_subplot(gs[1, 1])

    row_by_abbr = {r["abbr"]: r for _, r in df.iterrows()}

    # --- Panel a: Equal-area Tile Grid Map (50 states + D.C. + GU/VI inset) ---
    ax_map.set_facecolor(PALETTE["paper"])
    tile_w, tile_h = 0.90, 0.90
    pad_x, pad_y = (1.0 - tile_w) / 2, (1.0 - tile_h) / 2

    def draw_tile(col: float, row_y: float, abbr: str) -> None:
        r = row_by_abbr[abbr]
        fc, tc, ec, lw = tile_style(r["sec10105_rate_fy25"], bool(r["delayed_fy29_flag"]))
        y_bot = 7.0 - row_y  # flip row 0 to top
        rect = mpatches.FancyBboxPatch(
            (col + pad_x, y_bot + pad_y),
            tile_w, tile_h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            zorder=2,
        )
        ax_map.add_patch(rect)
        if bool(r["delayed_fy29_flag"]):
            # Subtle white corner notch + double dark outline for grayscale distinguishability
            tri = mpatches.Polygon(
                [
                    (col + pad_x + tile_w * 0.72, y_bot + pad_y + tile_h),
                    (col + pad_x + tile_w, y_bot + pad_y + tile_h),
                    (col + pad_x + tile_w, y_bot + pad_y + tile_h * 0.72),
                ],
                closed=True, facecolor="#FFFFFF", edgecolor="none", zorder=3,
            )
            ax_map.add_patch(tri)

        cx = col + 0.5
        abbr_lbl = f"{abbr}{'†' if r['delayed_fy29_flag'] else ''}"
        ax_map.text(cx, y_bot + 0.74, abbr_lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=tc, zorder=4)

        dol_str = fmt_dollars(r["combined_annual_exposure_usd"], zero_benefit=(r["sec10105_rate_fy25"] == 0))
        ax_map.text(cx, y_bot + 0.46, dol_str, ha="center", va="center", fontsize=7.0, fontweight="medium", color=tc, zorder=4, parse_math=False)

        sub_c = "#F2E2DA" if tc == "#FFFFFF" else PALETTE["muted"]
        ax_map.text(cx, y_bot + 0.21, f"{r['per_fy25']:.1f}%", ha="center", va="center", fontsize=7.0, color=sub_c, zorder=4)

    for abbr, (c, ry) in TILE_GRID.items():
        draw_tile(c, ry, abbr)

    # Territories inset (GU, VI) neatly separated from state HI (col 0) by a subtle vertical rule
    ax_map.plot([1.25, 1.25], [0.06, 0.94], color=PALETTE["context"], linewidth=0.8, zorder=2)
    draw_tile(1.55, 7.0, "GU")
    draw_tile(2.55, 7.0, "VI")
    ax_map.text(2.55, 1.07, "U.S. territories", ha="center", va="bottom", fontsize=7.0, color=PALETTE["muted"])

    ax_map.set_xlim(-0.1, 12.1)
    ax_map.set_ylim(-0.65, 8.25)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # Compact headline stats callout directly in empty Gulf / Mountain whitespace
    df51 = df[df["is_50_plus_dc"]]
    n_ge6_51 = int((df51["sec10105_rate_fy25"] >= 0.05).sum())
    tot_ben_b = df["sec10105_benefit_liability_usd"].sum() / 1e9
    tot_adm_b = df["sec10106_admin_shift_usd"].sum() / 1e9
    tot_all_b = df["combined_annual_exposure_usd"].sum() / 1e9
    share_ge6 = 100.0 * df.loc[df["per_fy25"] >= 6.0, "snap_benefits_fy25_usd"].sum() / df["snap_benefits_fy25_usd"].sum()

    ax_map.text(
        6.1, 7.82,
        f"{n_ge6_51} of 51 states+D.C. breach \u22656% PER ({share_ge6:.1f}% of U.S. SNAP benefits)\n"
        f"New annual state burden: ${tot_all_b:.2f}B/yr "
        f"(${tot_ben_b:.2f}B benefit + ${tot_adm_b:.2f}B admin)",
        ha="center", va="top", fontsize=8.0, color=PALETTE["ink"],
        linespacing=1.35, parse_math=False,
    )

    # Panel a label & subtitle
    ax_map.text(
        0.0, 1.04, "a", transform=ax_map.transAxes,
        fontsize=10.0, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom",
    )
    ax_map.text(
        0.032, 1.04,
        "State exposure under P.L. 119-21 (tile shows combined Sec. 10105 + Sec. 10106 cost & FY25 PER)",
        transform=ax_map.transAxes, fontsize=8.5, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom",
    )

    # Tier legend neatly below map
    legend_items = [
        ("0% ben. (<6%, admin only)", PALETTE["tier0"], "#B8B8B8", 0.7),
        ("5% (6\u2013<8%)", PALETTE["tier5"], "#8C6D1F", 0.8),
        ("10% (8\u2013<10%)", PALETTE["tier10"], "#7D5400", 0.85),
        ("15% (\u226510%, FY28)", PALETTE["tier15"], "#7A2D00", 0.9),
        ("\u2020 15% deferred FY29 (\u226513.33%)", PALETTE["tier15_delay"], PALETTE["ink"], 1.5),
    ]
    handles = [
        mpatches.Patch(facecolor=fc, edgecolor=ec, linewidth=lw, label=lbl)
        for lbl, fc, ec, lw in legend_items
    ]
    leg = ax_map.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.50, -0.055),
        ncol=3, frameon=False, fontsize=7.2, handlelength=1.1, handleheight=1.1, columnspacing=1.0,
    )
    for t in leg.get_texts():
        t.set_color(PALETTE["ink"])

    # --- Panel b: Top 12 Per-Capita Strain (Stacked Benefit + Admin) ---
    top12 = (
        df[df["is_50_plus_dc"]]
        .sort_values("combined_per_capita_usd", ascending=False)
        .head(12)
        .iloc[::-1]
        .copy()
    )
    y_pos = np.arange(len(top12))
    ben_pc = top12["sec10105_benefit_liability_usd"] / top12["pop_2025"]
    adm_pc = top12["sec10106_admin_shift_usd"] / top12["pop_2025"]
    tot_pc = ben_pc + adm_pc

    ax_bar.barh(y_pos, ben_pc, height=0.66, color=PALETTE["bar_benefit"], edgecolor="none", zorder=3, label="Sec. 10105 benefit share")
    ax_bar.barh(y_pos, adm_pc, left=ben_pc, height=0.66, color=PALETTE["admin_blue"], edgecolor="none", zorder=3, label="Sec. 10106 admin cut")

    us_pop_51 = df51["pop_2025"].sum()
    us_pc = df51["combined_annual_exposure_usd"].sum() / us_pop_51
    ax_bar.axvline(us_pc, color=PALETTE["muted"], linestyle="--", linewidth=1.0, zorder=2)
    ax_bar.text(
        us_pc + 1.5, len(top12) - 0.15,
        f"U.S. avg: ${us_pc:.0f}/res",
        fontsize=7.2, color=PALETTE["muted"], va="center", ha="left", parse_math=False,
    )

    lbls = [
        f"{r['state'].replace('District of Columbia', 'D.C.')}{' \u2020' if r['delayed_fy29_flag'] else ''}"
        for _, r in top12.iterrows()
    ]
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(lbls, fontsize=7.5, color=PALETTE["ink"])
    for idx_y, (_, r) in enumerate(top12.iterrows()):
        dol = fmt_dollars(r["combined_annual_exposure_usd"])
        ax_bar.text(
            tot_pc.iloc[idx_y] + 1.5, idx_y,
            f"${tot_pc.iloc[idx_y]:.0f} ({dol})",
            va="center", ha="left", fontsize=7.2, color=PALETTE["ink"], parse_math=False,
        )

    ax_bar.set_xlim(0, 134)
    ax_bar.set_ylim(-0.65, len(top12) + 0.35)
    ax_bar.set_xlabel("Combined annual state cost ($ per resident)", fontsize=8.0, color=PALETTE["ink"], parse_math=False)
    ax_bar.tick_params(axis="x", labelsize=7.5, colors=PALETTE["muted"])
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color(PALETTE["context"])
    ax_bar.spines["bottom"].set_color(PALETTE["context"])
    ax_bar.xaxis.grid(True, color=PALETTE["rule"], linewidth=0.5, zorder=1)
    ax_bar.set_axisbelow(True)

    ax_bar.legend(loc="lower right", frameon=False, fontsize=7.2, bbox_to_anchor=(1.0, 0.02))
    ax_bar.text(
        -0.25, 1.07, "b", transform=ax_bar.transAxes,
        fontsize=10.0, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom",
    )
    ax_bar.text(
        -0.19, 1.07, "Top 12 per-resident burden (total $ in parentheses)",
        transform=ax_bar.transAxes, fontsize=8.5, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom", parse_math=False,
    )

    # --- Panel c: FY2024 -> FY2025 Cliff Movers ---
    cliff_abbrs = ["AK", "DE", "IL", "NY", "FL", "MA", "MN", "HI", "NC", "NJ", "KY"]
    cdf = df.set_index("abbr").loc[cliff_abbrs].iloc[::-1].copy()
    cy = np.arange(len(cdf))

    # Tier shaded regions across horizontal PER axis (cover full xlim cleanly)
    ax_cliff.axvspan(3.5, 6.0, color=PALETTE["tier0"], alpha=0.85, zorder=0)
    ax_cliff.axvspan(6.0, 8.0, color=PALETTE["tier5"], alpha=0.28, zorder=0)
    ax_cliff.axvspan(8.0, 10.0, color=PALETTE["tier10"], alpha=0.18, zorder=0)
    ax_cliff.axvspan(10.0, 13.333, color=PALETTE["tier15"], alpha=0.14, zorder=0)
    ax_cliff.axvspan(13.333, 30.0, color=PALETTE["tier15_delay"], alpha=0.12, zorder=0)

    for thresh, tlab in [(6.0, "6%"), (8.0, "8%"), (10.0, "10%"), (13.333, "13.3%\u2020")]:
        ax_cliff.axvline(thresh, color="#7A7A7A", linestyle=":" if thresh == 13.333 else "-", linewidth=0.75, zorder=1)
        ax_cliff.text(thresh, len(cdf) - 0.15, tlab, ha="center", va="bottom", fontsize=7.0, color=PALETTE["muted"])

    for idx_c, (abbr_c, rc) in enumerate(cdf.iterrows()):
        p24, p25 = rc["per_fy24"], rc["per_fy25"]
        rising = p25 > p24
        col = PALETTE["up_vermilion"] if rising else PALETTE["down_blue"]
        ax_cliff.plot([p24, p25], [idx_c, idx_c], color=col, linewidth=1.6, zorder=3)
        ax_cliff.scatter([p24], [idx_c], s=26, facecolor="#FFFFFF", edgecolor=col, linewidth=1.2, zorder=4)
        ax_cliff.scatter([p25], [idx_c], s=32, facecolor=col, edgecolor=col, linewidth=0.8, zorder=5)

        hi_x = max(p24, p25)
        ax_cliff.text(
            hi_x + 0.45, idx_c,
            f"{p24:.1f} -> {p25:.1f}%",
            va="center", ha="left", fontsize=7.0, color=col,
        )

    ax_cliff.set_yticks(cy)
    ax_cliff.set_yticklabels([f"{a} ({cdf.loc[a, 'sec10105_rate_fy25']*100:.0f}%)" for a in cdf.index], fontsize=7.5, color=PALETTE["ink"])
    ax_cliff.set_xlim(3.5, 30.0)
    ax_cliff.set_ylim(-0.6, len(cdf) + 0.50)
    ax_cliff.set_xlabel("Official USDA Payment Error Rate (%, hollow = FY24, filled = FY25)", fontsize=8.0, color=PALETTE["ink"])
    ax_cliff.tick_params(axis="x", labelsize=7.5, colors=PALETTE["muted"])
    ax_cliff.tick_params(axis="y", length=0)
    ax_cliff.spines["top"].set_visible(False)
    ax_cliff.spines["right"].set_visible(False)
    ax_cliff.spines["left"].set_color(PALETTE["context"])
    ax_cliff.spines["bottom"].set_color(PALETTE["context"])

    ax_cliff.text(
        -0.25, 1.07, "c", transform=ax_cliff.transAxes,
        fontsize=10.0, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom",
    )
    ax_cliff.text(
        -0.19, 1.07, "FY2024 to FY2025 error-rate shifts across statutory cliffs",
        transform=ax_cliff.transAxes, fontsize=8.5, fontweight="bold", color=PALETTE["ink"], ha="left", va="bottom",
    )

    fig.savefig(f"{save_stem}.svg", format="svg", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(f"{save_stem}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


plot_figure1(df, FIG_DIR / "hr1-policy-stakes-map")
print("Saved Figure 1:", FIG_DIR / "hr1-policy-stakes-map.svg", FIG_DIR / "hr1-policy-stakes-map.png")

# %% [markdown]
# ## 4. Figure 2: Budget-Share Stress & 3-Year Stability Check (`hr1-budget-strain-rankings`)
#
# Pairs (a) **Combined Cost as % of FY2024 State General Expenditure** (measuring direct state-budget squeeze across states with different tax/spending bases) against (b) **Top-12 Absolute Dollar Liability ($B/yr)** comparing the primary FY2025 PER baseline against the 3-year (`FY2023–FY2025`) average PER sensitivity check.

# %%
def plot_figure2(df: pd.DataFrame, save_stem: Path) -> None:
    df51 = df[df["is_50_plus_dc"]].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300, facecolor=PALETTE["paper"])
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.14, wspace=0.40)

    # Panel a: % of State General Expenditure
    top_pct = df51.sort_values("combined_pct_gen_exp", ascending=False).head(12).iloc[::-1]
    y1 = np.arange(len(top_pct))
    ben_pct = top_pct["benefit_liab_pct_gen_exp"]
    adm_pct = top_pct["combined_pct_gen_exp"] - ben_pct

    ax1.barh(y1, ben_pct, height=0.65, color=PALETTE["bar_benefit"], label="Sec. 10105 benefit share")
    ax1.barh(y1, adm_pct, left=ben_pct, height=0.65, color=PALETTE["admin_blue"], label="Sec. 10106 admin cut")
    ax1.set_yticks(y1)
    ax1.set_yticklabels([f"{r['state']}{' \u2020' if r['delayed_fy29_flag'] else ''}" for _, r in top_pct.iterrows()], fontsize=8.0, color=PALETTE["ink"])
    for idx_i, (_, r) in enumerate(top_pct.iterrows()):
        ax1.text(r["combined_pct_gen_exp"] + 0.015, idx_i, f"{r['combined_pct_gen_exp']:.2f}%", va="center", fontsize=7.5, color=PALETTE["ink"])

    ax1.set_xlim(0, 1.24)
    ax1.set_xlabel("Combined cost as % of FY2024 State General Expenditure", fontsize=8.0, color=PALETTE["ink"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.xaxis.grid(True, color=PALETTE["rule"], linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="both", labelsize=7.5)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.02, 0.04), frameon=False, fontsize=7.5)
    ax1.text(-0.22, 1.06, "a", transform=ax1.transAxes, fontsize=10.0, fontweight="bold", color=PALETTE["ink"])
    ax1.text(-0.14, 1.06, "Budget squeeze (% of state general expenditure)", transform=ax1.transAxes, fontsize=8.5, fontweight="bold", color=PALETTE["ink"])

    # Panel b: Top 12 Absolute Dollars ($M): FY25 PER vs 3-yr Mean Sensitivity (+ Admin)
    top_abs = df51.sort_values("combined_annual_exposure_usd", ascending=False).head(12).iloc[::-1]
    y2 = np.arange(len(top_abs))
    h = 0.34
    ben25_m = top_abs["sec10105_benefit_liability_usd"] / 1e6
    ben3_m = top_abs["sec10105_benefit_liability_3yr_usd"] / 1e6
    adm_m = top_abs["sec10106_admin_shift_usd"] / 1e6

    ax2.barh(y2 + h / 2, ben25_m, height=h, color=PALETTE["bar_benefit"], label="Benefit share (FY25 PER)")
    ax2.barh(y2 + h / 2, adm_m, left=ben25_m, height=h, color=PALETTE["admin_blue"], label="Admin cut (Sec. 10106)")
    ax2.barh(y2 - h / 2, ben3_m, height=h, color=PALETTE["ref_gray"], label="Benefit share (FY23\u201325 mean PER)")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([f"{r['state']}{' \u2020' if r['delayed_fy29_flag'] else ''}" for _, r in top_abs.iterrows()], fontsize=8.0, color=PALETTE["ink"])
    for idx_i, (_, r) in enumerate(top_abs.iterrows()):
        v = r["combined_annual_exposure_usd"] / 1e6
        lbl = f"${v / 1000:.2f}B" if v >= 1000 else f"${v:.0f}M"
        ax2.text(max(v, r["sec10105_benefit_liability_3yr_usd"] / 1e6) + 55, idx_i, lbl, va="center", fontsize=7.5, color=PALETTE["ink"], parse_math=False)

    ax2.set_xlim(0, 3050)
    ax2.set_xlabel("Annual unbudgeted state cost ($ millions)", fontsize=8.0, color=PALETTE["ink"], parse_math=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.xaxis.grid(True, color=PALETTE["rule"], linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="both", labelsize=7.5)
    ax2.legend(loc="lower right", frameon=False, fontsize=7.2)
    ax2.text(-0.22, 1.06, "b", transform=ax2.transAxes, fontsize=10.0, fontweight="bold", color=PALETTE["ink"])
    ax2.text(-0.14, 1.06, "Largest dollar exposures & 3-yr PER sensitivity", transform=ax2.transAxes, fontsize=8.5, fontweight="bold", color=PALETTE["ink"])

    fig.savefig(f"{save_stem}.svg", format="svg", bbox_inches="tight", pad_inches=0.10)
    fig.savefig(f"{save_stem}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


plot_figure2(df, FIG_DIR / "hr1-budget-strain-rankings")
print("Saved Figure 2:", FIG_DIR / "hr1-budget-strain-rankings.svg", FIG_DIR / "hr1-budget-strain-rankings.png")
print("\nTop 10 states in processed CSV:")
df[["abbr", "state", "per_fy24", "per_fy25", "sec10105_rate_fy25", "delayed_fy29_flag", "sec10105_benefit_liability_usd", "sec10106_admin_shift_usd", "combined_annual_exposure_usd", "combined_per_capita_usd", "combined_pct_gen_exp"]].head(12)
