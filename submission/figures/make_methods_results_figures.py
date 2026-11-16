"""Figure 2: how often each reader flags the income group in question, by what the file shows.

Every plotted value is recomputed from the logged reader calls (repeat 1, the primary repeat):
  experiments/design1_v2/results.jsonl            three readers, 264 files
  experiments/design1_v2_amount/amount-results.jsonl         omission-sized credit, ambiguous label
  experiments/design1_v2_amount_clear/clear-results.jsonl    omission-sized credit, clear label

Run from the repository root:
    .venv/bin/python submission/figures/make_methods_results_figures.py
Writes fig2-reader-flag-rates.png and fig2-reader-flag-rates.svg next to this script and refuses to
overwrite an existing file unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = {"earned", "unearned", "shelter", "utilities"}

# ---------------------------------------------------------------- data

def valid_output(rec: dict) -> bool:
    """One report_flags call that passes the tool schema, names only the four groups, each at most once."""
    if not rec.get("valid"):
        return False
    flags = (rec.get("parsed") or {}).get("flags")
    if not isinstance(flags, list):
        return False
    groups = [f.get("group") for f in flags if isinstance(f, dict)]
    return len(groups) == len(flags) and all(g in GROUPS for g in groups) and len(set(groups)) == len(groups)


def load(path: str) -> list[dict]:
    rows = []
    for line in (ROOT / path).read_text().splitlines():
        r = json.loads(line)
        if r["repeat"] != 1:
            continue
        ok = valid_output(r)
        flagged = {f["group"] for f in r["parsed"]["flags"]} if ok else set()
        rows.append({"reader": r["reader"], "cls": r["cls"], "state": r["evidence_state"], "valid": ok,
                     "hit": r["target_group"] in flagged})
    return rows


RECORDS = (load("experiments/design1_v2/results.jsonl")
           + load("experiments/design1_v2_amount/amount-results.jsonl")
           + load("experiments/design1_v2_amount_clear/clear-results.jsonl"))


def cell(readers: list[str], cls: str, state: str) -> tuple[int, int]:
    """Flags on the target group (k) among valid outputs (n), pooled over the given readers."""
    sel = [r for r in RECORDS if r["reader"] in readers and r["cls"] == cls and r["state"] == state and r["valid"]]
    return sum(r["hit"] for r in sel), len(sel)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


MAIN = ["deepseek-v4-flash", "qwen3.5-397b", "gemma-4-31b-it"]

PANEL_A = [  # omitted income: flag on the omitted group
    ("No evidence in the file", "omission", "none"),
    ("Ambiguous deposits", "omission", "deposits_ambiguous"),
    ("Clear deposits", "omission", "deposits_clear"),
    ("Lead: cross-check line", "omission", "lead"),
]
PANEL_B = [  # correct households: flag on the group in question
    ("Documented, income reported", "truthful_documented", "documented"),
    ("True zero, rent only", "zero_rent", "none"),
    ("True zero, small credit,\nclear label", "zero_stray", "stray_clear"),
    ("True zero, small credit,\nambiguous label", "zero_stray", "stray_ambiguous"),
    ("True zero, omission-sized credit,\nclear label (exploratory)", "zero_stray", "stray_clear_amount"),
    ("True zero, omission-sized credit,\nambiguous label (exploratory)", "zero_stray", "stray_amount"),
]

# Okabe-Ito colours, with a different marker shape per reader so colour is never the only cue.
STYLE = {
    "pooled": ("#000000", "D", "Pooled, three readers"),
    "deepseek-v4-flash": ("#0072B2", "o", "deepseek-v4-flash"),
    "qwen3.5-397b": ("#E69F00", "s", "qwen3.5-397b"),
    "gemma-4-31b-it": ("#009E73", "^", "gemma-4-31b-it"),
}

# ---------------------------------------------------------------- drawing

def series_for(panel_readers: list[str], pooled: bool) -> list[str]:
    return (["pooled"] if pooled else []) + panel_readers


def draw_panel(ax, rows, readers, pooled, title, xlabel=None, counts="pooled"):
    series = series_for(readers, pooled)
    m = len(series)
    step = 0.17
    offsets = [-(m - 1) / 2 * step + i * step for i in range(m)]
    n_rows = len(rows)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(-2.5, 102.5)
    values = {}
    for yi, (label, cls, state) in enumerate(rows):
        ax.axhline(yi + 0.5, color="#dddddd", lw=0.6, zorder=0) if yi < n_rows - 1 else None
        for off, s in zip(offsets, series):
            who = MAIN if s == "pooled" else [s]
            k, n = cell(who, cls, state)
            lo, hi = wilson(k, n)
            colour, marker, _ = STYLE[s]
            size = 5.2 if s == "pooled" else 3.8
            ax.plot([lo * 100, hi * 100], [yi + off, yi + off], color=colour, lw=1.3 if s == "pooled" else 1.0, zorder=2,
                    solid_capstyle="butt")
            ax.plot([k / n * 100], [yi + off], marker=marker, ms=size, color=colour, mec="white" if s != "pooled" else colour,
                    mew=0.4, zorder=3, ls="none", clip_on=False)
            if counts == "all" or s == "pooled":
                ax.text(1.03, yi + off, f"{k} of {n}", transform=ax.get_yaxis_transform(), color=colour, fontsize=7.4,
                        va="center", ha="left", fontweight="bold" if s == "pooled" else "normal")
            values[(label, s)] = (k, n, lo, hi)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.6, length=2.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, loc="left", x=-0.505, fontsize=8.4, fontweight="bold", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.8)
    return values


def main(force: bool) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 6.0], hspace=0.55, left=0.30, right=0.87, top=0.80, bottom=0.095)
    axA, axB = (fig.add_subplot(gs[i]) for i in range(2))
    va = draw_panel(axA, PANEL_A, MAIN, True, "A. Files that omit income: share flagged on the omitted group (higher is better)")
    vb = draw_panel(axB, PANEL_B, MAIN, True, "B. Files that report income correctly: share flagged on the group in question (lower is better)",
                    xlabel="Share of files flagged, with Wilson 95% interval")
    for ax in (axA, axB):
        ax.tick_params(axis="x", labelbottom=True)

    fig.suptitle("Readers flag omitted income when the file shows it, and flag correct households\n"
                 "when a credit under an ambiguous label looks like income", x=0.02, y=0.992, ha="left", fontsize=10.2,
                 fontweight="bold", linespacing=1.25)
    handles = [Line2D([0], [0], marker=STYLE[s][1], color=STYLE[s][0], lw=1.2, ms=5, mec="white" if s != "pooled" else STYLE[s][0],
                      label=STYLE[s][2]) for s in ["pooled"] + MAIN]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.905), ncol=4, frameon=False, fontsize=7.8,
               handletextpad=0.4, columnspacing=1.2, title="Reader", title_fontsize=7.8, alignment="left")
    for ext in ("png", "svg"):
        out = HERE / f"fig2-reader-flag-rates.{ext}"
        if out.exists() and not force:
            raise SystemExit(f"{out.name} exists; pass --force to overwrite")
        fig.savefig(out, dpi=300 if ext == "png" else None, facecolor="white")
    # Print what was plotted so the numbers can be checked against the prose.
    for name, vals in (("A", va), ("B", vb)):
        for (label, s), (k, n, lo, hi) in vals.items():
            print(f"{name}\t{label.replace(chr(10), ' ')}\t{s}\t{k}/{n}\t{lo:.3f}-{hi:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
