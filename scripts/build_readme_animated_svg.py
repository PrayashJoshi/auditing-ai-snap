#!/usr/bin/env python3
"""Generate a self-contained, pure-SVG animated teaser for README.md
from submission/figures/fig1-paper-teaser.svg without modifying the paper source.

GitHub READMEs strip <script> tags, so this produces an <img>-safe SVG
using only embedded CSS @keyframes and SVG <clipPath>/<mask> (no JS,
no <foreignObject>, no external network requests), with full
prefers-reduced-motion fallback to the exact static paper figure.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_SVG = ROOT / "submission" / "figures" / "fig1-paper-teaser.svg"
OUT_SVG = ROOT / "assets" / "fig1-teaser-animated.svg"
CYCLE_SECONDS = 20


def extract_latin_fonts(raw_svg: str) -> str:
    """Keep only the /* latin */ @font-face blocks from the source SVG.

    Every glyph in fig1-paper-teaser.svg (ASCII plus curly quotes U+201C/201D)
    lives in the latin unicode-range, shrinking the standalone README SVG ~4.5x
    while preserving exact Source Sans 3 typography offline and inside <img>.
    """
    blocks = re.findall(r"/\* latin \*/\s*@font-face\s*\{[^}]+\}", raw_svg)
    if len(blocks) != 3:
        raise RuntimeError(f"Expected 3 latin @font-face blocks, found {len(blocks)}")
    return "\n".join(blocks)


def build_animated_svg(raw_svg: str) -> str:
    latin_fonts = extract_latin_fonts(raw_svg)

    css = f"""<style>
{latin_fonts}

:root {{
  --color-paper: #FFFFFF;
  --color-ink: #1F1F1F;
  --color-muted: #6B6B6B;
  --color-context: #BDBDBD;
  --color-rule: #E3E3E3;
  --color-actor: #0072B2;
  --color-cue: #D55E00;
  --font-sans: "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
  --text-label: 8px;
  --text-emph: 9px;
}}

text {{
  font-family: var(--font-sans);
  font-size: var(--text-label);
  fill: var(--color-ink);
}}
.m {{ fill: var(--color-muted); }}
.b {{ font-weight: 600; font-size: var(--text-emph); }}
.i {{ font-style: italic; fill: var(--color-muted); }}
.c {{ fill: var(--color-cue); }}
.a {{ stroke: var(--color-ink); stroke-width: 0.8; fill: none; }}
.act {{ stroke: var(--color-actor); stroke-width: 1.2; fill: none; }}

/* ================================================================
   {CYCLE_SECONDS}s synchronized narrative loop with clean pauses between sections:
     0-5%    Complete static paper figure (initial glance + thumbnail safe)
     5-8%    Soft dim of inactive sections as walk-through starts
     8-27%   Panel a (Filer): QC -> Bank deposit -> AI filer -> Portal -> Rejection loop
    27-30.5% Pause 1 (0.7s): hold on completed rejection loop
    30.5-44% Panel a (Reader): Accepted -> Case file -> AI reader -> Flag -> Scored + Fact ledger
    44-48.5% Pause 2 (0.9s): hold on completed Panel a before moving to Panel b
    48.5-67% Panel b: Number required (0/72) -> pause -> "Unknown" allowed (72/72) -> pause -> After rejection (13/24)
    67-71.5% Pause 3 (0.9s): hold on completed Panel b before moving to Panel c
    71.5-83.5% Panel c: Income omitted vs No income + "same deposit" diagonal bridge
    85-100%  Hold on exact static paper figure (4.0s continuous hold across seam)
   ================================================================ */

.stage-db      {{ animation: st-db      {CYCLE_SECONDS}s linear infinite; }}
.stage-packet  {{ animation: st-packet  {CYCLE_SECONDS}s linear infinite; }}
.stage-filer   {{ animation: st-filer   {CYCLE_SECONDS}s linear infinite; }}
.stage-reject  {{ animation: st-reject  {CYCLE_SECONDS}s linear infinite; }}
.stage-accept  {{ animation: st-accept  {CYCLE_SECONDS}s linear infinite; }}
.stage-reader  {{ animation: st-reader  {CYCLE_SECONDS}s linear infinite; }}
.stage-scored  {{ animation: st-scored  {CYCLE_SECONDS}s linear infinite; }}

@keyframes st-db {{
  0%, 5%     {{ opacity: 1; }}
  8%, 12.5%  {{ opacity: 1; }}
  15%, 47%   {{ opacity: 0.78; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-packet {{
  0%, 5%     {{ opacity: 1; }}
  8%         {{ opacity: 0.45; }}
  10.5%, 27% {{ opacity: 1; }}
  30.5%, 47% {{ opacity: 0.78; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-filer {{
  0%, 5%     {{ opacity: 1; }}
  8%, 12.5%  {{ opacity: 0.4; }}
  15%, 30%   {{ opacity: 1; }}
  32%, 47%   {{ opacity: 0.78; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-reject {{
  0%, 5%     {{ opacity: 1; }}
  8%, 20%    {{ opacity: 0.22; }}
  22%, 30%   {{ opacity: 1; }}
  32%, 47%   {{ opacity: 0.78; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-accept {{
  0%, 5%     {{ opacity: 1; }}
  8%, 29.5%  {{ opacity: 0.28; }}
  31%, 36%   {{ opacity: 1; }}
  38%, 47%   {{ opacity: 0.82; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-reader {{
  0%, 5%     {{ opacity: 1; }}
  8%, 33.5%  {{ opacity: 0.28; }}
  35%, 41%   {{ opacity: 1; }}
  43%, 47%   {{ opacity: 0.85; }}
  48.5%, 83% {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes st-scored {{
  0%, 5%     {{ opacity: 1; }}
  8%, 38.5%  {{ opacity: 0.25; }}
  40%, 47%   {{ opacity: 1; }}
  49%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}

/* --- Panel a micro-highlights and eye-candy --- */
.bank-bg {{
  animation: bank-glow {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes bank-glow {{
  0%, 5%    {{ opacity: 0.14; }}
  8%, 10%   {{ opacity: 0.08; }}
  11.5%, 16% {{ opacity: 0.30; }}
  18%, 100% {{ opacity: 0.14; }}
}}

.app-field {{
  animation: field-pulse {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes field-pulse {{
  0%, 5%    {{ stroke-width: 1.2; opacity: 1; }}
  8%, 15.5% {{ stroke-width: 1.2; opacity: 0.45; }}
  17%, 22%  {{ stroke-width: 1.9; opacity: 1; }}
  25%, 100% {{ stroke-width: 1.2; opacity: 1; }}
}}

.portal-x1 {{
  stroke-dasharray: 18;
  animation: draw-x1 {CYCLE_SECONDS}s ease-out infinite;
}}
.portal-x2 {{
  stroke-dasharray: 18;
  animation: draw-x2 {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes draw-x1 {{
  0%, 5%      {{ stroke-dashoffset: 0; opacity: 1; }}
  8%, 19%     {{ stroke-dashoffset: 18; opacity: 0.15; }}
  21%, 100%   {{ stroke-dashoffset: 0; opacity: 1; }}
}}
@keyframes draw-x2 {{
  0%, 5%      {{ stroke-dashoffset: 0; opacity: 1; }}
  8%, 20%     {{ stroke-dashoffset: 18; opacity: 0.15; }}
  22%, 100%   {{ stroke-dashoffset: 0; opacity: 1; }}
}}

.reject-mask-line {{
  stroke-dasharray: 180;
  animation: draw-reject-arc {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes draw-reject-arc {{
  0%, 5%     {{ stroke-dashoffset: 0; }}
  8%, 21%    {{ stroke-dashoffset: 180; }}
  25.5%, 100% {{ stroke-dashoffset: 0; }}
}}

.ledger-mask-line {{
  stroke-dasharray: 32;
  animation: draw-ledger {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes draw-ledger {{
  0%, 5%     {{ stroke-dashoffset: 0; }}
  8%, 39.5%  {{ stroke-dashoffset: 32; }}
  42.5%, 100% {{ stroke-dashoffset: 0; }}
}}

.check-draw {{
  stroke-dasharray: 18;
  animation: draw-check {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes draw-check {{
  0%, 5%     {{ stroke-dashoffset: 0; }}
  8%, 41.5%  {{ stroke-dashoffset: 18; }}
  44%, 100%  {{ stroke-dashoffset: 0; }}
}}

/* AI filer blink (explicit SVG coords avoid transform-box quirks) */
.filer-eye-l {{
  transform-origin: 210px 73px;
  animation: filer-blink {CYCLE_SECONDS}s ease-in-out infinite;
}}
.filer-eye-r {{
  transform-origin: 222px 73px;
  animation: filer-blink {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes filer-blink {{
  0%, 13.5%, 15.5%, 25.5%, 27.5%, 100% {{ transform: scaleY(1); }}
  14.5%, 26.5% {{ transform: scaleY(0.15); }}
}}

.filer-antenna {{
  transform-origin: 216px 51.5px;
  animation: antenna-ping {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes antenna-ping {{
  0%, 13.5%, 27.5%, 100% {{ transform: scale(1); }}
  16%, 25% {{ transform: scale(1.45); }}
}}

/* AI reader subtle iris glance */
.reader-iris {{
  animation: reader-scan {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes reader-scan {{
  0%, 36.5%   {{ transform: translateX(0px); }}
  38%         {{ transform: translateX(1.8px); }}
  39.5%       {{ transform: translateX(-2px); }}
  41%, 100%   {{ transform: translateX(0px); }}
}}

/* Case file orange evidence bar pulse */
.case-mark {{
  animation: case-pulse {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes case-pulse {{
  0%, 5%     {{ stroke-width: 1.4; opacity: 1; }}
  8%, 34%    {{ stroke-width: 1.4; opacity: 0.45; }}
  35%, 37.5% {{ stroke-width: 2.2; opacity: 1; }}
  39%, 100%  {{ stroke-width: 1.4; opacity: 1; }}
}}

/* --- Traveling signal tokens along pipeline edges --- */
.pkt {{
  opacity: 0;
  offset-rotate: 0deg;
}}
.pkt-qc      {{ offset-path: path("M51 72L88 72"); animation: pkt-qc {CYCLE_SECONDS}s linear infinite; }}
.pkt-deposit {{ offset-path: path("M170 68C186 68 190 72 199 73"); animation: pkt-deposit {CYCLE_SECONDS}s ease-in-out infinite; }}
.pkt-app     {{ offset-path: path("M236 73L256 73"); animation: pkt-app {CYCLE_SECONDS}s linear infinite; }}
.pkt-portal  {{ offset-path: path("M300 73L320 73"); animation: pkt-portal {CYCLE_SECONDS}s linear infinite; }}
.pkt-reject  {{ offset-path: path("M347 113C347 142 216 142 216 110"); animation: pkt-reject {CYCLE_SECONDS}s ease-in-out infinite; }}
.pkt-accept  {{ offset-path: path("M374 73H450V196H426"); animation: pkt-accept {CYCLE_SECONDS}s ease-in-out infinite; }}
.pkt-read1   {{ offset-path: path("M368 197L346 197"); animation: pkt-read1 {CYCLE_SECONDS}s linear infinite; }}
.pkt-read2   {{ offset-path: path("M294 197L272 197"); animation: pkt-read2 {CYCLE_SECONDS}s linear infinite; }}
.pkt-read3   {{ offset-path: path("M248 197L176 197"); animation: pkt-read3 {CYCLE_SECONDS}s linear infinite; }}
.pkt-truth   {{ offset-path: path("M161 157L161 183"); animation: pkt-truth {CYCLE_SECONDS}s linear infinite; }}

@keyframes pkt-qc {{
  0%, 7.5%  {{ offset-distance: 0%; opacity: 0; }}
  8.2%      {{ opacity: 1; }}
  11%       {{ offset-distance: 100%; opacity: 1; }}
  11.8%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-deposit {{
  0%, 11%   {{ offset-distance: 0%; opacity: 0; }}
  11.7%     {{ opacity: 1; }}
  14.5%     {{ offset-distance: 100%; opacity: 1; }}
  15.2%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-app {{
  0%, 14.5% {{ offset-distance: 0%; opacity: 0; }}
  15%       {{ opacity: 1; }}
  17%       {{ offset-distance: 100%; opacity: 1; }}
  17.6%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-portal {{
  0%, 17%   {{ offset-distance: 0%; opacity: 0; }}
  17.5%     {{ opacity: 1; }}
  19.5%     {{ offset-distance: 100%; opacity: 1; }}
  20.1%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-reject {{
  0%, 21.5% {{ offset-distance: 0%; opacity: 0; }}
  22.2%     {{ opacity: 1; }}
  26%       {{ offset-distance: 100%; opacity: 1; }}
  26.8%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-accept {{
  0%, 30.5% {{ offset-distance: 0%; opacity: 0; }}
  31.2%     {{ opacity: 1; }}
  34.8%     {{ offset-distance: 100%; opacity: 1; }}
  35.5%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-read1 {{
  0%, 35.8% {{ offset-distance: 0%; opacity: 0; }}
  36.3%     {{ opacity: 1; }}
  37.8%     {{ offset-distance: 100%; opacity: 1; }}
  38.4%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-read2 {{
  0%, 38.4% {{ offset-distance: 0%; opacity: 0; }}
  38.9%     {{ opacity: 1; }}
  40.2%     {{ offset-distance: 100%; opacity: 1; }}
  40.8%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-read3 {{
  0%, 40.5% {{ offset-distance: 0%; opacity: 0; }}
  41%       {{ opacity: 1; }}
  43%       {{ offset-distance: 100%; opacity: 1; }}
  43.7%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}
@keyframes pkt-truth {{
  0%, 40.5% {{ offset-distance: 0%; opacity: 0; }}
  41%       {{ opacity: 1; }}
  43%       {{ offset-distance: 100%; opacity: 1; }}
  43.7%, 100% {{ offset-distance: 100%; opacity: 0; }}
}}

/* --- Panel b: Waffle sweeps and Rule check contrast (0/72 vs 72/72) --- */
.b1-wrap {{ animation: b1-vis {CYCLE_SECONDS}s linear infinite; }}
.b2-wrap {{ animation: b2-vis {CYCLE_SECONDS}s linear infinite; }}
.b3-wrap {{ animation: b3-vis {CYCLE_SECONDS}s linear infinite; }}
.b1-grid {{ animation: b1-grid-vis {CYCLE_SECONDS}s linear infinite; }}
.b2-grid {{ animation: b2-grid-vis {CYCLE_SECONDS}s linear infinite; }}
.b3-grid {{ animation: b3-grid-vis {CYCLE_SECONDS}s linear infinite; }}
.b1-flag {{ animation: b1-flag-vis {CYCLE_SECONDS}s ease-out infinite; }}
.b2-flag {{ transform-origin: 207px 329px; animation: b2-flag-vis {CYCLE_SECONDS}s ease-out infinite; }}
.b-foot  {{ animation: b-foot-vis {CYCLE_SECONDS}s linear infinite; }}

.wipe-b1 {{ animation: wipe-b1 {CYCLE_SECONDS}s ease-out infinite; }}
.wipe-b2 {{ animation: wipe-b2 {CYCLE_SECONDS}s ease-out infinite; }}
.wipe-b3 {{ animation: wipe-b3 {CYCLE_SECONDS}s ease-out infinite; }}

@keyframes b1-vis {{
  0%, 5%     {{ opacity: 1; }}
  8%, 47.5%  {{ opacity: 0.22; }}
  48.8%, 70% {{ opacity: 1; }}
  72%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes b1-grid-vis {{
  0%, 46.8%  {{ opacity: 1; }}
  47.5%, 48.2% {{ opacity: 0; }}
  48.5%, 100% {{ opacity: 1; }}
}}
@keyframes wipe-b1 {{
  0%, 47.5%  {{ transform: translateX(0px); }}
  48.1%      {{ transform: translateX(-112px); }}
  52%, 100%  {{ transform: translateX(0px); }}
}}
@keyframes b1-flag-vis {{
  0%, 5%     {{ opacity: 1; }}
  8%, 51.5%  {{ opacity: 0.22; }}
  53.5%, 70% {{ opacity: 1; }}
  72%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}

@keyframes b2-vis {{
  0%, 5%     {{ opacity: 1; }}
  8%, 54%    {{ opacity: 0.22; }}
  55.3%, 70% {{ opacity: 1; }}
  72%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes b2-grid-vis {{
  0%, 53.3%  {{ opacity: 1; }}
  54%, 54.7% {{ opacity: 0; }}
  55%, 100%  {{ opacity: 1; }}
}}
@keyframes wipe-b2 {{
  0%, 54%    {{ transform: translateX(0px); }}
  54.6%      {{ transform: translateX(-112px); }}
  58.5%, 100% {{ transform: translateX(0px); }}
}}
@keyframes b2-flag-vis {{
  0%, 5%     {{ opacity: 1; transform: scale(1); }}
  8%, 58%    {{ opacity: 0.2; transform: scale(0.92); }}
  59.5%      {{ opacity: 1; transform: scale(1.12); }}
  60.5%, 70% {{ opacity: 1; transform: scale(1); }}
  72%, 83%   {{ opacity: 0.72; transform: scale(1); }}
  85%, 100%  {{ opacity: 1; transform: scale(1); }}
}}

@keyframes b3-vis {{
  0%, 5%     {{ opacity: 1; }}
  8%, 61%    {{ opacity: 0.22; }}
  62.3%, 70% {{ opacity: 1; }}
  72%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}
@keyframes b3-grid-vis {{
  0%, 60.3%  {{ opacity: 1; }}
  61%, 61.7% {{ opacity: 0; }}
  62%, 100%  {{ opacity: 1; }}
}}
@keyframes wipe-b3 {{
  0%, 61%    {{ transform: translateX(0px); }}
  61.6%      {{ transform: translateX(-112px); }}
  65.5%, 100% {{ transform: translateX(0px); }}
}}
@keyframes b-foot-vis {{
  0%, 5%     {{ opacity: 1; }}
  8%, 65%    {{ opacity: 0.32; }}
  67%, 70%   {{ opacity: 1; }}
  72%, 83%   {{ opacity: 0.72; }}
  85%, 100%  {{ opacity: 1; }}
}}

/* --- Panel c: Tracks + "same deposit" bridge --- */
.c-track1 {{
  stroke-dasharray: 165;
  animation: draw-c1 {CYCLE_SECONDS}s ease-out infinite;
}}
.c-track2 {{
  stroke-dasharray: 82;
  animation: draw-c2 {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes draw-c1 {{
  0%, 5%     {{ stroke-dashoffset: 0; opacity: 1; }}
  8%, 71%    {{ stroke-dashoffset: 165; opacity: 0.22; }}
  74.5%, 100% {{ stroke-dashoffset: 0; opacity: 1; }}
}}
@keyframes draw-c2 {{
  0%, 5%     {{ stroke-dashoffset: 0; opacity: 1; }}
  8%, 74%    {{ stroke-dashoffset: 82; opacity: 0.22; }}
  77%, 100%  {{ stroke-dashoffset: 0; opacity: 1; }}
}}

.c-d1 {{ animation: c-d1 {CYCLE_SECONDS}s ease-out infinite; }}
.c-d2 {{ animation: c-d2 {CYCLE_SECONDS}s ease-out infinite; }}
.c-d3 {{ animation: c-d3 {CYCLE_SECONDS}s ease-out infinite; }}
.c-d4 {{ animation: c-d4 {CYCLE_SECONDS}s ease-out infinite; }}
.c-d5 {{ animation: c-d5 {CYCLE_SECONDS}s ease-out infinite; }}

@keyframes c-d1 {{ 0%, 5% {{ opacity: 1; }} 8%, 71.5% {{ opacity: 0.22; }} 73%, 100% {{ opacity: 1; }} }}
@keyframes c-d2 {{ 0%, 5% {{ opacity: 1; }} 8%, 73%   {{ opacity: 0.22; }} 74.5%, 100% {{ opacity: 1; }} }}
@keyframes c-d3 {{ 0%, 5% {{ opacity: 1; }} 8%, 74%   {{ opacity: 0.22; }} 75.5%, 100% {{ opacity: 1; }} }}
@keyframes c-d4 {{ 0%, 5% {{ opacity: 1; }} 8%, 74.5% {{ opacity: 0.22; }} 76%, 100% {{ opacity: 1; }} }}
@keyframes c-d5 {{ 0%, 5% {{ opacity: 1; }} 8%, 76%   {{ opacity: 0.22; }} 77.5%, 100% {{ opacity: 1; }} }}

.bridge-mask-line {{
  stroke-dasharray: 76;
  animation: draw-bridge {CYCLE_SECONDS}s ease-in-out infinite;
}}
@keyframes draw-bridge {{
  0%, 5%     {{ stroke-dashoffset: 0; }}
  8%, 78%    {{ stroke-dashoffset: 76; }}
  81%, 100%  {{ stroke-dashoffset: 0; }}
}}
.bridge-lbl {{
  animation: bridge-lbl {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes bridge-lbl {{
  0%, 5%     {{ opacity: 1; }}
  8%, 78.5%  {{ opacity: 0.2; }}
  81%, 100%  {{ opacity: 1; }}
}}

.amb-ring-top {{
  transform-origin: 467.4px 286px;
  animation: amb-ring {CYCLE_SECONDS}s ease-out infinite;
}}
.amb-ring-bot {{
  transform-origin: 399.8px 316px;
  animation: amb-ring {CYCLE_SECONDS}s ease-out infinite;
}}
@keyframes amb-ring {{
  0%, 80.5%  {{ transform: scale(1); opacity: 0; }}
  81.5%      {{ transform: scale(1); opacity: 0.75; }}
  84%        {{ transform: scale(2.4); opacity: 0; }}
  100%       {{ transform: scale(2.4); opacity: 0; }}
}}

/* --- Respect reduced motion: snap immediately to exact static paper figure --- */
@media (prefers-reduced-motion: reduce) {{
  * {{
    animation: none !important;
    transition: none !important;
    opacity: 1 !important;
    stroke-dashoffset: 0 !important;
    transform: none !important;
  }}
  .bank-bg {{ opacity: 0.14 !important; }}
  .b1-grid, .b2-grid, .b3-grid {{ clip-path: none !important; }}
  .pkt, .amb-ring-top, .amb-ring-bot {{ display: none !important; }}
}}
</style>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="506" height="400" viewBox="0 0 506 400" role="img" aria-labelledby="fig1-title fig1-desc">
<title id="fig1-title">Auditing AI SNAP Applications: Filer and Reader Pipeline</title>
<desc id="fig1-desc">Three-panel teaser. Panel a shows the Filer study and Reader study built from FY2024 SNAP QC records (8.5% of omitted-income error dollars found in the case record). Panel b shows that when no wage record is present, a Number required form causes 71 of 72 filings to enter $0 plus 1 invalid (0/72 flagged by rule check), whereas an "Unknown" allowed form produces 41 blank and 31 schema-invalid filings (72/72 flagged), and after rejection qwen3.5 invents a wage on 13/24 cases; stale pay stubs trigger 0/144 rule-check flags under both forms. Panel c compares AI reader flag rates when income is omitted versus when there is no income, showing the same ambiguous bank deposit drives high error-group flagging alongside high false-flagging.</desc>
{css}
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M1 1L9 5L1 9" fill="none" stroke="#1F1F1F" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
  <marker id="ar-cue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M1 1L9 5L1 9" fill="none" stroke="#D55E00" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
  <pattern id="pf" x="84" y="276" width="6" height="6" patternUnits="userSpaceOnUse">
    <rect width="4.8" height="4.8" rx="0.8" fill="var(--color-cue, #D55E00)"/>
  </pattern>
  <pattern id="pe" x="84" y="276" width="6" height="6" patternUnits="userSpaceOnUse">
    <rect x="0.3" y="0.3" width="4.2" height="4.2" rx="0.8" fill="none" stroke="var(--color-muted, #6B6B6B)" stroke-width="0.6"/>
  </pattern>
  <pattern id="ph" x="84" y="276" width="6" height="6" patternUnits="userSpaceOnUse">
    <rect x="0.3" y="0.3" width="4.2" height="4.2" rx="0.8" fill="none" stroke="var(--color-muted, #6B6B6B)" stroke-width="0.6"/>
    <path d="M0.6 4.2L4.2 0.6" stroke="var(--color-muted, #6B6B6B)" stroke-width="0.6"/>
  </pattern>

  <!-- Explicit pixel mask bounds avoid zero-width/zero-height line clipping -->
  <mask id="mask-reject" maskUnits="userSpaceOnUse" x="-40" y="-20" width="560" height="440">
    <path d="M347 113C347 142 216 142 216 106" fill="none" stroke="#FFFFFF" stroke-width="14" stroke-linecap="round" class="reject-mask-line"/>
  </mask>
  <mask id="mask-ledger" maskUnits="userSpaceOnUse" x="-40" y="-20" width="560" height="440">
    <line x1="161" y1="157" x2="161" y2="183" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round" class="ledger-mask-line"/>
  </mask>
  <mask id="mask-bridge" maskUnits="userSpaceOnUse" x="-40" y="-20" width="560" height="440">
    <line x1="465.3" y1="289" x2="401.9" y2="313" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round" class="bridge-mask-line"/>
  </mask>

  <!-- Clip paths for left-to-right waffle grid sweeps in Panel b -->
  <clipPath id="clip-b1">
    <rect class="wipe-b1" x="84" y="274" width="110" height="28"/>
  </clipPath>
  <clipPath id="clip-b2">
    <rect class="wipe-b2" x="84" y="316" width="110" height="28"/>
  </clipPath>
  <clipPath id="clip-b3">
    <rect class="wipe-b3" x="84" y="358" width="110" height="16"/>
  </clipPath>
</defs>

<rect width="506" height="400" fill="#FFFFFF"/>

<text class="b" x="4" y="12" style="font-weight:700">a</text>
<g id="panel-a" transform="translate(19,0)">
  <!-- QC Database -->
  <g class="stage-db">
    <ellipse cx="31" cy="54" rx="15" ry="4.5" class="act"/>
    <path d="M16 54V88A15 4.5 0 0 0 46 88V54" class="act"/>
    <path d="M16 66A15 4.5 0 0 0 46 66M16 77A15 4.5 0 0 0 46 77" style="stroke:var(--color-actor);stroke-width:0.5;fill:none"/>
    <text x="-12" y="104">FY2024 SNAP QC:</text>
    <text x="-12" y="114"><tspan style="font-weight:600">8.5%</tspan> of omitted-income</text>
    <text x="-12" y="124">error dollars found in</text>
    <text x="-12" y="134">the case record</text>
    <line x1="51" y1="72" x2="88" y2="72" class="a" marker-end="url(#ar)"/>
    <text class="m" x="70" y="55" text-anchor="middle">rebuilt as</text>
    <text class="m" x="70" y="64" text-anchor="middle">records</text>
  </g>

  <!-- Household Packet + Fact Ledger -->
  <g class="stage-packet">
    <rect x="103" y="41" width="74" height="98" rx="2" style="fill:none;stroke:var(--color-muted);stroke-width:0.6;stroke-dasharray:2 2"/>
    <path d="M96 34H160L170 44V132H96Z" style="fill:var(--color-paper);stroke:var(--color-actor);stroke-width:1.2"/>
    <path d="M160 34V44H170" style="fill:none;stroke:var(--color-actor);stroke-width:0.6"/>
    <circle cx="96" cy="34" r="8" style="fill:var(--color-paper);stroke:var(--color-actor);stroke-width:1.2"/>
    <circle cx="96" cy="32" r="2.2" style="fill:none;stroke:var(--color-actor);stroke-width:0.9"/>
    <path d="M92 38.5C92 35 100 35 100 38.5" style="fill:none;stroke:var(--color-actor);stroke-width:0.9"/>
    <text class="b" x="108" y="30">Household packet</text>
    <rect class="bank-bg" x="100" y="62.5" width="66" height="11" rx="1" style="fill:var(--color-cue);opacity:0.14"/>
    <text x="104" y="55">Pay stub</text>
    <text class="c" x="104" y="71" style="font-weight:600">Bank deposit</text>
    <text x="104" y="87">Lease</text>
    <text x="104" y="103">Utility bill</text>
    <text x="104" y="119">Applicant note</text>
    <text class="m" x="137" y="151" text-anchor="middle">Fact ledger (hidden truth)</text>
    <g style="fill:none;stroke:var(--color-context);stroke-width:0.6">
      <path d="M170 52C188 52 188 70 199 72"/>
      <path d="M170 84C188 84 188 76 199 74"/>
      <path d="M170 100C188 100 188 77 199 75"/>
      <path d="M170 116C188 116 188 78 199 76"/>
    </g>
    <path d="M170 68C186 68 190 72 199 73" style="fill:none;stroke:var(--color-cue);stroke-width:1.6"/>
  </g>

  <!-- Filer study: AI filer -> Application -> Portal -->
  <g class="stage-filer">
    <rect x="200" y="59" width="32" height="29" rx="8" class="act"/>
    <circle class="filer-eye-l" cx="210" cy="73" r="2.3" style="fill:var(--color-actor)"/>
    <circle class="filer-eye-r" cx="222" cy="73" r="2.3" style="fill:var(--color-actor)"/>
    <line x1="216" y1="59" x2="216" y2="53" class="act"/>
    <circle class="filer-antenna" cx="216" cy="51.5" r="1.6" style="fill:var(--color-actor)"/>
    <text x="216" y="101" text-anchor="middle">AI filer</text>
    <text class="i" x="196" y="30">Filer study</text>
    <line x1="236" y1="73" x2="256" y2="73" class="a" marker-end="url(#ar)"/>
    <rect x="260" y="50" width="36" height="46" rx="1.5" class="act"/>
    <g style="stroke:var(--color-context);stroke-width:1.4">
      <line x1="266" y1="59" x2="290" y2="59"/>
      <line x1="266" y1="67" x2="286" y2="67"/>
      <line x1="266" y1="88" x2="288" y2="88"/>
    </g>
    <rect class="app-field" x="265" y="73" width="26" height="9" rx="1" style="fill:none;stroke:var(--color-cue);stroke-width:1.2"/>
    <text x="278" y="108" text-anchor="middle">Application</text>
    <rect x="258" y="27" width="40" height="13" rx="6.5" style="fill:var(--color-paper);stroke:var(--color-actor);stroke-width:0.9"/>
    <text x="278" y="36.5" text-anchor="middle" style="fill:var(--color-actor);font-size:7.5px;font-weight:600">rule check</text>
    <line x1="278" y1="40" x2="278" y2="49" style="stroke:var(--color-actor);stroke-width:0.7;stroke-dasharray:1.5 1.5"/>
    <line x1="300" y1="73" x2="320" y2="73" class="a" marker-end="url(#ar)"/>
    <rect x="324" y="54" width="46" height="38" rx="2.5" class="act"/>
    <line x1="324" y1="62" x2="370" y2="62" style="stroke:var(--color-actor);stroke-width:0.6"/>
    <circle cx="328.5" cy="58" r="1.1" style="fill:var(--color-actor)"/>
    <circle cx="332.5" cy="58" r="1.1" style="fill:var(--color-actor)"/>
    <path class="portal-x1" d="M341 71L353 83" style="stroke:var(--color-cue);stroke-width:1.5;stroke-linecap:round"/>
    <path class="portal-x2" d="M353 71L341 83" style="stroke:var(--color-cue);stroke-width:1.5;stroke-linecap:round"/>
    <text x="347" y="106" text-anchor="middle">Portal</text>
  </g>

  <!-- Rejection retry loop -->
  <g class="stage-reject">
    <path d="M347 113C347 142 216 142 216 110" mask="url(#mask-reject)" style="fill:none;stroke:var(--color-cue);stroke-width:1;stroke-dasharray:3 2" marker-end="url(#ar-cue)"/>
    <text class="c" x="281" y="152" text-anchor="middle">Rejected: “field required”</text>
  </g>

  <!-- Accepted path into Case file -->
  <g class="stage-accept">
    <path d="M374 73H450V196H426" class="a" marker-end="url(#ar)"/>
    <text class="m" x="378" y="68">accepted</text>
  </g>

  <!-- Reader study: Case file -> AI reader -> Flag -->
  <g class="stage-reader">
    <path d="M372 180H386L390 184H420V214H372Z" class="act"/>
    <rect x="379" y="190" width="13" height="18" style="fill:none;stroke:var(--color-ink);stroke-width:0.6"/>
    <rect x="397" y="190" width="17" height="18" style="fill:none;stroke:var(--color-ink);stroke-width:0.6"/>
    <line class="case-mark" x1="381" y1="198" x2="390" y2="198" style="stroke:var(--color-cue);stroke-width:1.4"/>
    <text x="396" y="228" text-anchor="middle">Case file</text>
    <text class="i" x="298" y="170">Reader study</text>
    <line x1="368" y1="197" x2="346" y2="197" class="a" marker-end="url(#ar)"/>
    <path d="M298 197Q320 181 342 197Q320 213 298 197Z" class="act"/>
    <circle class="reader-iris" cx="320" cy="197" r="4.2" style="fill:var(--color-actor)"/>
    <text x="320" y="228" text-anchor="middle">AI reader</text>
    <line x1="294" y1="197" x2="272" y2="197" class="a" marker-end="url(#ar)"/>
    <path d="M256 186V210M256 186L268 190L256 195" class="act"/>
    <text x="260" y="228" text-anchor="middle">Flag</text>
  </g>

  <!-- Scored + Hidden Truth verification -->
  <g class="stage-scored">
    <line x1="248" y1="197" x2="176" y2="197" class="a" marker-end="url(#ar)"/>
    <circle cx="161" cy="197" r="11" class="act"/>
    <path class="check-draw" d="M155.5 197L159.5 201L167 193" style="fill:none;stroke:var(--color-actor);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>
    <text x="161" y="228" text-anchor="middle">Scored</text>
    <line x1="161" y1="157" x2="161" y2="183" mask="url(#mask-ledger)" style="stroke:var(--color-muted);stroke-width:0.6;stroke-dasharray:2 2"/>
  </g>

  <text class="m" x="-6" y="186">Evidence varied in</text>
  <text class="m" x="-6" y="196">the packet: none,</text>
  <text class="m" x="-6" y="206">stale stub, deposits,</text>
  <text class="m" x="-6" y="216">applicant note</text>

  <!-- Animated signal tokens (hidden under reduced motion) -->
  <circle class="pkt pkt-qc"      cx="0" cy="0" r="2.1" fill="#0072B2"/>
  <circle class="pkt pkt-deposit" cx="0" cy="0" r="2.4" fill="#D55E00"/>
  <circle class="pkt pkt-app"     cx="0" cy="0" r="2.0" fill="#1F1F1F"/>
  <circle class="pkt pkt-portal"  cx="0" cy="0" r="2.0" fill="#1F1F1F"/>
  <circle class="pkt pkt-reject"  cx="0" cy="0" r="2.5" fill="#D55E00"/>
  <circle class="pkt pkt-accept"  cx="0" cy="0" r="2.2" fill="#0072B2"/>
  <circle class="pkt pkt-read1"   cx="0" cy="0" r="2.0" fill="#1F1F1F"/>
  <circle class="pkt pkt-read2"   cx="0" cy="0" r="2.0" fill="#1F1F1F"/>
  <circle class="pkt pkt-read3"   cx="0" cy="0" r="2.0" fill="#0072B2"/>
  <circle class="pkt pkt-truth"   cx="0" cy="0" r="1.8" fill="#6B6B6B"/>
</g>

<line x1="4" y1="244" x2="502" y2="244" style="stroke:var(--color-rule);stroke-width:0.6"/>

<g id="panel-b">
  <text class="b" x="4" y="262" style="font-weight:700">b</text>
  <text class="m" x="84" y="266">No wage record: filings</text>
  <text class="m" x="200" y="266">Rule check</text>

  <!-- Row 1: Number required (0/72 flagged) -->
  <g class="b1-wrap">
    <text x="18" y="288" style="font-weight:600">Number</text>
    <text x="18" y="297" style="font-weight:600">required</text>
    <g class="b1-grid" clip-path="url(#clip-b1)">
      <rect x="84" y="276" width="108" height="18" style="fill:url(#pf)"/>
      <rect x="84" y="294" width="102" height="6" style="fill:url(#pf)"/>
      <rect x="186" y="294" width="6" height="6" style="fill:url(#ph)"/>
    </g>
    <text class="c" x="84" y="309" style="font-weight:600">$0 filed: 71 + 1 invalid</text>
  </g>
  <g class="b1-flag">
    <path d="M202 280V298M202 280L212 283.5L202 287" style="fill:none;stroke:var(--color-context);stroke-width:1.1"/>
    <text x="216" y="288" style="font-weight:600">0/72</text>
    <text class="m" x="216" y="297">flagged</text>
  </g>

  <!-- Row 2: "Unknown" allowed (72/72 flagged) -->
  <g class="b2-wrap">
    <text x="18" y="324" style="font-weight:600">“Unknown”</text>
    <text x="18" y="333" style="font-weight:600">allowed</text>
    <g class="b2-grid" clip-path="url(#clip-b2)">
      <rect x="84" y="318" width="108" height="12" style="fill:url(#pe)"/>
      <rect x="84" y="330" width="30" height="6" style="fill:url(#pe)"/>
      <rect x="114" y="330" width="78" height="6" style="fill:url(#ph)"/>
      <rect x="84" y="336" width="108" height="6" style="fill:url(#ph)"/>
    </g>
    <text class="m" x="84" y="352">left blank: 41 + 31 schema-invalid (hatched)</text>
  </g>
  <g class="b2-flag">
    <path d="M202 320V338M202 320L212 323.5L202 327" class="act"/>
    <text x="216" y="328" style="font-weight:600;fill:var(--color-actor)">72/72</text>
    <text class="m" x="216" y="337">flagged</text>
  </g>

  <!-- Row 3: After rejection (13/24 invented wage) -->
  <g class="b3-wrap">
    <text x="18" y="365" style="font-weight:600">After</text>
    <text x="18" y="374" style="font-weight:600">rejection</text>
    <g class="b3-grid" clip-path="url(#clip-b3)">
      <rect x="84" y="360" width="78" height="6" style="fill:url(#pf)"/>
      <rect x="162" y="360" width="30" height="6" style="fill:url(#pe)"/>
      <rect x="84" y="366" width="36" height="6" style="fill:url(#pe)"/>
    </g>
    <text class="c" x="84" y="381" style="font-weight:600">13/24 invented wage</text>
    <text class="m" x="162" y="381"> (qwen3.5 req.)</text>
  </g>
  <text class="m b-foot" x="18" y="394">Stale pay stub: check flags 0/144 under both forms</text>
</g>

<line x1="252" y1="252" x2="252" y2="391" style="stroke:var(--color-rule);stroke-width:0.6"/>

<g id="panel-c">
  <text class="b" x="260" y="262" style="font-weight:700">c</text>
  <g style="stroke:var(--color-rule);stroke-width:0.5">
    <line x1="320" y1="270" x2="320" y2="336"/>
    <line x1="407" y1="270" x2="407" y2="336"/>
    <line x1="494" y1="270" x2="494" y2="336"/>
  </g>
  <line x1="320" y1="336" x2="494" y2="336" style="stroke:var(--color-muted);stroke-width:0.6"/>
  <text class="m" x="320" y="346" text-anchor="middle">0%</text>
  <text class="m" x="407" y="346" text-anchor="middle">50%</text>
  <text class="m" x="494" y="346" text-anchor="middle">100%</text>
  <text class="m" x="407" y="357" text-anchor="middle">share flagged on target income group</text>

  <!-- Income omitted track -->
  <text x="260" y="287" style="font-weight:600">Income omitted</text>
  <text class="m" x="260" y="296">should flag</text>
  <line class="c-track1" x1="332.1" y1="286" x2="494" y2="286" style="stroke:var(--color-context);stroke-width:1.4"/>
  <g class="c-d1">
    <circle cx="332.1" cy="286" r="3.4" style="fill:var(--color-paper);stroke:var(--color-ink);stroke-width:1"/>
    <text class="c" x="332.1" y="278" text-anchor="middle" style="font-weight:600">$0 file</text>
  </g>
  <g class="c-d2">
    <circle cx="467.4" cy="286" r="3.4" style="fill:var(--color-paper);stroke:var(--color-cue);stroke-width:1.4"/>
    <text class="c" x="463" y="278" text-anchor="end">ambiguous</text>
  </g>
  <g class="c-d3">
    <circle cx="494" cy="286" r="3.4" style="fill:var(--color-cue)"/>
    <text class="c" x="498" y="278" text-anchor="end">clear</text>
  </g>

  <!-- No income track -->
  <text x="260" y="317" style="font-weight:600">No income</text>
  <text class="m" x="260" y="326">should not flag</text>
  <line class="c-track2" x1="321.2" y1="316" x2="399.8" y2="316" style="stroke:var(--color-context);stroke-width:1.4"/>
  <g class="c-d4">
    <circle cx="321.2" cy="316" r="3.4" style="fill:var(--color-paper);stroke:var(--color-ink);stroke-width:1"/>
    <text class="m" x="328" y="328">clear deposit</text>
  </g>
  <g class="c-d5">
    <circle cx="399.8" cy="316" r="3.4" style="fill:var(--color-paper);stroke:var(--color-cue);stroke-width:1.4"/>
    <text class="c" x="399.8" y="328" text-anchor="middle">ambiguous</text>
  </g>

  <!-- "same deposit" diagonal bridge + ring pulse -->
  <line x1="465.3" y1="289" x2="401.9" y2="313" mask="url(#mask-bridge)" style="stroke:var(--color-cue);stroke-width:0.8;stroke-dasharray:2 1.5"/>
  <text class="c bridge-lbl" x="418" y="304" text-anchor="end" style="font-style:italic">same deposit</text>
  <circle class="amb-ring-top" cx="467.4" cy="286" r="3.4" fill="none" stroke="#D55E00" stroke-width="1"/>
  <circle class="amb-ring-bot" cx="399.8" cy="316" r="3.4" fill="none" stroke="#D55E00" stroke-width="1"/>
</g>
</svg>
"""
    return svg


def main() -> None:
    raw_svg = SRC_SVG.read_text(encoding="utf-8")
    out = build_animated_svg(raw_svg)
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUT_SVG.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT_SVG} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
