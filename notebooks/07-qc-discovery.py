# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 How QC found each error, the risk score on a matched face, and the false-flag rate 80% precision needs (no LLM calls)
#
# - **Question:** (1) What share of weighted error dollars is client-caused unreported income, and how often did QC find such a variance from the case record (DISCOV), against agency errors? (2) Is the risk score's AUC of 0.451 on the 151 omissions filed at $0 a selection effect of comparing them with all error-free households? (3) What false-flag rate gives 80% precision at the prevalences in question, and what can a zero-flag review of 24, 48 or 96 negatives rule out?
# - **Expected:** 22.1% of weighted error dollars client-caused unreported income, 8.5% of those dollars found from the record, 59.7% for agency errors. Households: 8.6% of the 185 omissions (16 found from the case record, 117 from the recipient interview); 10.0% and 6.1% for two other household cuts; about 808 client-caused unreported-income households. Comparison pool with the same face about 11,080 (34.7% of error-free households). Required false-flag rate at 80% recall: 0.09% at 0.46% prevalence, 0.5% at 2.47%. Wilson upper bound 7.4% at 0 of 48.
# - **Method:** (1) variance-level table of the FY2024 QC public file, all 53 jurisdictions, errors above $56, dollars FYWGT x AMOUNT, cause classes from the AGENCY codes, client-caused unreported income = client cause and NATURE 35; household DISCOV breakdowns under three weightings. (2) notebook 02's section-2 risk score (imported from `snapqc.review_sim` and `snapqc.features`), the 151 scored from the filed application, compared with error-free households in the same 41 states whose target group and net income are $0, pairs matched on target group; bootstrap 95% interval, seed 20261115. (3) closed form f = r p / (4 (1 - p)); Wilson bound z^2 / (n + z^2).
# - **Result:** (1) client-caused unreported income 22.35% of weighted error dollars, 8.50% of it found from the case record; agency errors 50.07% of dollars, 59.62% of agency dollars found from the case record. 185 omissions: 16 (8.65%) found from the case record, 117 from the recipient interview. 984 client-caused unreported-income households in the 53 jurisdictions: 10.98% by households, 9.37% FYWGT-weighted, 9.70% dollar-weighted; 785 in the 41 states: 8.92%, 8.27%, 9.00%. (2) the 0.451 (0.410 to 0.492) is reproduced. Against same-face households matched on target group the AUC is 0.628 (0.586 to 0.670); earned 0.601, unearned 0.698. Pool: 12,449 error-free households with net income $0 (39.0%); earned $0 and net $0 10,618 (33.2%), unearned $0 and net $0 8,563 (26.8%). (3) at 80% recall the required false-flag rate is 0.092% at 0.46% and 0.507% at 2.47%; Wilson upper bounds 13.8%, 7.41%, 3.85% at 24, 48, 96.
# - **Mismatch:** expected 22.1% vs 22.35%; 808 households vs 984 (no tested definition gave 808); the expected 59.7% matches the share of agency dollars found from the case record (59.62%), not the agency share of dollars (50.07%); the expected 10.0% and 6.1% could not be matched to a defined cut (closest: 10.98% and 9.0% to 9.7%); comparison pool 11,080 (34.7%) not matched by any face definition tried (net $0 alone 12,449; earned $0 and net $0 10,618). The base of 31,935 error-free households matches 11,080 / 0.347.
# - **Feeds:** `rq.md` RQ2 (where existing review rules fall short: how QC finds unreported income, and the risk score on omissions) and RQ3 (the false-flag rate a better rule needs for 80% precision).
# - **Open questions:** which household cuts the expected 10.0% and 6.1% refer to; which face definition gives 11,080; whether 0.628, between the two thresholds (0.65 and about 0.5), is reported as the matched figure beside 0.451.

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from snapqc import features as ft
from snapqc import review_sim as rs
from snapqc.data import ROOT, SINGLE_SUA_STATES, load_fy2024, per_variance, results_dir

OUT = results_dir("07-qc-discovery")
SEED = 20261115
TOL = rs.TOLERANCE[2024]  # $56
CLIENT = [1, 2, 3, 4, 7]
AGENCY_CAUSED = [8, *range(10, 26)]
CASE_RECORD = [1, 2]  # DISCOV 1 and 2: QC found the variance from the case record
DISCOV_LABEL = {
    1: "Variance clearly identified from case record (not automated match)",
    2: "From case record (automated match)",
    3: "From recipient interview",
    4: "Employer (present or former)",
    5: "Financial institution, insurance company, or other business",
    6: "Landlord",
    7: "Government agency or public records, not automated match",
    8: "Government agency or public records, automated match",
    9: "Other",
}

full = load_fy2024()
arm = pd.read_csv(ROOT / "results" / "01-data-checks" / "3d-human-arm.csv", index_col="row")
omission = arm[arm.group.isin(["earned", "unearned"]) & (arm.filed_value <= 2) & (arm.verified_value > 0)]
assert len(full) == 44891 and full.STATENAME.nunique() == 53 and len(arm) == 616
assert len(omission) == 185

# %% [markdown]
# ## 1. Where QC found each kind of error
#
# One row per variance (ELEMENT1..9 and the parallel NATURE, AGENCY, AMOUNT, DISCOV columns), all 53 jurisdictions, in error households (STATUS 2 or 3, AMTERR above $56). Dollars = FYWGT x AMOUNT of the variance. Cause classes from the AGENCY (variance cause) codes of the FY2024 SNAP QC technical documentation: client 1, 2, 3, 4, 7; agency 8 and 10 to 25; 26 "not required to be reported"; 99 "other". "Client-caused unreported income" = a client cause with NATURE 35 (unreported source of income); every NATURE 35 variance in this set sits on an income element.

# %%
cols = ["ELEMENT", "NATURE", "AGENCY", "AMOUNT", "DISCOV"]
var = pd.concat([
    full[[f"{c}{i}" for c in cols]].set_axis(cols, axis=1).assign(hh=full.index)
    for i in range(1, 10)
], ignore_index=True).dropna(subset=["ELEMENT"])
var = var.join(full[["STATUS", "AMTERR", "FYWGT", "STATENAME"]], on="hh")
var = var[var.STATUS.isin([2, 3]) & (var.AMTERR > TOL)].copy()
var["dollars"] = var.FYWGT * var.AMOUNT.abs()
var["case_record"] = var.DISCOV.isin(CASE_RECORD)
unrep = var.AGENCY.isin(CLIENT) & (var.NATURE == 35)
income_el = var.ELEMENT.between(311, 346) | (var.ELEMENT == 350)
assert income_el[unrep].all()
var["cause"] = np.select(
    [unrep, var.AGENCY.isin(CLIENT), var.AGENCY.isin(AGENCY_CAUSED), var.AGENCY == 26, var.AGENCY == 99],
    ["client-caused unreported income", "client-caused, other", "agency", "not required to be reported (26)", "other (99)"],
    "no cause code")

by_cause = var.groupby("cause").agg(variances=("hh", "size"), households=("hh", "nunique"), weighted_dollars=("dollars", "sum"),
                                    dollars_case_record=("dollars", lambda d: d[var.loc[d.index, "case_record"]].sum()))
by_cause["share of weighted error dollars"] = (by_cause.weighted_dollars / by_cause.weighted_dollars.sum()).round(4)
by_cause["share of the class's dollars QC found from the case record"] = (by_cause.dollars_case_record / by_cause.weighted_dollars).round(4)
by_cause = by_cause.drop(columns="dollars_case_record").sort_values("weighted_dollars", ascending=False)
by_cause["weighted_dollars"] = by_cause.weighted_dollars.round(0)
by_cause.to_csv(OUT / "1a-dollars-by-cause.csv", index_label="cause class")
by_cause

# %% [markdown]
# ### 1b. DISCOV breakdown by household
#
# (a) the 185 omissions of the human arm (one variance each; DISCOV1). (b) every household in the file with a client-caused unreported-income variance in an error above $56 (all 53 jurisdictions); a household with more than one such variance takes the DISCOV of its first. (c) the same, restricted to the 41 states of the human arm.

# %%
first_unrep = var[unrep].groupby("hh").first()
uh = pd.DataFrame({"DISCOV": first_unrep.DISCOV, "FYWGT": first_unrep.FYWGT, "AMTERR": first_unrep.AMTERR,
                   "state": first_unrep.STATENAME})
uh["case_record"] = var[unrep].groupby("hh").case_record.any()  # any such variance found from the case record
cuts = {
    "omissions, human arm (185)": omission.rename(columns={"DISCOV1": "DISCOV"}),
    f"client-caused unreported income, 53 jurisdictions ({len(uh)})": uh,
    f"client-caused unreported income, 41 states ({int(uh.state.isin(SINGLE_SUA_STATES).sum())})": uh[uh.state.isin(SINGLE_SUA_STATES)],
}
discov = pd.DataFrame({name: d.DISCOV.astype(int).value_counts().reindex(range(1, 10), fill_value=0) for name, d in cuts.items()})
discov.index = [f"{c} {DISCOV_LABEL[c]}" for c in discov.index]
discov.loc["households"] = discov.sum()
discov.to_csv(OUT / "1b-discov-by-household.csv", index_label="DISCOV")
discov

# %% [markdown]
# Share found from the case record (DISCOV 1 or 2) under three weightings: households, FYWGT, and FYWGT x AMTERR (weighted error dollars).

# %%
def shares(d):
    rec = d.get("case_record", d.DISCOV.isin(CASE_RECORD)).astype(bool)
    w, dw = d.FYWGT, d.FYWGT * d.AMTERR
    return {"households": len(d), "found from case record": int(rec.sum()),
            "share, households": round(rec.mean(), 4),
            "share, weighted (FYWGT)": round(float(w[rec].sum() / w.sum()), 4),
            "share, weighted error dollars (FYWGT x AMTERR)": round(float(dw[rec].sum() / dw.sum()), 4)}


case_record_cuts = pd.DataFrame({n: shares(d) for n, d in cuts.items()}).T
case_record_cuts.to_csv(OUT / "1c-case-record-share-by-cut.csv", index_label="cut")
case_record_cuts

# %% [markdown]
# ## 2. The risk score on the 151 against comparison households with the same face
#
# Section 2 of notebook 02: three gradient-boosted models fitted on BGL's frame for FY2022-23 (`snapqc.review_sim`), the features built by `snapqc.features` from the filed application, so the 151 omissions filed at $0 are scored as filed. Comparison households: the 41 states, no error above $56, with a section 1 score, as in notebook 02. "Same face": the comparison household shows what the 151 show on the form, the target group at $0 and net income (FSNETINC) at $0. The target group is earned for 103 of the 151 and unearned for 48, so each omission is paired only with comparison households whose own amount in that group is $0. The matched AUC is the share of such pairs where the omission scores higher (ties count half); the bootstrap resamples omissions and comparison households within each group (2,000 draws).

# %%
SNAP_QC = ROOT / "third_party" / "snap_qc"
frame = rs.read_bgl_frame(SNAP_QC / "additional_data" / "reg_model_data.rds")
models = rs.fit_risk_models(frame, train_years=(2022, 2023))
risk = rs.fit_risk(frame, score_year=2024, models=models)
c49 = full[~full.STATENAME.isin(rs.OUTSIDE)]
tables = ft.load_tables(SNAP_QC)
app_qc = ft.application_from_qc(c49)
covered = arm.index[arm.in_151]
assert covered.isin(omission.index).all() and len(covered) == 151
app_filed = ft.file_amounts(app_qc.loc[covered], arm.amount[covered], arm.filed_value[covered], arm.element[covered])
pos_score = rs.score_risk(models, ft.bgl_features(app_filed, tables), app_filed["size"])
s1 = c49.HHLDNO.map(risk)
assert pos_score.notna().all()

# %%
in41 = c49.STATENAME.isin(SINGLE_SUA_STATES)
error_free = in41 & ~(c49.AMTERR > TOL)
net0 = c49.FSNETINC == 0
group0 = {"earned": c49.FSEARN.fillna(0) == 0, "unearned": c49.FSUNEARN.fillna(0) == 0}
face = {g: error_free & net0 & z for g, z in group0.items()}
any_face = face["earned"] | face["unearned"]
pool = pd.DataFrame({
    "error-free households, 41 states": {"households": int(error_free.sum()), "with a score": int((error_free & s1.notna()).sum())},
    "net income $0": {"households": int((error_free & net0).sum()), "with a score": int((error_free & net0 & s1.notna()).sum())},
    "earned $0 and net income $0 (face for earned omissions)": {"households": int(face["earned"].sum()), "with a score": int((face["earned"] & s1.notna()).sum())},
    "unearned $0 and net income $0 (face for unearned omissions)": {"households": int(face["unearned"].sum()), "with a score": int((face["unearned"] & s1.notna()).sum())},
    "either face": {"households": int(any_face.sum()), "with a score": int((any_face & s1.notna()).sum())},
}).T
pool["share of error-free households"] = (pool.households / error_free.sum()).round(4)
pool.to_csv(OUT / "2a-comparison-pool.csv", index_label="comparison households")
pool

# %%
pos_group = arm.group[covered]
pos = {g: pos_score[pos_group == g].to_numpy() for g in face}
neg = {g: s1[face[g] & s1.notna()].to_numpy() for g in face}
neg_all = s1[error_free & s1.notna()].to_numpy()


def pair_auc(pos, neg):
    """AUC over pairs within each group: share of (omission, comparison) pairs with the omission scored higher."""
    wins = pairs = 0.0
    for g in pos:
        p, n = pos[g], neg[g]
        r = rankdata(np.concatenate([p, n]))
        wins += r[:len(p)].sum() - len(p) * (len(p) + 1) / 2
        pairs += len(p) * len(n)
    return wins / pairs


def boot(stat, pos, neg, rng, n_boot=2000):
    draws = [stat({g: rng.choice(pos[g], len(pos[g])) for g in pos}, {g: rng.choice(neg[g], len(neg[g])) for g in neg})
             for _ in range(n_boot)]
    return np.quantile(draws, [0.025, 0.975])


rng = np.random.default_rng(SEED)
rows = {}
for name, p, n in [
    ("all error-free comparison households (notebook 02 figure)", {"all": pos_score.to_numpy()}, {"all": neg_all}),
    ("same face, matched by target group", pos, neg),
    ("same face, earned omissions only", {"earned": pos["earned"]}, {"earned": neg["earned"]}),
    ("same face, unearned omissions only", {"unearned": pos["unearned"]}, {"unearned": neg["unearned"]}),
]:
    lo, hi = boot(pair_auc, p, n, rng)
    rows[name] = {"omissions": sum(len(v) for v in p.values()), "comparison households": sum(len(v) for v in n.values()),
                  "AUC": round(pair_auc(p, n), 3), "95% interval low": round(lo, 3), "95% interval high": round(hi, 3)}
matched = pd.DataFrame(rows).T
assert abs(roc_auc_score(np.r_[np.ones(151), np.zeros(len(neg_all))], np.r_[pos_score, neg_all]) - matched.iloc[0].AUC) < 1e-3
matched.to_csv(OUT / "2b-matched-auc.csv", index_label="comparison")
matched

# %% [markdown]
# ## 3. The false-flag rate 80% precision needs
#
# Precision = r p / (r p + f (1 - p)) for recall r, prevalence p and false-flag rate f (share of negatives flagged). Setting precision to 0.8 gives f = r p (1 - 0.8) / (0.8 (1 - p)) = r p / (4 (1 - p)). Prevalence runs from 0.46% to 2.47% with three evenly spaced points between. Beside it: the Wilson 95% upper bound on f when 0 of n negatives are flagged, z^2 / (n + z^2) with z = 1.96.

# %%
prevalence = np.linspace(0.0046, 0.0247, 5)
recall = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
needed = pd.DataFrame({f"recall {r:.1f}": r * prevalence / (4 * (1 - prevalence)) for r in recall},
                      index=pd.Index([f"{p:.4%}" for p in prevalence], name="prevalence"))
needed.round(6).to_csv(OUT / "3a-false-flag-rate-for-80pct-precision.csv")
(needed * 100).round(3).astype(str) + "%"

# %%
z = norm.ppf(0.975)
wilson = pd.DataFrame({"negatives reviewed": [24, 48, 96]})
wilson["flags"] = 0
wilson["Wilson 95% upper bound on false-flag rate"] = (z**2 / (wilson["negatives reviewed"] + z**2)).round(4)
wilson.to_csv(OUT / "3b-wilson-upper-bound-zero-flags.csv", index=False)
wilson
