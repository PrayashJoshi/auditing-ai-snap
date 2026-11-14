"""Review simulation of Chohlas-Wood (arXiv 2611.03213), reimplemented from the paper's text.

The paper's simulation code is not public. Its reference [9] (github.com/giannella/snap_qc) is the Better
Government Lab (BGL) risk-model code, which the paper says it adapted. So this module rebuilds the four
review rules from the paper's Section 2 and Appendices B to D:

- risk only: cases in risk order, the state's top-k categories checked in each case (k = 1..12);
- AI only: every flag, category by category in the state's historical error-dollar order;
- combined (Section 2.3): all flags first, flagged cases in risk order; then category checks (top k) on cases
  with no flag, in risk order. Flagged cases get flag checks only;
- combined, risk order throughout (Appendix D.9): cases in risk order; in each of the top-k categories, check
  the flags if there are any, otherwise do a category check.

Constants below are the paper's main assumptions. Where the paper is silent, the choice made here is marked
"not stated in the paper".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import RAW, per_variance

# Section 1: the 48 contiguous states and DC.
OUTSIDE = ["Alaska", "Hawaii", "Guam", "Virgin Islands"]
TOLERANCE = {2022: 48, 2023: 54, 2024: 56}  # BGL additional_data/year_data.csv; FY2024 $56 is also in the paper
THRESHOLD = 0.06  # cost-sharing PER threshold

# Section 2.1 and 2.2 main assumptions.
CAT_MINUTES, CAT_SUCCESS = 10, 0.5
FLAG_MINUTES, FLAG_SUCCESS = 5, 0.8
FLAG_RECALL, FLAG_PRECISION = 0.8, 0.8
STATE_COST_PER_HOUR = 30  # $40/h less 25% federal reimbursement
N_DRAWS = 200

# Table C1 of the paper (Appendix C): element of variance -> review category.
CATEGORY_ELEMENTS = {
    "Earned income": [311, 314, 321],
    "Unearned income": [331, 332, 333, 334, 335, 336, 342, 343, 344, 345, 346, 350],
    "Shelter and utilities": [362, 363, 364],
    "Household composition": [140, 150, 170],
    "Self-employment": [312],
    "Medical expenses": [365],
    "Care and child support": [323, 366],
    "Citizenship and students": [111, 130],
    "Other computation and reporting": [520, 530, 540, 542, 560, 810, 820],
    "Work and disqualifications": [151, 160, 161, 162, 163, 164, 165, 166],
    "Income totals and standard deduction": [361, 371, 372],
    "Assets": [211, 212, 213, 214, 221, 222, 224, 225],
}
CATEGORIES = list(CATEGORY_ELEMENTS)
ELEMENT_CATEGORY = {e: i for i, els in enumerate(CATEGORY_ELEMENTS.values()) for e in els}

# Table B1 of the paper (Appendix B): primary causes of variance (AGENCYn) assumed NOT detectable. All others, and a missing
# cause, are assumed detectable.
NOT_DETECTABLE = {1, 3, 4, 7, 8, 26, 99}

# BGL v2 household-size model (INCL_find_inclusion_rules_by_hh_size_v2.R): strata, features, engine settings.
FEATURES = [
    "HH_size_n", "children_i", "elderly_disabled_i", "total_deductions_by_hh_size", "expedited_i",
    "bbce_state_i", "rawben_rel_max", "medical_deductions", "shelter_expenses_by_hh_size", "utilities",
    "married", "homeless", "earned_by_hh_size", "unearned_by_hh_size", "gross_by_hh_size", "percent_abawd",
    "unc_rawben_rel_max", "months_since_cert_n", "count_divisible_by_100",
]
BGL_XGB = dict(nrounds=1000, max_depth=4, eta=0.02, subsample=0.20)


def hh_stratum(size: pd.Series) -> pd.Series:
    return pd.cut(size, [-np.inf, 1, 3, np.inf], labels=["1", "2-3", "4+"])


# ---------------------------------------------------------------- data


def load_qc(year: int) -> pd.DataFrame:
    """One fiscal year of the public-use file, 48 contiguous states and DC, with the columns the simulation uses."""
    cols = ["HHLDNO", "STATENAME", "HWGT", "FYWGT", "RAWBEN", "AMTERR", "STATUS"]
    cols += per_variance("ELEMENT") + per_variance("AGENCY") + per_variance("AMOUNT")
    df = pd.read_csv(RAW / f"fy{year}" / f"qc_pub_fy{year}.csv", usecols=cols, low_memory=False)
    return df[~df.STATENAME.isin(OUTSIDE)].reset_index(drop=True)


def variances(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded variance: household row, category, detectable, and its share of AMTERR in dollars.

    The paper does not say how it splits a household's error dollars across its variances (not stated in the
    paper). AMOUNTn sums to zero for about half of households with a recorded error (about a fifth of those
    above the tolerance), so: split AMTERR in proportion to AMOUNTn when they sum above zero, else equally
    across the recorded variances.
    """
    el = df[per_variance("ELEMENT")].to_numpy(float)
    ag = df[per_variance("AGENCY")].to_numpy(float)
    am = np.nan_to_num(df[per_variance("AMOUNT")].to_numpy(float))
    rec = ~np.isnan(el) & (df.AMTERR.to_numpy()[:, None] > 0)
    am = np.where(rec, am, 0)
    tot = am.sum(1, keepdims=True)
    share = np.where(tot > 0, am / np.where(tot > 0, tot, 1), rec / np.maximum(rec.sum(1, keepdims=True), 1))
    hh, slot = np.nonzero(rec)
    out = pd.DataFrame({
        "hh": hh,
        "cat": pd.Series(el[hh, slot]).map(ELEMENT_CATEGORY).to_numpy(),
        "detectable": ~np.isin(ag[hh, slot], list(NOT_DETECTABLE)),  # NaN cause -> detectable
        "dollars": share[hh, slot] * df.AMTERR.to_numpy()[hh],
    })
    if out.cat.isna().any():
        raise ValueError(f"unmapped element codes: {sorted(set(el[hh, slot][out.cat.isna()]))}")
    out["cat"] = out.cat.astype(int)
    return out


def historical_category_order(years=(2022, 2023), above_tolerance_only=False) -> pd.DataFrame:
    """Per state, categories ranked by total error dollars in the FY2022-23 records (weighted by HWGT).

    Returns a state x category table of ranks (0 = largest). Ties (e.g. no dollars) are broken by the national
    order (not stated in the paper). The paper does not say whether "total error dollars" includes households
    at or below the year's tolerance (not stated in the paper): by default they are included;
    `above_tolerance_only=True` leaves them out.
    """
    parts = []
    for y in years:
        df = load_qc(y)
        v = variances(df)
        if above_tolerance_only:
            v = v[(df.AMTERR.to_numpy() > TOLERANCE[y])[v.hh]].copy()
        v["state"] = df.STATENAME.to_numpy()[v.hh]
        v["wd"] = v.dollars * df.HWGT.to_numpy()[v.hh]
        parts.append(v[["state", "cat", "wd"]])
    tot = pd.concat(parts).groupby(["state", "cat"]).wd.sum().unstack(fill_value=0)
    tot = tot.reindex(columns=range(len(CATEGORIES)), fill_value=0)
    national = tot.sum().rank(ascending=False, method="first")
    key = tot - national.to_numpy() * 1e-9  # national order breaks ties
    return key.rank(axis=1, ascending=False, method="first").astype(int) - 1


# ---------------------------------------------------------------- simulation


class Simulation:
    """Expected review time and error reduction per household, for every rule and scope, from N_DRAWS draws.

    `df` is the FY2024 file (48 states + DC); `in_sample` marks households with a risk score; `risk` is the score
    (NaN outside the sample); `rank` is historical_category_order().
    Households outside the sample are never reviewed, but their errors stay in the state totals (Section 1).

    A household whose error is at or below the tolerance still has its variances on file. The paper does not
    say whether such a variance can draw a flag (not stated in the paper). By default it can: the flag costs a
    flag check and counts as real when the false-flag rate is set, but correcting it removes no counted error.
    `flag_below_tolerance=False` lets only households above the tolerance draw real flags.
    """

    def __init__(self, df, in_sample, risk, rank, tol=TOLERANCE[2024], n_draws=N_DRAWS, seed=0,
                 flag_below_tolerance=True):
        self.df, self.tol = df, tol
        self.state = df.STATENAME.to_numpy()
        self.in_sample = np.asarray(in_sample, bool)
        self.risk = np.asarray(risk, float)
        self.w = df.HWGT.to_numpy(float)          # PER weights
        self.w_cost = df.FYWGT.to_numpy(float)    # one review per household per year (not stated in the paper)
        amterr = df.AMTERR.to_numpy(float)
        self.base = np.where(amterr > tol, amterr, 0.0)  # counted error before review
        rng = np.random.default_rng(seed)
        n, K = len(df), len(CATEGORIES)

        v = variances(df)
        v = v[self.in_sample[v.hh]].reset_index(drop=True)
        v["rank"] = rank.reindex(self.state[v.hh]).to_numpy()[np.arange(len(v)), v.cat]
        if not flag_below_tolerance:
            v["detectable"] = v.detectable & (amterr[v.hh.to_numpy()] > tol)
        self.v = v
        # unallocated dollars (error households with no recorded variance) can never be corrected
        self.fixed = amterr - np.bincount(v.hh, v.dollars, minlength=n) * 1.0
        self.fixed[~self.in_sample] = amterr[~self.in_sample]

        # false flags: Poisson per state and category, mean set so 80% of expected flags are real (Section 2.2)
        det = v[v.detectable]
        real = pd.Series(self.w[det.hh]).groupby([self.state[det.hh], det.cat.to_numpy()]).sum()
        real = real.unstack(fill_value=0).reindex(columns=range(K), fill_value=0) * FLAG_RECALL
        wsum = pd.Series(self.w[self.in_sample]).groupby(self.state[self.in_sample]).sum()
        lam = (real * (1 - FLAG_PRECISION) / FLAG_PRECISION).div(wsum, axis=0)
        self.lam = lam.reindex(sorted(set(self.state)), fill_value=0)
        lam_hc = self.lam.reindex(self.state).to_numpy()          # n x K
        n_det = np.zeros((n, K))
        np.add.at(n_det, (det.hh.to_numpy(), det.cat.to_numpy()), 1)
        self.n_det, self.lam_hc = n_det, lam_hc
        rank_hc = rank.reindex(self.state).to_numpy()               # n x K

        # draws for real-error events
        m = len(v)
        pair_id, pairs = pd.factorize(v.hh.to_numpy() * K + v.cat.to_numpy())
        self.pair_hh, pair_cat = pairs // K, pairs % K
        P = len(pairs)
        flagged = v.detectable.to_numpy()[:, None] & (rng.random((m, n_draws)) < FLAG_RECALL)
        flag_fix = rng.random((m, n_draws)) < FLAG_SUCCESS
        cat_fix = (rng.random((P, n_draws)) < CAT_SUCCESS)[pair_id]
        q_false = 1 - np.exp(-lam_hc[self.pair_hh, pair_cat])     # a false flag lands in this pair's category
        false_any = (rng.random((P, n_draws)) < q_false[:, None])
        to_pair = sp.csr_matrix((np.ones(m), (pair_id, np.arange(m))), shape=(P, m))
        real_in_pair = (to_pair @ flagged.astype(float)) > 0      # P x R
        to_hh = sp.csr_matrix((np.ones(m), (v.hh, np.arange(m))), shape=(n, m))
        no_real_flag_hh = (to_hh @ flagged.astype(float)) == 0     # n x R
        self._to_hh = sp.csr_matrix((v.dollars.to_numpy(), (v.hh, np.arange(m))), shape=(n, m))
        rk = v["rank"].to_numpy()[:, None]
        flag_corr = flagged & flag_fix

        def red(corrected, weight=None):
            """Expected reduction in counted error per household, given a corrected mask (m x R)."""
            remaining = self.fixed[:, None] + self._to_hh @ (~corrected).astype(float)
            counted = np.where(remaining > tol, remaining, 0.0)
            gain = self.base[:, None] - counted
            if weight is not None:
                gain = gain * weight
            return gain.mean(1)

        self.ks = range(1, K + 1)
        # risk only
        self.risk_red = {k: red(cat_fix & (rk < k)) for k in self.ks}
        # AI only: cumulative reduction after the first j categories (j = 1..K) of the state's order
        self.ai_red = {j: red(flag_corr & (rk < j)) for j in self.ks}
        self.ai_flags = n_det * FLAG_RECALL + lam_hc               # expected flags per household x category
        self.rank_hc = rank_hc
        # combined, main (Section 2.3). Real-flag reductions need no false-flag term; a case with no flag at
        # all also needs no false flag anywhere, probability exp(-sum lambda), independent of the draws.
        p_no_false = np.exp(-lam_hc.sum(1))
        self.comb1_red = red(flag_corr)
        self.comb1_time = FLAG_MINUTES * self.ai_flags.sum(1)
        p_no_flag = p_no_false * ((1 - FLAG_RECALL) ** n_det.sum(1))
        self.comb2_red = {k: red(cat_fix & (rk < k), no_real_flag_hh) * p_no_false for k in self.ks}
        self.comb2_time = {k: CAT_MINUTES * k * p_no_flag for k in self.ks}
        # combined, risk order throughout (Appendix D.9)
        flags_in_cat = real_in_pair[pair_id] | false_any[pair_id]
        d9_corr = (flagged & flag_fix & flags_in_cat) | (~flags_in_cat & cat_fix)
        self.d9_red = {k: red(d9_corr & (rk < k)) for k in self.ks}
        p_none_c = (1 - FLAG_RECALL) ** n_det * np.exp(-lam_hc)      # no flag at all in category
        per_cat = FLAG_MINUTES * self.ai_flags + CAT_MINUTES * p_none_c
        self.d9_time = {k: (per_cat * (rank_hc < k)).sum(1) for k in self.ks}

    # ------------------------------------------------------------ curves

    def _state(self, s):
        idx = np.flatnonzero(self.state == s)
        tot_err = (self.w[idx] * self.base[idx]).sum()
        tot_ben = (self.w[idx] * self.df.RAWBEN.to_numpy(float)[idx]).sum()
        samp = idx[self.in_sample[idx]]
        order = samp[np.argsort(-self.risk[samp], kind="stable")]
        return order, tot_err, tot_ben

    def _curve(self, s, units):
        """units: list of (weighted minutes, weighted reduction) arrays, in queue order.
        Returns annual state cost ($) and PER after each unit, starting at (0, PER before review)."""
        _, tot_err, tot_ben = self._state(s)
        minutes = np.cumsum(np.concatenate([[0.0]] + [u[0] for u in units]))
        gain = np.cumsum(np.concatenate([[0.0]] + [u[1] for u in units]))
        return minutes / 60 * STATE_COST_PER_HOUR, (tot_err - gain) / tot_ben

    def _unit(self, h, minutes, reduction):
        return minutes * self.w_cost[h], reduction * self.w[h]

    def curves(self, s, rule):
        """Dict scope -> (cost, PER) for one state and rule ('risk', 'ai', 'combined', 'combined_d9')."""
        o, _, _ = self._state(s)
        if rule == "risk":
            return {k: self._curve(s, [self._unit(o, np.full(len(o), CAT_MINUTES * k), self.risk_red[k][o])])
                    for k in self.ks}
        if rule == "combined":
            ph1 = self._unit(o, self.comb1_time[o], self.comb1_red[o])
            return {k: self._curve(s, [ph1, self._unit(o, self.comb2_time[k][o], self.comb2_red[k][o])])
                    for k in self.ks}
        if rule == "combined_d9":
            return {k: self._curve(s, [self._unit(o, self.d9_time[k][o], self.d9_red[k][o])]) for k in self.ks}
        if rule == "ai":
            # Flags within a category are checked in random order. In expectation, cost and reduction then rise
            # linearly inside a category, so one queue unit per category is exact at category ends and linear
            # in between.
            ranks = self.rank_hc[o[0]]
            units, prev = [], np.zeros(len(o))
            for j in self.ks:
                c = int(np.flatnonzero(ranks == j - 1)[0])
                t, r = self._unit(o, FLAG_MINUTES * self.ai_flags[o, c], self.ai_red[j][o] - prev)
                units.append((np.array([t.sum()]), np.array([r.sum()])))
                prev = self.ai_red[j][o]
            return {len(CATEGORIES): self._curve(s, units)}
        raise ValueError(rule)

    def reach_cost(self, s, rule):
        """Lowest annual state review cost at which PER falls below 6%, best scope chosen afterwards (Section 3).
        np.inf if no scope reaches it. Returns (cost, scope)."""
        best = (np.inf, None)
        for k, (cost, per) in self.curves(s, rule).items():
            hit = np.flatnonzero(per < THRESHOLD)
            if len(hit):
                i = hit[0]
                if i == 0:
                    c = 0.0
                else:  # linear inside the unit that crosses the threshold
                    c = cost[i - 1] + (cost[i] - cost[i - 1]) * (per[i - 1] - THRESHOLD) / (per[i - 1] - per[i])
                if c < best[0]:
                    best = (c, k)
        return best

    def envelope(self, s, rule, hours_grid):
        """Lowest PER across scopes at each review-hours level (hours per 1,000 scored households per year)."""
        order, _, _ = self._state(s)
        per_1000 = self.w_cost[order].sum() / 1000
        out = np.full(len(hours_grid), np.inf)
        for cost, per in self.curves(s, rule).values():
            hours = cost / STATE_COST_PER_HOUR / per_1000
            out = np.minimum(out, np.interp(hours_grid, hours, per, right=per[-1]))
        return out


# ---------------------------------------------------------------- risk score


def read_bgl_frame(path, extra=()) -> pd.DataFrame:
    """BGL's munged frame (additional_data/reg_model_data.rds): reconstructed pre-review inputs per case.
    `extra` names further columns to keep as text (for example `correctednotes`, BGL's label for what its
    reconstruction changed)."""
    import rdata

    df = rdata.read_rds(str(path))
    df.columns = [str(c) for c in df.columns]
    keep = ["hhldno", "state_name", "fiscal_year", "cert_HH_size_FS_n", "over_threshold",
            "total_error_amount", "hwgt", *FEATURES]
    df = df[keep + list(extra)].copy()
    for c in extra:
        df[c] = df[c].astype(str)
    df["homeless"] = df.homeless.astype(str).map({"TRUE": 1.0, "FALSE": 0.0})
    for c in keep[2:]:
        df[c] = pd.to_numeric(df[c].astype("object"), errors="coerce").astype(float)
    df["hhldno"] = pd.to_numeric(df.hhldno).astype("int64")
    df["state_name"] = df.state_name.astype(str)
    return df


STRATA = ["1", "2-3", "4+"]


def fit_risk_models(frame: pd.DataFrame, train_years=(2022, 2023), seed=117) -> dict:
    """One fitted model per household-size stratum (1, 2-3, 4+), trained on cases with complete inputs.

    Engine: scikit-learn HistGradientBoostingClassifier at BGL's xgboost settings (1,000 rounds, depth 4,
    learning rate 0.02). xgboost itself could not load here (no OpenMP runtime), and scikit-learn has no row
    subsampling, so BGL's subsample = 0.20 is not applied.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    ok = frame[FEATURES].notna().all(axis=1) & frame.over_threshold.notna()
    fr = frame[ok & frame.fiscal_year.isin(train_years)]
    stratum = hh_stratum(fr.cert_HH_size_FS_n)
    models = {}
    for s in STRATA:
        tr = stratum == s
        models[s] = HistGradientBoostingClassifier(
            max_iter=BGL_XGB["nrounds"], max_depth=BGL_XGB["max_depth"], learning_rate=BGL_XGB["eta"],
            max_leaf_nodes=None, l2_regularization=1.0, early_stopping=False, random_state=seed)
        models[s].fit(fr.loc[tr, FEATURES], fr.loc[tr, "over_threshold"].astype(int))
    return models


def score_risk(models: dict, features: pd.DataFrame, size: pd.Series) -> pd.Series:
    """Risk score for each row of `features` (the 19 inputs), from the model of its household-size stratum.
    NaN where an input is missing. Indexed like `features`."""
    out = pd.Series(np.nan, index=features.index, name="risk")
    ok = features[FEATURES].notna().all(axis=1).to_numpy()
    stratum = hh_stratum(pd.Series(np.asarray(size, float), index=features.index))
    for s in STRATA:
        m = ok & (stratum == s).to_numpy()
        if m.any():
            out[m] = models[s].predict_proba(features.loc[m, FEATURES])[:, 1]
    return out


def fit_risk(frame: pd.DataFrame, train_years=(2022, 2023), score_year=2024, seed=117, models=None) -> pd.Series:
    """Risk score for score_year cases with complete inputs, from `fit_risk_models` (fitted here unless
    `models` is given). Returns a Series indexed by hhldno."""
    models = models or fit_risk_models(frame, train_years, seed)
    fr = frame[(frame.fiscal_year == score_year) & frame.over_threshold.notna()]
    risk = score_risk(models, fr[FEATURES], fr.cert_HH_size_FS_n)
    return pd.Series(risk.to_numpy(), index=fr.hhldno.to_numpy(), name="risk").dropna()
