"""Toy review simulator: a caseload of household x fact-group pairs, every review rule as a queue of checks,
and one engine that spends a budget of reviewer minutes down a queue.

The review model is the one in `review_sim` (Chohlas-Wood 2026, Sections 2.1 and 2.2): a flag check takes 5
minutes and corrects a flagged error with probability 0.8; a category check takes 10 minutes and corrects the
errors in that fact group with probability 0.5. Each pair is checked at most once by a rule. Whether a check
succeeds is drawn once per pair and check type and stored on the caseload, so every rule faces the same luck.

Rules (each returns a Queue):
- random_order: every pair, in random order;
- risk_only: category checks in risk order, flags ignored;
- flags_only: flag checks only, fact group by fact group;
- gold_standard: the paper's combined rule (Section 2.3);
- combined_risk_order: the paper's alternative combined rule (Appendix D.9);
- dollars_per_minute: one queue of flag and category checks, ordered by expected error dollars corrected per
  reviewer minute, from rates given per fact group (`true_rates`) or per pair (`toy_posterior_rates`);
- oracle: knows where every error is and how large, not whether a check will succeed.

Capture is the error dollars a rule corrects at a budget divided by the dollars the oracle corrects at the
same budget (per `rq.md`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .review_sim import CAT_MINUTES, CAT_SUCCESS, FLAG_MINUTES, FLAG_PRECISION, FLAG_RECALL, FLAG_SUCCESS

GROUPS = ["earned", "unearned", "shelter", "utilities"]
CATEGORY, FLAG = 0, 1  # kinds of check


@dataclass
class Caseload:
    """n households x G fact groups. Arrays are (n, G) unless noted."""
    error: np.ndarray       # error dollars in the pair, 0 where there is none
    flag: np.ndarray        # bool: the pair carries a flag
    documented: np.ndarray  # bool: the packet holds a document for the fact group
    risk: np.ndarray        # (n,) risk score, higher = reviewed earlier
    flag_ok: np.ndarray     # bool: a check priced as a flag check succeeds on this pair
    cat_ok: np.ndarray      # bool: a check priced as a category check succeeds on this pair

    @property
    def n(self):
        return self.error.shape[0]


@dataclass
class Queue:
    """Checks in the order a rule makes them: household row, fact-group column, kind (CATEGORY or FLAG)."""
    hh: np.ndarray
    group: np.ndarray
    kind: np.ndarray

    def __post_init__(self):
        self.hh, self.group, self.kind = (np.asarray(x, int) for x in (self.hh, self.group, self.kind))
        pair = self.hh * (self.group.max(initial=0) + 1) + self.group
        assert len(np.unique(pair)) == len(pair), "a pair is checked at most once"


def toy_caseload(rng, n=400, prevalence=0.2, group_share=(0.35, 0.22, 0.32, 0.11), error_dollars=(100, 200, 400),
                 risk_auc=0.75, undocumented_error_share=0.93, recall=FLAG_RECALL, precision=FLAG_PRECISION) -> Caseload:
    """A synthetic caseload with planted errors and synthetic flags.

    - Each household has one error with probability `prevalence`, in a fact group drawn from `group_share`, with
      dollars drawn from `error_dollars`.
    - Risk score: standard normal, shifted up for error households so that its AUC is `risk_auc`.
    - Flags: each error is flagged with probability `recall`; false flags fall on error-free pairs at the rate
      that makes the expected share of real flags equal `precision`.
    - Documents: an error household lacks the document for its error's group with probability
      `undocumented_error_share`, and otherwise has every document; an error-free household lacks the document
      for one group drawn from `group_share`.
    """
    G = len(group_share)
    share = np.asarray(group_share, float) / np.sum(group_share)
    has_error = rng.random(n) < prevalence
    group = rng.choice(G, size=n, p=share)
    error = np.zeros((n, G))
    error[has_error, group[has_error]] = rng.choice(np.asarray(error_dollars, float), size=int(has_error.sum()))
    risk = rng.normal(size=n) + np.sqrt(2) * norm.ppf(risk_auc) * has_error

    real = (error > 0) & (rng.random((n, G)) < recall)
    clean = error == 0
    q_false = (error > 0).sum() * recall * (1 - precision) / precision / max(clean.sum(), 1)
    flag = real | (clean & (rng.random((n, G)) < q_false))

    documented = np.ones((n, G), bool)
    missing = np.where(has_error, group, rng.choice(G, size=n, p=share))
    lacks = ~has_error | (rng.random(n) < undocumented_error_share)
    documented[np.flatnonzero(lacks), missing[lacks]] = False
    return Caseload(error, flag, documented, risk,
                    flag_ok=rng.random((n, G)) < FLAG_SUCCESS, cat_ok=rng.random((n, G)) < CAT_SUCCESS)


# ---------------------------------------------------------------- engine


def priced_as_category(c: Caseload, hh, group, kind, undocumented_flag_as="flag"):
    """True where the check costs 10 minutes and succeeds with probability 0.5: every category check, and a
    flag check on an undocumented fact group when such flags are priced as category checks."""
    assert undocumented_flag_as in ("flag", "category")
    as_cat = np.asarray(kind) == CATEGORY
    if undocumented_flag_as == "category":
        as_cat = as_cat | ~c.documented[hh, group]
    return as_cat


def spend(c: Caseload, queue: Queue, budgets, undocumented_flag_as="flag", expected=False) -> dict:
    """Work down the queue with each budget (total reviewer minutes; scalar or array).

    The check the budget runs out on is credited in proportion to the minutes spent on it. Returns arrays
    over budgets: dollars corrected, minutes used, checks made and flag checks made. With `expected=True` the
    stored luck is replaced by the success probabilities.
    """
    budgets = np.atleast_1d(np.asarray(budgets, float))
    as_cat = priced_as_category(c, queue.hh, queue.group, queue.kind, undocumented_flag_as)
    minutes = np.where(as_cat, CAT_MINUTES, FLAG_MINUTES).astype(float)
    if expected:
        ok = np.where(as_cat, CAT_SUCCESS, FLAG_SUCCESS)
    else:
        ok = np.where(as_cat, c.cat_ok[queue.hh, queue.group], c.flag_ok[queue.hh, queue.group])
    gain = ok * c.error[queue.hh, queue.group]
    before = np.concatenate([[0.0], np.cumsum(minutes)[:-1]])
    done = np.clip((budgets[:, None] - before[None, :]) / minutes[None, :], 0, 1)  # budgets x checks
    return {
        "corrected": done @ gain,
        "minutes": done @ minutes,
        "checks": done.sum(1),
        "flag_checks": done @ (queue.kind == FLAG).astype(float),
    }


# ---------------------------------------------------------------- rules


def _by_risk(c: Caseload):
    return np.argsort(-c.risk, kind="stable")


def _pairs(hh_order, groups, keep):
    """Pairs (household, group) for households in `hh_order` and groups in `groups` order, where keep[h, g]."""
    hh = np.repeat(hh_order, len(groups))
    gr = np.tile(np.asarray(groups, int), len(hh_order))
    m = keep[hh, gr]
    return hh[m], gr[m]


def random_order(c: Caseload, rng) -> Queue:
    """Every pair in random order: a flag check where there is a flag, a category check elsewhere."""
    n, G = c.error.shape
    p = rng.permutation(n * G)
    hh, gr = p // G, p % G
    return Queue(hh, gr, np.where(c.flag[hh, gr], FLAG, CATEGORY))


def risk_only(c: Caseload, scope, group_order) -> Queue:
    """Households in risk order; a category check on each of the first `scope` fact groups. Flags are ignored."""
    hh, gr = _pairs(_by_risk(c), group_order[:scope], np.ones_like(c.flag))
    return Queue(hh, gr, np.full(len(hh), CATEGORY))


def flags_only(c: Caseload, rng, group_order) -> Queue:
    """Flag checks only: fact groups in `group_order`, flags inside a group in random order."""
    hh, gr = [], []
    for g in group_order:
        h = rng.permutation(np.flatnonzero(c.flag[:, g]))
        hh.append(h)
        gr.append(np.full(len(h), g))
    hh, gr = np.concatenate(hh), np.concatenate(gr)
    return Queue(hh, gr, np.full(len(hh), FLAG))


def gold_standard(c: Caseload, scope, group_order) -> Queue:
    """The paper's combined rule: every flag check first, households in risk order; then category checks on
    the first `scope` fact groups of households with no flag, in risk order."""
    order = _by_risk(c)
    f_hh, f_gr = _pairs(order, group_order, c.flag)
    unflagged = order[~c.flag[order].any(axis=1)]
    c_hh, c_gr = _pairs(unflagged, group_order[:scope], np.ones_like(c.flag))
    return Queue(np.concatenate([f_hh, c_hh]), np.concatenate([f_gr, c_gr]),
                 np.concatenate([np.full(len(f_hh), FLAG), np.full(len(c_hh), CATEGORY)]))


def combined_risk_order(c: Caseload, scope, group_order) -> Queue:
    """The paper's Appendix D.9 rule: households in risk order; in each of the first `scope` fact groups, a
    flag check where there is a flag and a category check otherwise."""
    hh, gr = _pairs(_by_risk(c), group_order[:scope], np.ones_like(c.flag))
    return Queue(hh, gr, np.where(c.flag[hh, gr], FLAG, CATEGORY))


def dollars_per_minute(c: Caseload, precision, miss_rate, mean_dollars, undocumented_flag_as="flag") -> Queue:
    """One queue of every flag check and every category check, ordered by expected dollars corrected per minute.

    `precision` is the share of flagged pairs that hold an error, `miss_rate` the share of unflagged pairs that
    hold one, `mean_dollars` the mean error size; each is a scalar or one value per fact group (or per pair).
    Expected dollars = rate x mean dollars x success probability; ties are broken by risk score.
    """
    n, G = c.error.shape
    hh, gr = np.divmod(np.arange(n * G), G)
    kind = np.where(c.flag[hh, gr], FLAG, CATEGORY)
    full = lambda x: np.broadcast_to(np.asarray(x, float), (n, G))[hh, gr]
    rate = np.where(kind == FLAG, full(precision), full(miss_rate))
    as_cat = priced_as_category(c, hh, gr, kind, undocumented_flag_as)
    value = rate * full(mean_dollars) * np.where(as_cat, CAT_SUCCESS / CAT_MINUTES, FLAG_SUCCESS / FLAG_MINUTES)
    o = np.lexsort((-c.risk[hh], -value))
    return Queue(hh[o], gr[o], kind[o])


def oracle(c: Caseload, undocumented_flag_as="flag") -> Queue:
    """Only the pairs that hold an error, largest expected dollars corrected per minute first. The check is a
    flag check where the pair is flagged and a category check otherwise."""
    hh, gr = np.nonzero(c.error > 0)
    kind = np.where(c.flag[hh, gr], FLAG, CATEGORY)
    as_cat = priced_as_category(c, hh, gr, kind, undocumented_flag_as)
    value = c.error[hh, gr] * np.where(as_cat, CAT_SUCCESS / CAT_MINUTES, FLAG_SUCCESS / FLAG_MINUTES)
    o = np.argsort(-value, kind="stable")
    return Queue(hh[o], gr[o], kind[o])


def true_rates(c: Caseload) -> dict:
    """Per fact group, from the caseload itself: flag precision, miss rate (share of unflagged pairs with an
    error) and mean error dollars."""
    err = c.error > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "precision": np.nan_to_num((err & c.flag).sum(0) / c.flag.sum(0)),
            "miss_rate": np.nan_to_num((err & ~c.flag).sum(0) / (~c.flag).sum(0)),
            "mean_dollars": np.nan_to_num(c.error.sum(0) / err.sum(0)),
        }


def toy_posterior_rates(c: Caseload, prevalence=0.2, group_share=(0.35, 0.22, 0.32, 0.11), risk_auc=0.75,
                        recall=FLAG_RECALL, precision=FLAG_PRECISION) -> dict:
    """Per pair, the probability that it holds an error given the household's risk score and the pair's own
    flag, under the model `toy_caseload` draws from (same arguments). Returned as `precision` (used where the
    pair is flagged) and `miss_rate` (where it is not), both (n, G). Flags on the household's other fact groups
    are not used.
    """
    share = np.asarray(group_share, float) / np.sum(group_share)
    shift = np.sqrt(2) * norm.ppf(risk_auc)
    odds = prevalence / (1 - prevalence) * np.exp(shift * c.risk - shift ** 2 / 2)  # error household vs not, given risk
    p = (odds / (1 + odds))[:, None] * share[None, :]                               # error in this pair
    q_false = (c.error > 0).sum() * recall * (1 - precision) / precision / max((c.error == 0).sum(), 1)
    return {
        "precision": p * recall / (p * recall + (1 - p) * q_false),
        "miss_rate": p * (1 - recall) / (p * (1 - recall) + (1 - p) * (1 - q_false)),
    }
