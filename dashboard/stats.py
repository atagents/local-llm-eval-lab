"""Statistical rigor for eval scores — turns bare means into CIs + significance.

Eval scores are random variables (judge noise dominates on small sets). A "+0.3 on
40 cases" is usually not real. This module answers "is that delta real?" with a
bootstrap confidence interval and a PAIRED significance test, and gates ship
decisions on the CI lower bound — not the point estimate.

No LLM, no network, no GPU. scipy (BCa bootstrap, Wilcoxon, permutation) + statsmodels
(McNemar, Wilson, Holm), both BSD. Consumed by the dashboard Overview/Compare tabs,
the regression CI gate (#7), and report generation.

Self-test is the success criterion: run `python stats.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

_N_BOOT = 10000


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n: int
    method: str
    note: str = ""

    def fmt(self) -> str:
        return f"{self.point:.2f} [{self.lo:.2f}, {self.hi:.2f}] (n={self.n})"

    def as_dict(self) -> dict:
        return asdict(self)


def mean_ci(scores, kind: str = "score", confidence: float = 0.95) -> CI:
    """Mean (or pass-rate) with a 95% CI.
    kind='rate' -> Wilson interval on a 0/1 proportion; else BCa bootstrap on the mean.
    Degenerate inputs (n<2 or zero variance) return a zero-width CI with a note."""
    x = np.asarray([s for s in scores if s is not None], dtype=float)
    n = int(x.size)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0, "none", "no data")
    point = float(x.mean())
    if n < 2 or np.allclose(x, x[0]):
        return CI(point, point, point, n, "degenerate", "n<2 or zero variance")

    if kind == "rate":
        k = int(round(x.sum()))
        lo, hi = proportion_confint(k, n, alpha=1 - confidence, method="wilson")
        return CI(point, float(lo), float(hi), n, "wilson")
    try:
        res = stats.bootstrap((x,), np.mean, confidence_level=confidence,
                              n_resamples=_N_BOOT, method="BCa")
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("non-finite BCa interval")
        return CI(point, lo, hi, n, "bootstrap-BCa")
    except Exception:  # noqa: BLE001  (e.g. degenerate acceleration) -> percentile fallback
        boot = np.array([np.mean(np.random.default_rng(i).choice(x, n, replace=True))
                         for i in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return CI(point, float(lo), float(hi), n, "bootstrap-pct", "BCa fell back to percentile")


@dataclass
class Compare:
    delta: float          # mean(candidate) - mean(baseline); >0 = candidate better
    ci_lo: float
    ci_hi: float
    p: float
    test: str
    verdict: str          # IMPROVE | FLAT | REGRESS
    n: int
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _paired(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"compare_two needs PAIRED arrays (join by case_id); got {a.shape} vs {b.shape}")
    return a, b


def compare_two(candidate, baseline, *, binary: bool = False,
                floor: float = 0.0, confidence: float = 0.95) -> Compare:
    """Paired comparison of `candidate` vs `baseline` (same length, aligned by case_id).
    delta>0 means candidate scored higher. Verdict gates on the CI bound, not the point:
      IMPROVE  if ci_lo >  floor      (real, meaningful gain)
      REGRESS  if ci_hi < -floor
      FLAT     otherwise              (could be judge noise)
    binary=True: 0/1 outcomes -> McNemar (exact when discordant < 25)."""
    a, b = _paired(candidate, baseline)
    n = int(a.size)
    delta = float(a.mean() - b.mean())
    d = a - b

    if np.allclose(d, 0):
        return Compare(0.0, 0.0, 0.0, 1.0, "none", "FLAT", n, "identical paired scores")

    # CI on the paired difference (BCa bootstrap; percentile fallback)
    try:
        res = stats.bootstrap((d,), np.mean, confidence_level=confidence,
                              n_resamples=_N_BOOT, method="BCa")
        ci_lo, ci_hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
            raise ValueError("non-finite")
    except Exception:  # noqa: BLE001
        boot = np.array([np.mean(np.random.default_rng(i).choice(d, n, replace=True))
                         for i in range(2000)])
        ci_lo, ci_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

    if binary:
        ai, bi = (a > 0.5).astype(int), (b > 0.5).astype(int)
        n11 = int(np.sum((ai == 1) & (bi == 1)))
        n10 = int(np.sum((ai == 1) & (bi == 0)))
        n01 = int(np.sum((ai == 0) & (bi == 1)))
        n00 = int(np.sum((ai == 0) & (bi == 0)))
        discordant = n10 + n01
        res = mcnemar([[n11, n10], [n01, n00]], exact=discordant < 25)
        p, test = float(res.pvalue), f"mcnemar({'exact' if discordant < 25 else 'chi2'})"
    else:
        pt = stats.permutation_test((a, b), lambda x, y: np.mean(x - y),
                                    permutation_type="samples", n_resamples=_N_BOOT,
                                    alternative="two-sided")
        p, test = float(pt.pvalue), "permutation(paired)"

    verdict = "IMPROVE" if ci_lo > floor else "REGRESS" if ci_hi < -floor else "FLAT"
    return Compare(delta, ci_lo, ci_hi, p, test, verdict, n)


def holm(pvals) -> dict:
    """Holm-Bonferroni correction for >2 comparisons. Returns adjusted p + reject mask."""
    reject, p_adj, _, _ = multipletests(list(pvals), method="holm")
    return {"p_adjusted": [float(x) for x in p_adj], "reject": [bool(x) for x in reject]}


def underpowered(point: float, n: int, half_width_max: float = 0.1) -> bool:
    """Crude power flag for a rate: 95% CI half-width ~ 1.96*sqrt(p(1-p)/n)."""
    if n < 1:
        return True
    p = min(max(point, 0.0), 1.0)
    half = 1.96 * (p * (1 - p) / n) ** 0.5
    return half > half_width_max


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # A) real +0.5 lift on n=200 -> IMPROVE, CI lower bound > 0
    base = np.clip(rng.normal(3.0, 1.0, 200), 1, 5)
    cand = np.clip(base + 0.5 + rng.normal(0, 0.3, 200), 1, 5)
    A = compare_two(cand, base)
    print("A (+0.5, n=200):", A.verdict, f"delta={A.delta:.3f} CI=[{A.ci_lo:.3f},{A.ci_hi:.3f}] "
          f"p={A.p:.4f} {A.test}")
    assert A.verdict == "IMPROVE" and A.ci_lo > 0, A

    # B) tiny +0.05 on n=20 -> FLAT, CI crosses 0 (judge noise)
    base2 = np.clip(rng.normal(3.0, 1.0, 20), 1, 5)
    cand2 = np.clip(base2 + 0.05 + rng.normal(0, 1.0, 20), 1, 5)
    B = compare_two(cand2, base2)
    print("B (+0.05, n=20): ", B.verdict, f"delta={B.delta:.3f} CI=[{B.ci_lo:.3f},{B.ci_hi:.3f}] "
          f"p={B.p:.4f} {B.test}")
    assert B.verdict == "FLAT" and B.ci_lo <= 0 <= B.ci_hi, B

    # C) binary pass-rate, clear paired improvement -> McNemar IMPROVE
    bb = (rng.random(80) < 0.6).astype(float)
    cc = bb.copy()
    flip = np.where(bb == 0)[0][:15]      # turn 15 fails into passes, none the other way
    cc[flip] = 1.0
    C = compare_two(cc, bb, binary=True)
    print("C (binary +15): ", C.verdict, f"delta={C.delta:.3f} CI=[{C.ci_lo:.3f},{C.ci_hi:.3f}] "
          f"p={C.p:.4f} {C.test}")
    assert C.verdict == "IMPROVE" and C.p < 0.05, C

    # D) mean_ci + underpowered flag
    m = mean_ci([5, 4, 5, 3, 5, 4, 5, 5])
    print("D mean_ci:", m.fmt(), m.method)
    r = mean_ci([1, 1, 0, 1, 1, 0], kind="rate")
    print("D rate_ci:", r.fmt(), r.method, "| underpowered(n=6)?", underpowered(r.point, r.n))
    h = holm([0.001, 0.04, 0.20])
    print("D holm:", h)

    print("\nstats.py self-test: PASS  (+0.5/n=200 -> IMPROVE ci_lo>0; +0.05/n=20 -> FLAT crosses 0; "
          "binary -> McNemar IMPROVE)")
