"""Forecast engines for Narrows v0.1.

Direction (down / stable / up) as three probabilities:
- Climatology: historical frequencies ("nothing changes" reference).
- Engine A, rules: each rule is a condition; its historical outcome frequencies are
  shrunk toward the base rates with weight k, then active rules are averaged,
  weighted by their number of past cases.
- Engine B, ordinal logistic regression with an L2 penalty on standardised inputs:
  P(down) = s(c1 - x.b), P(down or stable) = s(c2 - x.b).

Magnitude (median and 80% range of the % move):
- A: the outcomes of the k most similar past weeks (nearest neighbours).
- B: linear quantile regression, one line per quantile (10%, 50%, 90%).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import QuantileRegressor

QUANTILES = (0.1, 0.5, 0.9)

# Rule book v0.1 (thresholds in %). Written before the backtest, from the
# 9 Nov exploration; to be reviewed by Ethan.
RULES = {
    "R-02a EC fell >10% in 4 weeks": lambda f: f["ec_4w"] < -10,
    "R-02b EC rose >8% in 1 week": lambda f: f["ec_1w"] > 8,
    "R-02c EC fell >8% in 1 week": lambda f: f["ec_1w"] < -8,
    "R-03a price rose >5% in 4 weeks": lambda f: f["mom_4w"] > 5,
    "R-03b price fell >5% in 4 weeks": lambda f: f["mom_4w"] < -5,
    "R-04 Bab el-Mandeb traffic -20% vs last year": lambda f: f["traffic_bab"] < -20,
    "R-05 Suez traffic +10% vs last year": lambda f: f["traffic_suez"] > 10,
    "R-06 Red Sea / Aden hostile news > 3 sd above normal": lambda f: f["news_redsea"] > 3,
    "R-07 Hormuz hostile news > 3 sd above normal": lambda f: f["news_hormuz"] > 3,
}


def classify(y: np.ndarray, band: float) -> np.ndarray:
    return np.where(y > band, 2, np.where(y < -band, 0, 1))


def freqs(c: np.ndarray) -> np.ndarray:
    return np.bincount(c.astype(int), minlength=3) / max(len(c), 1)


# --- direction -----------------------------------------------------------------

def climatology(train_c: np.ndarray) -> np.ndarray:
    return freqs(train_c)


def rules_engine(train: pd.DataFrame, train_c: np.ndarray, now: pd.Series, k: float = 30.0):
    base = freqs(train_c)
    probs, weights, active = [], [], []
    for name, cond in RULES.items():
        try:
            if not bool(cond(now)):
                continue
        except (KeyError, TypeError):
            continue
        m = cond(train).fillna(False).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        p = (n * freqs(train_c[m]) + k * base) / (n + k)
        probs.append(p); weights.append(n); active.append(name)
    if not probs:
        return base, []
    return np.average(probs, axis=0, weights=weights), active


class OrdinalLogit:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, c: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        p = Z.shape[1]

        def nll(theta):
            b, c1, gap = theta[:p], theta[p], np.exp(theta[p + 1])
            s = Z @ b
            P0 = expit(c1 - s)
            P01 = expit(c1 + gap - s)
            pr = np.choose(c, [P0, P01 - P0, 1 - P01])
            return -np.log(np.clip(pr, 1e-9, None)).sum() + self.l2 * (b ** 2).sum()

        x0 = np.zeros(p + 2); x0[p] = -0.5
        self.theta = minimize(nll, x0, method="L-BFGS-B").x
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = len(self.mu)
        z = (x - self.mu) / self.sd
        s = z @ self.theta[:p]
        c1, c2 = self.theta[p], self.theta[p] + np.exp(self.theta[p + 1])
        P0, P01 = expit(c1 - s), expit(c2 - s)
        return np.array([P0, P01 - P0, 1 - P01])

    def weights(self, names):
        return dict(zip(names, np.round(self.theta[:len(names)], 3)))


# --- magnitude -----------------------------------------------------------------

def knn_quantiles(X: np.ndarray, y: np.ndarray, x: np.ndarray, k: int = 40) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    d = (((X - mu) / sd - (x - mu) / sd) ** 2).sum(1)
    idx = np.argsort(d)[:min(k, len(y))]
    return np.quantile(y[idx], QUANTILES)


def quantile_regression(X: np.ndarray, y: np.ndarray, x: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z, z = (X - mu) / sd, ((x - mu) / sd).reshape(1, -1)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in QUANTILES:
            m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(Z, y)
            out.append(m.predict(z)[0])
    return np.sort(np.array(out))


# --- scores --------------------------------------------------------------------

def rps(p: np.ndarray, c: int) -> float:
    """Ranked probability score for 3 ordered classes: 0 is perfect, lower is better."""
    o = np.zeros(3); o[c] = 1
    return float(((np.cumsum(p)[:2] - np.cumsum(o)[:2]) ** 2).sum() / 2)


def pinball(q: np.ndarray, y: float) -> float:
    return float(np.mean([max(t * (y - v), (t - 1) * (y - v)) for t, v in zip(QUANTILES, q)]))
