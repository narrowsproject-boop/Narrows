"""Final comparison on every week since Aug 2024 (no week dropped), Rotterdam and Genoa.

Fixes an over-confidence problem of ridge->classes: it turned in-sample residuals into
probabilities, which are too small when training is short. Here residuals are
leave-one-out residuals, e_i / (1 - h_ii), the error the model would have made on
week i had it not seen it (closed form for ridge).
Candidates: ridge LOO (pooled), multinomial logistic (pooled), their average, and the
current champion from champion_record.csv (same weeks).

Usage: python scripts/final_comparison.py -> data/model/final_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narrows.common import DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import build, load_config  # noqa: E402

OUT = DATA / "model"
R = ["ec_premium", "ec_4w", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"]
TARGETS = ["shanghai_rotterdam", "shanghai_genoa"]
START = pd.Timestamp("2024-06-01")


def ridge_probs(X, y, x, band, alpha=3.0, loo=True):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd]); z = np.concatenate([[1.0], (x - mu) / sd])
    P = np.eye(Z.shape[1]) * alpha; P[0, 0] = 0  # no penalty on the intercept
    A = np.linalg.inv(Z.T @ Z + P)
    beta = A @ Z.T @ y
    res = y - Z @ beta
    if loo:
        hii = np.einsum("ij,jk,ik->i", Z, A, Z)
        res = res / (1 - hii)
    sims = z @ beta + res
    p = np.array([(sims < -band).mean(), ((sims >= -band) & (sims <= band)).mean(), (sims > band).mean()])
    return p, beta, mu, sd


def run():
    cfg = load_config()
    df = build(); df = df[~df["covid"]]
    rows, coefs = [], []
    for target in TARGETS:
        other = [t for t in TARGETS if t != target][0]
        for h in cfg["horizons_weeks"]:
            yc = f"y_{h}w"; band = cfg["stability_band_pct"][target][str(h)]; band_o = cfg["stability_band_pct"][other][str(h)]
            g = df[df.target == target].sort_values("week"); go = df[df.target == other]
            for _, now in g[(g.week >= START) & g[yc].notna()].iterrows():
                t = now.week
                if now[R].isna().any():
                    continue
                a = g[(g.week + pd.Timedelta(weeks=h) <= t) & g[yc].notna()].dropna(subset=R)
                b = go[(go.week + pd.Timedelta(weeks=h) <= t) & go[yc].notna()].dropna(subset=R)
                if len(a) < 40:
                    continue
                X = np.vstack([np.hstack([a[R].to_numpy(float), np.zeros((len(a), 1))]), np.hstack([b[R].to_numpy(float), np.ones((len(b), 1))])])
                y = np.concatenate([a[yc].to_numpy(float), b[yc].to_numpy(float)])
                c = np.concatenate([E.classify(a[yc].to_numpy(float), band), E.classify(b[yc].to_numpy(float), band_o)])
                x = np.append(now[R].to_numpy(float), 0.0)
                actual = float(now[yc]); k = int(E.classify(np.array([actual]), band)[0])
                base = {"target": target, "h": h, "week": t, "actual": actual, "class": k, "n_train": len(a)}
                p_old, *_ = ridge_probs(X, y, x, band, loo=False)
                p_loo, beta, mu, sd = ridge_probs(X, y, x, band, loo=True)
                mu2, sd2 = X.mean(0), X.std(0) + 1e-9
                ml = LogisticRegression(C=0.5, max_iter=4000).fit((X - mu2) / sd2, c)
                p_ml = np.zeros(3); p_ml[ml.classes_.astype(int)] = ml.predict_proba(((x - mu2) / sd2).reshape(1, -1))[0]
                for name, p in (("ridge pooled (old, in-sample residuals)", p_old), ("ridge pooled LOO", p_loo),
                                ("multilogit pooled", p_ml), ("average ridge LOO + multilogit", (p_loo + p_ml) / 2)):
                    rows.append({**base, "method": name, "p_down": p[0], "p_stable": p[1], "p_up": p[2]})
                coefs.append({"target": target, "h": h, "week": t, **dict(zip(["intercept"] + R + ["is_other_route"], beta))})
    d = pd.DataFrame(rows)
    ch = pd.read_csv(OUT / "champion_record.csv", parse_dates=["week"])
    ch = ch[(ch.version == "with EC") & ch.target.isin(TARGETS)].assign(method="champion (logit+boosting+forest)")
    nc = ch.assign(method="ref no change")
    d = pd.concat([d, ch[d.columns.intersection(ch.columns)], nc[["target", "h", "week", "actual", "class", "method"]]], ignore_index=True)
    return d, pd.DataFrame(coefs), ch


if __name__ == "__main__":
    d, coefs, ch = run()
    # no-change reference probabilities = the champion record's climatology column
    P = d[["p_down", "p_stable", "p_up"]].to_numpy()
    d["rps"] = [E.rps(p, c) if np.isfinite(p).all() else np.nan for p, c in zip(P, d["class"])]
    ref = ch.set_index(["target", "h", "week"])["rps_no_change"]
    m = d["method"] == "ref no change"
    d.loc[m, "rps"] = d[m].set_index(["target", "h", "week"]).index.map(ref).to_numpy()
    d["hit"] = np.where(m, d["class"] == 1, P.argmax(1) == d["class"].to_numpy())
    full = d.groupby(["target", "h", "week"])["method"].transform("nunique")
    d = d[full == full.max()]
    d.to_csv(OUT / "final_forecasts.csv", index=False, float_format="%.5f")
    coefs.to_csv(OUT / "final_ridge_coefficients.csv", index=False, float_format="%.4f")
    d["period"] = pd.cut(d["week"], [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), pd.Timestamp("2025-06-30"),
                                     pd.Timestamp("2025-12-31"), pd.Timestamp("2027-01-01")],
                         labels=["2024 (cold start)", "2025 H1", "2025 H2", "2026"])
    s = d.pivot_table(index="method", columns="period", values="rps", aggfunc="mean", observed=True)
    s["all weeks"] = d.groupby("method")["rps"].mean()
    s["hit all"] = d.groupby("method")["hit"].mean()
    per = d.pivot_table(index="method", columns=["target", "h"], values="rps", aggfunc="mean")
    per.columns = [f"{a.split('_')[1]} {b}w" for a, b in per.columns]
    out = s.join(per).sort_values("all weeks")
    out.to_csv(OUT / "final_summary.csv", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    print("weeks per series:", d.groupby(["target", "h"])["week"].nunique().to_dict())
