"""Full walk-forward record of the champion model on every tracked index.

Champion = average of three models (ordinal logit l2=10, gradient boosting, random
forest). Magnitude = average of kNN (k=40), quantile regression and boosting quantile.
Two versions, because EC futures only exist since Aug 2023:
- "with EC" (the champion): reduced inputs, forecasts from mid-2024 (40 weeks to learn);
- "without EC": socle inputs (trend, straits traffic, Brent, control route), from 2020.
Covid regime (Jul 2020 - Jun 2022) excluded. No leakage: training uses only weeks whose
outcome was known on the forecast Thursday. Trees refitted every 2 weeks.

Usage: python scripts/champion_report.py  -> data/model/champion_record*.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import QuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narrows.common import DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import build, load_config  # noqa: E402

warnings.filterwarnings("ignore")
OUT = DATA / "model"
REFIT = 2
TARGETS = ["shanghai_rotterdam", "shanghai_genoa", "shanghai_los_angeles", "ec_front"]
SOCLE = ["mom_1w", "mom_4w", "traffic_bab", "traffic_suez", "traffic_cape", "brent_4w", "control_4w"]


def inputs(target: str, version: str) -> list[str]:
    if version == "with EC":
        if target == "ec_front":
            return ["ec_premium", "ec_slope", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"]
        return ["ec_premium", "ec_4w", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"]
    cols = SOCLE if target != "shanghai_los_angeles" else [c for c in SOCLE if c != "control_4w"]
    return cols


def run(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for version in ("with EC", "without EC"):
        for target in TARGETS:
            if version == "without EC" and target == "ec_front":
                continue
            g = df[df["target"] == target].sort_values("week").reset_index(drop=True)
            cols = inputs(target, version)
            for h in cfg["horizons_weeks"]:
                band = cfg["stability_band_pct"][target][str(h)]
                ycol = f"y_{h}w"
                print(version, target, h, flush=True)
                trees, last = None, None
                for _, now in g.iterrows():
                    t = now["week"]
                    if pd.isna(now[ycol]) or now[cols].isna().any():
                        continue
                    tr = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()].dropna(subset=cols)
                    if len(tr) < (40 if version == "with EC" else 50):
                        continue
                    X, y = tr[cols].to_numpy(float), tr[ycol].to_numpy(float)
                    c = E.classify(y, band)
                    if len(np.unique(c)) < 3:
                        continue
                    x = now[cols].to_numpy(float)
                    p1 = E.OrdinalLogit(l2=10).fit(X, c).predict(x)
                    if last is None or (t - last).days >= 7 * REFIT:
                        last = t
                        b = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                           l2_regularization=1.0, min_samples_leaf=10).fit(X, c)
                        f = RandomForestClassifier(n_estimators=150, min_samples_leaf=8, max_features="sqrt",
                                                   random_state=0, n_jobs=2).fit(X, c)
                        hq = [HistGradientBoostingRegressor(loss="quantile", quantile=a, max_depth=2, learning_rate=0.05,
                                                            max_iter=150, min_samples_leaf=10).fit(X, y) for a in E.QUANTILES]
                        mu, sd = X.mean(0), X.std(0) + 1e-9
                        qr = [QuantileRegressor(quantile=a, alpha=0.05, solver="highs").fit((X - mu) / sd, y) for a in E.QUANTILES]
                        trees = (b, f, hq, mu, sd, qr)
                    b, f, hq, mu, sd, qr = trees

                    def pp(m):
                        p = np.zeros(3); p[m.classes_.astype(int)] = m.predict_proba(x.reshape(1, -1))[0]
                        return p
                    p = (p1 + pp(b) + pp(f)) / 3
                    q_knn = E.knn_quantiles(X, y, x, k=40)
                    q_hgb = np.sort([m.predict(x.reshape(1, -1))[0] for m in hq])
                    q_qr = np.sort([m.predict(((x - mu) / sd).reshape(1, -1))[0] for m in qr])
                    q = (q_knn + q_hgb + q_qr) / 3
                    clim = E.climatology(c)
                    trend = float(now["mom_4w"]) * h / 4
                    p_tr = np.full(3, 0.2); p_tr[int(E.classify(np.array([trend]), band)[0])] = 0.6
                    actual = float(now[ycol])
                    k = int(E.classify(np.array([actual]), band)[0])
                    rows.append({"version": version, "target": target, "h": h, "week": t, "actual": actual, "class": k,
                                 "p_down": p[0], "p_stable": p[1], "p_up": p[2],
                                 "q10": q[0], "q50": q[1], "q90": q[2],
                                 "rps": E.rps(p, k), "hit": int(np.argmax(p) == k),
                                 "rps_no_change": E.rps(clim, k), "hit_no_change": int(k == 1),
                                 "rps_trend": E.rps(p_tr, k), "hit_trend": int(int(E.classify(np.array([trend]), band)[0]) == k),
                                 "abs_err": abs(q[1] - actual), "abs_err_no_change": abs(actual),
                                 "covered": int(q[0] <= actual <= q[2])})
    return pd.DataFrame(rows)


def summarise(r: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    s = r.groupby(by).agg(weeks=("rps", "size"), start=("week", "min"), end=("week", "max"),
                          rps=("rps", "mean"), rps_no_change=("rps_no_change", "mean"), rps_trend=("rps_trend", "mean"),
                          hit=("hit", "mean"), hit_no_change=("hit_no_change", "mean"), hit_trend=("hit_trend", "mean"),
                          mae=("abs_err", "mean"), mae_no_change=("abs_err_no_change", "mean"),
                          coverage80=("covered", "mean")).reset_index()
    s["gain_vs_no_change_pct"] = (1 - s["rps"] / s["rps_no_change"]) * 100
    return s


if __name__ == "__main__":
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    r = run(df, cfg)
    r.to_csv(OUT / "champion_record.csv", index=False, float_format="%.4f")
    s = summarise(r, ["version", "target", "h"])
    r["year"] = r["week"].dt.year
    y = summarise(r, ["version", "target", "h", "year"])
    s.to_csv(OUT / "champion_record_summary.csv", index=False, float_format="%.4f")
    y.to_csv(OUT / "champion_record_by_year.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print(s.round(3).to_string(index=False))
    print(y.round(3).to_string(index=False))
