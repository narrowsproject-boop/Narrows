"""Test and learn v2: richer inputs (detailed news, container traffic in 6 straits,
Cape diversion share, carrier share prices) and more complex models (gradient
boosting, random forest, regularised logit, ensembles), against the v0.1 champion.

Same protocol as test_and_learn.py: walk-forward, no leakage, methods chosen on the
selection period and scored on a holdout that played no part in the choice.
Tree models are refitted every REFIT weeks (on data known at the refit date) to keep
the run short; forecasts are still made every week.

Usage: python scripts/test_and_learn_v2.py
Writes data/model/tl2_*.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narrows.common import DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import build, load_config  # noqa: E402
import scripts.test_and_learn as TL  # noqa: E402

warnings.filterwarnings("ignore")
OUT = DATA / "model"
REFIT = 4
TARGETS = ["shanghai_rotterdam", "shanghai_genoa"]


def feature_sets(cols_all: list[str]) -> dict[str, tuple[str, list[str]]]:
    reduced = ["ec_premium", "ec_4w", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"]
    traffic = [c for c in cols_all if c.startswith("tc_")] + ["cape_share", "cape_share_4w"]
    news_core = [c for c in cols_all if c.startswith("nw_") and c.endswith(("_pressure", "_shock"))] + ["nw_x_redsea_confirmed"]
    news_all = [c for c in cols_all if c.startswith("nw_")]
    equity = [c for c in cols_all if c.startswith("eq_")]
    socle = ["mom_1w", "mom_4w", "brent_4w", "control_4w"]
    ec = ["ec_1w", "ec_4w", "ec_premium", "ec_slope"]
    return {
        "champion(reduced)": ("market", reduced),
        "reduced+traffic": ("market", reduced + traffic),
        "reduced+news": ("market", reduced + news_core),
        "reduced+equity": ("market", reduced + equity),
        "reduced+traffic+news+equity": ("market", reduced + traffic + news_core + equity),
        "everything": ("market", list(dict.fromkeys(socle + ec + traffic + news_all + equity + ["traffic_bab", "traffic_suez", "traffic_cape"]))),
        "socle_rich(no EC)": ("socle", socle + traffic + news_core + equity),
    }


def models_direction():
    return {
        "logit l2=10": lambda: E.OrdinalLogit(l2=10),
        "logit l2=30": lambda: E.OrdinalLogit(l2=30),
        "boosting": lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                           l2_regularization=1.0, min_samples_leaf=10),
        "forest": lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=8, max_features="sqrt",
                                                 random_state=0, n_jobs=2),
    }


def proba(model, X, c, x):
    if isinstance(model, E.OrdinalLogit):
        return model.fit(X, c).predict(x)
    model.fit(X, c)
    p = np.zeros(3)
    p[model.classes_.astype(int)] = model.predict_proba(x.reshape(1, -1))[0]
    return p


def hgb_quantiles(X, y, x):
    q = []
    for a in E.QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_depth=2, learning_rate=0.05,
                                          max_iter=150, min_samples_leaf=10).fit(X, y)
        q.append(m.predict(x.reshape(1, -1))[0])
    return np.sort(np.array(q))


def run(df: pd.DataFrame, cfg: dict):
    cols_all = list(df.columns)
    fsets = feature_sets(cols_all)
    drows, mrows = [], []
    for target in TARGETS:
        g = df[df["target"] == target].sort_values("week").reset_index(drop=True)
        for h in cfg["horizons_weeks"]:
            band = cfg["stability_band_pct"][target][str(h)]
            ycol = f"y_{h}w"
            print(f"{target} h={h}", flush=True)
            cache = {}  # (fset, model) -> (fit_week, fitted object or function)
            for wi, now in g.iterrows():
                t = now["week"]
                if t < TL.PERIODS["socle"][0] or pd.isna(now[ycol]):
                    continue
                known = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()]
                if len(known) < 40:
                    continue
                actual = float(now[ycol])
                base = {"target": target, "h": h, "week": t, "actual": actual,
                        "class": int(E.classify(np.array([actual]), band)[0])}
                yk = known[ycol].to_numpy(float)
                ck = E.classify(yk, band)
                spread = np.quantile(yk, E.QUANTILES) - np.median(yk)
                drows.append({**base, "track": "ref", "method": "ref_no_change", "p": E.climatology(ck)})
                trend = float(now["mom_4w"]) * h / 4 if pd.notna(now["mom_4w"]) else 0.0
                p_tr = np.full(3, 0.2); p_tr[int(E.classify(np.array([trend]), band)[0])] = 0.6
                drows.append({**base, "track": "ref", "method": "ref_trend", "p": p_tr})
                mrows.append({**base, "track": "ref", "method": "ref_no_change", "q": spread.copy()})
                mrows.append({**base, "track": "ref", "method": "ref_trend", "q": trend + spread})
                for fname, (track, cols) in fsets.items():
                    tr = known.dropna(subset=cols)
                    if len(tr) < TL.MIN_TRAIN[track] or now[cols].isna().any():
                        continue
                    X, y = tr[cols].to_numpy(float), tr[ycol].to_numpy(float)
                    c = E.classify(y, band)
                    if len(np.unique(c)) < 3:
                        continue
                    x = now[cols].to_numpy(float)
                    ps = {}
                    for mname, make in models_direction().items():
                        key = (fname, mname)
                        tree = mname in ("boosting", "forest")
                        if tree and key in cache and (t - cache[key][0]).days < 7 * REFIT:
                            m = cache[key][1]
                            p = np.zeros(3); p[m.classes_.astype(int)] = m.predict_proba(x.reshape(1, -1))[0]
                        else:
                            m = make()
                            p = proba(m, X, c, x)
                            if tree:
                                cache[key] = (t, m)
                        ps[mname] = p
                        drows.append({**base, "track": track, "method": f"{mname} | {fname}", "p": p})
                    drows.append({**base, "track": track, "method": f"ensemble(logit+boosting+forest) | {fname}",
                                  "p": (ps["logit l2=10"] + ps["boosting"] + ps["forest"]) / 3})
                    # magnitude
                    qk = E.knn_quantiles(X, y, x, k=40)
                    mrows.append({**base, "track": track, "method": f"kNN k=40 | {fname}", "q": qk})
                    key = (fname, "hgbq")
                    if key in cache and (t - cache[key][0]).days < 7 * REFIT:
                        ms = cache[key][1]
                        qh = np.sort(np.array([mm.predict(x.reshape(1, -1))[0] for mm in ms]))
                    else:
                        ms = [HistGradientBoostingRegressor(loss="quantile", quantile=a, max_depth=2, learning_rate=0.05,
                                                            max_iter=150, min_samples_leaf=10).fit(X, y) for a in E.QUANTILES]
                        cache[key] = (t, ms)
                        qh = np.sort(np.array([mm.predict(x.reshape(1, -1))[0] for mm in ms]))
                    mrows.append({**base, "track": track, "method": f"boosting-quantile | {fname}", "q": qh})
                    mrows.append({**base, "track": track, "method": f"ensemble(kNN+boosting) | {fname}", "q": (qk + qh) / 2})
    d = pd.DataFrame(drows)
    P = np.vstack(d.pop("p").to_numpy()); d["p_down"], d["p_stable"], d["p_up"] = P.T
    m = pd.DataFrame(mrows)
    Q = np.vstack(m.pop("q").to_numpy()); m["q10"], m["q50"], m["q90"] = Q.T
    return d, m


def big_moves(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Holdout scores on the weeks with a large move (more than twice the stability band)."""
    rows = []
    for (target, h), g in d.groupby(["target", "h"]):
        band = cfg["stability_band_pct"][target][str(h)]
        hold = g[g["week"] >= TL.PERIODS["market"][1]]
        big = hold[hold["actual"].abs() > 2 * band]
        s = big.groupby("method").agg(n=("rps", "size"), rps=("rps", "mean"), hit=("hit", "mean")).reset_index()
        s.insert(0, "h", h); s.insert(0, "target", target)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    d, m = run(df, cfg)
    m = TL.add_conformal(m)
    d, m = TL.score(d, m)
    d.to_csv(OUT / "tl2_forecasts_direction.csv", index=False, float_format="%.4f")
    m.to_csv(OUT / "tl2_forecasts_magnitude.csv", index=False, float_format="%.4f")
    choice, scores = TL.choose(d, m)
    scores.to_csv(OUT / "tl2_scores.csv", index=False, float_format="%.4f")
    choice.to_csv(OUT / "tl2_choice.csv", index=False, float_format="%.4f")
    big_moves(d, cfg).to_csv(OUT / "tl2_big_moves.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(choice.round(3).to_string(index=False))
