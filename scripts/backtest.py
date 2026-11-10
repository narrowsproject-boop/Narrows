"""Walk-forward backtest of the Narrows v0.1 candidates.

For every Thursday in the test window, each model is estimated only on weeks whose
outcome was already known that Thursday (week + horizon <= Thursday), Covid regime
excluded, then forecasts the next 1 and 4 weeks. Nothing from the future is used.

Usage: python scripts/backtest.py
Writes data/model/backtest_forecasts.csv and data/model/backtest_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narrows.common import DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import build, load_config  # noqa: E402

SOCLE = ["mom_1w", "mom_4w", "traffic_bab", "traffic_suez", "traffic_cape", "brent_4w", "control_4w"]
MARKET_EXTRA = ["ec_1w", "ec_4w", "ec_premium", "ec_slope"]
TEST_START = pd.Timestamp("2023-01-05")
COMMON_START = pd.Timestamp("2024-06-06")  # first week where market models have ~40 weeks to learn from
MIN_TRAIN = {"socle": 60, "market": 40}


def feature_set(target: str, kind: str) -> list[str]:
    if kind == "socle":
        return SOCLE
    extra = ["ec_premium", "ec_slope"] if target == "ec_front" else MARKET_EXTRA
    return SOCLE + extra


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    rows = []
    for (target, g) in df.groupby("target"):
        g = g.sort_values("week").reset_index(drop=True)
        for h in cfg["horizons_weeks"]:
            band = cfg["stability_band_pct"][target][str(h)]
            ycol = f"y_{h}w"
            for i, now in g.iterrows():
                t = now["week"]
                if t < TEST_START or pd.isna(now[ycol]):
                    continue
                known = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()]
                actual = now[ycol]
                c_act = int(E.classify(np.array([actual]), band)[0])
                base_row = {"target": target, "h": h, "week": t, "actual": actual, "class": c_act}

                def add(model, p, q, active=""):
                    rows.append({**base_row, "model": model, "p_down": p[0], "p_stable": p[1],
                                 "p_up": p[2], "q10": q[0], "q50": q[1], "q90": q[2], "rules": active})

                # references
                y_known = known[ycol].to_numpy()
                if len(y_known) < (MIN_TRAIN["market"] if target == "ec_front" else MIN_TRAIN["socle"]):
                    continue
                c_known = E.classify(y_known, band)
                clim = E.climatology(c_known)
                spread = np.quantile(y_known, E.QUANTILES) - np.median(y_known)
                add("ref_no_change", clim, np.array([0, 0, 0]) + spread)
                trend = now["mom_4w"] * h / 4 if pd.notna(now["mom_4w"]) else 0.0
                tc = int(E.classify(np.array([trend]), band)[0])
                p_tr = np.full(3, 0.2); p_tr[tc] = 0.6
                add("ref_trend", p_tr, trend + spread)

                for kind in ("socle", "market"):
                    cols = feature_set(target, kind)
                    tr = known.dropna(subset=cols)
                    if len(tr) < MIN_TRAIN[kind] or now[cols].isna().any():
                        continue
                    X, y = tr[cols].to_numpy(float), tr[ycol].to_numpy(float)
                    c = E.classify(y, band)
                    x = now[cols].to_numpy(float)
                    pA, active = E.rules_engine(tr, c, now)
                    qA = E.knn_quantiles(X, y, x)
                    add(f"A_{kind}", pA, qA, " | ".join(active))
                    pB = E.OrdinalLogit(l2=1.0).fit(X, c).predict(x)
                    qB = E.quantile_regression(X, y, x)
                    add(f"B_{kind}", pB, qB)
    fc = pd.DataFrame(rows)
    fc.to_csv(DATA / "model" / "backtest_forecasts.csv", index=False, float_format="%.4f")
    return fc, summarise(fc)


def summarise(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    P = fc[["p_down", "p_stable", "p_up"]].to_numpy()
    fc["rps"] = [E.rps(p, c) for p, c in zip(P, fc["class"])]
    fc["hit"] = P.argmax(1) == fc["class"].to_numpy()
    fc["abs_err"] = (fc["q50"] - fc["actual"]).abs()
    fc["covered"] = (fc["actual"] >= fc["q10"]) & (fc["actual"] <= fc["q90"])
    fc["width"] = fc["q90"] - fc["q10"]
    fc["pinball"] = [E.pinball(np.array([a, b, c]), y)
                     for a, b, c, y in zip(fc["q10"], fc["q50"], fc["q90"], fc["actual"])]
    # direction of the "no change" reference is always "stable"
    nc = fc["model"] == "ref_no_change"
    fc.loc[nc, "hit"] = fc.loc[nc, "class"] == 1
    out = []
    for window, start in (("since_2023", TEST_START), ("since_mid_2024", COMMON_START)):
        w = fc[fc["week"] >= start]
        s = (w.groupby(["target", "h", "model"])
              .agg(n=("rps", "size"), rps=("rps", "mean"), hit_rate=("hit", "mean"),
                   mae=("abs_err", "mean"), coverage80=("covered", "mean"),
                   width80=("width", "mean"), pinball=("pinball", "mean"))
              .reset_index())
        s.insert(0, "window", window)
        out.append(s)
    summ = pd.concat(out, ignore_index=True)
    summ.to_csv(DATA / "model" / "backtest_summary.csv", index=False, float_format="%.3f")
    return summ


if __name__ == "__main__":
    _, s = run()
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
