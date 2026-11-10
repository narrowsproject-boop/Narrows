"""Test and learn: every candidate method, every index, walk-forward since 2019.

Direction candidates (3 probabilities): references, engine A rules (k), engine B
ordinal logit (l2), each on several input sets. Magnitude candidates (10/50/90%):
nearest neighbours (k), quantile regression (alpha), each optionally widened by
conformal calibration on its own past errors.

Anti-overfitting: methods are chosen on a selection period and only then scored
on a holdout period that played no part in the choice.
- "socle" track (no EC futures inputs): select 2020 -> Jun 2024, holdout Jul 2024 ->
- "market" track (EC futures since Aug 2023): select Jun 2024 -> Jun 2025, holdout Jul 2025 ->

Usage: python scripts/test_and_learn.py
Writes data/model/tl_forecasts_direction.csv, tl_forecasts_magnitude.csv,
tl_scores.csv and tl_choice.csv.
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

OUT = DATA / "model"
SOCLE = ["mom_1w", "mom_4w", "traffic_bab", "traffic_suez", "traffic_cape", "brent_4w", "control_4w"]
NEWS = ["news_redsea", "news_suez", "news_hormuz"]


def feature_sets(target: str) -> dict[str, tuple[str, list[str]]]:
    """name -> (track, columns)"""
    ec_extra = ["ec_premium", "ec_slope"] if target == "ec_front" else ["ec_1w", "ec_4w", "ec_premium", "ec_slope"]
    reduced = (["ec_premium", "ec_slope", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"] if target == "ec_front"
               else ["ec_premium", "ec_4w", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"])
    socle = [c for c in SOCLE if not (target == "shanghai_los_angeles" and c == "control_4w")]
    return {
        "socle": ("socle", socle),
        "socle+news": ("socle", socle + NEWS),
        "market": ("market", socle + ec_extra),
        "market+news": ("market", socle + ec_extra + NEWS),
        "reduced": ("market", reduced),
        "reduced+news": ("market", reduced + ["news_redsea", "news_hormuz"]),
    }


PERIODS = {
    "socle": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-07-01")),   # select from, holdout from
    "market": (pd.Timestamp("2024-06-01"), pd.Timestamp("2025-07-01")),
}
MIN_TRAIN = {"socle": 50, "market": 40}
A_K = [10, 30, 60]
B_L2 = [0.3, 1, 3, 10]
KNN_K = [20, 40, 80]
QR_ALPHA = [0.01, 0.05, 0.2]


def run_grid(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    drows, mrows = [], []
    for target, g in df.groupby("target"):
        g = g.sort_values("week").reset_index(drop=True)
        fsets = feature_sets(target)
        for h in cfg["horizons_weeks"]:
            band = cfg["stability_band_pct"][target][str(h)]
            ycol = f"y_{h}w"
            print(f"{target} h={h}", flush=True)
            for _, now in g.iterrows():
                t = now["week"]
                if t < PERIODS["socle"][0] or pd.isna(now[ycol]):
                    continue
                known = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()]
                if len(known) < MIN_TRAIN["market"]:
                    continue
                actual = float(now[ycol])
                base = {"target": target, "h": h, "week": t, "actual": actual,
                        "class": int(E.classify(np.array([actual]), band)[0])}
                yk = known[ycol].to_numpy(float)
                ck = E.classify(yk, band)
                spread = np.quantile(yk, E.QUANTILES) - np.median(yk)
                # references (both tracks)
                drows.append({**base, "track": "ref", "method": "ref_no_change", "p": E.climatology(ck)})
                trend = float(now["mom_4w"]) * h / 4 if pd.notna(now["mom_4w"]) else 0.0
                p_tr = np.full(3, 0.2); p_tr[int(E.classify(np.array([trend]), band)[0])] = 0.6
                drows.append({**base, "track": "ref", "method": "ref_trend", "p": p_tr})
                mrows.append({**base, "track": "ref", "method": "ref_no_change", "q": spread.copy()})
                mrows.append({**base, "track": "ref", "method": "ref_trend", "q": trend + spread})
                for fname, (track, cols) in fsets.items():
                    tr = known.dropna(subset=cols)
                    if len(tr) < MIN_TRAIN[track] or now[cols].isna().any():
                        continue
                    X, y = tr[cols].to_numpy(float), tr[ycol].to_numpy(float)
                    c = E.classify(y, band)
                    x = now[cols].to_numpy(float)
                    for k in A_K:
                        p, _ = E.rules_engine(tr, c, now[list(dict.fromkeys(cols + ["ec_1w", "ec_4w"]))] if track == "market"
                                              else now[cols], k=k)
                        drows.append({**base, "track": track, "method": f"A k={k} | {fname}", "p": p})
                    if len(np.unique(c)) == 3:
                        for l2 in B_L2:
                            p = E.OrdinalLogit(l2=l2).fit(X, c).predict(x)
                            drows.append({**base, "track": track, "method": f"B l2={l2} | {fname}", "p": p})
                    for k in KNN_K:
                        mrows.append({**base, "track": track, "method": f"kNN k={k} | {fname}",
                                      "q": E.knn_quantiles(X, y, x, k=k)})
                    for a in QR_ALPHA:
                        mrows.append({**base, "track": track, "method": f"QR a={a} | {fname}",
                                      "q": E.quantile_regression(X, y, x, alpha=a)})
    d = pd.DataFrame(drows)
    P = np.vstack(d.pop("p").to_numpy())
    d["p_down"], d["p_stable"], d["p_up"] = P[:, 0], P[:, 1], P[:, 2]
    m = pd.DataFrame(mrows)
    Q = np.vstack(m.pop("q").to_numpy())
    m["q10"], m["q50"], m["q90"] = Q[:, 0], Q[:, 1], Q[:, 2]
    return d, m


def add_conformal(m: pd.DataFrame, min_scores: int = 20) -> pd.DataFrame:
    """Widen each method's 80% range by the 80th percentile of its own past misses
    (conformalised quantile regression). Uses only forecasts already resolved."""
    out = []
    for (target, h, method), g in m.groupby(["target", "h", "method"]):
        g = g.sort_values("week").reset_index(drop=True)
        score = np.maximum(g["q10"] - g["actual"], g["actual"] - g["q90"]).to_numpy()
        res_week = (g["week"] + pd.Timedelta(weeks=int(h))).to_numpy()
        widen = []
        for t in g["week"].to_numpy():
            s = score[res_week <= t]
            widen.append(np.quantile(s, 0.8 * (1 + 1 / len(s))) if len(s) >= min_scores else np.nan)
        c = g.copy()
        c["method"] = method + " +conformal"
        w = np.array(widen)
        c["q10"], c["q90"] = c["q10"] - w, c["q90"] + w
        out.append(c.dropna(subset=["q10"]))
    return pd.concat([m] + out, ignore_index=True)


def score(d: pd.DataFrame, m: pd.DataFrame):
    P = d[["p_down", "p_stable", "p_up"]].to_numpy()
    d["rps"] = [E.rps(p, c) for p, c in zip(P, d["class"])]
    d["hit"] = P.argmax(1) == d["class"].to_numpy()
    nc = d["method"] == "ref_no_change"
    d.loc[nc, "hit"] = d.loc[nc, "class"] == 1
    m["abs_err"] = (m["q50"] - m["actual"]).abs()
    m["covered"] = (m["actual"] >= m["q10"]) & (m["actual"] <= m["q90"])
    m["width"] = m["q90"] - m["q10"]
    m["pinball"] = [E.pinball(np.array(q), y) for q, y in zip(m[["q10", "q50", "q90"]].to_numpy(), m["actual"])]
    return d, m


def period_of(week, track):
    sel, hold = PERIODS[track]
    return "holdout" if week >= hold else ("select" if week >= sel else None)


def choose(d: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per target, horizon and track: best direction method on the selection period
    (lowest RPS), best magnitude method (lowest pinball among ranges covering 72-88%),
    then their holdout scores next to the references on the same weeks."""
    rows, allscores = [], []
    for track in ("socle", "market"):
        for (target, h), gd in d.groupby(["target", "h"]):
            cand = gd[gd["track"].isin([track, "ref"])].copy()
            cand["period"] = [period_of(w, track) for w in cand["week"]]
            cand = cand.dropna(subset=["period"])
            # keep only weeks where every method of this track exists
            nmeth = cand.groupby("method")["week"].nunique()
            full = cand.groupby("week")["method"].nunique()
            cand = cand[cand["week"].isin(full[full == full.max()].index)]
            s = cand.groupby(["period", "method"]).agg(n=("rps", "size"), rps=("rps", "mean"),
                                                       hit=("hit", "mean")).reset_index()
            s.insert(0, "track", track); s.insert(0, "h", h); s.insert(0, "target", target)
            allscores.append(s.assign(kind="direction"))
            sel = s[(s["period"] == "select") & ~s["method"].str.startswith("ref")]
            if sel.empty:
                continue
            best = sel.sort_values("rps").iloc[0]["method"]
            hold = s[s["period"] == "holdout"].set_index("method")
            selx = s[s["period"] == "select"].set_index("method")
            # quarter stability on holdout vs no-change
            hq = cand[cand["period"] == "holdout"]
            a = hq[hq["method"] == best].set_index("week")["rps"]
            b = hq[hq["method"] == "ref_no_change"].set_index("week")["rps"]
            q = (b - a).groupby((b - a).index.to_period("Q")).mean()
            row = {"target": target, "h": h, "track": track, "direction_method": best,
                   "sel_n": int(selx.loc[best, "n"]), "sel_rps": selx.loc[best, "rps"],
                   "sel_rps_ref": selx.loc["ref_no_change", "rps"],
                   "hold_n": int(hold.loc[best, "n"]) if best in hold.index else 0,
                   "hold_rps": hold.loc[best, "rps"] if best in hold.index else np.nan,
                   "hold_rps_no_change": hold["rps"].get("ref_no_change", np.nan),
                   "hold_rps_trend": hold["rps"].get("ref_trend", np.nan),
                   "hold_hit": hold.loc[best, "hit"] if best in hold.index else np.nan,
                   "hold_hit_no_change": hold["hit"].get("ref_no_change", np.nan),
                   "hold_hit_trend": hold["hit"].get("ref_trend", np.nan),
                   "hold_quarters_better": f"{int((q > 0).sum())}/{len(q)}"}
            # best holdout method in hindsight (for information only)
            hb = hold.drop([i for i in hold.index if i.startswith("ref")]).sort_values("rps")
            row["hindsight_best"] = hb.index[0] if len(hb) else ""
            row["hindsight_rps"] = hb["rps"].iloc[0] if len(hb) else np.nan
            # magnitude
            gm = m[(m["target"] == target) & (m["h"] == h) & m["track"].isin([track, "ref"])].copy()
            gm["period"] = [period_of(w, track) for w in gm["week"]]
            gm = gm.dropna(subset=["period"])
            full = gm.groupby("week")["method"].nunique()
            gm = gm[gm["week"].isin(full[full == full.max()].index)]
            sm = gm.groupby(["period", "method"]).agg(n=("pinball", "size"), pinball=("pinball", "mean"),
                                                      mae=("abs_err", "mean"), coverage=("covered", "mean"),
                                                      width=("width", "mean")).reset_index()
            sm.insert(0, "track", track); sm.insert(0, "h", h); sm.insert(0, "target", target)
            allscores.append(sm.assign(kind="magnitude"))
            ms = sm[(sm["period"] == "select") & ~sm["method"].str.startswith("ref")]
            ok = ms[ms["coverage"].between(0.72, 0.88)]
            pick = (ok if len(ok) else ms).sort_values("pinball").iloc[0]["method"]
            mh = sm[sm["period"] == "holdout"].set_index("method")
            row.update({"magnitude_method": pick,
                        "hold_pinball": mh["pinball"].get(pick, np.nan),
                        "hold_pinball_no_change": mh["pinball"].get("ref_no_change", np.nan),
                        "hold_mae": mh["mae"].get(pick, np.nan),
                        "hold_mae_no_change": mh["mae"].get("ref_no_change", np.nan),
                        "hold_coverage": mh["coverage"].get(pick, np.nan),
                        "hold_width": mh["width"].get(pick, np.nan)})
            rows.append(row)
    return pd.DataFrame(rows), pd.concat(allscores, ignore_index=True)


if __name__ == "__main__":
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    if len(sys.argv) > 1 and sys.argv[1] == "--rescore":
        d = pd.read_csv(OUT / "tl_forecasts_direction.csv", parse_dates=["week"])
        m = pd.read_csv(OUT / "tl_forecasts_magnitude.csv", parse_dates=["week"])
        m = m[~m["method"].str.endswith("+conformal")]
    else:
        d, m = run_grid(df, cfg)
    m = add_conformal(m)
    d, m = score(d, m)
    d.to_csv(OUT / "tl_forecasts_direction.csv", index=False, float_format="%.4f")
    m.to_csv(OUT / "tl_forecasts_magnitude.csv", index=False, float_format="%.4f")
    choice, scores = choose(d, m)
    scores.to_csv(OUT / "tl_scores.csv", index=False, float_format="%.4f")
    choice.to_csv(OUT / "tl_choice.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(choice.round(3).to_string(index=False))
