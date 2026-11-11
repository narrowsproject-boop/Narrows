"""Model search, round 2: new ideas plus a meta-model that picks models by itself.

New candidates (Rotterdam and Genoa, 1 and 4 weeks, same no-leak walk-forward):
- pooled training: learn from Rotterdam AND Genoa weeks together (twice the examples),
  with a route flag;
- detailed EC curve: spreads between the 1st, 2nd and 3rd contracts, 3-day move of the
  main contract, its 4-week volatility;
- regress-then-classify variants (ridge strength, local residuals).
Meta-models (no peeking): every Thursday they look only at forecasts already resolved
and either follow the best recent model, average the 5 best recent ones, or weight all
models by their recent error. Candidates = the ~60 models of model_search.py + these.

Usage: python scripts/model_search2.py   -> data/model/ms2_*.csv
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrows.common import CLEAN, DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import EC, build, load_config  # noqa: E402
import model_search as MS  # noqa: E402

warnings.filterwarnings("ignore")
OUT = DATA / "model"
REDUCED = MS.REDUCED
CURVE = ["ec_spread_12", "ec_spread_23", "ec_3d", "ec_vol_4w"]
REFIT = 2


def curve_features(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    ec = EC()
    rows = {}
    for w in weeks:
        d0 = ec.last_day(w)
        if d0 is None:
            continue
        alive = sorted([c for c in ec.settle.columns if ec.expiry(c) >= w and pd.notna(ec.settle.at[d0, c])], key=ec.expiry)
        if len(alive) < 3:
            continue
        v = [ec.settle.at[d0, c] for c in alive[:3]]
        main, _ = ec.pick(w, until=w)
        s = ec.settle[main].loc[:d0].dropna()
        r = np.log(s).diff().dropna()
        rows[w] = {"ec_spread_12": (v[1] / v[0] - 1) * 100, "ec_spread_23": (v[2] / v[1] - 1) * 100,
                   "ec_3d": (s.iloc[-1] / s.iloc[-4] - 1) * 100 if len(s) > 4 else np.nan,
                   "ec_vol_4w": r.iloc[-20:].std() * np.sqrt(250) * 100 if len(r) > 20 else np.nan}
    return pd.DataFrame(rows).T


def scaled(m):
    return make_pipeline(StandardScaler(), m)


def candidates():
    """name -> (inputs, pooled, kind, factory)"""
    C = {}
    for pooled in (False, True):
        tag = " pooled" if pooled else ""
        for inp_name, cols in (("reduced", REDUCED), ("reduced+curve", REDUCED + CURVE)):
            if not pooled and inp_name == "reduced":
                continue  # already in round 1
            C[f"ordlogit l2=10{tag} | {inp_name}"] = (cols, pooled, "ord", None)
            C[f"multilogit L2 C=0.5{tag} | {inp_name}"] = (cols, pooled, "clf",
                lambda: scaled(LogisticRegression(C=0.5, max_iter=4000)))
            C[f"SVM C=1{tag} | {inp_name}"] = (cols, pooled, "clf", lambda: scaled(SVC(C=1.0, probability=True, random_state=0)))
            C[f"MLP 16{tag} | {inp_name}"] = (cols, pooled, "clf",
                lambda: scaled(MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0, max_iter=3000, random_state=0)))
            C[f"ridge->classes a=3{tag} | {inp_name}"] = (cols, pooled, "reg", 3.0)
            C[f"champion-style ensemble{tag} | {inp_name}"] = (cols, pooled, "ens", None)
    for a in (1.0, 10.0, 30.0):
        C[f"ridge->classes a={a:g} | reduced"] = (REDUCED, False, "reg", a)
    C["ridge->classes local residuals | reduced"] = (REDUCED, False, "reg_local", 3.0)
    C["ridge->classes local residuals pooled | reduced"] = (REDUCED, True, "reg_local", 3.0)
    return C


def fit_model(kind, factory, X, y, c, band):
    """Fit once; return a function that turns one input row into 3 probabilities."""
    if kind == "ord":
        m = E.OrdinalLogit(l2=10).fit(X, c)
        return m.predict

    def proba(m, x):
        cls = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
        p = np.zeros(3); p[np.asarray(cls).astype(int)] = m.predict_proba(x.reshape(1, -1))[0]
        return p
    if kind == "clf":
        m = factory().fit(X, c)
        return lambda x: proba(m, x)
    if kind == "ens":
        lo = E.OrdinalLogit(l2=10).fit(X, c)
        b = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                           min_samples_leaf=10).fit(X, c)
        f = RandomForestClassifier(n_estimators=150, min_samples_leaf=8, max_features="sqrt", random_state=0, n_jobs=1).fit(X, c)
        return lambda x: (lo.predict(x) + proba(b, x) + proba(f, x)) / 3
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    m = Ridge(alpha=factory).fit(Z, y)
    res_all = y - m.predict(Z)

    def pr(x):
        z = (x - mu) / sd
        res = res_all
        if kind == "reg_local":
            res = res_all[np.argsort(((Z - z) ** 2).sum(1))[:40]]
        sims = m.predict(z.reshape(1, -1))[0] + res
        return np.array([(sims < -band).mean(), ((sims >= -band) & (sims <= band)).mean(), (sims > band).mean()])
    return pr


def run_series(args):
    target, h, df, cfg = args
    other = "shanghai_genoa" if target == "shanghai_rotterdam" else "shanghai_rotterdam"
    ycol = f"y_{h}w"
    band = cfg["stability_band_pct"][target][str(h)]
    band_o = cfg["stability_band_pct"][other][str(h)]
    g = df[df["target"] == target].sort_values("week").reset_index(drop=True)
    go = df[df["target"] == other].sort_values("week").reset_index(drop=True)
    rows, cache = [], {}
    for t in g[(g["week"] >= MS.START) & g[ycol].notna()]["week"]:
        now = g[g["week"] == t].iloc[0]
        own = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()]
        oth = go[(go["week"] + pd.Timedelta(weeks=h) <= t) & go[ycol].notna()]
        actual = float(now[ycol]); k = int(E.classify(np.array([actual]), band)[0])
        refit = (t - cache.get("_t", pd.Timestamp("1900-01-01"))).days >= 7 * REFIT
        if refit:
            cache = {"_t": t}
        for name, (cols, pooled, kind, factory) in candidates().items():
            if now[cols].isna().any():
                continue
            a = own.dropna(subset=cols)
            if len(a) < MS.MIN_TRAIN:
                continue
            Xa, ya = a[cols].to_numpy(float), a[ycol].to_numpy(float)
            ca = E.classify(ya, band)
            x = now[cols].to_numpy(float)
            if pooled:
                b = oth.dropna(subset=cols)
                Xa = np.vstack([np.hstack([Xa, np.zeros((len(Xa), 1))]), np.hstack([b[cols].to_numpy(float), np.ones((len(b), 1))])])
                yb = b[ycol].to_numpy(float)
                ya = np.concatenate([ya, yb])
                ca = np.concatenate([ca, E.classify(yb, band_o)])
                x = np.append(x, 0.0)
            if len(np.unique(ca)) < 3:
                continue
            light = kind in ("ord", "reg", "reg_local")
            try:
                if light or name not in cache:
                    cache[name] = fit_model(kind, factory, Xa, ya, ca, band)
                p = cache[name](x)
            except Exception:
                continue
            if light:
                cache.pop(name, None)
            p = np.clip(p, 1e-6, 1); p = p / p.sum()
            rows.append({"target": target, "h": h, "week": t, "actual": actual, "class": k, "method": name,
                         "p_down": p[0], "p_stable": p[1], "p_up": p[2]})
    print(f"done {target} h={h}: {len(rows)}", flush=True)
    return rows


# ------------------------------------------------------------- meta-models

def meta(d: pd.DataFrame, window: int = 52, min_res: int = 15) -> pd.DataFrame:
    out = []
    methods = [m for m in d["method"].unique() if not m.startswith("ref")]
    for (target, h), g in d.groupby(["target", "h"]):
        piv = {c: g.pivot_table(index="week", columns="method", values=c) for c in ("p_down", "p_stable", "p_up", "rps")}
        weeks = piv["rps"].index.sort_values()
        cls = g.groupby("week")["class"].first(); act = g.groupby("week")["actual"].first()
        for t in weeks:
            resolved = weeks[(weeks + pd.Timedelta(weeks=int(h))) <= t][-window:]
            if len(resolved) < min_res:
                continue
            past = piv["rps"].loc[resolved, methods].mean()
            avail = [m for m in past.dropna().index if pd.notna(piv["p_up"].at[t, m])]
            past = past[avail].sort_values()
            P = np.vstack([piv[c].loc[t, past.index].to_numpy() for c in ("p_down", "p_stable", "p_up")]).T
            def add(name, p):
                out.append({"target": target, "h": h, "week": t, "actual": act[t], "class": cls[t], "method": name,
                            "p_down": p[0], "p_stable": p[1], "p_up": p[2]})
            add("META follow the leader (best of last 52 weeks)", P[0])
            add("META average of 5 best of last 52 weeks", P[:5].mean(0))
            for tau in (0.005, 0.02):
                w = np.exp(-(past.to_numpy() - past.iloc[0]) / tau); w /= w.sum()
                add(f"META weighted by recent error (tau={tau})", (P * w[:, None]).sum(0))
    return pd.DataFrame(out)


def main():
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    df = df.merge(curve_features(weeks), left_on="week", right_index=True, how="left")
    jobs = [(t, h, df, cfg) for t in MS.TARGETS for h in cfg["horizons_weeks"]]
    with Pool(2) as pool:
        rows = sum(pool.map(run_series, jobs), [])
    new = pd.DataFrame(rows)
    new.to_csv(OUT / "ms2_forecasts_new.csv", index=False, float_format="%.5f")
    old = MS.add_champion(pd.read_csv(OUT / "ms_forecasts_raw.csv", parse_dates=["week"]))
    d = pd.concat([old, new], ignore_index=True)
    d = MS.score(d)
    d = MS.common(d)
    m = MS.score(meta(d))
    d = pd.concat([d, m], ignore_index=True)
    d = MS.common(d)
    d.to_csv(OUT / "ms2_forecasts_all.csv", index=False, float_format="%.5f")
    d["period"] = np.where(d["week"] >= MS.HOLDOUT, "holdout", "select")
    s = d.groupby(["method", "period"]).agg(n=("rps", "size"), rps=("rps", "mean"), hit=("hit", "mean")).unstack("period")
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    full = d.groupby("method").agg(rps_all=("rps", "mean"), hit_all=("hit", "mean"))
    per = d[d["period"] == "holdout"].groupby(["method", "target", "h"])["rps"].mean().unstack(["target", "h"])
    per.columns = [f"hold_{a.split('_')[1]}_{b}w" for a, b in per.columns]
    table = s.join(full).join(per).sort_values("rps_select")
    table.to_csv(OUT / "ms2_ranking.csv", float_format="%.4f")
    boots = []
    cands = list(table.index[:5]) + [m for m in table.index if m.startswith("META")]
    for mth in dict.fromkeys(cands):
        for ref in ("ref_no_change", MS.CHAMPION):
            if mth != ref:
                g_, lo, hi, pr = MS.bootstrap_gain(d, mth, ref)
                boots.append({"method": mth, "vs": ref, "rps_gain": g_, "ci90_low": lo, "ci90_high": hi, "p_better": pr})
    pd.DataFrame(boots).to_csv(OUT / "ms2_bootstrap.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 20)
    print(table.round(3).head(30).to_string())
    print(table.loc[[i for i in table.index if i.startswith(("META", "ref", "ensemble"))]].round(3).to_string())
    print(pd.DataFrame(boots).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
