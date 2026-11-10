"""Model search: many model families and input recipes, scored walk-forward.

Targets: Drewry Shanghai -> Rotterdam and -> Genoa, horizons 1 and 4 weeks, on the
"market" track (EC futures inputs exist from Aug 2023, so Covid plays no part here).
Protocol (unchanged): every forecast uses only weeks whose outcome was known on that
Thursday; models are refitted every REFIT weeks on data known at the refit date.
Selection period: Jun 2024 -> Jun 2025. Holdout: Jul 2025 -> today. The top 3 are
ranked on the selection period only (mean RPS over the 4 series), then reported on
the holdout with block-bootstrap confidence intervals.

Usage: python scripts/model_search.py            (full run, ~1 h on 2 CPUs)
       python scripts/model_search.py --rescore  (reuse saved forecasts)
Writes data/model/ms_*.csv
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import HuberRegressor, LogisticRegression, QuantileRegressor, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narrows.common import CLEAN, DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import build, load_config  # noqa: E402

warnings.filterwarnings("ignore")
OUT = DATA / "model"
TARGETS = ["shanghai_rotterdam", "shanghai_genoa"]
START = pd.Timestamp("2024-06-01")
HOLDOUT = pd.Timestamp("2025-07-01")
REFIT = 2
MIN_TRAIN = 40
CHAMPION = "ensemble(logit+boosting+forest) | reduced"
V01 = "ordlogit l2=10 | reduced"
REDUCED = ["ec_premium", "ec_4w", "mom_1w", "mom_4w", "traffic_cape", "brent_4w"]


# ---------------------------------------------------------------- extra inputs

def news_halflives(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Shipping-zone news pressure with several memory lengths (half-lives in days)."""
    n = pd.read_csv(CLEAN / "news_events_by_zone.csv", parse_dates=["date"])
    n = n[n["event_class"].isin(["conflict", "coercion", "threat"])]
    zones = {"route": ["red_sea_bab_el_mandeb", "gulf_of_aden", "suez_canal", "hormuz_gulf"],
             "redsea": ["red_sea_bab_el_mandeb", "gulf_of_aden"]}
    out = pd.DataFrame(index=weeks)
    wed = weeks - pd.Timedelta(days=1)
    for name, zs in zones.items():
        s = n[n["zone"].isin(zs)].groupby("date")["events"].sum().asfreq("D").fillna(0)
        z = ((s - s.rolling(365, min_periods=180).mean().shift(1)) / s.rolling(365, min_periods=180).std().shift(1)).clip(-5, 30)
        for hl in (3, 7, 28, 56):
            out[f"nh_{name}_hl{hl}"] = z.clip(lower=0).ewm(halflife=hl).mean().reindex(wed).values
    return out


def blocks(cols: list[str]) -> dict[str, list[str]]:
    traffic = [c for c in cols if c.startswith("tc_")] + ["cape_share", "cape_share_4w"]
    news = [c for c in cols if c.startswith("nw_") and c.endswith(("_pressure", "_shock"))] + ["nw_x_redsea_confirmed"]
    equity = [c for c in cols if c.startswith("eq_")]
    news_eng = ["nh_route_hl7", "nh_route_hl28", "nh_redsea_hl3", "nh_redsea_hl56", "nw_x_redsea_confirmed"]
    pool = list(dict.fromkeys(REDUCED + ["ec_1w", "ec_slope", "traffic_bab", "traffic_suez", "control_4w"]
                              + traffic + news + equity + news_eng))
    return {"reduced": REDUCED, "traffic": traffic, "news": news, "equity": equity, "news_eng": news_eng, "pool": pool,
            "rich": traffic + news + equity}


# ---------------------------------------------------------------- model zoo

def scaled(m):
    return make_pipeline(StandardScaler(), m)


def zoo() -> dict[str, dict]:
    """name -> spec. kind: clf (sklearn classifier), ord (ordinal logit), reg (regress
    then classify), plus input recipe and training options."""
    Z = {}
    R = "reduced"
    for l2 in (1, 3, 10, 30):
        Z[f"ordlogit l2={l2} | reduced"] = dict(kind="ord", l2=l2, inputs=R)
    Z["ordlogit l2=10 | reduced, rolling 78w"] = dict(kind="ord", l2=10, inputs=R, window=78)
    Z["ordlogit l2=10 | reduced, recency hl=52w"] = dict(kind="ord", l2=10, inputs=R, halflife=52)
    Z["ordlogit l2=10 | reduced, recency hl=26w"] = dict(kind="ord", l2=10, inputs=R, halflife=26)
    for blk in ("traffic", "news", "equity", "news_eng"):
        Z[f"ordlogit l2=10 | reduced+{blk}"] = dict(kind="ord", l2=10, inputs=f"reduced+{blk}")
        Z[f"ordlogit l2=30 | reduced+{blk}"] = dict(kind="ord", l2=30, inputs=f"reduced+{blk}")
    Z["ordlogit l2=10 | reduced+PCA3(rich)"] = dict(kind="ord", l2=10, inputs="reduced+pca")
    Z["ordlogit l2=10 | forward selection (pool)"] = dict(kind="ord", l2=10, inputs="forward")
    Z["ordlogit l2=10 | top-6 mutual info (pool)"] = dict(kind="ord", l2=10, inputs="mi")
    Z["ordlogit l2=10 | regime (crisis x reduced)"] = dict(kind="ord", l2=10, inputs="regime")
    for C in (0.1, 0.5, 2.0):
        for r in (0.0, 0.5, 1.0):
            nm = {0.0: "L2", 0.5: "elastic-net", 1.0: "L1"}[r]
            Z[f"multilogit {nm} C={C} | reduced"] = dict(kind="clf", inputs=R,
                make=lambda C=C, r=r: scaled(LogisticRegression(C=C, l1_ratio=r, solver="saga", max_iter=4000)))
    Z["multilogit L1 C=0.1 | pool (built-in selection)"] = dict(kind="clf", inputs="pool",
        make=lambda: scaled(LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=4000)))
    Z["multilogit L1 C=0.5 | pool (built-in selection)"] = dict(kind="clf", inputs="pool",
        make=lambda: scaled(LogisticRegression(C=0.5, l1_ratio=1.0, solver="saga", max_iter=4000)))
    Z["LDA shrinkage | reduced"] = dict(kind="clf", inputs=R,
        make=lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    Z["LDA shrinkage | pool"] = dict(kind="clf", inputs="pool",
        make=lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    Z["naive Bayes | reduced"] = dict(kind="clf", inputs=R, make=lambda: scaled(GaussianNB()))
    for k in (15, 30, 60):
        Z[f"kNN k={k} | reduced"] = dict(kind="clf", inputs=R,
            make=lambda k=k: scaled(KNeighborsClassifier(n_neighbors=k, weights="distance")))
    for C in (0.3, 1.0):
        Z[f"SVM rbf C={C} | reduced"] = dict(kind="clf", inputs=R,
            make=lambda C=C: scaled(SVC(C=C, probability=True, random_state=0)))
    for leaf in (5, 10, 20):
        Z[f"forest leaf={leaf} | reduced"] = dict(kind="clf", inputs=R,
            make=lambda leaf=leaf: RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
                                                          random_state=0, n_jobs=1))
    Z["forest leaf=10 | reduced+rich"] = dict(kind="clf", inputs="reduced+rich",
        make=lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features="sqrt", random_state=0, n_jobs=1))
    Z["extra trees leaf=10 | reduced"] = dict(kind="clf", inputs=R,
        make=lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", random_state=0, n_jobs=1))
    Z["extra trees leaf=10 | reduced+rich"] = dict(kind="clf", inputs="reduced+rich",
        make=lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", random_state=0, n_jobs=1))
    for depth in (1, 2, 3):
        for lr in (0.03, 0.1):
            Z[f"boosting depth={depth} lr={lr} | reduced"] = dict(kind="clf", inputs=R,
                make=lambda d=depth, lr=lr: HistGradientBoostingClassifier(max_depth=d, learning_rate=lr, max_iter=150,
                                                                           l2_regularization=1.0, min_samples_leaf=10))
    Z["boosting depth=2 lr=0.03 | reduced+rich"] = dict(kind="clf", inputs="reduced+rich",
        make=lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, l2_regularization=1.0, min_samples_leaf=10))
    for hid, a in (((8,), 1.0), ((16,), 3.0)):
        Z[f"MLP {hid[0]} units alpha={a} | reduced"] = dict(kind="clf", inputs=R,
            make=lambda hid=hid, a=a: scaled(MLPClassifier(hidden_layer_sizes=hid, alpha=a, max_iter=3000, random_state=0)))
    for reg, mk in (("ridge", lambda: Ridge(alpha=3.0)), ("huber", lambda: HuberRegressor(alpha=1.0, max_iter=500)),
                    ("median QR", lambda: QuantileRegressor(quantile=0.5, alpha=0.05, solver="highs"))):
        for logm in (False, True):
            Z[f"regress->classes {reg}{' log' if logm else ''} | reduced"] = dict(kind="reg", inputs=R, make=lambda mk=mk: scaled(mk()), log=logm)
    Z["regress->classes ridge | reduced+news_eng"] = dict(kind="reg", inputs="reduced+news_eng", make=lambda: scaled(Ridge(alpha=3.0)), log=False)
    # the current champion, rebuilt from its three parts in this run
    Z["_part logit"] = dict(kind="ord", l2=10, inputs=R)
    Z["_part boosting"] = dict(kind="clf", inputs=R, make=lambda: HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.05, max_iter=150, l2_regularization=1.0, min_samples_leaf=10))
    Z["_part forest"] = dict(kind="clf", inputs=R, make=lambda: RandomForestClassifier(
        n_estimators=150, min_samples_leaf=8, max_features="sqrt", random_state=0, n_jobs=1))
    return Z


# ---------------------------------------------------------------- fitting

def resolve_inputs(spec, B, tr, ycol, band):
    inp = spec["inputs"]
    if inp == "reduced":
        return B["reduced"], None
    if inp.startswith("reduced+") and inp != "reduced+pca":
        return list(dict.fromkeys(B["reduced"] + B[inp.split("+", 1)[1]])), None
    if inp == "pool":
        return B["pool"], None
    if inp == "reduced+pca":
        return B["reduced"], "pca"
    if inp == "regime":
        return B["reduced"], "regime"
    pool = [c for c in B["pool"] if tr[c].notna().all()]
    X = tr[pool].to_numpy(float)
    c = E.classify(tr[ycol].to_numpy(float), band)
    if inp == "mi":
        mi = mutual_info_classif(X, c, random_state=0)
        return [pool[i] for i in np.argsort(mi)[::-1][:6]], None
    if inp == "forward":  # greedy forward selection, inner split = last 30% of training weeks
        cut = int(len(tr) * 0.7)
        chosen, best = [], np.inf
        for _ in range(6):
            trial = {}
            for f in pool:
                if f in chosen:
                    continue
                cols = chosen + [f]
                Xa, Xb = tr[cols].to_numpy(float)[:cut], tr[cols].to_numpy(float)[cut:]
                if len(np.unique(c[:cut])) < 3:
                    break
                m = E.OrdinalLogit(l2=10).fit(Xa, c[:cut])
                trial[f] = np.mean([E.rps(m.predict(x), cc) for x, cc in zip(Xb, c[cut:])])
            if not trial:
                break
            f, s = min(trial.items(), key=lambda kv: kv[1])
            if s >= best - 1e-4:
                break
            chosen.append(f); best = s
        return (chosen or B["reduced"]), None
    raise ValueError(inp)


class Fitted:
    """A fitted candidate that turns one row of inputs into 3 probabilities."""

    def __init__(self, spec, B, tr, ycol, band):
        self.spec = spec
        cols, extra = resolve_inputs(spec, B, tr, ycol, band)
        self.cols, self.extra = cols, extra
        tr = tr.dropna(subset=cols)
        y = tr[ycol].to_numpy(float)
        c = E.classify(y, band)
        self.band = band
        X = self._design(tr, fit=True, B=B)
        w = None
        if spec.get("halflife"):
            age = (tr["week"].max() - tr["week"]).dt.days.to_numpy() / 7
            w = 0.5 ** (age / spec["halflife"])
        k = spec["kind"]
        if k == "ord":
            self.m = E.OrdinalLogit(l2=spec["l2"]).fit(X, c, w)
        elif k == "clf":
            self.m = spec["make"]().fit(X, c)
        elif k == "reg":
            yt = np.log1p(y / 100) if spec.get("log") else y
            self.m = spec["make"]().fit(X, yt)
            pred = self.m.predict(X)
            self.resid = yt - pred
        self.ok = len(np.unique(c)) == 3

    def _design(self, df, fit=False, B=None):
        X = df[self.cols].to_numpy(float)
        if self.extra == "pca":
            R = df[B["rich"] if B else self.rich_cols].to_numpy(float)
            if fit:
                self.rich_cols = B["rich"]
                self.rmu, self.rsd = np.nanmean(R, 0), np.nanstd(R, 0) + 1e-9
                Rz = np.nan_to_num((R - self.rmu) / self.rsd)
                self.pca = PCA(n_components=3, random_state=0).fit(Rz)
            Rz = np.nan_to_num((R - self.rmu) / self.rsd)
            X = np.hstack([X, self.pca.transform(Rz)])
        if self.extra == "regime":
            crisis = ((df["mom_4w"].abs() > 15) | (df["nw_redsea_pressure"].fillna(0) > 1.5)).to_numpy(float)[:, None]
            X = np.hstack([X, crisis, X * crisis])
        return X

    def predict(self, row: pd.DataFrame) -> np.ndarray | None:
        if row[self.cols].isna().any(axis=None):
            return None
        x = self._design(row)
        k = self.spec["kind"]
        if k == "ord":
            return self.m.predict(x[0])
        if k == "clf":
            p = np.zeros(3)
            cls = getattr(self.m, "classes_", None)
            if cls is None:
                cls = self.m[-1].classes_
            p[np.asarray(cls).astype(int)] = self.m.predict_proba(x)[0]
            return p
        mu = self.m.predict(x)[0]
        sims = mu + self.resid
        if self.spec.get("log"):
            sims = np.expm1(sims) * 100
        b = self.band
        return np.array([(sims < -b).mean(), ((sims >= -b) & (sims <= b)).mean(), (sims > b).mean()])


def run_series(args):
    target, h, df, cfg = args
    zoo_ = zoo()
    band = cfg["stability_band_pct"][target][str(h)]
    ycol = f"y_{h}w"
    g = df[df["target"] == target].sort_values("week").reset_index(drop=True)
    B = blocks(list(g.columns))
    need = list(dict.fromkeys(REDUCED))
    rows, fitted, last_fit = [], {}, None
    weeks = g[(g["week"] >= START) & g[ycol].notna()]["week"].tolist()
    for t in weeks:
        now = g[g["week"] == t]
        known = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()].dropna(subset=need)
        if len(known) < MIN_TRAIN:
            continue
        actual = float(now[ycol].iloc[0])
        base = {"target": target, "h": h, "week": t, "actual": actual,
                "class": int(E.classify(np.array([actual]), band)[0])}
        ck = E.classify(known[ycol].to_numpy(float), band)
        rows.append({**base, "method": "ref_no_change", "p": E.climatology(ck)})
        trend = float(now["mom_4w"].iloc[0]) * h / 4
        p_tr = np.full(3, 0.2); p_tr[int(E.classify(np.array([trend]), band)[0])] = 0.6
        rows.append({**base, "method": "ref_trend", "p": p_tr})
        refit = last_fit is None or (t - last_fit).days >= 7 * REFIT
        if refit:
            last_fit = t
        for name, spec in zoo_.items():
            if refit:
                tr = known
                if spec.get("window"):
                    tr = tr[tr["week"] > t - pd.Timedelta(weeks=spec["window"])]
                try:
                    f = Fitted(spec, B, tr, ycol, band)
                    fitted[name] = f if f.ok else None
                except Exception as e:  # a failing candidate is skipped, not fatal
                    fitted[name] = None
            f = fitted.get(name)
            if f is None:
                continue
            try:
                p = f.predict(now)
            except Exception:
                p = None
            if p is not None and np.isfinite(p).all():
                rows.append({**base, "method": name, "p": np.clip(p, 1e-6, 1) / np.clip(p, 1e-6, 1).sum()})
    print(f"done {target} h={h}: {len(rows)} forecasts", flush=True)
    return rows


# ---------------------------------------------------------------- post-processing

def to_frame(rows):
    d = pd.DataFrame(rows)
    P = np.vstack(d.pop("p").to_numpy()); d["p_down"], d["p_stable"], d["p_up"] = P.T
    return d


def add_champion(d):
    parts = d[d["method"].isin(["_part logit", "_part boosting", "_part forest"])]
    piv = parts.pivot_table(index=["target", "h", "week", "actual", "class"], columns="method",
                            values=["p_down", "p_stable", "p_up"])
    ch = pd.DataFrame({c: piv[c].mean(axis=1) for c in ["p_down", "p_stable", "p_up"]}).reset_index()
    ch["method"] = CHAMPION
    return pd.concat([d[~d["method"].str.startswith("_part")], ch], ignore_index=True)


def score(d):
    P = d[["p_down", "p_stable", "p_up"]].to_numpy()
    d["rps"] = [E.rps(p, c) for p, c in zip(P, d["class"])]
    d["hit"] = P.argmax(1) == d["class"].to_numpy()
    nc = d["method"] == "ref_no_change"
    d.loc[nc, "hit"] = d.loc[nc, "class"] == 1
    return d


def add_calibrated(d, names, min_past=20):
    """Temperature scaling on the model's own past forecasts (already resolved)."""
    out = []
    for (target, h, name), g in d[d["method"].isin(names)].groupby(["target", "h", "method"]):
        g = g.sort_values("week").reset_index(drop=True)
        P = g[["p_down", "p_stable", "p_up"]].to_numpy()
        res = (g["week"] + pd.to_timedelta(7 * g["h"], unit="D")).to_numpy()
        new = []
        for i, t in enumerate(g["week"].to_numpy()):
            past = res <= t
            if past.sum() < min_past:
                new.append(None); continue
            best_T, best = 1.0, np.inf
            for T in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0):
                Q = P[past] ** (1 / T); Q /= Q.sum(1, keepdims=True)
                ll = -np.log(Q[np.arange(past.sum()), g["class"].to_numpy()[past]] + 1e-9).mean()
                if ll < best:
                    best, best_T = ll, T
            q = P[i] ** (1 / best_T); new.append(q / q.sum())
        c = g.copy(); c["method"] = name + " +calibrated"
        keep = [i for i, v in enumerate(new) if v is not None]
        c = c.iloc[keep].copy()
        Q = np.vstack([new[i] for i in keep]); c["p_down"], c["p_stable"], c["p_up"] = Q.T
        out.append(c)
    return pd.concat([d] + out, ignore_index=True) if out else d


def common(d):
    full = d.groupby(["target", "h", "week"])["method"].transform("nunique")
    mx = d.groupby(["target", "h"])["method"].transform("nunique")
    return d[full == mx]


def rank_selection(d):
    sel = d[(d["week"] < HOLDOUT)]
    s = sel.groupby(["method", "target", "h"])["rps"].mean().unstack(["target", "h"])
    s = s.dropna()
    return s.mean(axis=1).sort_values()


def bootstrap_gain(d, a, b, block=8, n=2000, seed=0):
    """Mean RPS of b minus a on the holdout (positive = a better), pooled over the 4
    series week by week, with a block bootstrap 90% interval."""
    hold = d[d["week"] >= HOLDOUT]
    x = hold[hold["method"] == a].groupby("week")["rps"].mean()
    y = hold[hold["method"] == b].groupby("week")["rps"].mean()
    diff = (y - x).dropna().sort_index().to_numpy()
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(diff) / block))
    sims = []
    for _ in range(n):
        starts = rng.integers(0, len(diff) - block + 1, nb)
        sims.append(np.concatenate([diff[s:s + block] for s in starts])[:len(diff)].mean())
    lo, hi = np.percentile(sims, [5, 95])
    return diff.mean(), lo, hi, float((np.array(sims) > 0).mean())


def averages_of_best(d, ranking, ks=(3, 5)):
    out = []
    names = [m for m in ranking.index if not m.startswith("ref") and "+calibrated" not in m]
    for k in ks:
        top = names[:k]
        g = d[d["method"].isin(top)]
        a = g.groupby(["target", "h", "week", "actual", "class"])[["p_down", "p_stable", "p_up"]].mean().reset_index()
        a["method"] = f"average of top-{k} (chosen on selection period)"
        out.append(a)
    return pd.concat([d] + out, ignore_index=True)


def magnitude(df, cfg):
    """Magnitude candidates on the reduced inputs (the direction tests showed extra
    inputs add noise): kNN, quantile regression, boosting quantile, their average."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_and_learn as TL  # noqa: E402
    rows = []
    for target in TARGETS:
        g = df[df["target"] == target].sort_values("week").reset_index(drop=True)
        for h in cfg["horizons_weeks"]:
            ycol = f"y_{h}w"
            cache, last = {}, None
            for t in g[(g["week"] >= START) & g[ycol].notna()]["week"]:
                now = g[g["week"] == t]
                tr = g[(g["week"] + pd.Timedelta(weeks=h) <= t) & g[ycol].notna()].dropna(subset=REDUCED)
                if len(tr) < MIN_TRAIN or now[REDUCED].isna().any(axis=None):
                    continue
                X, y = tr[REDUCED].to_numpy(float), tr[ycol].to_numpy(float)
                x = now[REDUCED].to_numpy(float)[0]
                base = {"target": target, "h": h, "week": t, "actual": float(now[ycol].iloc[0]), "track": "market"}
                spread = np.quantile(y, E.QUANTILES) - np.median(y)
                rows.append({**base, "method": "ref_no_change", "q": spread})
                q = {}
                for k in (20, 40, 80):
                    q[f"kNN k={k}"] = E.knn_quantiles(X, y, x, k=k)
                if last is None or (t - last).days >= 7 * REFIT:
                    last = t
                    cache["hgb"] = [HistGradientBoostingRegressor(loss="quantile", quantile=a, max_depth=2,
                                    learning_rate=0.05, max_iter=150, min_samples_leaf=10).fit(X, y) for a in E.QUANTILES]
                    cache["qr"] = (X.mean(0), X.std(0) + 1e-9,
                                   [QuantileRegressor(quantile=a, alpha=0.05, solver="highs").fit((X - X.mean(0)) / (X.std(0) + 1e-9), y)
                                    for a in E.QUANTILES])
                q["boosting-quantile"] = np.sort([m.predict(x.reshape(1, -1))[0] for m in cache["hgb"]])
                mu, sd, ms = cache["qr"]
                q["quantile regression"] = np.sort([m.predict(((x - mu) / sd).reshape(1, -1))[0] for m in ms])
                q["average kNN40+QR+boosting"] = (q["kNN k=40"] + q["quantile regression"] + q["boosting-quantile"]) / 3
                for k, v in q.items():
                    rows.append({**base, "method": k, "q": np.asarray(v)})
    m = pd.DataFrame(rows)
    Q = np.vstack(m.pop("q").to_numpy()); m["q10"], m["q50"], m["q90"] = Q.T
    m = TL.add_conformal(m)
    m["abs_err"] = (m["q50"] - m["actual"]).abs()
    m["covered"] = (m["actual"] >= m["q10"]) & (m["actual"] <= m["q90"])
    m["width"] = m["q90"] - m["q10"]
    m["pinball"] = [E.pinball(np.array(q), y) for q, y in zip(m[["q10", "q50", "q90"]].to_numpy(), m["actual"])]
    full = m.groupby(["target", "h", "week"])["method"].transform("nunique")
    m = m[full == full.max()]
    m["period"] = np.where(m["week"] >= HOLDOUT, "holdout", "select")
    return m.groupby(["period", "method"]).agg(n=("pinball", "size"), pinball=("pinball", "mean"),
                                               mae=("abs_err", "mean"), coverage80=("covered", "mean"),
                                               width80=("width", "mean")).reset_index()


def main(rescore=False):
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].merge(news_halflives(pd.DatetimeIndex(sorted(df["week"].unique()))),
                                left_on="week", right_index=True, how="left")
    if rescore:
        d = pd.read_csv(OUT / "ms_forecasts_raw.csv", parse_dates=["week"])
    else:
        jobs = [(t, h, df, cfg) for t in TARGETS for h in cfg["horizons_weeks"]]
        with Pool(2) as pool:
            rows = sum(pool.map(run_series, jobs), [])
        d = to_frame(rows)
        d.to_csv(OUT / "ms_forecasts_raw.csv", index=False, float_format="%.5f")
    d = add_champion(d)
    d = score(d)
    d = common(d)
    ranking = rank_selection(d)
    top_single = [m for m in ranking.index if not m.startswith("ref")][:8]
    d = add_calibrated(d, top_single + [CHAMPION])
    d = averages_of_best(d, ranking)
    d = score(d)
    d = common(d)
    ranking = rank_selection(d)
    # per-method summary
    d["period"] = np.where(d["week"] >= HOLDOUT, "holdout", "select")
    summ = d.groupby(["method", "period", "target", "h"]).agg(n=("rps", "size"), rps=("rps", "mean"),
                                                             hit=("hit", "mean")).reset_index()
    summ.to_csv(OUT / "ms_scores.csv", index=False, float_format="%.4f")
    hold = d[d["period"] == "holdout"].groupby(["method", "target", "h"])["rps"].mean().unstack(["target", "h"])
    table = pd.DataFrame({"select_rps": ranking, "holdout_rps": hold.mean(axis=1).reindex(ranking.index)})
    for col in hold.columns:
        table[f"hold_{col[0].split('_')[1]}_{col[1]}w"] = hold[col].reindex(ranking.index)
    hit = d[d["period"] == "holdout"].groupby("method")["hit"].mean()
    table["holdout_hit"] = hit.reindex(ranking.index)
    # quarter stability vs no change, big moves
    hd = d[d["period"] == "holdout"]
    nc = hd[hd["method"] == "ref_no_change"].set_index(["target", "h", "week"])["rps"]
    qs, big = {}, {}
    bands = load_config()["stability_band_pct"]
    hd = hd.assign(bigmove=[abs(a) > 2 * bands[t][str(h)] for a, t, h in zip(hd["actual"], hd["target"], hd["h"])])
    for mname, g in hd.groupby("method"):
        x = g.set_index(["target", "h", "week"])["rps"]
        diff = (nc - x).dropna().groupby(level="week").mean()
        q = diff.groupby(diff.index.to_period("Q")).mean()
        qs[mname] = f"{int((q > 0).sum())}/{len(q)}"
        big[mname] = g[g["bigmove"]]["rps"].mean()
    table["quarters_beating_no_change"] = pd.Series(qs)
    table["holdout_rps_big_moves"] = pd.Series(big)
    table.index.name = "method"
    table.to_csv(OUT / "ms_ranking.csv", float_format="%.4f")
    top3 = [m for m in ranking.index if not m.startswith("ref")][:3]
    boots = []
    for m in top3 + [CHAMPION]:
        for ref in ("ref_no_change", CHAMPION, V01):
            if m == ref:
                continue
            g, lo, hi, pr = bootstrap_gain(d, m, ref)
            boots.append({"method": m, "vs": ref, "rps_gain": g, "ci90_low": lo, "ci90_high": hi, "p_better": pr})
    pd.DataFrame(boots).to_csv(OUT / "ms_bootstrap.csv", index=False, float_format="%.4f")
    mag = magnitude(df, cfg)
    mag.to_csv(OUT / "ms_magnitude.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print(table.round(3).to_string())
    print(pd.DataFrame(boots).round(4).to_string(index=False))
    print(mag.round(3).to_string(index=False))


if __name__ == "__main__":
    main(rescore=len(sys.argv) > 1 and sys.argv[1] == "--rescore")
