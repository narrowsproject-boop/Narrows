"""Model search, round 3: does calendar, the other Drewry routes, or real container
capacity through Suez and the Cape improve the round-2 winner?

Round-2 winner: ridge regression of the % move, turned into 3 probabilities with its
past residuals, trained on Rotterdam and Genoa together ("ridge->classes a=3 pooled").
New input blocks, all available since 2019 and known on the Thursday:
- calendar: week of year (sin/cos), weeks to Chinese New Year, number of 1st/15th of the
  month inside the horizon (the usual dates for carriers' GRI / FAK increases);
- other routes: Drewry composite, transpacific, backhaul and transatlantic moves;
- capacity: container capacity (TEU-weighted) through Suez + Cape vs last year, and its
  recent speed of change, as a stand-in for blank sailings (no free history).
Same protocol: walk-forward, no leakage, selection Jun 2024-Jun 2025, holdout after.

Usage: python scripts/model_search3.py  -> data/model/ms3_*.csv
"""
from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narrows.common import CLEAN, DATA  # noqa: E402
from narrows.model import engines as E  # noqa: E402
from narrows.model.features import PORTWATCH_LAG_DAYS, build, load_config, pct  # noqa: E402
import model_search as MS  # noqa: E402
import model_search2 as M2  # noqa: E402

warnings.filterwarnings("ignore")
OUT = DATA / "model"
REDUCED = MS.REDUCED
CNY = pd.to_datetime(["2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
                      "2025-01-29", "2026-02-17", "2027-02-06"])


def calendar(weeks: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    out = pd.DataFrame(index=weeks)
    woy = weeks.isocalendar().week.to_numpy().astype(float)
    out["cal_sin"], out["cal_cos"] = np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52)
    nxt = [min([c for c in CNY if c >= w - pd.Timedelta(days=21)], default=CNY[-1]) for w in weeks]
    out["cal_weeks_to_cny"] = np.clip([(c - w).days / 7 for c, w in zip(nxt, weeks)], -3, 12)
    out["cal_gri_dates"] = [sum(1 for d in pd.date_range(w + pd.Timedelta(days=1), w + pd.Timedelta(weeks=h)) if d.day in (1, 15))
                            for w in weeks]
    return out


def routes(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.read_csv(CLEAN / "drewry_wci.csv", parse_dates=["date"])
    p = d.pivot_table(index="date", columns="route", values="usd_per_40ft").sort_index()
    p = p.reindex(weeks, method="ffill", tolerance=pd.Timedelta(days=3))
    out = pd.DataFrame(index=weeks)
    for r, name in (("composite", "rt_composite"), ("los_angeles_shanghai", "rt_backhaul_tp"),
                    ("rotterdam_shanghai", "rt_backhaul_ae"), ("rotterdam_new_york", "rt_transatlantic")):
        if r in p:
            out[f"{name}_1w"] = pct(p[r], p[r].shift(1))
            out[f"{name}_4w"] = pct(p[r], p[r].shift(4))
    return out


def capacity(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.read_csv(CLEAN / "portwatch_straits.csv", parse_dates=["date"])
    s = s[s["strait"].isin(["Suez Canal", "Cape of Good Hope"])].groupby("date")["capacity_container"].sum().asfreq("D")
    m14, m28, m364 = s.rolling(14, min_periods=10).mean(), s.rolling(28, min_periods=20).mean(), s.rolling(364, min_periods=300).mean()
    lag = weeks - pd.Timedelta(days=PORTWATCH_LAG_DAYS)
    return pd.DataFrame({"cap_ae_vs_year": pct(m28, m364).reindex(lag).values,
                         "cap_ae_speed": pct(m14, m28.shift(14)).reindex(lag).values}, index=weeks)


BLOCKS = {
    "calendar": ["cal_sin", "cal_cos", "cal_weeks_to_cny", "cal_gri_dates"],
    "routes": ["rt_composite_1w", "rt_composite_4w", "rt_backhaul_ae_4w", "rt_backhaul_tp_4w", "rt_transatlantic_4w"],
    "capacity": ["cap_ae_vs_year", "cap_ae_speed"],
}


def candidates():
    C = {}
    sets = {"reduced": REDUCED}
    for b, cols in BLOCKS.items():
        sets[f"reduced+{b}"] = REDUCED + cols
    sets["reduced+calendar+capacity"] = REDUCED + BLOCKS["calendar"] + BLOCKS["capacity"]
    sets["reduced+all new"] = REDUCED + BLOCKS["calendar"] + BLOCKS["routes"] + BLOCKS["capacity"]
    for name, cols in sets.items():
        for a in (3.0, 30.0):
            C[f"ridge->classes a={a:g} pooled | {name}"] = (cols, True, "reg", a)
        C[f"multilogit L2 C=0.5 pooled | {name}"] = (cols, True, "clf",
            lambda: M2.scaled(M2.LogisticRegression(C=0.5, max_iter=4000)))
    return C


def prepare():
    cfg = load_config()
    df = build()
    df = df[~df["covid"]].copy()
    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    df = df.merge(routes(weeks), left_on="week", right_index=True, how="left")
    df = df.merge(capacity(weeks), left_on="week", right_index=True, how="left")
    parts = []
    for h in cfg["horizons_weeks"]:
        c = calendar(weeks, h)
        parts.append(df.merge(c, left_on="week", right_index=True, how="left").assign(_h=h))
    return cfg, parts


def job(args):
    target, h, df, cfg = args
    M2.candidates = candidates  # reuse round-2 machinery with this round's candidates
    return M2.run_series((target, h, df, cfg))


def main():
    cfg, parts = prepare()
    jobs = [(t, h, p, cfg) for p in parts for h in [int(p["_h"].iloc[0])] for t in MS.TARGETS]
    with Pool(2) as pool:
        rows = sum(pool.map(job, jobs), [])
    d = pd.DataFrame(rows)
    ref = pd.read_csv(OUT / "ms2_forecasts_all.csv", parse_dates=["week"])
    keep = ["ref_no_change", "ref_trend", MS.CHAMPION]
    d = pd.concat([d, ref[ref["method"].isin(keep)][d.columns.intersection(ref.columns)]], ignore_index=True)
    d = MS.score(d)
    d.to_csv(OUT / "ms3_forecasts.csv", index=False, float_format="%.5f")
    base = "ridge->classes a=3 pooled | reduced"
    d["period"] = np.where(d["week"] >= MS.HOLDOUT, "holdout", "select")
    key = ["target", "h", "week"]
    b = d[d["method"] == base].set_index(key)["rps"].rename("rps_base")
    rows, boots = [], []
    for m, g in d.groupby("method"):
        x = g.set_index(key).join(b, how="inner").reset_index()  # same weeks as the base
        for per_name, xx in x.groupby("period"):
            rows.append({"method": m, "period": per_name, "weeks": xx["week"].nunique(), "rps": xx["rps"].mean(),
                         "rps_base_same_weeks": xx["rps_base"].mean(), "hit": xx["hit"].mean(),
                         "last_week": xx["week"].max().date()})
        if m == base:
            continue
        for per_name in ("select", "holdout"):
            xx = x[x["period"] == per_name]
            diff = (xx.groupby("week")["rps_base"].mean() - xx.groupby("week")["rps"].mean()).sort_index().to_numpy()
            if len(diff) < 16:
                continue
            rng = np.random.default_rng(0); nb = int(np.ceil(len(diff) / 8))
            sims = [np.concatenate([diff[s0:s0 + 8] for s0 in rng.integers(0, len(diff) - 7, nb)])[:len(diff)].mean() for _ in range(2000)]
            boots.append({"method": m, "period": per_name, "gain_vs_round2_winner": diff.mean(),
                          "ci90_low": np.percentile(sims, 5), "ci90_high": np.percentile(sims, 95), "p_better": float(np.mean(np.array(sims) > 0))})
    table = pd.DataFrame(rows).pivot_table(index="method", columns="period",
                                           values=["rps", "rps_base_same_weeks", "hit", "weeks"]).round(4)
    table.columns = [f"{a}_{b}" for a, b in table.columns]
    table = table.sort_values("rps_select")
    table.to_csv(OUT / "ms3_ranking.csv")
    boots = pd.DataFrame(boots)
    boots.to_csv(OUT / "ms3_bootstrap.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(table.round(3).to_string())
    print(boots.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
