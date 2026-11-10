"""Weekly feature panel for the Narrows model (v0.1).

One row per Thursday (Drewry release week) and per forecast target.
Every feature uses only data that was public on that Thursday:
- Drewry: the value published that Thursday and earlier ones.
- EC futures: settlements up to the last trading day on or before Thursday.
- PortWatch: daily transits up to Thursday minus PORTWATCH_LAG_DAYS (the IMF
  publishes with a delay of about a week).
- Brent: closes up to Thursday.
Targets (future moves) are stored in separate columns prefixed "y_" and are
never used as inputs.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common import CLEAN, CONFIG, DATA

PORTWATCH_LAG_DAYS = 10
ROUTES = ["shanghai_rotterdam", "shanghai_genoa", "shanghai_los_angeles"]
STRAITS = {"Bab el-Mandeb Strait": "bab", "Suez Canal": "suez", "Cape of Good Hope": "cape"}


def load_config() -> dict:
    return json.loads((CONFIG / "model.json").read_text())


def pct(a, b):
    return (a / b - 1.0) * 100.0


def _drewry() -> pd.DataFrame:
    d = pd.read_csv(DATA / "model" / "drewry_weekly.csv", parse_dates=["week"])
    return d.pivot(index="week", columns="route", values="usd_per_40ft").sort_index()


def _traffic(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "portwatch_straits.csv", parse_dates=["date"])
    p = p[p["strait"].isin(STRAITS)]
    daily = p.pivot_table(index="date", columns="strait", values="n_total").rename(columns=STRAITS)
    daily = daily.asfreq("D")
    m28 = daily.rolling(28, min_periods=20).mean()
    m364 = daily.rolling(364, min_periods=300).mean()
    ratio = pct(m28, m364)  # last 4 weeks vs last year, in %
    out = {}
    for w in weeks:
        cut = w - pd.Timedelta(days=PORTWATCH_LAG_DAYS)
        r = ratio.loc[:cut]
        out[w] = r.iloc[-1] if len(r) else pd.Series(dtype=float)
    return pd.DataFrame(out).T.add_prefix("traffic_").reindex(weeks)


def _brent(weeks: pd.DatetimeIndex) -> pd.Series:
    m = pd.read_csv(CLEAN / "markets.csv", parse_dates=["date"])
    b = m[m["series"] == "brent_usd_bbl"].set_index("date")["value"].sort_index()
    now = b.reindex(weeks, method="ffill")
    before = b.reindex(weeks - pd.Timedelta(days=28), method="ffill").values
    return pd.Series(pct(now.values, before), index=weeks, name="brent_4w")


class EC:
    """Same-contract EC settlements, so that a roll never looks like a price move."""

    def __init__(self):
        c = pd.read_csv(CLEAN / "ec_contracts.csv", parse_dates=["date"])
        c = c[c["settle"] > 0]
        self.settle = c.pivot(index="date", columns="contract", values="settle").sort_index()
        self.hold = c.pivot(index="date", columns="contract", values="hold").sort_index()
        self.dates = self.settle.index

    @staticmethod
    def expiry(contract: str) -> pd.Timestamp:
        # ECyymm expires in the last days of its month; treat the 20th as the cut-off.
        return pd.Timestamp(2000 + int(contract[2:4]), int(contract[4:6]), 20)

    def last_day(self, t):
        d = self.dates[self.dates <= t]
        return d[-1] if len(d) else None

    def pick(self, t, until=None):
        """Most traded contract on the last day <= t, alive until `until`."""
        d0 = self.last_day(t)
        if d0 is None:
            return None, None
        h = self.hold.loc[d0].dropna()
        if until is not None:
            h = h[[self.expiry(c) >= until for c in h.index]]
        return (h.idxmax() if len(h) else None), d0

    def value(self, contract, t):
        d = self.last_day(t)
        if contract is None or d is None or contract not in self.settle:
            return np.nan
        s = self.settle[contract].loc[:d].dropna()
        return s.iloc[-1] if len(s) and (d - s.index[-1]).days < 7 else np.nan


def build() -> pd.DataFrame:
    cfg = load_config()
    dw = _drewry()
    weeks = dw.index
    ec = EC()
    feats = pd.DataFrame(index=weeks)
    feats = feats.join(_traffic(weeks)).join(_brent(weeks))

    # EC features on the main contract
    ec1, ec4, slope, ec_basis = [], [], [], []
    front = pd.read_csv(CLEAN / "ec_front.csv", parse_dates=["date"]).set_index("date")["curve_slope_pct"]
    for w in weeks:
        c, d0 = ec.pick(w, until=w)
        if c is None:
            ec1.append(np.nan); ec4.append(np.nan); slope.append(np.nan); ec_basis.append(np.nan)
            continue
        v0 = ec.value(c, w)
        ec1.append(pct(v0, ec.value(c, w - pd.Timedelta(days=7))))
        ec4.append(pct(v0, ec.value(c, w - pd.Timedelta(days=28))))
        s = front.loc[:w]
        slope.append(s.iloc[-1] if len(s) else np.nan)
        ec_basis.append(v0)
    feats["ec_1w"], feats["ec_4w"], feats["ec_slope"] = ec1, ec4, slope
    ec_level = pd.Series(ec_basis, index=weeks)

    rows = []
    for target in ["shanghai_rotterdam", "shanghai_genoa", "ec_front"]:
        f = feats.copy()
        f["target"] = target
        if target == "ec_front":
            lvl = ec_level
            f["mom_1w"], f["mom_4w"] = feats["ec_1w"], feats["ec_4w"]
        else:
            lvl = dw[target]
            f["mom_1w"] = pct(lvl, lvl.shift(1))
            f["mom_4w"] = pct(lvl, lvl.shift(4))
        # EC level vs Drewry Rotterdam, deviation from its own 26-week median (premium/discount)
        ratio = np.log(ec_level / dw["shanghai_rotterdam"])
        f["ec_premium"] = (ratio - ratio.rolling(26, min_periods=12).median()) * 100
        la = dw["shanghai_los_angeles"]
        f["control_4w"] = pct(la, la.shift(4))
        f["covid"] = (weeks >= cfg["covid_regime"][0]) & (weeks <= cfg["covid_regime"][1])
        for h in cfg["horizons_weeks"]:
            if target == "ec_front":
                ys = []
                for w in weeks:
                    c, _ = ec.pick(w, until=w + pd.Timedelta(days=7 * h))
                    ys.append(pct(ec.value(c, w + pd.Timedelta(days=7 * h)), ec.value(c, w))
                              if c and w + pd.Timedelta(days=7 * h) <= ec.dates[-1] else np.nan)
                f[f"y_{h}w"] = ys
            else:
                f[f"y_{h}w"] = pct(lvl.shift(-h), lvl)
        rows.append(f.reset_index(names="week"))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "model" / "features_weekly.csv", index=False, float_format="%.4f")
    return out


if __name__ == "__main__":
    df = build()
    print(df.groupby("target")[["mom_4w", "ec_4w", "traffic_bab", "y_1w", "y_4w"]].count())
