"""Build the clean weekly Drewry panel the model uses.

- Every value is aligned to the Thursday of its week (some releases are dated Friday).
- Missing weeks are filled by linear interpolation and flagged, never silently.
  Christmas/New Year weeks are a known Drewry gap (no release).
"""
from __future__ import annotations

import pandas as pd

from .common import CLEAN, DATA, log

OUT = DATA / "model" / "drewry_weekly.csv"


def build() -> pd.DataFrame:
    d = pd.read_csv(CLEAN / "drewry_wci.csv")
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"] - pd.to_timedelta((d["date"].dt.dayofweek - 3) % 7, unit="D")
    d = d.sort_values(["route", "week", "date"]).drop_duplicates(["route", "week"], keep="last")
    frames = []
    for route, g in d.groupby("route"):
        idx = pd.date_range(g["week"].min(), g["week"].max(), freq="7D")
        s = g.set_index("week")["usd_per_40ft"].reindex(idx)
        filled = s.isna()
        s = s.interpolate(limit_area="inside")
        frames.append(pd.DataFrame({"week": idx.strftime("%Y-%m-%d"), "route": route,
                                    "usd_per_40ft": s.round(0).values,
                                    "filled": filled.values}))
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    log.info("Weekly Drewry panel: %d rows, %d filled weeks", len(out), int(out["filled"].sum()))
    return out


if __name__ == "__main__":
    build()
