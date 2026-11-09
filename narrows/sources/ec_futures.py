"""INE Containerized Freight Index (Europe) futures, ticker EC.

- Every listed contract since launch (18 Aug 2023), daily, from Sina via akshare.
- From these we derive: the front-month series (predicted index) and the curve
  (front vs. next contracts, a model signal).
Contract months: Feb, Apr, Jun, Aug, Oct, Dec.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..common import CLEAN, EC_START, log, merge_csv

MONTHS = [2, 4, 6, 8, 10, 12]


def contract_codes(today: date | None = None) -> list[str]:
    today = today or date.today()
    codes = []
    for year in range(2023, today.year + 2):
        for m in MONTHS:
            d = date(year, m, 1)
            if d < date(2023, 10, 1):  # first listed contract was EC2310
                continue
            if (d.year - today.year) * 12 + (d.month - today.month) > 14:
                continue
            codes.append(f"EC{str(year)[2:]}{m:02d}")
    return codes


def fetch_contracts() -> pd.DataFrame:
    import akshare as ak

    frames = []
    for code in contract_codes():
        try:
            df = ak.futures_zh_daily_sina(symbol=code)
        except Exception as e:  # unlisted or not yet traded
            log.info("EC %s skipped: %s", code, e)
            continue
        if df is None or df.empty:
            continue
        df = df.rename(columns=str.lower)
        df["contract"] = code
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        frames.append(df[["date", "contract", "open", "high", "low", "close", "volume", "hold", "settle"]])
    if not frames:
        raise RuntimeError("No EC contract data returned")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["date"] >= EC_START]
    merge_csv(out, CLEAN / "ec_contracts.csv", ["date", "contract"])
    log.info("EC contracts: %d rows, %d contracts, last %s", len(out), out["contract"].nunique(), out["date"].max())
    return out


def derive(contracts: pd.DataFrame) -> pd.DataFrame:
    """Front month = contract with highest open interest each day; curve = settle of next two expiries."""
    c = contracts.copy()
    for col in ("settle", "close", "hold"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[c["settle"] > 0]  # Sina sometimes returns 0 for a missing settlement
    c["expiry"] = pd.to_datetime("20" + c["contract"].str[2:4] + "-" + c["contract"].str[4:6] + "-01")
    rows = []
    for d, g in c.groupby("date"):
        g = g.sort_values("expiry")
        front = g.loc[g["hold"].idxmax()] if g["hold"].notna().any() else g.iloc[0]
        after = g[g["expiry"] > front["expiry"]]
        nxt = after.iloc[0] if len(after) else None
        rows.append({
            "date": d,
            "front_contract": front["contract"],
            "front_settle": front["settle"],
            "next_contract": nxt["contract"] if nxt is not None else "",
            "next_settle": nxt["settle"] if nxt is not None else "",
            "curve_slope_pct": round((nxt["settle"] / front["settle"] - 1) * 100, 3) if nxt is not None and front["settle"] else "",
        })
    out = pd.DataFrame(rows)
    merge_csv(out, CLEAN / "ec_front.csv", ["date"])
    log.info("EC front/curve: %d days", len(out))
    return out


def fetch() -> pd.DataFrame:
    return derive(fetch_contracts())
