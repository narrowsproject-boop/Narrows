"""Context series collected from now on (no free history): HARPEX and bunker fuel prices.

Shown on the site, accumulated for a later model version. Best effort: a failure here never
blocks the daily run.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from ..common import CLEAN, get, log, merge_csv

HARPEX_URL = "https://www.harperpetersen.com/harpex"
BUNKER_URL = "https://shipandbunker.com/prices"


def harpex() -> pd.DataFrame:
    """The page shows the last ~4 weekly HARPEX values in a small table."""
    tables = pd.read_html(get(HARPEX_URL).text)
    rows = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if not any("harpex" in c for c in cols):
            continue
        date_col = t.columns[0]
        val_col = t.columns[[i for i, c in enumerate(cols) if "harpex" in c][0]]
        for _, r in t.iterrows():
            try:
                d = pd.to_datetime(str(r[date_col]), dayfirst=True).strftime("%Y-%m-%d")
                v = float(str(r[val_col]).replace(",", ""))
            except (ValueError, TypeError):
                continue
            rows.append({"date": d, "harpex": v})
    if not rows:
        raise RuntimeError("HARPEX table not found")
    df = pd.DataFrame(rows)
    merge_csv(df, CLEAN / "harpex.csv", ["date"])
    log.info("HARPEX: %d rows, last %s", len(df), df["date"].max())
    return df


def bunker() -> pd.DataFrame:
    """Today's Singapore and Rotterdam prices (USD/mt) as shown on the public prices page."""
    from bs4 import BeautifulSoup

    text = re.sub(r"\s+", " ", BeautifulSoup(get(BUNKER_URL).text, "lxml").get_text(" "))
    rows = []
    for port in ("Singapore", "Rotterdam"):
        m = re.search(port + r"[^$]{0,80}?\$\s?([\d,]+\.\d{2})", text)
        if m:
            rows.append({"date": date.today().isoformat(), "port": port, "usd_per_mt": float(m.group(1).replace(",", "")), "grade": "VLSFO (headline, unconfirmed)"})
    if not rows:
        raise RuntimeError("Bunker prices not found on page")
    df = pd.DataFrame(rows)
    merge_csv(df, CLEAN / "bunker.csv", ["date", "port"])
    log.info("Bunker: %s", rows)
    return df
