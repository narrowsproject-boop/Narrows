"""Context series collected from now on (no free history): HARPEX and bunker fuel prices.

Shown on the site, accumulated for a later model version. Best effort: a failure here never
blocks the daily run.
"""
from __future__ import annotations

import re
from datetime import date
from io import StringIO

import pandas as pd

from ..common import CLEAN, get, log, merge_csv

HARPEX_URL = "https://www.harperpetersen.com/harpex"
BUNKER_URL = "https://shipandbunker.com/prices"


def harpex() -> pd.DataFrame:
    """The page shows the last ~4 weekly HARPEX values in a small table."""
    tables = pd.read_html(StringIO(get(HARPEX_URL).text))
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


CANCELLED_URL = "https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/cancelled-sailings-tracker"


def cancelled_sailings() -> pd.DataFrame:
    """Drewry's weekly Cancelled Sailings Tracker (free headline): blank sailings announced
    for the next 5 weeks on the East-West trades, cancellation rate, and Asia-North
    Europe/Med share. No free history: accumulated from now on for a later model version."""
    from bs4 import BeautifulSoup

    text = re.sub(r"\s+", " ", BeautifulSoup(get(CANCELLED_URL).text, "lxml").get_text(" "))
    text = text.replace("–", "-").replace("—", "-")
    d = re.search(r"Weekly analysis:\s*(\d{1,2} \w{3,9} \d{4})", text)
    n = re.search(r"(\d+)\s+(?:blank|cancelled)\s+sailings", text, re.I)
    tot = re.search(r"out of (?:a total of )?(\d+)\s+(?:planned|scheduled)", text, re.I)
    rate = re.search(r"(\d+)%\s+cancellation rate", text, re.I)
    ane = re.search(r"Asia-(?:North )?Europe(?:\s*/\s*|\s*&\s*|\s+and\s+)Med\w*[^%]{0,40}?(\d+)%", text, re.I)
    wk = re.search(r"week (\d+)[^)]*\)\s*(?:and|to|through)\s*week (\d+)", text, re.I)
    if not (d and n):
        raise RuntimeError("Cancelled sailings figures not found on page")
    row = {"date": pd.to_datetime(d.group(1)).strftime("%Y-%m-%d"),
           "blank_sailings": int(n.group(1)),
           "scheduled_sailings": int(tot.group(1)) if tot else None,
           "cancellation_rate_pct": int(rate.group(1)) if rate else None,
           "asia_europe_share_pct": int(ane.group(1)) if ane else None,
           "week_from": int(wk.group(1)) if wk else None, "week_to": int(wk.group(2)) if wk else None}
    df = pd.DataFrame([row])
    merge_csv(df, CLEAN / "cancelled_sailings.csv", ["date"])
    log.info("Cancelled sailings: %s", row)
    return df
