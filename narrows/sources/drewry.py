"""Drewry World Container Index: the composite and its 8 East-West routes.

Forecast routes: Shanghai -> Rotterdam, Shanghai -> Genoa (Los Angeles is the control). The composite and the
other routes (backhauls, New York, transatlantic) are context for the dashboard.

Published every Thursday at drewry.co.uk. Three ways in, tried in order:
  1. akshare.drewry_wci_index (reads the Infogram chart behind Drewry's page, multi-year weekly history)
  2. the Drewry page text for the current week ("Shanghai–Rotterdam ... to $2,443 per 40ft")
  3. backfill from weekly press reprints of Drewry's release (see backfill_from_press)
Unit: USD per 40ft container, spot.
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import CLEAN, HISTORY_START, get, log, merge_csv

PAGE = "https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry"
ROUTES = {
    "shanghai_rotterdam": {"akshare": "shanghai-rotterdam", "pattern": r"Shanghai\s*(?:–|-|to)\s*Rotterdam"},
    "shanghai_genoa": {"akshare": "shanghai-genoa", "pattern": r"Shanghai\s*(?:–|-|to)\s*Genoa"},
    "shanghai_los_angeles": {"akshare": "shanghai-los angeles", "pattern": r"Shanghai\s*(?:–|-|to)\s*Los Angeles"},
}
# context series: akshare only (the weekly page text is parsed for the three core routes)
EXTRA = {
    "composite": "composite",
    "rotterdam_shanghai": "rotterdam-shanghai",
    "los_angeles_shanghai": "los angeles-shanghai",
    "shanghai_new_york": "shanghai-new york",
    "new_york_rotterdam": "new york-rotterdam",
    "rotterdam_new_york": "rotterdam-new york",
}
OUT = CLEAN / "drewry_wci.csv"
KEYS = ["date", "route"]


def from_akshare() -> pd.DataFrame:
    import akshare as ak

    def one(route: str, symbol: str) -> pd.DataFrame:
        df = ak.drewry_wci_index(symbol=symbol)
        df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "usd_per_40ft"})
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["route"] = route
        df["source"] = "drewry_infogram"
        return df[["date", "route", "usd_per_40ft", "source"]]

    frames = [one(route, cfg["akshare"]) for route, cfg in ROUTES.items()]
    for route, symbol in EXTRA.items():
        try:
            frames.append(one(route, symbol))
        except Exception as e:  # context only: never fail the run for these
            log.warning("Drewry %s not available: %s", route, e)
    return pd.concat(frames, ignore_index=True)


VALUE_RE = r"[^$]{0,120}?(?:to|at)\s*(?:US)?\$\s?([\d,]{3,6})\s*per\s*40\s?ft"


# patterns for the context series in the weekly release text, so the daily run keeps them current
PAGE_EXTRA = {
    "composite": r"(?:WCI|World Container Index)\)?\s*composite",
    "rotterdam_shanghai": r"Rotterdam\s*(?:–|-|to)\s*Shanghai",
    "los_angeles_shanghai": r"Los Angeles\s*(?:–|-|to)\s*Shanghai",
    "shanghai_new_york": r"Shanghai\s*(?:–|-|to)\s*New York",
    "new_york_rotterdam": r"New York\s*(?:–|-|to)\s*Rotterdam",
    "rotterdam_new_york": r"Rotterdam\s*(?:–|-|to)\s*New York",
}


def parse_release(text: str) -> dict[str, int]:
    """Extract route values from one weekly Drewry release text (Drewry page or a press reprint)."""
    text = re.sub(r"\s+", " ", text)
    found = {}
    for route, pattern in PAGE_EXTRA.items():
        m = re.search(pattern + VALUE_RE, text, flags=re.I)
        if m:
            found[route] = int(m.group(1).replace(",", ""))
    for route, cfg in ROUTES.items():
        m = re.search(cfg["pattern"] + VALUE_RE, text, flags=re.I)
        if m:
            found[route] = int(m.group(1).replace(",", ""))
    return found


def parse_release_date(text: str) -> str | None:
    m = re.search(r"assessment for Thursday,?\s*(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})", text)
    if not m:
        return None
    return pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}", dayfirst=True).strftime("%Y-%m-%d")


def from_page() -> pd.DataFrame:
    from bs4 import BeautifulSoup

    html = get(PAGE).text
    text = BeautifulSoup(html, "lxml").get_text(" ")
    d = parse_release_date(text)
    vals = parse_release(text)
    if not d or not vals:
        raise RuntimeError("Could not parse the Drewry page this week")
    return pd.DataFrame([{"date": d, "route": r, "usd_per_40ft": v, "source": "drewry_page"} for r, v in vals.items()])


def fetch() -> pd.DataFrame:
    """Chart history (akshare) and this week's page text; both are merged, the newest value wins.

    The chart can lag the page by several weeks, so the page is always read too.
    """
    errors, got = [], []
    for fn in (from_akshare, from_page):
        try:
            df = fn()
            df = df[df["date"] >= HISTORY_START]
            if not df.empty:
                merge_csv(df, OUT, KEYS)
                got.append(df)
                log.info("Drewry via %s: %d rows, %s -> %s", fn.__name__, len(df), df["date"].min(), df["date"].max())
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
            log.warning("Drewry %s failed: %s", fn.__name__, e)
    if not got:
        raise RuntimeError("; ".join(errors))
    return pd.concat(got, ignore_index=True)


# ---------- press backfill (fills weeks the chart does not cover) ----------
PRESS_SEARCH = "https://www.hellenicshippingnews.com/page/{page}/?s=Drewry+World+Container+Index"


def backfill_from_press(max_pages: int = 60) -> pd.DataFrame:
    """Crawl press reprints of Drewry's weekly release and parse route values. Slow, run once."""
    from bs4 import BeautifulSoup

    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        try:
            soup = BeautifulSoup(get(PRESS_SEARCH.format(page=page)).text, "lxml")
        except Exception as e:
            log.info("press search stopped at page %d: %s", page, e)
            break
        links = {a["href"] for a in soup.select("h2 a, h3 a") if "drewry" in a.get("href", "").lower() or "container-index" in a.get("href", "").lower()}
        if not links:
            break
        for url in links - seen:
            seen.add(url)
            try:
                text = BeautifulSoup(get(url).text, "lxml").get_text(" ")
            except Exception:
                continue
            d, vals = parse_release_date(text), parse_release(text)
            if d and vals:
                rows += [{"date": d, "route": r, "usd_per_40ft": v, "source": "press:" + url} for r, v in vals.items()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["date"] >= HISTORY_START]
        # never overwrite values that came straight from Drewry
        existing = pd.read_csv(OUT, dtype=str) if OUT.exists() else pd.DataFrame(columns=KEYS)
        have = set(zip(existing.get("date", []), existing.get("route", [])))
        df = df[[(d, r) not in have for d, r in zip(df["date"], df["route"])]]
        merge_csv(df, OUT, KEYS)
    log.info("Drewry press backfill: %d new rows", len(df))
    return df
