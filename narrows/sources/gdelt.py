"""GDELT DOC 2.0 API: news volume per family (history) and today's articles (daily).

- history(): daily article volume per news family since 2019 (timelinevolraw), the raw
  material for the past "pressure" scores. Scoring itself comes in phase 2.
- articles(): yesterday's matching articles per family (title, url, domain, date), the raw
  material for the short notes.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..common import CLEAN, CONFIG, RAW, log, merge_csv, session

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def families() -> dict:
    return json.loads((CONFIG / "news_families.json").read_text())["families"]


PAUSE = 6  # GDELT asks for at most one request every 5 seconds


def _call(params: dict, sess) -> dict:
    for attempt in range(5):
        time.sleep(PAUSE * (1 + 2 * attempt))
        r = sess.get(API, params={**params, "format": "json"}, timeout=90)
        if r.status_code != 429:
            break
        log.info("GDELT busy (429), waiting before retry %d", attempt + 1)
    if r.status_code != 200:
        raise RuntimeError(f"GDELT HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()
    except ValueError:
        raise RuntimeError(f"GDELT non-JSON answer: {r.text[:200]}")


def history(start: str = "2019-01-01") -> pd.DataFrame:
    sess, frames = session(), []
    end = datetime.now(timezone.utc)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    for key, fam in families().items():
        # one call per year keeps answers small and avoids timeouts
        y0 = start_dt
        while y0 < end:
            y1 = min(y0.replace(year=y0.year + 1), end)
            js = _call({"query": fam["query"] + " sourcelang:english", "mode": "timelinevolraw",
                        "startdatetime": y0.strftime("%Y%m%d%H%M%S"), "enddatetime": y1.strftime("%Y%m%d%H%M%S")}, sess)
            for series in js.get("timeline", []):
                for p in series.get("data", []):
                    frames.append({"date": p["date"][:8], "family": key, "articles": p.get("value"), "all_articles": p.get("norm")})
            y0 = y1
        log.info("GDELT history %s done", key)
    df = pd.DataFrame(frames)
    if df.empty:
        raise RuntimeError("GDELT history returned nothing")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    merge_csv(df, CLEAN / "news_volume.csv", ["date", "family"])
    return df


def articles(hours: int = 24) -> pd.DataFrame:
    sess, rows = session(), []
    for key, fam in families().items():
        js = _call({"query": fam["query"] + " sourcelang:english", "mode": "artlist", "maxrecords": 250,
                    "timespan": f"{hours}h", "sort": "datedesc"}, sess)
        for a in js.get("articles", []):
            rows.append({"seen": a.get("seendate"), "family": key, "title": a.get("title"), "url": a.get("url"),
                         "domain": a.get("domain"), "country": a.get("sourcecountry")})
    df = pd.DataFrame(rows)
    day = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d")
    if not df.empty:
        merge_csv(df, RAW / "news" / f"{day}.csv", ["url", "family"])
    log.info("GDELT articles: %d", len(df))
    return df
