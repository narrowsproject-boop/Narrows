"""Shared helpers: paths, HTTP session, CSV merge, run status."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
CONFIG = ROOT / "config"
STATUS_FILE = DATA / "status.json"

HISTORY_START = "2019-01-01"
EC_START = "2023-08-18"  # INE EC futures listing date

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

log = logging.getLogger("narrows")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-GB,en;q=0.9"})
    return s


def get(url: str, *, params: dict | None = None, retries: int = 3, timeout: int = 60, sess=None) -> requests.Response:
    sess = sess or session()
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {r.url}")
        except requests.RequestException as e:  # network hiccup
            last = e
        time.sleep(2 * attempt)
    raise last  # type: ignore[misc]


def merge_csv(df: pd.DataFrame, path: Path, keys: list[str]) -> pd.DataFrame:
    """Append new rows to an existing CSV, newest value wins on duplicate keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        df = pd.concat([old, df.astype(str)], ignore_index=True)
    df = df.astype(str).drop_duplicates(subset=keys, keep="last").sort_values(keys)
    df.to_csv(path, index=False)
    return df


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(results: dict) -> None:
    status = {}
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text())
    status.update(results)
    status["_last_run"] = now_utc()
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2, sort_keys=True))
