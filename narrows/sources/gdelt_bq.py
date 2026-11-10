"""GDELT through Google BigQuery (public dataset gdelt-bq), instead of the throttled DOC API.

Credentials come from two GitHub secrets, exposed as environment variables:
  GCP_SA_KEY   the service-account JSON key (BigQuery Job User role)
  GCP_PROJECT  the Google Cloud project id that runs (and is billed for) the queries

BigQuery sandbox: free, 1 TB scanned per month. Every query is dry-run first and refused
above MAX_GB, so a mistake can never burn the monthly quota.

- events_history(): daily geolocated event counts near our routes since 2019 (GDELT Events)
- articles(days):   matching articles of the last days per news family, with title and location (GDELT GKG)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..common import CLEAN, CONFIG, HISTORY_START, RAW, log, merge_csv

MAX_GB = 250

# lat_min, lat_max, lon_min, lon_max
ZONES = {
    "suez_canal": (29.5, 31.6, 31.8, 33.0),
    "red_sea_bab_el_mandeb": (11.0, 29.5, 32.0, 45.5),
    "gulf_of_aden": (10.0, 15.5, 43.0, 52.0),
    "hormuz_gulf": (23.0, 30.5, 47.0, 60.0),
    "cape_of_good_hope": (-37.0, -30.0, 15.0, 30.0),
    "malacca_singapore": (-2.0, 7.0, 95.0, 105.0),
    "china_coast": (20.0, 41.0, 108.0, 123.0),
    "north_europe_ports": (48.0, 56.0, -5.0, 12.0),
    "mediterranean": (30.0, 46.0, -6.0, 36.0),
}

# CAMEO root codes grouped into what matters for shipping
EVENT_CLASSES = {
    "conflict": ("18", "19", "20"),   # assault, fight, mass violence
    "coercion": ("17",),              # seize, blockade, sanctions-like coercion
    "threat": ("13",),
    "protest": ("14",),               # includes strikes
    "cooperation": ("03", "04", "05", "06"),  # intent, consult, diplomatic and material cooperation
}


def _client():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    key = os.environ.get("GCP_SA_KEY")
    project = os.environ.get("GCP_PROJECT")
    if not key or not project:
        raise RuntimeError("GCP_SA_KEY / GCP_PROJECT secrets are not set")
    creds = service_account.Credentials.from_service_account_info(json.loads(key))
    return bigquery.Client(project=project, credentials=creds)


def _run(sql: str, params: list) -> pd.DataFrame:
    from google.cloud import bigquery

    client = _client()
    dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params))
    gb = dry.total_bytes_processed / 1e9
    log.info("BigQuery dry run: %.1f GB", gb)
    if gb > MAX_GB:
        raise RuntimeError(f"Query would scan {gb:.0f} GB, above the {MAX_GB} GB safety limit")
    return client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()


def _zone_case(lat: str, lon: str) -> str:
    whens = [f"WHEN {lat} BETWEEN {a} AND {b} AND {lon} BETWEEN {c} AND {d} THEN '{z}'" for z, (a, b, c, d) in ZONES.items()]
    return "CASE " + " ".join(whens) + " END"


def _class_case(col: str) -> str:
    whens = [f"WHEN {col} IN ({','.join(repr(c) for c in codes)}) THEN '{k}'" for k, codes in EVENT_CLASSES.items()]
    return "CASE " + " ".join(whens) + " ELSE 'other' END"


def events_history(start: str = HISTORY_START, end: str | None = None) -> pd.DataFrame:
    from google.cloud import bigquery

    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sql = f"""
    WITH e AS (
      SELECT
        PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) AS date,
        {_zone_case('ActionGeo_Lat', 'ActionGeo_Long')} AS zone,
        {_class_case('EventRootCode')} AS event_class,
        NumArticles, GoldsteinScale
      FROM `gdelt-bq.gdeltv2.events_partitioned`
      WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start) AND TIMESTAMP(@end)
        AND ActionGeo_Lat IS NOT NULL
    )
    SELECT date, zone, event_class,
           COUNT(*) AS events, SUM(NumArticles) AS articles, ROUND(AVG(GoldsteinScale), 2) AS goldstein_avg
    FROM e
    WHERE zone IS NOT NULL AND date BETWEEN DATE(@start) AND DATE(@end)
    GROUP BY date, zone, event_class
    """
    params = [bigquery.ScalarQueryParameter("start", "STRING", start), bigquery.ScalarQueryParameter("end", "STRING", end)]
    df = _run(sql, params)
    if df.empty:
        raise RuntimeError("BigQuery events returned no rows")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    merge_csv(df, CLEAN / "news_events_by_zone.csv", ["date", "zone", "event_class"])
    log.info("GDELT events by zone: %d rows, %s -> %s", len(df), df["date"].min(), df["date"].max())
    return df


# ---------- articles: GDELT DOC-style queries translated to regular expressions ----------
def query_to_regexes(query: str) -> list[str]:
    """'(a OR "b c") (d OR e)' -> ['\\b(a|b c)\\b', '\\b(d|e)\\b']: every group must match."""
    groups = re.findall(r"\(([^()]*)\)", query)
    out = []
    for g in groups:
        terms = [t.strip().strip('"').lower() for t in re.split(r"\s+OR\s+", g) if t.strip()]
        terms = [re.escape(t).replace(r"\ ", r"[\s_-]+") for t in terms]
        out.append(r"\b(" + "|".join(terms) + r")\b")
    return out


def articles(days: int = 1) -> pd.DataFrame:
    from google.cloud import bigquery

    fams = json.loads((CONFIG / "news_families.json").read_text())["families"]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    family_cols = []
    for key, fam in fams.items():
        conds = " AND ".join(f"REGEXP_CONTAINS(txt, r'{rx}')" for rx in query_to_regexes(fam["query"]))
        family_cols.append(f"IF({conds}, '{key}', NULL)")
    sql = f"""
    WITH g AS (
      SELECT
        DATE AS ts, DocumentIdentifier AS url, SourceCommonName AS domain,
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
        SPLIT(V2Locations, ';')[SAFE_OFFSET(0)] AS loc0,
        LOWER(CONCAT(IFNULL(REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>'), ''), ' ',
                     REGEXP_REPLACE(IFNULL(DocumentIdentifier, ''), r'[-_/]', ' '), ' ',
                     IFNULL(V2Organizations, ''))) AS txt
      FROM `gdelt-bq.gdeltv2.gkg_partitioned`
      WHERE _PARTITIONTIME >= TIMESTAMP(@start)
    ),
    f AS (
      SELECT ts, url, domain, title, loc0,
             ARRAY(SELECT x FROM UNNEST([{', '.join(family_cols)}]) AS x WHERE x IS NOT NULL) AS families
      FROM g
    )
    SELECT ts, url, domain, title, family,
           SAFE_CAST(SPLIT(loc0, '#')[SAFE_OFFSET(5)] AS FLOAT64) AS lat,
           SAFE_CAST(SPLIT(loc0, '#')[SAFE_OFFSET(6)] AS FLOAT64) AS lon,
           SPLIT(loc0, '#')[SAFE_OFFSET(1)] AS place
    FROM f, UNNEST(families) AS family
    """
    params = [bigquery.ScalarQueryParameter("start", "STRING", start.strftime("%Y-%m-%d %H:%M:%S"))]
    df = _run(sql, params)
    if df.empty:
        log.info("GDELT articles: none")
        return df
    df["seen"] = pd.to_datetime(df["ts"].astype(str), format="%Y%m%d%H%M%S", errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df = df.drop(columns=["ts"]).drop_duplicates(["url", "family"])
    name = f"{end.strftime('%Y-%m-%d')}.csv" if days == 1 else f"recent_{days}d_{end.strftime('%Y-%m-%d')}.csv"
    merge_csv(df, RAW / "news" / name, ["url", "family"])
    events(df).to_csv(RAW / "news" / name.replace(".csv", "_events.csv"), index=False)
    log.info("GDELT articles (%dd): %d rows, %d families", days, len(df), df["family"].nunique())
    return df


def events(df: pd.DataFrame) -> pd.DataFrame:
    """Group articles that are the same story: same family and same first 7 title words."""
    import html

    d = df.copy()
    d["title"] = d["title"].fillna("").map(html.unescape).str.strip()
    d = d[d["title"].str.len() > 25]
    d["event_key"] = (d["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
                      .str.split().str[:7].str.join(" "))
    g = (d.sort_values("seen").groupby(["family", "event_key"])
         .agg(first_seen=("seen", "first"), articles=("url", "count"), title=("title", "first"),
              url=("url", "first"), place=("place", "first"), lat=("lat", "first"), lon=("lon", "first"))
         .reset_index().sort_values(["articles", "first_seen"], ascending=[False, True]))
    return g


def articles_15d() -> pd.DataFrame:
    return articles(days=15)
