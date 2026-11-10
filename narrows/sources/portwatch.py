"""IMF PortWatch: daily transit calls through straits and daily port calls.

Public ArcGIS FeatureServer, max 1000 records per request, so we page with resultOffset.
"""
from __future__ import annotations

import pandas as pd

from ..common import CLEAN, HISTORY_START, get, log, merge_csv, session

BASE = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
CHOKEPOINTS_URL = f"{BASE}/Daily_Chokepoints_Data/FeatureServer/0/query"
PORTS_URL = f"{BASE}/Daily_Ports_Data/FeatureServer/0/query"

# Straits on the Asia -> Europe corridor (PortWatch names)
CHOKEPOINTS = [
    "Suez Canal", "Bab el-Mandeb Strait", "Cape of Good Hope", "Malacca Strait",
    "Strait of Hormuz", "Gibraltar Strait", "Bosporus Strait", "Taiwan Strait",
]
# Ports on the corridor (PortWatch names)
PORTS = [
    "Rotterdam", "Antwerp", "Hamburg", "Genoa", "Piraeus",
    "Singapore", "Colombo", "Jeddah", "Shanghai", "Tanjung Pelepas",
]


def _fetch(url: str, where: str, fields: str, order: bool = True) -> pd.DataFrame:
    sess = session()
    rows, offset = [], 0
    while True:
        params = {"where": where, "outFields": fields, "resultOffset": offset, "resultRecordCount": 1000, "f": "json"}
        if order:
            params["orderByFields"] = "date ASC"
        js = get(url, params=params, sess=sess).json()
        if "error" in js:
            raise RuntimeError(js["error"])
        feats = js.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < 1000 and not js.get("exceededTransferLimit"):
            break
        offset += len(feats)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Chokepoints return epoch milliseconds; Ports uses a DateOnly field returned as text
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _names_clause(field: str, names: list[str]) -> str:
    quoted = ",".join("'" + n.replace("'", "''") + "'" for n in names)
    return f"{field} IN ({quoted})"


def chokepoints(start: str = HISTORY_START) -> pd.DataFrame:
    where = f"{_names_clause('portname', CHOKEPOINTS)} AND date >= DATE '{start}'"
    df = _fetch(CHOKEPOINTS_URL, where, "date,portid,portname,n_total,n_container,n_tanker,n_dry_bulk,capacity,capacity_container")
    if df.empty:
        raise RuntimeError("PortWatch chokepoints returned no rows")
    df = df.rename(columns={"portname": "strait"})
    merge_csv(df, CLEAN / "portwatch_straits.csv", ["date", "strait"])
    log.info("PortWatch straits: %d rows, last %s", len(df), df["date"].max())
    return df


def ports(start: str = HISTORY_START) -> pd.DataFrame:
    """One request per port and year (at most 366 rows): this layer rejects deep pagination."""
    from datetime import date as _date

    frames = []
    for port in PORTS:
        for year in range(int(start[:4]), _date.today().year + 1):
            where = f"portname = '{port}' AND year = {year}"
            try:
                part = _fetch(PORTS_URL, where, "*", order=False)
            except Exception as e:
                log.warning("PortWatch port %s %d failed: %s", port, year, e)
                continue
            if not part.empty:
                frames.append(part)
    if not frames:
        raise RuntimeError("PortWatch ports returned no rows")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["date"] >= start].drop(columns=[c for c in ("ObjectId",) if c in df.columns])
    df = df.rename(columns={"portname": "port"})
    merge_csv(df, CLEAN / "portwatch_ports.csv", ["date", "port"])
    log.info("PortWatch ports: %d rows, %d ports, last %s", len(df), df["port"].nunique(), df["date"].max())
    return df
