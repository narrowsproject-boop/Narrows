"""Export one Excel workbook for Power BI (web or desktop): one sheet per table, tidy format.

  python scripts/export_powerbi.py  ->  data/powerbi/narrows_powerbi.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from narrows.common import CLEAN, DATA  # noqa: E402

OUT = DATA / "powerbi" / "narrows_powerbi.xlsx"

ROUTES = pd.DataFrame([
    {"route": "shanghai_rotterdam", "route_label": "Shanghai → Rotterdam", "role": "Forecast", "corridor": "Asia → North Europe", "via": "Suez or Cape of Good Hope"},
    {"route": "shanghai_genoa", "route_label": "Shanghai → Genoa", "role": "Forecast", "corridor": "Asia → Mediterranean", "via": "Suez or Cape of Good Hope"},
    {"route": "shanghai_los_angeles", "route_label": "Shanghai → Los Angeles", "role": "Control", "corridor": "Transpacific", "via": "Pacific (no Suez, no Hormuz)"},
])

# Reference shocks for annotations. Dates are the trigger events; see notes for precision.
EVENTS = pd.DataFrame([
    {"event": "Covid boom (regime flag)", "start": "2020-07-01", "end": "2022-06-30", "type": "Demand and congestion", "zone": "Global", "note": "Flagged as a separate regime in the model"},
    {"event": "Ever Given blocks Suez", "start": "2021-03-23", "end": "2021-03-29", "type": "Accident", "zone": "Suez Canal", "note": "Grounded 23 Mar, refloated 29 Mar 2021"},
    {"event": "Russia invades Ukraine", "start": "2022-02-24", "end": "", "type": "Conflict", "zone": "Black Sea", "note": ""},
    {"event": "Panama Canal drought restrictions", "start": "2023-06-01", "end": "2024-02-29", "type": "Weather and climate", "zone": "Panama Canal", "note": "Approximate window"},
    {"event": "Houthi attacks begin (Galaxy Leader seized)", "start": "2023-11-19", "end": "", "type": "Conflict", "zone": "Red Sea", "note": "Carriers divert via the Cape from mid-December 2023"},
    {"event": "War with Iran begins", "start": "2026-02-28", "end": "", "type": "Conflict", "zone": "Strait of Hormuz", "note": "Hormuz traffic collapses"},
    {"event": "MSC restores Suez transits", "start": "2026-08-25", "end": "", "type": "Carrier decision", "zone": "Suez Canal", "note": "Start of the large-scale Suez return"},
])


def main() -> None:
    sheets = {}
    w = pd.read_csv(DATA / "model" / "drewry_weekly.csv")
    w["week"] = pd.to_datetime(w["week"])
    w = w.sort_values(["route", "week"])
    w["change_1w_pct"] = w.groupby("route")["usd_per_40ft"].pct_change(1).mul(100).round(2)
    w["change_4w_pct"] = w.groupby("route")["usd_per_40ft"].pct_change(4).mul(100).round(2)
    sheets["drewry_weekly"] = w

    ec = pd.read_csv(CLEAN / "ec_front.csv")
    ec["date"] = pd.to_datetime(ec["date"])
    sheets["ec_futures_daily"] = ec

    pw = pd.read_csv(CLEAN / "portwatch_straits.csv")
    pw["date"] = pd.to_datetime(pw["date"])
    pw = pw.sort_values(["strait", "date"])
    pw["n_total_7d_avg"] = pw.groupby("strait")["n_total"].transform(lambda s: s.rolling(7, min_periods=1).mean()).round(1)
    base = pw[(pw["date"] >= "2019-01-01") & (pw["date"] < "2020-01-01")].groupby("strait")["n_total"].mean()
    pw["vs_2019_pct"] = (pw["n_total_7d_avg"] / pw["strait"].map(base) - 1).mul(100).round(1)
    sheets["straits_daily"] = pw[["date", "strait", "n_total", "n_container", "n_tanker", "n_dry_bulk", "n_total_7d_avg", "vs_2019_pct"]]

    m = pd.read_csv(CLEAN / "markets.csv")
    m["date"] = pd.to_datetime(m["date"])
    sheets["markets_daily"] = m

    sheets["dim_routes"] = ROUTES
    ev = EVENTS.copy()
    ev["start"] = pd.to_datetime(ev["start"])
    ev["end"] = pd.to_datetime(ev["end"], errors="coerce")
    sheets["dim_events"] = ev

    cal = pd.DataFrame({"date": pd.date_range("2019-01-01", pd.Timestamp.today().normalize(), freq="D")})
    cal["year"] = cal["date"].dt.year
    cal["month"] = cal["date"].dt.month
    cal["month_name"] = cal["date"].dt.strftime("%b")
    cal["week_thursday"] = cal["date"] - pd.to_timedelta((cal["date"].dt.dayofweek - 3) % 7, unit="D")
    cal["covid_regime"] = cal["date"].between("2020-07-01", "2022-06-30")
    sheets["dim_date"] = cal

    readme = pd.DataFrame({"table": list(sheets) , "description": [
        "Drewry WCI spot rate per 40ft container, one row per route and Thursday. filled = TRUE when the week was interpolated (no Drewry release).",
        "INE Asia–Europe container freight futures: front contract, next contract, curve slope in %.",
        "IMF PortWatch daily transits per strait, 7-day average and change vs 2019 average.",
        "Daily closes: Brent (USD/bbl), Maersk B (DKK), Hapag-Lloyd (EUR).",
        "Route labels and roles (forecast or control).",
        "Reference shocks for chart annotations.",
        "Calendar table: link drewry_weekly[week] to dim_date[week_thursday] or the daily tables to dim_date[date].",
    ]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(OUT, {k: len(v) for k, v in sheets.items()})


if __name__ == "__main__":
    main()
