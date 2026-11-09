# Narrows

**Geopolitical news → Asia–Europe container freight forecasts.**

Narrows reads the news, checks it against ship traffic and futures markets, and calls where
Asia–Europe container freight rates will be in 1 and 4 weeks. Every call is committed to this
repository before the outcome is known, then scored against reality and against two naive
baselines ("no change" and "trend continues").

> Status: **phase 1, data collection** (November 2026). First locked forecast planned for Thursday 26 November 2026.

## What is tracked

| Role | Series | Source | Frequency | History |
| --- | --- | --- | --- | --- |
| Forecast | Drewry WCI Shanghai → Rotterdam | drewry.co.uk | Weekly (Thu) | 2019 → |
| Forecast | Drewry WCI Shanghai → Genoa | drewry.co.uk | Weekly (Thu) | 2019 → |
| Forecast | INE Containerized Freight Index (Europe) futures, front month | INE via Sina | Daily | Aug 2023 → |
| Control | Drewry WCI Shanghai → Los Angeles | drewry.co.uk | Weekly (Thu) | 2019 → |
| Signal | Geolocated events near the routes, and matching articles per news family | GDELT via Google BigQuery | Daily | 2019 → |
| Signal | Strait transits (Suez, Bab el-Mandeb, Cape, Malacca…) | IMF PortWatch | Daily, updated weekly | 2019 → |
| Signal | Port calls (Rotterdam, Genoa, Singapore, Colombo…) | IMF PortWatch | Daily, updated weekly | 2019 → |
| Signal | EC futures curve (front vs next contract) | INE via Sina | Daily | Aug 2023 → |
| Signal | Brent crude | Yahoo Finance | Daily | 2019 → |
| Context | Maersk B, Hapag-Lloyd share prices | Yahoo Finance | Daily | 2019 → |
| Context | HARPEX container ship charter index | harperpetersen.com | Weekly (Fri) | Nov 2026 → |
| Context | Bunker fuel, Singapore and Rotterdam | shipandbunker.com | Daily | Nov 2026 → |

All sources are free and public. HARPEX and bunker history are paid, so they are collected from now on and will only enter the model once enough history exists.

## Repository layout

```
config/news_families.json   news families, queries, zones, intensity scale (draft v0)
narrows/sources/            one module per data source
scripts/collect.py          daily and backfill runs, writes data/status.json
data/clean/                 tidy CSV files, one per source
data/raw/news/              daily matching articles from GDELT
.github/workflows/          daily collection and one-off history backfill
```

## Run locally

```bash
pip install -r requirements.txt
python scripts/collect.py backfill   # once
python scripts/collect.py daily
```

## Method notes

- History starts in January 2019; the Covid period (2020 to mid-2022) is flagged as a separate regime.
- The model has two stages: a base layer trained since 2019, and a market correction using EC futures since August 2023.
- Stable = a move between −2% and +2% (to be set per index and horizon).
- Data from Yahoo Finance and Sina are unofficial endpoints; the daily run fails loudly if a core source stops answering.
