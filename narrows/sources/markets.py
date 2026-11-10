"""Daily market series from Yahoo Finance: Brent (model signal), carrier stocks (context)."""
from __future__ import annotations

import pandas as pd

from ..common import CLEAN, HISTORY_START, log, merge_csv

TICKERS = {
    "BZ=F": "brent_usd_bbl",          # Brent crude front-month future
    "MAERSK-B.CO": "maersk_b_dkk",    # A.P. Moller-Maersk B
    "HLAG.DE": "hapag_lloyd_eur",     # Hapag-Lloyd
    "HO=F": "heating_oil_usd_gal",    # NY Harbor ULSD (diesel/gasoil) future, proxy for marine fuel
    "EURUSD=X": "eurusd",             # euro in dollars, to convert freight into EUR per container
}


def fetch(start: str = HISTORY_START) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for ticker, name in TICKERS.items():
        hist = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=False)
        if hist.empty:
            raise RuntimeError(f"Yahoo returned no data for {ticker}")
        s = hist["Close"].rename("value").reset_index()
        s["date"] = pd.to_datetime(s["Date"]).dt.strftime("%Y-%m-%d")
        s["series"] = name
        frames.append(s[["date", "series", "value"]])
    df = pd.concat(frames, ignore_index=True)
    df["value"] = df["value"].round(4)
    merge_csv(df, CLEAN / "markets.csv", ["date", "series"])
    log.info("Markets: %d rows, last %s", len(df), df["date"].max())
    return df
