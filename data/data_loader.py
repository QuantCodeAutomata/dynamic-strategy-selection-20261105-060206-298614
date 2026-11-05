"""Market and macroeconomic data loading for the dynamic strategy selection replication.

Three public sources are used:

1. The Kenneth R. French Data Library (public CSV download, no API key) for the momentum
   (MOM), value (HML) and profitability (RMW) factor returns (Section 4.1, "Data").
2. FRED, via ``pandas_datareader`` (public, no API key), for the macroeconomic state
   variables: FEDFUNDS, GS10, GS1, BAA, AAA, CPILFENS and headline CPI (Section S.5.1).
3. The QuantX Dataloader REST API for daily U.S. common-stock bars, used to build the
   defensive low-volatility factor (DEF) described in Section 4.1. CRSP daily data (the
   source the paper uses for DEF) requires a WRDS subscription that is not available in
   this environment; see "Deviations from the paper" in results/RESULTS.md for the
   resulting substitution and its effect on the universe.

Downloaded data is never committed: everything here is cached under ``data/cache/``
(.gitignore'd) and reloaded from cache on repeated runs.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src import config

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}


class DataLoaderError(RuntimeError):
    """Raised when a data source fails or returns an unexpected shape."""


# ---------------------------------------------------------------------------
# Kenneth R. French Data Library
# ---------------------------------------------------------------------------

def _download_french_zip(url: str, cache_name: str) -> pd.DataFrame:
    """Download and parse one Kenneth French Data Library CSV-in-zip file, monthly frequency.

    Returns a DataFrame indexed by a monthly ``PeriodIndex`` with percentage-point columns as
    published (the caller converts to decimals).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=False).pipe(_period_index)

    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise DataLoaderError(f"French Data Library download failed: HTTP {response.status_code} for {url}")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        inner_name = zf.namelist()[0]
        raw_text = zf.read(inner_name).decode("latin-1")

    lines = raw_text.splitlines()
    # The monthly block starts at the header row (a line beginning with a comma or leading blank
    # then "Mkt-RF") and ends at the first blank line or a row introducing annual data.
    header_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith(",") or "Mkt-RF" in ln)
    data_lines = []
    for ln in lines[header_idx:]:
        stripped = ln.strip()
        if not stripped:
            break
        first_token = stripped.split(",")[0]
        if len(first_token) == 6 and first_token.isdigit():  # YYYYMM monthly row
            data_lines.append(ln)
        elif data_lines:
            break  # left the monthly block (annual block follows)
    header_line = lines[header_idx]
    columns = [c.strip() for c in header_line.split(",")]
    columns[0] = "date"
    table_text = "\n".join([",".join(columns)] + data_lines)
    df = pd.read_csv(io.StringIO(table_text))
    df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.to_period("M")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_csv(cache_path)
    return df


def _period_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def load_french_factors() -> pd.DataFrame:
    """Monthly MOM, HML, RMW and the market excess return (Mkt-RF), in decimal units.

    Source: Kenneth R. French Data Library, "F-F_Research_Data_Factors" (market, value,
    profitability omitted -- RMW is in the 5-factor file) and "F-F_Momentum_Factor"
    (Section 4.1: "Obtain MOM, HML, RMW from French Data Library").
    """
    three = _download_french_zip(config.FRENCH_3FACTOR_URL, "ff3_factors.csv")
    five = _download_french_zip(config.FRENCH_5FACTOR_URL, "ff5_factors.csv")
    mom = _download_french_zip(config.FRENCH_MOM_URL, "ff_mom.csv")

    out = pd.DataFrame(index=three.index)
    out["MKT_RF"] = three["Mkt-RF"] / 100.0
    out["RF"] = three["RF"] / 100.0
    out["HML"] = three["HML"] / 100.0
    out["RMW"] = five.reindex(three.index)["RMW"] / 100.0
    mom_col = [c for c in mom.columns if "Mom" in c or "WML" in c][0]
    out["MOM"] = mom.reindex(three.index)[mom_col] / 100.0
    out = out.dropna(subset=["MKT_RF", "HML", "RMW", "MOM"])
    logger.info("French factors loaded: %d months, %s to %s", len(out), out.index.min(), out.index.max())
    return out


# ---------------------------------------------------------------------------
# FRED macroeconomic series
# ---------------------------------------------------------------------------

def load_fred_series(start: str, end: str) -> pd.DataFrame:
    """Monthly FRED series FEDFUNDS, GS10, GS1, BAA, AAA, CPILFENS, CPIAUCNS (Section S.5.1).

    Uses ``pandas_datareader``, a public source requiring no API key.
    """
    from pandas_datareader import data as pdr

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fred_{start}_{end}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df.index = pd.PeriodIndex(df.index, freq="M")
        return df

    frames = {}
    for series in config.FRED_SERIES:
        s = pdr.DataReader(series, "fred", start, end)
        frames[series] = s[series]
    df = pd.DataFrame(frames)
    df.index = pd.PeriodIndex(df.index, freq="M")
    df = df.groupby(level=0).last()
    df.to_csv(cache_path)
    logger.info("FRED series loaded: %s, %d months, %s to %s", list(df.columns), len(df),
                df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# QuantX Dataloader: daily U.S. common-stock bars, used to build the DEF proxy
# ---------------------------------------------------------------------------

def _headers() -> dict:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataLoaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set in the environment")
    return {"Authorization": f"Bearer {dataloader_key}", "X-Massive-Api-Key": massive_key}


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(method, DATALOADER_API_ROOT + route, json=body, headers=headers, timeout=900)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2**attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataLoaderError(f"{route}: HTTP {response.status_code}: {error}")
    return response.json()


def _tidy_bars(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataLoaderError(f"{ticker}: no {field} column in the Dataloader answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict]]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    if not loaded:
        return pd.DataFrame(), details
    result = answer["result"]
    if not result.get("data_stored"):
        wide = pd.DataFrame(result["ohlcv_df"]["rows"])
    else:
        export = _request("POST", "/save-dataframe", {
            "result_id": result["result_id"], "ticker": f"batch{batch}", "dataset_name": "ohlcv",
            "timeframe": request["timeframe"], "start_date": request["start_date"],
            "end_date": request["end_date"], "file_format": "csv", "data_kind": "ohlcv",
        })
        link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
        csv = requests.get(link["result"]["url"], timeout=600)
        csv.raise_for_status()
        wide = pd.read_csv(io.BytesIO(csv.content))
    return _tidy_bars(wide, details), details


def load_ohlcv(tickers: list[str], start_date: str, end_date: str, timeframe: str = "1d",
               use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Daily bars for ``tickers`` from the QuantX Dataloader, dividend-adjusted by default.

    Returns tidy rows (timestamp, date, ticker, open, high, low, close, volume) and a metadata
    dict with ``ticker_details``, ``failed`` and ``non_massive_source`` (the Yahoo backstop and
    any other non-Massive source, recorded for "Deviations from the paper").
    """
    request = {"timeframe": timeframe, "start_date": start_date, "end_date": end_date,
               "source": "massive", "adjust_dividends": True, "yahoo_backstop": True}
    tickers = list(dict.fromkeys(tickers))
    digest = hashlib.sha256(",".join(sorted(tickers)).encode()).hexdigest()[:16]
    key = f"ohlcv_{timeframe}_{start_date}_{end_date}_{digest}"
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
                                 enumerate(batches)))
    frames = [bars for bars, _ in results if not bars.empty]
    details = [d for _, batch_details in results for d in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataLoaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request, "ticker_details": details, "failed": failed,
        "non_massive_source": {d["ticker"]: d["data_source"] for d in details
                                if d.get("data_source") not in (None, "massive")},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


# ---------------------------------------------------------------------------
# Defensive low-volatility factor (DEF), substituting the paper's CRSP-based construction
# ---------------------------------------------------------------------------

def build_def_factor(tickers: list[str], start_date: str, end_date: str) -> pd.Series:
    """Monthly returns of a 2x3-style low-volatility-minus-high-volatility factor.

    Substitutes for the paper's CRSP-based Fama-French-style 2x3 volatility-sorted
    portfolio (Section 4.1, motivated by Novy-Marx 2014): CRSP daily data requires a WRDS
    subscription unavailable in this environment, so daily bars are instead loaded from the
    QuantX Dataloader for a liquidity-ranked universe of long-history U.S. common stocks
    (``EXP1_UNIVERSE_SIZE_RUN`` in src/config.py). Each month, stocks are ranked by trailing
    daily-return volatility (``EXP1_DEF_VOL_WINDOW_DAYS``); the factor is long the bottom
    tercile and short the top tercile, equally weighted, rebalanced monthly, using only
    information available before the start of the holding month (no look-ahead).
    """
    bars, meta = load_ohlcv(tickers, start_date, end_date, timeframe="1d")
    bars = bars.sort_values(["ticker", "date"])
    bars["ret"] = bars.groupby("ticker")["close"].pct_change()
    panel = bars.pivot(index="date", columns="ticker", values="ret")
    vol = panel.rolling(config.EXP1_DEF_VOL_WINDOW_DAYS, min_periods=config.EXP1_DEF_VOL_WINDOW_DAYS // 2).std()

    month_ends = panel.resample("ME").last().index
    monthly_ret = (1.0 + panel).resample("ME").prod() - 1.0  # monthly compounded return per stock
    vol_month_end = vol.reindex(panel.index).resample("ME").last()  # trailing vol known at month end

    def_returns = {}
    for i in range(1, len(month_ends)):
        formation_date = month_ends[i - 1]  # volatility known at the end of the prior month
        holding_date = month_ends[i]
        vol_row = vol_month_end.loc[formation_date].dropna()
        if len(vol_row) < 10:
            continue
        low_cut = vol_row.quantile(config.EXP1_DEF_LOW_QUANTILE)
        high_cut = vol_row.quantile(config.EXP1_DEF_HIGH_QUANTILE)
        long_names = vol_row[vol_row <= low_cut].index
        short_names = vol_row[vol_row >= high_cut].index
        ret_row = monthly_ret.loc[holding_date]
        long_ret = ret_row.reindex(long_names).dropna().mean()
        short_ret = ret_row.reindex(short_names).dropna().mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            continue
        def_returns[holding_date] = long_ret - short_ret

    series = pd.Series(def_returns).sort_index()
    series.index = pd.PeriodIndex(series.index, freq="M")
    series.name = "DEF"
    logger.info("DEF factor built: %d months from %d tickers (%d failed: %s)", len(series),
                len(tickers), len(meta["failed"]), list(meta["failed"])[:5])
    return series


def default_def_universe() -> list[str]:
    """A fixed, liquidity-oriented universe of long-history large-cap U.S. common stocks used to
    build the DEF proxy (see ``build_def_factor``). Selected by hand from long-standing S&P 500
    constituents with a multi-decade trading history, approximating the paper's liquidity-ranked
    cut of the full CRSP universe (see "Deviations from the paper" in results/RESULTS.md)."""
    tickers = [
        "XOM", "CVX", "IBM", "GE", "KO", "PG", "JNJ", "MRK", "PFE", "WMT", "MCD", "DIS", "HON",
        "CAT", "MMM", "UTX", "BA", "AXP", "JPM", "WFC", "C", "BAC", "USB", "T", "VZ", "CSCO",
        "INTC", "HPQ", "XRX", "DD", "DOW", "EMR", "ETN", "ITW", "PH", "ROK", "TXN", "ADM", "CL",
        "KMB", "HSY", "GIS", "K", "CPB", "HRL", "SYY", "CAG", "MO", "PM", "CVS", "WBA", "TGT",
        "HD", "LOW", "TJX", "ROST", "NKE", "VFC", "GPS", "LEG", "WHR", "F", "GM", "DE", "PCAR",
        "NSC", "UNP", "CSX", "FDX", "UPS", "LUV", "DAL", "AAL", "NOC", "LMT", "GD", "RTX", "ITT",
        "COP", "OXY", "SLB", "HAL", "BHI", "APD", "ECL", "PPG", "SHW", "NUE", "VMC", "MLM",
        "CMI", "DOV", "PNR", "SWK", "TAP", "STZ", "BF-B", "CLX", "CHD", "SJM", "MKC", "AVY",
        "BDX", "BAX", "ABT", "MDT", "SYK", "BSX", "ZBH", "CAH", "MCK", "ABC", "AEE", "AEP",
        "CNP", "DUK", "ED", "ETR", "EXC", "FE", "NEE", "PEG", "SO", "XEL", "D", "PPL", "WEC",
    ]
    return tickers
