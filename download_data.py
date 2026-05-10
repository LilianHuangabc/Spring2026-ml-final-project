"""
Download CSI500 data via akshare.

Produces three files under ./data/ :
  - constituents.csv      CSI500 constituents (as of download time)
  - prices.parquet        daily OHLCV for every constituent, forward-adjusted
  - index.parquet         daily OHLCV for the CSI500 index itself (benchmark)

Usage
-----
  # initial download (slow: ~500 API calls, expect 10-30 min)
  python download_data.py --start 20250101 --end 20260421

  # incremental update: resume from max date already in prices.parquet
  python download_data.py --update --end 20260430
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
CSI500_SYMBOL = "000905"


def fetch_constituents() -> pd.DataFrame:
    df = ak.index_stock_cons_csindex(symbol=CSI500_SYMBOL)
    # Columns are Chinese; normalize to what we need.
    rename = {
        "成分券代码": "stock_code",
        "成分券名称": "stock_name",
        "日期": "as_of_date",
    }
    df = df.rename(columns=rename)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    return df[["stock_code", "stock_name", "as_of_date"]]


def _exchange_prefix(code: str) -> str:
    # A-share listing: 6xx-xxx on SSE (Shanghai), everything else on SZSE.
    return "sh" if code.startswith("6") else "sz"


def fetch_stock_hist(code: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one stock via akshare's sina backend.

    We prefer stock_zh_a_daily over stock_zh_a_hist because the eastmoney
    backend used by the latter rate-limits aggressively for batch downloads.
    """
    symbol = f"{_exchange_prefix(code)}{code}"
    df = None
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    if df is None:
        print(f"  [warn] {symbol} failed after {retries} tries: {last_err}")
        return None
    if df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = code
    # sina returns turnover as a fraction (0..1-ish); pct_change is not provided,
    # so compute it from the adjusted close.
    df["pct_change"] = df["close"].pct_change() * 100.0
    keep = ["date", "stock_code", "open", "close", "high", "low", "volume", "amount", "turnover", "pct_change"]
    return df[[c for c in keep if c in df.columns]]


def fetch_index_hist(start: str, end: str) -> pd.DataFrame:
    # stock_zh_index_daily returns the full history; we filter client-side.
    # Uses sina backend; more stable than the eastmoney-backed index_zh_a_hist.
    df = ak.stock_zh_index_daily(symbol="sh000905")
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20250101", help="YYYYMMDD (ignored if --update)")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    p.add_argument("--update", action="store_true", help="incremental from max date in existing prices.parquet")
    p.add_argument("--sleep", type=float, default=0.1, help="seconds between stock requests")
    args = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    # Always refresh the constituent list and the benchmark index.
    print(">> Fetching CSI500 constituents...")
    cons = fetch_constituents()
    # constituents.csv is written at the end, filtered to stocks whose price
    # data actually came through. Stash the full list here for iteration.
    print(f"   {len(cons)} constituents from CSI500 index.")

    # Resolve start date (for --update, pick max date already cached).
    existing = None
    if args.update and (DATA_DIR / "prices.parquet").exists():
        existing = pd.read_parquet(DATA_DIR / "prices.parquet")
        max_date = existing["date"].max()
        start = (max_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        print(f">> Incremental update from {start} to {args.end}")
    else:
        start = args.start
        print(f">> Full download from {start} to {args.end}")

    print(">> Fetching CSI500 index benchmark...")
    idx = fetch_index_hist(start, args.end)
    if (DATA_DIR / "index.parquet").exists() and args.update:
        idx_old = pd.read_parquet(DATA_DIR / "index.parquet")
        idx = pd.concat([idx_old, idx]).drop_duplicates(subset=["date"]).sort_values("date")
    idx.to_parquet(DATA_DIR / "index.parquet", index=False)
    print(f"   {len(idx)} index rows.")

    print(">> Fetching constituent OHLCV...")
    frames = []
    ok_codes: list[str] = []
    n_fail = 0
    codes = cons["stock_code"].tolist()
    checkpoint_every = 100
    for i, code in enumerate(tqdm(codes)):
        df = fetch_stock_hist(code, start, args.end)
        if df is not None and not df.empty:
            frames.append(df)
            ok_codes.append(code)
        else:
            n_fail += 1
        # periodic checkpoint so a crash partway keeps partial progress
        if frames and (i + 1) % checkpoint_every == 0:
            tmp = pd.concat(frames, ignore_index=True)
            tmp.to_parquet(DATA_DIR / "prices.partial.parquet", index=False)
        time.sleep(args.sleep)

    if not frames:
        print(">> ERROR: no stocks downloaded successfully. Nothing to save.")
        return

    new_prices = pd.concat(frames, ignore_index=True)
    if existing is not None:
        prices = pd.concat([existing, new_prices], ignore_index=True)
        prices = prices.drop_duplicates(subset=["date", "stock_code"]).sort_values(["stock_code", "date"])
    else:
        prices = new_prices

    prices.to_parquet(DATA_DIR / "prices.parquet", index=False)
    partial = DATA_DIR / "prices.partial.parquet"
    if partial.exists():
        partial.unlink()

    # Write constituents.csv filtered to codes that we actually have prices for.
    # Merging with existing data on --update preserves codes present there too.
    final_codes = set(prices["stock_code"].unique())
    filtered = cons[cons["stock_code"].isin(final_codes)].reset_index(drop=True)
    filtered.to_csv(DATA_DIR / "constituents.csv", index=False)

    print(f">> Saved {len(prices):,} rows across {len(final_codes)} stocks to data/prices.parquet")
    print(f"   success: {len(ok_codes)}, failed: {n_fail}")
    print(f">> Saved {len(filtered)} constituents to data/constituents.csv (filtered to downloaded)")
    if n_fail:
        missing = sorted(set(codes) - final_codes)[:10]
        print(f"   dropped from universe (no price data): {missing}{'...' if n_fail > 10 else ''}")


if __name__ == "__main__":
    main()
