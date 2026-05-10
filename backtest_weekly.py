"""Backtest on clean Mon-Fri 5-trading-day weeks only.

Each window is a full Monday-to-Friday week with no holiday interruptions,
mirroring the structure of the live eval window (2026-05-11 to 2026-05-15).
"""
import warnings
warnings.filterwarnings("ignore")

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EMBARGO_TRADING_DAYS = 6


def run_one(asof: str, start: str, end: str) -> float:
    subprocess.run(
        ["python", "ensemble_submission.py", "--as-of", asof, "--out", "/tmp/sub_bt.csv"],
        capture_output=True, check=True,
    )
    out = subprocess.run(
        ["python", "score_submission.py", "/tmp/sub_bt.csv", "--start", start, "--end", end],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if "excess return" in line:
            return float(line.split(":")[1].replace("%", "").strip().replace("+", ""))
    return np.nan


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    dates = pd.to_datetime(pd.Series(np.sort(prices["date"].unique())))

    # Find every Monday in the data; check if Mon+4 trading days = the same Friday
    mondays = [d for d in dates if d.weekday() == 0]

    clean_weeks = []
    for mon in mondays:
        # The 5 trading days starting from mon
        idx = dates.searchsorted(mon)
        if idx + 4 >= len(dates):
            continue
        if idx - EMBARGO_TRADING_DAYS < 60:  # need enough training data
            continue
        friday_candidate = dates.iloc[idx + 4]
        expected_friday = mon + pd.Timedelta(days=4)
        # Only keep if Mon..Fri = exactly 5 consecutive trading days
        if friday_candidate == expected_friday:
            clean_weeks.append((mon, friday_candidate, dates.iloc[idx - EMBARGO_TRADING_DAYS]))

    # Take most recent 15 clean weeks
    clean_weeks = clean_weeks[-15:]

    print(f"Found {len(clean_weeks)} clean Mon-Fri weeks with embargo.\n")

    results = []
    for start_ts, end_ts, asof_ts in clean_weeks:
        start_s = start_ts.strftime("%Y%m%d")
        end_s   = end_ts.strftime("%Y%m%d")
        asof_s  = asof_ts.strftime("%Y%m%d")

        try:
            exc = run_one(asof_s, start_s, end_s)
            print(f"  asof={asof_s} Mon-Fri {start_s}..{end_s}  excess={exc:+.3f}%", flush=True)
            results.append({"start": start_ts, "end": end_ts, "excess": exc})
        except subprocess.CalledProcessError as e:
            print(f"  asof={asof_s}  FAILED", flush=True)

    df = pd.DataFrame(results).sort_values("start").reset_index(drop=True)
    print("\n" + "=" * 70)
    print(f"  {len(df)} clean Mon-Fri 5-trading-day weeks")
    print(f"  Mean excess    : {df['excess'].mean():+.3f}%")
    print(f"  Median excess  : {df['excess'].median():+.3f}%")
    print(f"  Std dev        : {df['excess'].std():.3f}%")
    print(f"  Win rate       : {(df['excess'] > 0).mean()*100:.0f}% ({(df['excess'] > 0).sum()}/{len(df)})")
    print(f"  Best week      : {df['excess'].max():+.3f}%")
    print(f"  Worst week     : {df['excess'].min():+.3f}%")
    print(f"  t-stat         : {df['excess'].mean() / (df['excess'].std() / np.sqrt(len(df))):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
