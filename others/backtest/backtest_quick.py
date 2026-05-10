"""Quick rolling backtest — recent 15 non-overlapping windows only."""
import warnings
warnings.filterwarnings("ignore")

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EMBARGO_TRADING_DAYS = 6
N_WINDOWS = 15


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
    dates = np.sort(prices["date"].unique())
    n = len(dates)

    results = []
    idx = n - 5
    count = 0
    while idx - EMBARGO_TRADING_DAYS >= 0 and count < N_WINDOWS:
        start_ts = pd.Timestamp(dates[idx])
        end_ts   = pd.Timestamp(dates[idx + 4])
        asof_ts  = pd.Timestamp(dates[idx - EMBARGO_TRADING_DAYS])

        start_s = start_ts.strftime("%Y%m%d")
        end_s   = end_ts.strftime("%Y%m%d")
        asof_s  = asof_ts.strftime("%Y%m%d")

        try:
            exc = run_one(asof_s, start_s, end_s)
            print(f"  asof={asof_s} window {start_s}..{end_s}  excess={exc:+.3f}%", flush=True)
            results.append({"start": start_ts, "excess": exc})
        except subprocess.CalledProcessError as e:
            print(f"  asof={asof_s}  FAILED", flush=True)

        idx -= 5
        count += 1

    df = pd.DataFrame(results).sort_values("start").reset_index(drop=True)
    print("\n" + "=" * 70)
    print(f"  {len(df)} windows")
    print(f"  Mean excess   : {df['excess'].mean():+.3f}%")
    print(f"  Median excess : {df['excess'].median():+.3f}%")
    print(f"  Std dev       : {df['excess'].std():.3f}%")
    print(f"  Win rate      : {(df['excess'] > 0).mean()*100:.0f}% ({(df['excess'] > 0).sum()}/{len(df)})")
    print(f"  Min (worst)   : {df['excess'].min():+.3f}%")
    print(f"  Max (best)    : {df['excess'].max():+.3f}%")
    print(f"  t-stat        : {df['excess'].mean() / (df['excess'].std() / np.sqrt(len(df))):.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
