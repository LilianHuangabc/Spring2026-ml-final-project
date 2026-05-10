"""Rolling backtest with strict 5-trading-day windows and no data leakage.

Leakage model: training row at date D has target = close(D+5)/close(D) - 1.
Eval window [start, end] uses prices from (start-1) through end.
No-leak requires D+5 < start-1 => D <= start - 6 trading days.
"""
import warnings
warnings.filterwarnings("ignore")

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EMBARGO_TRADING_DAYS = 6  # target horizon 5 + 1 buffer


def run_one(asof: str, start: str, end: str) -> tuple[float, str]:
    subprocess.run(
        ["python", "ensemble_submission.py", "--as-of", asof, "--out", "/tmp/sub_bt.csv"],
        capture_output=True, check=True,
    )
    out = subprocess.run(
        ["python", "score_submission.py", "/tmp/sub_bt.csv", "--start", start, "--end", end],
        capture_output=True, text=True, check=True,
    )
    excess = np.nan
    window_info = ""
    for line in out.stdout.splitlines():
        if line.startswith("Window:"):
            window_info = line.strip()
        if "excess return" in line:
            excess = float(line.split(":")[1].replace("%", "").strip().replace("+", ""))
    return excess, window_info


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    dates = np.sort(prices["date"].unique())
    n = len(dates)

    # Build eval windows stepping back every 5 trading days
    # Require: idx - EMBARGO_TRADING_DAYS >= 0, idx + 4 < n
    results = []
    idx = n - 5  # last full 5-day window ends at dates[n-1]
    while idx - EMBARGO_TRADING_DAYS >= 0:
        start_ts = pd.Timestamp(dates[idx])
        end_ts   = pd.Timestamp(dates[idx + 4])
        asof_ts  = pd.Timestamp(dates[idx - EMBARGO_TRADING_DAYS])

        start_s = start_ts.strftime("%Y%m%d")
        end_s   = end_ts.strftime("%Y%m%d")
        asof_s  = asof_ts.strftime("%Y%m%d")

        try:
            exc, win_info = run_one(asof_s, start_s, end_s)
            print(f"  asof={asof_s} -> {win_info}  excess={exc:+.3f}%")
            results.append({
                "asof": asof_ts, "start": start_ts, "end": end_ts,
                "excess": exc,
            })
        except subprocess.CalledProcessError as e:
            print(f"  asof={asof_s}  FAILED: {e.stderr.decode()[:200]}")

        idx -= 5  # step back 5 trading days (non-overlapping eval windows)

    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results).sort_values("start").reset_index(drop=True)
    print("\n" + "=" * 70)
    print(f"  {len(df)} non-overlapping 5-trading-day windows")
    print(f"  Embargo: {EMBARGO_TRADING_DAYS} trading days")
    print(f"  Mean excess    : {df['excess'].mean():+.3f}% per window")
    print(f"  Median excess  : {df['excess'].median():+.3f}%")
    print(f"  Std dev        : {df['excess'].std():.3f}%")
    print(f"  Win rate       : {(df['excess'] > 0).mean()*100:.0f}% ({(df['excess'] > 0).sum()}/{len(df)})")
    t = df['excess'].mean() / (df['excess'].std() / np.sqrt(len(df)))
    print(f"  t-stat         : {t:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
