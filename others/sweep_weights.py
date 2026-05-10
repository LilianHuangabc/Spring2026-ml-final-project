"""Sweep weighting schemes on top-50 picks to find best mean-excess / risk trade.

For each of 10 Mon-Fri windows:
  1. Generate top-50 picks from ensemble_submission.py (score-ranked)
  2. Re-weight the same 50 picks with each scheme below
  3. Score via score_submission.py
Aggregate: mean excess, std dev, Sharpe-like (mean/std), win rate per scheme.
"""
import warnings
warnings.filterwarnings("ignore")

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
EMBARGO_TRADING_DAYS = 6
N_WINDOWS = 10
MAX_WEIGHT = 0.10

# Weighting schemes to test. All operate on rank-ordered top-50 picks
# (rank 50 = best, rank 1 = weakest of the top-50).
SCHEMES: dict[str, callable] = {
    "equal":        lambda ranks, k: np.ones(k) / k,
    "rankexp_0.5":  lambda ranks, k: np.exp(ranks / k * 0.5),
    "rankexp_1.0":  lambda ranks, k: np.exp(ranks / k * 1.0),
    "rankexp_1.5":  lambda ranks, k: np.exp(ranks / k * 1.5),
    "rankexp_2.0":  lambda ranks, k: np.exp(ranks / k * 2.0),
    "rankexp_3.0":  lambda ranks, k: np.exp(ranks / k * 3.0),
    "linear":       lambda ranks, k: ranks.astype(float),
    "sqrt_rank":    lambda ranks, k: np.sqrt(ranks.astype(float)),
    "rank_squared": lambda ranks, k: ranks.astype(float) ** 2,
}


def normalize_with_cap(raw_w: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    w = raw_w / raw_w.sum()
    for _ in range(50):
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()


def score_excess(sub_path: str, start: str, end: str) -> float:
    out = subprocess.run(
        ["python", "score_submission.py", sub_path, "--start", start, "--end", end],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if "excess return" in line:
            return float(line.split(":")[1].replace("%", "").strip().replace("+", ""))
    return np.nan


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    dates = pd.to_datetime(pd.Series(np.sort(prices["date"].unique())))
    mondays = [d for d in dates if d.weekday() == 0]

    clean_weeks = []
    for mon in mondays:
        idx = dates.searchsorted(mon)
        if idx + 4 >= len(dates):
            continue
        if idx - EMBARGO_TRADING_DAYS < 60:
            continue
        friday = dates.iloc[idx + 4]
        if friday == mon + pd.Timedelta(days=4):
            clean_weeks.append((mon, friday, dates.iloc[idx - EMBARGO_TRADING_DAYS]))

    clean_weeks = clean_weeks[-N_WINDOWS:]
    print(f"Testing {len(SCHEMES)} schemes on {len(clean_weeks)} windows.\n")

    # Store {window_key: {scheme: excess_pct}}
    results: list[dict] = []

    for start_ts, end_ts, asof_ts in clean_weeks:
        start_s = start_ts.strftime("%Y%m%d")
        end_s   = end_ts.strftime("%Y%m%d")
        asof_s  = asof_ts.strftime("%Y%m%d")

        # 1. Generate picks once per window
        subprocess.run(
            ["python", "ensemble_submission.py", "--as-of", asof_s, "--out", "/tmp/sub_pick.csv"],
            capture_output=True, check=True,
        )
        sub = pd.read_csv("/tmp/sub_pick.csv", dtype={"stock_code": str})
        sub = sub.sort_values("weight")  # ascending weight => rank 1..k
        stock_codes = sub["stock_code"].values
        k = len(stock_codes)
        ranks = np.arange(1, k + 1)

        # 2. Test each scheme
        row = {"window": f"{start_s}-{end_s}"}
        for name, fn in SCHEMES.items():
            raw = fn(ranks, k)
            w = normalize_with_cap(raw.astype(float))
            out_df = pd.DataFrame({"stock_code": stock_codes, "weight": w})
            out_path = f"/tmp/sub_{name}.csv"
            out_df.to_csv(out_path, index=False)
            exc = score_excess(out_path, start_s, end_s)
            row[name] = exc
        results.append(row)
        print(f"  {start_s}-{end_s} done", flush=True)

    df = pd.DataFrame(results)

    # Summary
    print("\n" + "=" * 96)
    print(f"  WEIGHTING SCHEME SWEEP ({N_WINDOWS} Mon-Fri weeks)")
    print("=" * 96)
    print(f"  {'Scheme':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Sharpe':>8} {'Win%':>6} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 82)

    scheme_stats = []
    for name in SCHEMES.keys():
        excs = df[name].dropna().values
        mean = excs.mean()
        med  = np.median(excs)
        std  = excs.std()
        sharpe = mean / std if std > 0 else np.nan
        win  = (excs > 0).mean() * 100
        scheme_stats.append({
            "scheme": name, "mean": mean, "median": med, "std": std,
            "sharpe": sharpe, "win": win, "min": excs.min(), "max": excs.max(),
        })
        print(f"  {name:<15} {mean:>+7.3f}% {med:>+7.3f}% {std:>7.3f}% {sharpe:>+7.3f} "
              f"{win:>5.0f}% {excs.min():>+7.3f}% {excs.max():>+7.3f}%")

    print("  " + "-" * 82)

    # Rank by sharpe
    stats_df = pd.DataFrame(scheme_stats).sort_values("sharpe", ascending=False)
    print(f"\n  Best by Sharpe (risk-adjusted): {stats_df.iloc[0]['scheme']} "
          f"(sharpe={stats_df.iloc[0]['sharpe']:+.3f})")
    print(f"  Best by mean:                   {stats_df.sort_values('mean', ascending=False).iloc[0]['scheme']} "
          f"(mean={stats_df.sort_values('mean', ascending=False).iloc[0]['mean']:+.3f}%)")
    print(f"  Best by median:                 {stats_df.sort_values('median', ascending=False).iloc[0]['scheme']} "
          f"(median={stats_df.sort_values('median', ascending=False).iloc[0]['median']:+.3f}%)")
    print(f"  Best by win rate:               {stats_df.sort_values('win', ascending=False).iloc[0]['scheme']} "
          f"(win={stats_df.sort_values('win', ascending=False).iloc[0]['win']:.0f}%)")


if __name__ == "__main__":
    main()
