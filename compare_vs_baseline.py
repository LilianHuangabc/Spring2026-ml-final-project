"""Head-to-head: baseline (orig_features) vs ensemble. Reports excess return + IC.

Uses run_orig_baseline.py (baseline logic + orig_features) as the official
baseline, and ensemble_submission.py as the ensemble. Tests on clean Mon-Fri
5-trading-day windows with 6-day embargo. Computes IC via a separate helper
that mirrors score_submission.py conventions.
"""
import warnings
warnings.filterwarnings("ignore")

import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent / "data"
EMBARGO_TRADING_DAYS = 6
N_WINDOWS = 10


def score_excess(sub_path: str, start: str, end: str) -> float:
    """Run score_submission.py and return excess return in %."""
    out = subprocess.run(
        ["python", "score_submission.py", sub_path,
         "--start", start, "--end", end],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        if "excess return" in line:
            return float(line.split(":")[1].replace("%", "").strip().replace("+", ""))
    return np.nan


def realized_ic(sub_path: str, prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Spearman rank-IC between submission weights and realized per-stock returns.

    Uses same entry/exit convention as score_submission.py:
      entry = close(day_before_start), fallback to open(start)
      exit  = close(end)
    """
    sub = pd.read_csv(sub_path, dtype={"stock_code": str})
    sub["stock_code"] = sub["stock_code"].str.zfill(6)
    rets = []
    weights = []
    for code, w in zip(sub["stock_code"], sub["weight"]):
        df = prices[prices["stock_code"] == code].sort_values("date")
        before = df[df["date"] < start]
        in_window = df[(df["date"] >= start) & (df["date"] <= end)]
        if in_window.empty:
            continue
        if not before.empty:
            entry = before["close"].iloc[-1]
        else:
            entry = in_window["open"].iloc[0]
        exit_ = in_window["close"].iloc[-1]
        if entry <= 0 or pd.isna(entry) or pd.isna(exit_):
            continue
        rets.append(float(exit_ / entry - 1.0))
        weights.append(float(w))
    if len(rets) < 20:
        return np.nan
    r, _ = spearmanr(weights, rets)
    return float(r) if not np.isnan(r) else np.nan


def main():
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
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
    print(f"Testing on {len(clean_weeks)} clean Mon-Fri 5-day windows.")
    print(f"Baseline: run_orig_baseline.py (orig_features.py)")
    print(f"Ensemble: ensemble_submission.py (features.py)\n")

    hdr = f"{'Window':<24}  {'Base_Exc':>9} {'Ens_Exc':>9}  {'Base_IC':>8} {'Ens_IC':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for start_ts, end_ts, asof_ts in clean_weeks:
        start_s = start_ts.strftime("%Y%m%d")
        end_s   = end_ts.strftime("%Y%m%d")
        asof_s  = asof_ts.strftime("%Y%m%d")

        subprocess.run(
            ["python", "run_orig_baseline.py", "--as-of", asof_s, "--out", "/tmp/sub_base.csv"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["python", "ensemble_submission.py", "--as-of", asof_s, "--out", "/tmp/sub_ens.csv"],
            capture_output=True, check=True,
        )

        base_exc = score_excess("/tmp/sub_base.csv", start_s, end_s)
        ens_exc  = score_excess("/tmp/sub_ens.csv",  start_s, end_s)
        base_ic  = realized_ic("/tmp/sub_base.csv", prices, start_ts, end_ts)
        ens_ic   = realized_ic("/tmp/sub_ens.csv",  prices, start_ts, end_ts)

        rows.append({
            "window": f"{start_s}-{end_s}",
            "base_exc": base_exc, "ens_exc": ens_exc,
            "base_ic": base_ic,   "ens_ic": ens_ic,
        })
        print(f"{start_s}-{end_s}  {base_exc:>+8.3f}%  {ens_exc:>+8.3f}%"
              f"   {base_ic:>+7.4f}  {ens_ic:>+7.4f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  CROSS-VALIDATION SUMMARY (10 Mon-Fri Weeks)")
    print("=" * 80)
    print(f"  {'Metric':<32} {'Baseline':>12} {'Ensemble':>12}")
    print("  " + "-" * 58)
    print(f"  {'Mean excess return':<32} {df['base_exc'].mean():>+11.3f}% {df['ens_exc'].mean():>+11.3f}%")
    print(f"  {'Median excess return':<32} {df['base_exc'].median():>+11.3f}% {df['ens_exc'].median():>+11.3f}%")
    print(f"  {'Std dev excess':<32} {df['base_exc'].std():>12.3f}% {df['ens_exc'].std():>12.3f}%")
    print(f"  {'Win rate vs index':<32} {(df['base_exc']>0).mean()*100:>11.0f}% {(df['ens_exc']>0).mean()*100:>11.0f}%")
    print(f"  {'Mean IC':<32} {df['base_ic'].mean():>+12.4f} {df['ens_ic'].mean():>+12.4f}")
    print(f"  {'Median IC':<32} {df['base_ic'].median():>+12.4f} {df['ens_ic'].median():>+12.4f}")
    print(f"  {'IC > 0 frac':<32} {(df['base_ic']>0).mean()*100:>11.0f}% {(df['ens_ic']>0).mean()*100:>11.0f}%")
    print()
    diff_exc = df['ens_exc'] - df['base_exc']
    diff_ic  = df['ens_ic']  - df['base_ic']
    print(f"  Ensemble beats baseline on excess: {(diff_exc>0).sum()}/{len(df)} windows")
    print(f"  Ensemble beats baseline on IC    : {(diff_ic>0).sum()}/{len(df)} windows")
    print(f"  Avg excess advantage (Ens - Base): {diff_exc.mean():+.3f}%")
    print(f"  Avg IC advantage (Ens - Base)    : {diff_ic.mean():+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
