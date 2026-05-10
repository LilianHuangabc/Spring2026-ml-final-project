"""Quick factor eval on ALL computed features to find what's actually useful."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import build_features, training_frame, TARGET_COLUMN

DATA_DIR = Path(__file__).parent / "data"


def main():
    print(">> Loading + building features...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    panel = build_features(prices, index_df=index_df)
    df = panel.dropna(subset=[TARGET_COLUMN]).copy()

    last_date = df["date"].max()
    print(f"   Last date: {last_date.date()}, total rows: {len(df):,}\n")

    # Get all numeric columns that aren't metadata or target
    skip = {"stock_code", "date", "open", "high", "low", "close", "volume",
            "turnover", TARGET_COLUMN, "ret_1d"}
    candidates = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]

    # Compute IC over last 60 trading days (recent regime)
    dates = np.sort(df["date"].unique())
    recent_dates = dates[-60:]

    results = []
    for col in candidates:
        ics = []
        for d in recent_dates:
            day = df[df["date"] == d][[col, TARGET_COLUMN]].dropna()
            if len(day) < 30:
                continue
            r, _ = spearmanr(day[col], day[TARGET_COLUMN])
            if not np.isnan(r):
                ics.append(r)
        if len(ics) < 20:
            continue
        ic_mean = float(np.mean(ics))
        ic_std = float(np.std(ics))
        sign_pct = float(np.mean(np.array(ics) > 0))
        ir = ic_mean / (ic_std + 1e-6)
        results.append({
            "feature": col,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "sign_pct": sign_pct,
            "n_days": len(ics),
        })

    res_df = pd.DataFrame(results).sort_values("ir", ascending=False)
    print(f"{'Feature':<30} {'IC_mean':>8} {'IC_std':>8} {'IR':>8} {'Sign%':>8}")
    print("-" * 72)
    for _, row in res_df.head(40).iterrows():
        print(f"{row['feature']:<30} {row['ic_mean']:+.4f}   {row['ic_std']:.4f}   {row['ir']:+.3f}   {row['sign_pct']*100:.0f}%")

    print(f"\n\nBottom 10 (negative IC = momentum works as SHORT signal):")
    print(f"{'Feature':<30} {'IC_mean':>8} {'IC_std':>8} {'IR':>8} {'Sign%':>8}")
    print("-" * 72)
    for _, row in res_df.tail(10).iterrows():
        print(f"{row['feature']:<30} {row['ic_mean']:+.4f}   {row['ic_std']:.4f}   {row['ir']:+.3f}   {row['sign_pct']*100:.0f}%")


if __name__ == "__main__":
    main()
