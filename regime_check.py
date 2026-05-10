"""Check whether reversal factors still work in recent weeks."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import build_features, training_frame, TARGET_COLUMN

DATA_DIR = Path(__file__).parent / "data"

# Features to check — the 12 currently in use plus the momentum/vol features
# we removed (to see if regime has flipped)
CHECK_FEATURES = [
    # Current production features (reversal/flow)
    "rev_10d", "rev_20d", "rev_ma20",
    "rev_1d", "rev_3d",
    "rev_pvd_10d", "rev_pvd_5d",
    "rev_ofi_ma5", "vol_lead_ret_1d",
    "upper_shadow_ratio",
    "rev_vol_z_20d", "rev_vs_3d",
    # Removed momentum features (to check regime)
    "ret_5d", "ret_10d",
]


def ic_per_window(df: pd.DataFrame, col: str) -> dict:
    """IC over various recent windows."""
    if col not in df.columns:
        return {}

    dates = np.sort(df["date"].unique())
    out = {}
    for window_days in [20, 40, 60, 120, 9999]:  # 9999 = all history
        if window_days == 9999:
            cutoff = dates[0]
            label = "all"
        else:
            if window_days >= len(dates):
                continue
            cutoff = dates[-window_days]
            label = f"last_{window_days}d"

        sub = df[df["date"] >= cutoff]
        ics = []
        for d in np.sort(sub["date"].unique()):
            day = sub[sub["date"] == d][[col, TARGET_COLUMN]].dropna()
            if len(day) < 30:
                continue
            r, _ = spearmanr(day[col], day[TARGET_COLUMN])
            if not np.isnan(r):
                ics.append(r)
        if not ics:
            continue
        ic_mean = float(np.mean(ics))
        sign_pct = float(np.mean(np.array(ics) > 0))
        out[label] = (ic_mean, sign_pct, len(ics))
    return out


def main():
    print(">> Loading + building features...")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    panel = build_features(prices, index_df=index_df)
    df = panel.dropna(subset=[TARGET_COLUMN]).copy()

    last_date = df["date"].max()
    print(f"   Last date: {last_date.date()}, total rows: {len(df):,}\n")

    # Header
    windows = ["last_20d", "last_40d", "last_60d", "last_120d", "all"]
    print(f"{'Feature':<25} " + " ".join(f"{w:>15}" for w in windows))
    print("-" * (25 + 16 * len(windows)))

    for col in CHECK_FEATURES:
        result = ic_per_window(df, col)
        cells = []
        for w in windows:
            if w in result:
                ic, sign, n = result[w]
                cells.append(f"{ic:+.4f}/{sign*100:.0f}%")
            else:
                cells.append("--")
        print(f"{col:<25} " + " ".join(f"{c:>15}" for c in cells))

    print()
    print("Cell format: IC / sign_consistency%")
    print("Reversal features should have positive IC + >55% sign consistency.")
    print("If recent windows show IC near 0 or negative -> regime break.")


if __name__ == "__main__":
    main()
