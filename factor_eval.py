"""Factor evaluation pipeline: long-short, rolling IC, incremental signal, ranking."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, ttest_1samp

from features import (
    TARGET_COLUMN, FEATURE_COLUMNS,
    build_features, training_frame, _available_feat_cols,
)

DATA_DIR = Path(__file__).parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    panel = build_features(prices, index_df=index_df)
    return training_frame(panel)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Long-Short Test
# ─────────────────────────────────────────────────────────────────────────────

def long_short_test(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    results = []
    for col in feat_cols:
        daily_rets = []
        for _, grp in df.groupby("date"):
            vals = grp[[col, TARGET_COLUMN]].dropna()
            if len(vals) < 50:
                continue
            n = len(vals)
            top_q = vals[col].quantile(0.9)
            bot_q = vals[col].quantile(0.1)
            long_ret = vals.loc[vals[col] >= top_q, TARGET_COLUMN].mean()
            short_ret = vals.loc[vals[col] <= bot_q, TARGET_COLUMN].mean()
            daily_rets.append(long_ret - short_ret)

        if len(daily_rets) < 10:
            results.append({"feature": col, "ls_mean": np.nan, "ls_sharpe": np.nan, "ls_tstat": np.nan})
            continue

        arr = np.array(daily_rets)
        mean_r = arr.mean()
        std_r = arr.std() + 1e-9
        sharpe = mean_r / std_r * np.sqrt(252)
        tstat, _ = ttest_1samp(arr, 0)
        results.append({"feature": col, "ls_mean": mean_r, "ls_sharpe": sharpe, "ls_tstat": tstat})

    return pd.DataFrame(results).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling IC Stability
# ─────────────────────────────────────────────────────────────────────────────

def rolling_ic_stability(df: pd.DataFrame, feat_cols: list[str], window_days: int = 126) -> pd.DataFrame:
    dates = np.sort(df["date"].unique())
    results = []

    for col in feat_cols:
        ic_series = []
        for d in dates:
            grp = df.loc[df["date"] == d, [col, TARGET_COLUMN]].dropna()
            if len(grp) < 30:
                continue
            rho, _ = spearmanr(grp[col], grp[TARGET_COLUMN])
            if not np.isnan(rho):
                ic_series.append(rho)

        if len(ic_series) < 20:
            results.append({
                "feature": col, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "sign_consistency": np.nan,
            })
            continue

        arr = np.array(ic_series)
        ic_mean = arr.mean()
        ic_std = arr.std() + 1e-9
        icir = ic_mean / ic_std
        sign_cons = (arr > 0).mean()

        results.append({
            "feature": col,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "sign_consistency": sign_cons,
        })

    return pd.DataFrame(results).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Incremental Signal Test
# ─────────────────────────────────────────────────────────────────────────────

def incremental_signal_test(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in feat_cols if c in df.columns]
    X = df[cols].values
    y = df[TARGET_COLUMN].values

    model = lgb.LGBMRegressor(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        verbose=-1, n_jobs=-1, random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    residuals = y - preds

    dates = df["date"].values
    results = []
    for i, col in enumerate(cols):
        ics = []
        for d in np.unique(dates):
            mask = dates == d
            if mask.sum() < 30:
                continue
            feat_vals = X[mask, i]
            res_vals = residuals[mask]
            valid = ~(np.isnan(feat_vals) | np.isnan(res_vals))
            if valid.sum() < 20:
                continue
            rho, _ = spearmanr(feat_vals[valid], res_vals[valid])
            if not np.isnan(rho):
                ics.append(rho)
        inc_ic = float(np.mean(ics)) if ics else 0.0
        results.append({"feature": col, "incremental_ic": inc_ic})

    return pd.DataFrame(results).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Redundancy (pairwise correlation of cross-sectional ranks)
# ─────────────────────────────────────────────────────────────────────────────

def redundancy_analysis(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in feat_cols if c in df.columns]
    rank_df = df[cols].rank(pct=True)
    corr_matrix = rank_df.corr(method="spearman")

    results = []
    for col in cols:
        others = corr_matrix[col].drop(col).abs()
        max_corr = others.max()
        max_corr_with = others.idxmax()
        mean_corr = others.mean()
        n_high = (others > 0.7).sum()
        results.append({
            "feature": col,
            "max_abs_corr": max_corr,
            "max_corr_with": max_corr_with,
            "mean_abs_corr": mean_corr,
            "n_corr_above_0.7": n_high,
        })

    return pd.DataFrame(results).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Final Ranking
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranking(ls_df: pd.DataFrame, ic_df: pd.DataFrame, inc_df: pd.DataFrame) -> pd.DataFrame:
    merged = ls_df.join(ic_df, how="outer").join(inc_df, how="outer")

    icir = merged["icir"].abs().fillna(0)
    sharpe = merged["ls_sharpe"].abs().fillna(0)
    stability = merged["sign_consistency"].fillna(0.5)

    merged["score"] = icir * np.sqrt(sharpe.clip(lower=0) + 1e-6) * stability
    merged = merged.sort_values("score", ascending=False)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(">> Loading data and building features...")
    df = load_panel()
    feat_cols = _available_feat_cols(df)
    print(f"   {len(feat_cols)} features, {len(df):,} rows\n")

    print(">> [1/4] Long-Short Test...")
    ls_df = long_short_test(df, feat_cols)

    print(">> [2/4] Rolling IC Stability...")
    ic_df = rolling_ic_stability(df, feat_cols)

    print(">> [3/4] Incremental Signal Test...")
    inc_df = incremental_signal_test(df, feat_cols)

    print(">> [4/4] Redundancy Analysis...")
    red_df = redundancy_analysis(df, feat_cols)

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranking = compute_ranking(ls_df, ic_df, inc_df)
    ranking = ranking.join(red_df[["max_abs_corr", "max_corr_with", "n_corr_above_0.7"]], how="left")

    # ── Output ────────────────────────────────────────────────────────────────
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n" + "=" * 100)
    print("  TOP 10 ALPHA FEATURES (highest composite score)")
    print("=" * 100)
    top10 = ranking.head(10)[["score", "icir", "ls_sharpe", "sign_consistency", "incremental_ic", "ic_mean"]]
    print(top10.to_string())

    print("\n" + "=" * 100)
    print("  TOP 10 REDUNDANT FEATURES (high overlap with other features)")
    print("=" * 100)
    redundant = ranking.sort_values("max_abs_corr", ascending=False).head(10)[
        ["max_abs_corr", "max_corr_with", "n_corr_above_0.7", "score", "icir"]
    ]
    print(redundant.to_string())

    print("\n" + "=" * 100)
    print("  UNSTABLE / NOISY FEATURES (low sign consistency OR low |ICIR|)")
    print("=" * 100)
    noisy_mask = (ranking["sign_consistency"] < 0.55) | (ranking["icir"].abs() < 0.05)
    noisy = ranking[noisy_mask].sort_values("sign_consistency")[
        ["sign_consistency", "icir", "ic_mean", "ic_std", "ls_sharpe", "score"]
    ]
    print(noisy.to_string() if len(noisy) > 0 else "  (none)")

    print("\n" + "=" * 100)
    print("  FULL RANKING")
    print("=" * 100)
    full = ranking[["score", "icir", "ls_sharpe", "sign_consistency", "incremental_ic",
                    "ic_mean", "max_abs_corr", "max_corr_with"]]
    print(full.to_string())


if __name__ == "__main__":
    main()
