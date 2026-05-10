"""Compare LGB-only vs LGB+XGB blend on 12-fold purged walk-forward CV.

Blend method: IC-squared weighting per fold.
Run: python test_lgb_xgb_ensemble.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, _available_feat_cols,
)

DATA_DIR       = Path(__file__).parent / "data"
EMBARGO_DAYS   = FORWARD_HORIZON + 2
TEST_VAL_DAYS  = 15
INNER_VAL_DAYS = 40
MAX_WEIGHT     = 0.10
N_FOLDS        = 12
HALF_LIFE      = 45

LGB_PARAMS: dict = {
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 20,
    "n_estimators":      3000,
    "learning_rate":     0.015,
    "subsample":         0.75,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.05,
    "reg_lambda":        1.5,
    "min_split_gain":    0.01,
    "objective":         "huber",
    "alpha":             1.0,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}

XGB_PARAMS: dict = {
    "objective":         "reg:pseudohubererror",
    "huber_slope":       1.0,
    "eta":               0.015,
    "max_depth":         5,
    "min_child_weight":  10,
    "subsample":         0.75,
    "colsample_bytree":  0.7,
    "colsample_bylevel": 0.8,
    "gamma":             0.05,
    "reg_lambda":        1.5,
    "reg_alpha":         0.05,
    "nthread":           -1,
    "eval_metric":       "rmse",
    "tree_method":       "hist",
    "seed":              42,
}
XGB_ROUNDS = 3000
XGB_EARLY  = 50


def fit_sanitizer(df: pd.DataFrame, cols: list[str]) -> dict:
    stats: dict = {}
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        stats[col] = {"q01": s.quantile(0.01), "q99": s.quantile(0.99),
                      "mean": s.mean(), "std": s.std()}
    return stats


def apply_sanitizer(df: pd.DataFrame, cols: list[str], stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = df[col].clip(s["q01"], s["q99"])
            df[col] = (df[col] - s["mean"]) / (s["std"] + 1e-6)
    return df


def time_decay_weights(dates: pd.Series, half_life: int = HALF_LIFE) -> np.ndarray:
    days_ago = (dates.max() - dates).dt.days.to_numpy().astype(float)
    w = np.exp(-np.log(2) / half_life * days_ago)
    return w / w.mean()


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def rank_norm(arr: np.ndarray) -> np.ndarray:
    r = pd.Series(arr).rank(pct=True).to_numpy()
    return (r - 0.5) * 2


def train_lgb(train_df, val_df, cols, weights):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(
        train_df[cols], train_df[TARGET_COLUMN],
        sample_weight=weights,
        eval_set=[(val_df[cols], val_df[TARGET_COLUMN])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return m


def train_xgb(train_df, val_df, cols, weights):
    dtrain = xgb.DMatrix(train_df[cols], label=train_df[TARGET_COLUMN], weight=weights)
    dval   = xgb.DMatrix(val_df[cols],   label=val_df[TARGET_COLUMN])
    return xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=XGB_EARLY,
        verbose_eval=False,
    )


def run_cv(train_pool: pd.DataFrame, feat_cols: list[str]) -> None:
    all_dates  = np.sort(train_pool["date"].unique())
    total_days = len(all_dates)
    min_train  = 120

    folds: list[tuple] = []
    for k in range(N_FOLDS):
        ve = total_days - 1 - k * TEST_VAL_DAYS
        vs = ve - TEST_VAL_DAYS + 1
        if vs - EMBARGO_DAYS < min_train:
            break
        folds.append((pd.Timestamp(all_dates[vs]), pd.Timestamp(all_dates[ve])))
    folds = list(reversed(folds))

    lgb_ics:    list[float] = []
    xgb_ics:    list[float] = []
    blend_ics:  list[float] = []

    print(f"{'Fold':<5} {'Val window':<28} {'LGB IC':>8} {'XGB IC':>8} {'Blend IC':>9}")
    print("-" * 65)

    for k, (val_start, val_end) in enumerate(folds):
        vsi = int(np.searchsorted(all_dates, np.datetime64(val_start)))
        tei = vsi - EMBARGO_DAYS - 1
        if tei < min_train:
            continue

        isi  = max(0, tei - INNER_VAL_DAYS)
        istart = pd.Timestamp(all_dates[isi])
        tend   = pd.Timestamp(all_dates[tei])

        train_df  = train_pool[train_pool["date"] <  istart].copy()
        inner_val = train_pool[(train_pool["date"] >= istart) & (train_pool["date"] <= tend)].copy()
        val_df    = train_pool[(train_pool["date"] >= val_start) & (train_pool["date"] <= val_end)].copy()

        if len(train_df) < 500 or inner_val.empty or val_df.empty:
            continue

        stats     = fit_sanitizer(train_df, feat_cols)
        train_df  = apply_sanitizer(train_df, feat_cols, stats)
        inner_val = apply_sanitizer(inner_val, feat_cols, stats)
        val_df    = apply_sanitizer(val_df,    feat_cols, stats)

        sw = time_decay_weights(train_df["date"])
        cols = [c for c in feat_cols if c in train_df.columns]

        # LGB
        lgb_m  = train_lgb(train_df, inner_val, cols, sw)
        lgb_p  = rank_norm(lgb_m.predict(val_df[cols]))
        lgb_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), lgb_p, val_df["date"].to_numpy())

        # XGB
        xgb_m  = train_xgb(train_df, inner_val, cols, sw)
        xgb_p  = rank_norm(xgb_m.predict(xgb.DMatrix(val_df[cols])))
        xgb_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), xgb_p, val_df["date"].to_numpy())

        # IC-squared blend
        lgb_w2  = lgb_ic ** 2 if not np.isnan(lgb_ic) else 0.0
        xgb_w2  = xgb_ic ** 2 if not np.isnan(xgb_ic) else 0.0
        total_w = lgb_w2 + xgb_w2 + 1e-9
        blend_p = (lgb_w2 / total_w) * lgb_p + (xgb_w2 / total_w) * xgb_p
        blend_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), blend_p, val_df["date"].to_numpy())

        lgb_ics.append(lgb_ic)
        xgb_ics.append(xgb_ic)
        blend_ics.append(blend_ic)

        print(f"{k+1:<5} {str(val_start.date())+' → '+str(val_end.date()):<28}"
              f" {lgb_ic:>8.4f} {xgb_ic:>8.4f} {blend_ic:>9.4f}")

    print("-" * 65)
    def stats_row(ics, label):
        m  = float(np.nanmean(ics))
        ir = float(np.mean(ics) / (np.std(ics) + 1e-9)) if len(ics) >= 2 else float("nan")
        wins = sum(1 for ic in ics if ic > 0)
        print(f"  {label:<18} Mean IC={m:+.4f}  IC-IR={ir:.4f}  wins={wins}/{len(ics)}")
        return m

    print()
    lgb_mean   = stats_row(lgb_ics,   "LGB-only")
    xgb_mean   = stats_row(xgb_ics,   "XGB-only")
    blend_mean = stats_row(blend_ics, "LGB+XGB blend")

    print()
    winner = "LGB+XGB blend" if blend_mean > lgb_mean else "LGB-only"
    delta  = blend_mean - lgb_mean
    print(f"  >> Winner: {winner}  (delta={delta:+.4f})")


if __name__ == "__main__":
    print(">> Loading data...")
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")

    print(">> Building features...")
    panel      = build_features(prices, index_df=index_df)
    train_pool = training_frame(panel)
    feat_cols  = _available_feat_cols(train_pool)

    print(f">> {N_FOLDS}-fold CV  |  half_life={HALF_LIFE}d  |  {len(feat_cols)} features\n")
    run_cv(train_pool, feat_cols)
