"""XGBoost baseline for the CSI500 stock-selection competition — T+5 edition.

Aligned with baseline_ensemble.py conventions:
  - Same embargo / inner-val / test-val split logic
  - Same time-decay weight formula (half_life=45d default)
  - Same rank_ic metric
  - RankExp portfolio weighting (same as baseline_ensemble.py)
  - top_k=40 default

Usage
-----
  python baseline_xgboost.py --out submission_xgb.csv
  python baseline_xgboost.py --cv --out submission_xgb.csv
  python baseline_xgboost.py --cv --cv-folds 5 --half-life 45 --out submission_xgb.csv
  python baseline_xgboost.py --no-decay --out submission_xgb.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
    _available_feat_cols,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (kept in sync with baseline_ensemble.py)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

EMBARGO_DAYS   = FORWARD_HORIZON + 2   # 7 days for T+5
TEST_VAL_DAYS  = 15
INNER_VAL_DAYS = 40
MIN_STOCKS     = 30
MAX_WEIGHT     = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# Model hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAMS: dict = {
    "objective":             "reg:pseudohubererror",
    "huber_slope":           1.0,
    "eta":                   0.015,
    "max_depth":             5,
    "min_child_weight":      10,
    "subsample":             0.75,
    "colsample_bytree":      0.7,
    "colsample_bylevel":     0.8,
    "gamma":                 0.05,
    "reg_lambda":            1.5,
    "reg_alpha":             0.05,
    "nthread":               -1,
    "eval_metric":           "rmse",
    "tree_method":           "hist",
    "seed":                  42,
}

NUM_BOOST_ROUND    = 3000
EARLY_STOP_ROUNDS  = 50

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def fit_sanitizer(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    stats: dict = {}
    for col in feat_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "q01":  series.quantile(0.01),
            "q99":  series.quantile(0.99),
            "mean": series.mean(),
            "std":  series.std(),
        }
    return stats


def apply_sanitizer(df: pd.DataFrame, feat_cols: list[str], stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = df[col].clip(s["q01"], s["q99"])
            df[col] = (df[col] - s["mean"]) / (s["std"] + 1e-6)
    return df


def time_decay_weights(dates: pd.Series, half_life: int = 45) -> np.ndarray:
    """Exponential time-decay sample weights, normalized to mean=1."""
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.to_numpy().astype(float)
    w = np.exp(-np.log(2) / half_life * days_ago)
    return w / w.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    feat_cols:     list[str],
    sample_weight: np.ndarray | None = None,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(
        train_df[feat_cols],
        label=train_df[TARGET_COLUMN],
        weight=sample_weight,
    )
    dval = xgb.DMatrix(val_df[feat_cols], label=val_df[TARGET_COLUMN])

    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dval, "inner_val")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        verbose_eval=False,
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(scores: pd.Series, top_k: int = 40) -> pd.DataFrame:
    """RankExp weighting: w_i ∝ exp(rank_i / top_k * scale), top_k stocks."""
    top_stocks = scores.nlargest(top_k)
    ranks = top_stocks.rank(method="average")
    scale = 3.0
    raw_w = np.exp(ranks / top_k * scale)
    w = raw_w / raw_w.sum()
    w = w.clip(upper=MAX_WEIGHT)
    w = w / w.sum()
    return pd.DataFrame({"stock_code": w.index, "weight": w.values})

# ─────────────────────────────────────────────────────────────────────────────
# Purged Walk-Forward CV
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    train_pool:     pd.DataFrame,
    feat_cols:      list[str],
    n_folds:        int  = 5,
    half_life:      int  = 45,
    use_decay:      bool = True,
    min_train_days: int  = 120,
) -> tuple[float, float]:
    decay_desc = f"ON (half_life={half_life}d)" if use_decay else "OFF"
    print(f"\n>> Running {n_folds}-Fold Purged Walk-Forward CV "
          f"[embargo={EMBARGO_DAYS}d, time_decay={decay_desc}]...")

    all_dates  = np.sort(train_pool["date"].unique())
    total_days = len(all_dates)
    fold_ics: list[float] = []

    folds: list[tuple] = []
    for k in range(n_folds):
        val_end_idx   = total_days - 1 - k * TEST_VAL_DAYS
        val_start_idx = val_end_idx - TEST_VAL_DAYS + 1
        if val_start_idx - EMBARGO_DAYS < min_train_days:
            break
        folds.append((
            pd.Timestamp(all_dates[val_start_idx]),
            pd.Timestamp(all_dates[val_end_idx]),
        ))
    folds = list(reversed(folds))

    for k, (val_start, val_end) in enumerate(folds):
        val_start_idx = int(np.searchsorted(all_dates, np.datetime64(val_start)))
        train_end_idx = val_start_idx - EMBARGO_DAYS - 1

        if train_end_idx < min_train_days:
            print(f"  Fold {k+1}: skipped (insufficient training data)")
            continue

        inner_start_idx = max(0, train_end_idx - INNER_VAL_DAYS)
        inner_start     = pd.Timestamp(all_dates[inner_start_idx])
        train_end       = pd.Timestamp(all_dates[train_end_idx])

        train_df  = train_pool[train_pool["date"] <  inner_start].copy()
        inner_val = train_pool[
            (train_pool["date"] >= inner_start) &
            (train_pool["date"] <= train_end)
        ].copy()
        val_df    = train_pool[
            (train_pool["date"] >= val_start) &
            (train_pool["date"] <= val_end)
        ].copy()

        if len(train_df) < 500 or inner_val.empty or val_df.empty:
            print(f"  Fold {k+1}: skipped (too few rows)")
            continue

        stats     = fit_sanitizer(train_df, feat_cols)
        train_df  = apply_sanitizer(train_df, feat_cols, stats)
        inner_val = apply_sanitizer(inner_val, feat_cols, stats)
        val_df    = apply_sanitizer(val_df,    feat_cols, stats)

        sw    = time_decay_weights(train_df["date"], half_life) if use_decay else None
        model = train_model(train_df, inner_val, feat_cols, sample_weight=sw)

        preds = model.predict(xgb.DMatrix(val_df[feat_cols]))
        ic    = rank_ic(val_df[TARGET_COLUMN].to_numpy(), preds, val_df["date"].to_numpy())
        fold_ics.append(ic)

        print(f"  Fold {k+1}/{len(folds)}"
              f"  train < {inner_start.date()}"
              f"  val [{val_start.date()}, {val_end.date()}]"
              f"  IC = {ic:.4f}")

    mean_ic = float(np.nanmean(fold_ics)) if fold_ics else float("nan")
    ic_ir   = float(np.mean(fold_ics) / (np.std(fold_ics) + 1e-9)) if len(fold_ics) >= 2 else float("nan")
    print(f"\n[CV RESULT] Mean IC: {mean_ic:.4f}  |  IC IR: {ic_ir:.4f}\n")
    return mean_ic, ic_ir

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       type=str,  default="submission_xgb.csv")
    parser.add_argument("--top-k",     type=int,  default=50)
    parser.add_argument("--cv",        action="store_true")
    parser.add_argument("--cv-folds",  type=int,  default=5)
    parser.add_argument("--half-life", type=int,  default=45,
                        help="Time-decay half-life in trading days (default 45 ≈ 2 months)")
    parser.add_argument("--no-decay",  action="store_true")
    parser.add_argument("--as-of",     type=str,  default=None)
    args = parser.parse_args()

    use_decay = not args.no_decay

    print(">> Loading data...")
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")

    print(">> Building features (T+5 target)...")
    panel = build_features(prices, index_df=index_df)

    feat_cols  = _available_feat_cols(panel)
    if args.as_of is not None:
        as_of_date = pd.Timestamp(args.as_of)
        train_pool = training_frame(panel, max_date=as_of_date)
    else:
        as_of_date = pd.Timestamp(panel["date"].max())
        train_pool = training_frame(panel)

    print(f">> Feature columns ({len(feat_cols)}): {feat_cols}")
    print(f">> as_of_date: {as_of_date.date()}")
    print(f">> Training rows: {len(train_pool):,}")
    print(f">> Time-decay: {'ON half_life=' + str(args.half_life) + 'd' if use_decay else 'OFF'}")

    if args.cv:
        walk_forward_cv(
            train_pool, feat_cols,
            n_folds=args.cv_folds,
            half_life=args.half_life,
            use_decay=use_decay,
        )

    # ── Final submission ──────────────────────────────────────────────────────
    print(">> Training final model...")

    all_dates = np.sort(train_pool["date"].unique())

    inner_val_end_idx   = -(EMBARGO_DAYS + 1)
    inner_val_start_idx = inner_val_end_idx - INNER_VAL_DAYS + 1
    inner_val_start = pd.Timestamp(all_dates[inner_val_start_idx])
    inner_val_end   = pd.Timestamp(all_dates[inner_val_end_idx])

    final_train = train_pool[train_pool["date"] <  inner_val_start].copy()
    final_val   = train_pool[
        (train_pool["date"] >= inner_val_start) &
        (train_pool["date"] <= inner_val_end)
    ].copy()

    stats       = fit_sanitizer(final_train, feat_cols)
    final_train = apply_sanitizer(final_train, feat_cols, stats)
    final_val   = apply_sanitizer(final_val,   feat_cols, stats)

    sw    = time_decay_weights(final_train["date"], args.half_life) if use_decay else None
    model = train_model(final_train, final_val, feat_cols, sample_weight=sw)

    pred_df    = prediction_frame(panel, as_of=as_of_date)
    pred_df    = apply_sanitizer(pred_df, feat_cols, stats)
    raw_scores = model.predict(xgb.DMatrix(pred_df[feat_cols]))
    scores     = pd.Series(raw_scores, index=pred_df["stock_code"])
    scores     = (scores - scores.mean()) / (scores.std() + 1e-6)

    weights_df = build_portfolio(scores, top_k=args.top_k)
    weights_df.to_csv(args.out, index=False)
    print(f">> Submission saved → {args.out}  ({len(weights_df)} stocks)")


if __name__ == "__main__":
    main()
