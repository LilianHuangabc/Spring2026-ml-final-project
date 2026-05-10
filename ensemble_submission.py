"""LGB + XGB ensemble for CSI500 T+5 stock selection.

Blend: 0.5 * rank_norm(LGB) + 0.5 * rank_norm(XGB)
CV:    Purged walk-forward with embargo = FORWARD_HORIZON + 2 days.

Usage
-----
  python ensemble_submission.py --out submission_ensemble.csv
  python ensemble_submission.py --cv --out submission_ensemble.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
    _available_feat_cols,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

EMBARGO_DAYS   = FORWARD_HORIZON + 2
TEST_VAL_DAYS  = 15
INNER_VAL_DAYS = 40
MAX_WEIGHT     = 0.10

LGB_WEIGHT = 0.5
XGB_WEIGHT = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

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
    "colsample_bytree":  0.5,
    "colsample_bylevel": 0.8,
    "gamma":             0.05,
    "reg_lambda":        1.5,
    "reg_alpha":         0.05,
    "nthread":           -1,
    "eval_metric":       "rmse",
    "tree_method":       "hist",
    "seed":              137,
}
XGB_ROUNDS = 3000
XGB_EARLY  = 50

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def fit_sanitizer(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    stats: dict = {}
    for col in feat_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        stats[col] = {"q01": s.quantile(0.01), "q99": s.quantile(0.99),
                      "mean": s.mean(), "std": s.std()}
    return stats


def apply_sanitizer(df: pd.DataFrame, feat_cols: list[str], stats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = df[col].clip(s["q01"], s["q99"])
            df[col] = (df[col] - s["mean"]) / (s["std"] + 1e-6)
    return df


def time_decay_weights(dates: pd.Series, half_life: int = 45) -> np.ndarray:
    days_ago = (dates.max() - dates).dt.days.to_numpy().astype(float)
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


def rank_norm(arr: np.ndarray) -> np.ndarray:
    r = pd.Series(arr).rank(pct=True).to_numpy()
    return (r - 0.5) * 2

# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_lgb(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cols:     list[str],
    weights:  np.ndarray,
) -> lgb.LGBMRegressor:
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


def train_xgb(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cols:     list[str],
    weights:  np.ndarray,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(train_df[cols], label=train_df[TARGET_COLUMN], weight=weights)
    dval   = xgb.DMatrix(val_df[cols],   label=val_df[TARGET_COLUMN])
    return xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=XGB_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=XGB_EARLY,
        verbose_eval=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_cv(
    train_pool: pd.DataFrame,
    feat_cols:  list[str],
    n_folds:    int = 5,
    half_life:  int = 45,
    min_train_days: int = 120,
) -> None:
    all_dates  = np.sort(train_pool["date"].unique())
    total_days = len(all_dates)

    folds: list[tuple] = []
    for k in range(n_folds):
        ve = total_days - 1 - k * TEST_VAL_DAYS
        vs = ve - TEST_VAL_DAYS + 1
        if vs - EMBARGO_DAYS < min_train_days:
            break
        folds.append((pd.Timestamp(all_dates[vs]), pd.Timestamp(all_dates[ve])))
    folds = list(reversed(folds))

    lgb_ics, xgb_ics, blend_ics = [], [], []
    lgb_xgb_corrs: list[float] = []

    print(f"\n{'Fold':<5} {'Val window':<28} {'LGB IC':>8} {'XGB IC':>8} {'Blend IC':>9} {'LGB-XGB r':>10}")
    print("-" * 78)

    for k, (val_start, val_end) in enumerate(folds):
        vsi = int(np.searchsorted(all_dates, np.datetime64(val_start)))
        tei = vsi - EMBARGO_DAYS - 1
        if tei < min_train_days:
            continue

        isi    = max(0, tei - INNER_VAL_DAYS)
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

        cols = [c for c in feat_cols if c in train_df.columns]
        sw   = time_decay_weights(train_df["date"], half_life)

        lgb_m  = train_lgb(train_df, inner_val, cols, sw)
        lgb_p  = rank_norm(lgb_m.predict(val_df[cols]))
        lgb_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), lgb_p, val_df["date"].to_numpy())

        xgb_m  = train_xgb(train_df, inner_val, cols, sw)
        xgb_p  = rank_norm(xgb_m.predict(xgb.DMatrix(val_df[cols])))
        xgb_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), xgb_p, val_df["date"].to_numpy())

        blend_p  = LGB_WEIGHT * lgb_p + XGB_WEIGHT * xgb_p
        blend_ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), blend_p, val_df["date"].to_numpy())

        corr = float(np.corrcoef(lgb_p, xgb_p)[0, 1])

        lgb_ics.append(lgb_ic)
        xgb_ics.append(xgb_ic)
        blend_ics.append(blend_ic)
        lgb_xgb_corrs.append(corr)

        print(f"{k+1:<5} {str(val_start.date())+' -> '+str(val_end.date()):<28}"
              f" {lgb_ic:>8.4f} {xgb_ic:>8.4f} {blend_ic:>9.4f} {corr:>10.4f}")

    print("-" * 78)
    for label, ics in [("LGB-only", lgb_ics), ("XGB-only", xgb_ics), ("Blend 50/50", blend_ics)]:
        m  = float(np.nanmean(ics)) if ics else float("nan")
        ir = float(np.mean(ics) / (np.std(ics) + 1e-9)) if len(ics) >= 2 else float("nan")
        print(f"  {label:<18} Mean IC={m:+.4f}  IC-IR={ir:.4f}")
    mean_corr = float(np.mean(lgb_xgb_corrs)) if lgb_xgb_corrs else float("nan")
    print(f"  {'LGB-XGB corr':<18} Mean={mean_corr:.4f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(scores: pd.Series, top_k: int = 40) -> pd.DataFrame:
    top_stocks = scores.nlargest(top_k)
    ranks  = top_stocks.rank(method="average")
    raw_w  = np.exp(ranks / top_k * 3.0)
    w      = raw_w / raw_w.sum()
    w      = w.clip(upper=MAX_WEIGHT)
    w      = w / w.sum()
    return pd.DataFrame({"stock_code": w.index, "weight": w.values})

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       type=str,  default="submission_ensemble.csv")
    parser.add_argument("--top-k",     type=int,  default=40)
    parser.add_argument("--cv",        action="store_true")
    parser.add_argument("--cv-folds",  type=int,  default=5)
    parser.add_argument("--half-life", type=int,  default=45)
    parser.add_argument("--as-of",     type=str,  default=None)
    args = parser.parse_args()

    print(">> Loading data...")
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")

    print(">> Building features (T+5 excess-return target)...")
    panel = build_features(prices, index_df=index_df)

    feat_cols = _available_feat_cols(panel)
    if args.as_of is not None:
        as_of_date = pd.Timestamp(args.as_of)
        train_pool = training_frame(panel, max_date=as_of_date)
    else:
        as_of_date = pd.Timestamp(panel["date"].max())
        train_pool = training_frame(panel)

    print(f">> Feature columns ({len(feat_cols)})")
    print(f">> as_of_date: {as_of_date.date()}")
    print(f">> Training rows: {len(train_pool):,}")
    print(f">> Blend weights: LGB={LGB_WEIGHT}  XGB={XGB_WEIGHT}")

    if args.cv:
        walk_forward_cv(train_pool, feat_cols, n_folds=args.cv_folds, half_life=args.half_life)

    # ── Train final models ────────────────────────────────────────────────────
    print("\n>> Training final LGB + XGB models...")

    all_dates = np.sort(train_pool["date"].unique())

    iv_end_idx   = -(EMBARGO_DAYS + 1)
    iv_start_idx = iv_end_idx - INNER_VAL_DAYS + 1
    iv_start = pd.Timestamp(all_dates[iv_start_idx])
    iv_end   = pd.Timestamp(all_dates[iv_end_idx])

    final_train = train_pool[train_pool["date"] <  iv_start].copy()
    final_val   = train_pool[
        (train_pool["date"] >= iv_start) &
        (train_pool["date"] <= iv_end)
    ].copy()

    stats       = fit_sanitizer(final_train, feat_cols)
    final_train = apply_sanitizer(final_train, feat_cols, stats)
    final_val   = apply_sanitizer(final_val,   feat_cols, stats)

    cols = [c for c in feat_cols if c in final_train.columns]
    sw   = time_decay_weights(final_train["date"], args.half_life)

    lgb_model = train_lgb(final_train, final_val, cols, sw)
    xgb_model = train_xgb(final_train, final_val, cols, sw)
    print(f"  LGB best_iteration={lgb_model.best_iteration_}  XGB best_iteration={xgb_model.best_iteration}")

    # ── Predict & blend ───────────────────────────────────────────────────────
    pred_df = prediction_frame(panel, as_of=as_of_date)
    if pred_df.empty:
        raise ValueError(f"No features found for {as_of_date.date()}")

    pred_df   = apply_sanitizer(pred_df, feat_cols, stats)
    pred_cols = [c for c in cols if c in pred_df.columns]

    lgb_p = rank_norm(lgb_model.predict(pred_df[pred_cols]))
    xgb_p = rank_norm(xgb_model.predict(xgb.DMatrix(pred_df[pred_cols])))

    blended = LGB_WEIGHT * lgb_p + XGB_WEIGHT * xgb_p
    scores  = pd.Series(blended, index=pred_df["stock_code"])
    scores  = (scores - scores.mean()) / (scores.std() + 1e-6)

    weights_df = build_portfolio(scores, top_k=args.top_k)
    weights_df.to_csv(args.out, index=False)
    print(f">> Submission saved -> {args.out}  ({len(weights_df)} stocks)")


if __name__ == "__main__":
    main()
