"""LightGBM model for CSI500 T+5 prediction.

Architecture
------------
  Single LightGBM model (LGB-only) — empirically outperforms 3-model
  ensemble on this dataset; XGB/CatBoost dominated by LGB in IC-weighting.

  Time decay: half_life=45d — emphasises recent 2 months more aggressively,
  better aligned with the current bull-market regime.

  Portfolio: RankExp top_k=40 — higher conviction concentration, better
  risk-adjusted excess return vs top_k=50.

  CV: Purged Walk-Forward with embargo = FORWARD_HORIZON + 2 = 7 days

  激活环境：cd /Users/huangyuzhou/Desktop/ml-competition-sp26
  source venv/bin/activate

Usage
-----
  python baseline_ensemble.py --out submission.csv
  python baseline_ensemble.py --cv --out submission.csv
  python baseline_ensemble.py --cv --cv-folds 5 --half-life 45 --out submission.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import xgboost as xgb
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

# Embargo = horizon + 2 buffer days (prevents target leakage across folds)
EMBARGO_DAYS   = FORWARD_HORIZON + 2   # 7 days for T+5
TEST_VAL_DAYS  = 15                    # each CV fold test window
INNER_VAL_DAYS = 40                    # early-stopping validation window
MIN_STOCKS     = 30
MAX_WEIGHT     = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# Model hyper-parameters
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


def icir(ic_list: list[float]) -> float:
    arr = np.array([x for x in ic_list if not np.isnan(x)])
    if len(arr) < 2:
        return float("nan")
    return float(arr.mean() / (arr.std() + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LGBModel:
    lgb_model: lgb.LGBMRegressor
    feat_cols: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feat_cols if c in X.columns]
        p = self.lgb_model.predict(X[cols])
        r = pd.Series(p).rank(pct=True).to_numpy()
        return (r - 0.5) * 2   # rank-normalize to [-1, 1]


def train_stacking(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    feat_cols: list[str],
    half_life: int  = 45,
    use_decay: bool = True,
) -> LGBModel:
    cols    = [c for c in feat_cols if c in train_df.columns]
    weights = time_decay_weights(train_df["date"], half_life) if use_decay else None

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

    ic_val = rank_ic(
        val_df[TARGET_COLUMN].to_numpy(),
        m.predict(val_df[cols]),
        val_df["date"].to_numpy(),
    )
    print(f"    LGB inner_val IC={ic_val:.4f}  trees={m.best_iteration_}")
    return LGBModel(lgb_model=m, feat_cols=cols)


# ─────────────────────────────────────────────────────────────────────────────
# Purged Walk-Forward CV
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFCVResult:
    fold_ics:   list[float] = field(default_factory=list)
    fold_dates: list[str]   = field(default_factory=list)
    mean_ic:    float       = float("nan")
    ic_ir:      float       = float("nan")
    n_folds:    int         = 0

    def summary(self) -> str:
        lines = [
            "",
            "=" * 65,
            "  Purged Walk-Forward CV — Excess Return IC (5-Fold Breakdown)",
            "=" * 65,
            f"  {'Fold':<6} {'Date Range':<32} {'IC':>8}",
            "  " + "-" * 50,
        ]
        for i, (ic, dr) in enumerate(zip(self.fold_ics, self.fold_dates)):
            lines.append(f"  {i+1:<6} {dr:<32} {ic:>8.4f}")
        lines += [
            "  " + "-" * 50,
            f"  {'Mean IC':<38} {self.mean_ic:>8.4f}",
            f"  {'IC IR  (Mean/Std)':<38} {self.ic_ir:>8.4f}",
            "=" * 65,
        ]
        return "\n".join(lines)


def walk_forward_cv(
    train_pool:     pd.DataFrame,
    n_folds:        int  = 5,
    val_days:       int  = TEST_VAL_DAYS,
    embargo_days:   int  = EMBARGO_DAYS,
    min_train_days: int  = 120,
    half_life:      int  = 45,
    use_decay:      bool = True,
    verbose:        bool = True,
) -> WFCVResult:
    feat_cols  = _available_feat_cols(train_pool)
    all_dates  = np.sort(train_pool["date"].unique())
    total_days = len(all_dates)

    if verbose:
        decay_desc = f"ON (half_life={half_life}d)" if use_decay else "OFF"
        print(f"\n>> Running {n_folds}-Fold Purged Walk-Forward CV "
              f"[embargo={embargo_days}d, time_decay={decay_desc}]...")

    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(n_folds):
        val_end_idx   = total_days - 1 - k * val_days
        val_start_idx = val_end_idx - val_days + 1
        if val_start_idx - embargo_days < min_train_days:
            break
        folds.append((
            pd.Timestamp(all_dates[val_start_idx]),
            pd.Timestamp(all_dates[val_end_idx]),
        ))
    folds = list(reversed(folds))

    result = WFCVResult()

    for k, (val_start, val_end) in enumerate(folds):
        val_start_idx = int(np.searchsorted(all_dates, np.datetime64(val_start)))
        train_end_idx = val_start_idx - embargo_days - 1

        if train_end_idx < min_train_days:
            if verbose:
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
            if verbose:
                print(f"  Fold {k+1}: skipped (too few rows)")
            continue

        stats     = fit_sanitizer(train_df, feat_cols)
        train_df  = apply_sanitizer(train_df, feat_cols, stats)
        inner_val = apply_sanitizer(inner_val, feat_cols, stats)
        val_df    = apply_sanitizer(val_df, feat_cols, stats)

        model  = train_stacking(
            train_df, inner_val, feat_cols,
            half_life=half_life, use_decay=use_decay,
        )
        y_pred = model.predict(val_df)
        ic_val = rank_ic(
            val_df[TARGET_COLUMN].to_numpy(),
            y_pred,
            val_df["date"].to_numpy(),
        )

        result.fold_ics.append(ic_val)
        result.fold_dates.append(f"{val_start.date()} → {val_end.date()}")

        if verbose:
            print(
                f"  Fold {k+1}/{len(folds)}"
                f"  train < {inner_start.date()}"
                f"  val [{val_start.date()}, {val_end.date()}]"
                f"  IC = {ic_val:.4f}"
            )

    result.n_folds = len(result.fold_ics)
    result.mean_ic = float(np.nanmean(result.fold_ics)) if result.fold_ics else float("nan")
    result.ic_ir   = icir(result.fold_ics)

    if verbose:
        print(result.summary())

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(scores: pd.Series, top_k: int = 40) -> pd.DataFrame:
    """RankExp weighting: w_i ∝ exp(rank_i / top_k * scale), top_k stocks."""
    top_stocks = scores.nlargest(top_k)
    # Rank within top_k (1=worst, top_k=best)
    ranks = top_stocks.rank(method="average")
    scale = 3.0  # concentration parameter
    raw_w = np.exp(ranks / top_k * scale)
    w = raw_w / raw_w.sum()
    w = w.clip(upper=MAX_WEIGHT)
    w = w / w.sum()
    return pd.DataFrame({"stock_code": w.index, "weight": w.values})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       type=str,  default="submission.csv")
    parser.add_argument("--top-k",     type=int,  default=50)
    parser.add_argument("--cv",        action="store_true")
    parser.add_argument("--cv-folds",  type=int,  default=5)
    parser.add_argument("--half-life", type=int,  default=45,
                        help="Time-decay half-life in trading days (default 45 ≈ 2 months)")
    parser.add_argument("--no-decay",  action="store_true")
    parser.add_argument("--as-of",     type=str,  default=None,
                        help="Prediction date YYYYMMDD (default: last date in prices). "
                             "Set to the trading day BEFORE your eval window opens.")
    args = parser.parse_args()

    use_decay = not args.no_decay

    print(">> Loading data...")
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")

    print(">> Building features (T+5 excess-return target)...")
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
            train_pool,
            n_folds=args.cv_folds,
            half_life=args.half_life,
            use_decay=use_decay,
        )

    # ── Final submission ──────────────────────────────────────────────────────
    print("\n>> Training final LGB model...")

    all_dates = np.sort(train_pool["date"].unique())
    total     = len(all_dates)

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

    final_model = train_stacking(
        final_train, final_val, feat_cols,
        half_life=args.half_life, use_decay=use_decay,
    )

    pred_df = prediction_frame(panel, as_of=as_of_date)
    if pred_df.empty:
        raise ValueError(f"No features found for {as_of_date.date()}")

    pred_df = apply_sanitizer(pred_df, feat_cols, stats)
    raw_scores = final_model.predict(pred_df)
    scores = pd.Series(raw_scores, index=pred_df["stock_code"])
    scores = (scores - scores.mean()) / (scores.std() + 1e-6)

    weights_df = build_portfolio(scores, top_k=args.top_k)
    weights_df.to_csv(args.out, index=False)
    print(f">> Submission saved → {args.out}  ({len(weights_df)} stocks)")
