"""LightGBM + XGBoost ensemble with Purged Walk-Forward Cross-Validation.

Changes vs. uploaded version
------------------------------
1. walk_forward_cv()   — 新增 fit_sanitizer / apply_sanitizer，每折独立标准化   ← FIX (严重)
2. __main__            — 最终 submission 也走 sanitizer，保持训练/推断一致       ← FIX (严重)
3. as_of_date          — 改为动态取 panel 最新日期，不再硬编码 "2026-04-15"      ← FIX (严重)
4. train_ensemble()    — half_life_days 参数化（原来硬编码 120）                  ← FIX (次要)
5. walk_forward_cv()   — 透传 half_life_days 到 train_ensemble                   ← FIX (次要)
6. CLI                 — 新增 --out / --top-k / --half-life / --no-decay 参数    ← NEW

No pretrained weights are loaded anywhere.
Both models are trained from scratch on the caller's data.

Usage
-----
  # Standard submission run
  python baseline_ensemble.py --out submissions/week1.csv

  # Run CV first
  python baseline_ensemble.py --cv --out submissions/week1.csv

  # Disable time-decay weights (ablation study)
  python baseline_ensemble.py --cv --no-decay --out submissions/week1.csv

  # Tune half-life
  python baseline_ensemble.py --cv --half-life 90 --out submissions/week1.csv

Public API
----------
  train_ensemble(train_df, val_df, feat_cols)  -> EnsembleModel
  walk_forward_cv(train_pool, ...)             -> WFCVResult
  rank_ic(y_true, y_pred, dates)               -> float
  icir(ic_series)                              -> float
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

LGB_PARAMS: dict = {
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 40,
    "n_estimators":      1000,
    "learning_rate":     0.02,
    "subsample":         0.7,
    "subsample_freq":    1,
    "colsample_bytree":  0.6,
    "reg_alpha":         0.1,
    "reg_lambda":        2.0,
    "min_split_gain":    0.05,
    "objective":         "huber",
    "alpha":             1.5,
    "verbose":           -1,
    "n_jobs":            -1,
    "random_state":      42,
}

XGB_PARAMS: dict = {
    "n_estimators":      1000,
    "max_depth":         4,
    "learning_rate":     0.03,
    "subsample":         0.7,
    "colsample_bytree":  0.6,
    "colsample_bylevel": 0.8,
    "min_child_weight":  15,
    "gamma":             0.1,
    "reg_lambda":        2.0,
    "reg_alpha":         0.1,
    "tree_method":       "hist",
    "n_jobs":            -1,
    "random_state":      42,
    # ↑ early_stopping_rounds 故意不写在这里：
    #   XGBRegressor 的 early_stopping_rounds 是构造器参数，
    #   会在 fit() 里对 eval_set 的每一棵树都评估一次，
    #   inner_val 只有 40 天数据、噪音极大，30 轮容忍太少，
    #   导致 XGB 在 iter=0~22 就秒停，实际上啥都没学到。
    #   改为在 fit() 里传入更大的容忍轮数（见 train_ensemble）。
    "objective":         "reg:pseudohubererror",
    "huber_slope":       1.5,
}

# CV constants（与 baseline_xgboost.py 对齐）
TEST_VAL_DAYS  = 15
INNER_VAL_DAYS = 40
EMBARGO_DAYS   = FORWARD_HORIZON + 2   # 7 天

MIN_STOCKS = 30
MAX_WEIGHT = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 防泄露预处理（从 baseline_xgboost.py 对齐过来）
# ─────────────────────────────────────────────────────────────────────────────

def fit_sanitizer(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """在训练集上计算 q01/q99/mean/std，供 apply_sanitizer 使用。"""
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
    """用训练集统计量做截断 + z-score 标准化，防止测试集极端值干扰。"""
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = df[col].clip(s["q01"], s["q99"])
            df[col] = (df[col] - s["mean"]) / (s["std"] + 1e-6)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 时间衰减样本权重
# ─────────────────────────────────────────────────────────────────────────────

def get_time_decay_weights(dates_series: pd.Series, half_life_days: int = 120) -> np.ndarray:
    """
    以训练集最新日期为基准，对每个样本按指数衰减分配权重。

    公式：w = exp(-ln2 / half_life * days_ago)，均值归一化为 1.0。

    Parameters
    ----------
    dates_series   : 训练集的 'date' 列（pd.Series[datetime]）
    half_life_days : 衰减半衰期（交易日数）。
                     120d ≈ 半年，推荐搜索范围 [60, 90, 120, 180]。
    """
    max_date = dates_series.max()
    days_ago = (max_date - dates_series).dt.days.to_numpy().astype(float)
    weights  = np.exp(-np.log(2) / half_life_days * days_ago)
    return weights / weights.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleModel:
    """Holds a fitted LGB + XGB pair and blends their predictions."""
    lgb_model:  lgb.LGBMRegressor
    xgb_model:  xgb.XGBRegressor
    feat_cols:  list[str]
    lgb_weight: float = 0.5
    xgb_weight: float = 0.5

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        cols  = [c for c in self.feat_cols if c in X.columns]
        p_lgb = self.lgb_model.predict(X[cols])
        p_xgb = self.xgb_model.predict(X[cols])
        return self.lgb_weight * p_lgb + self.xgb_weight * p_xgb

    @classmethod
    def with_ic_weights(
        cls,
        lgb_model:  lgb.LGBMRegressor,
        xgb_model:  xgb.XGBRegressor,
        feat_cols:  list[str],
        lgb_ic:     float,
        xgb_ic:     float,
    ) -> "EnsembleModel":
        """
        用两个模型各自在 inner_val 上的 IC 动态决定混合权重。

        IC <= 0 的模型权重清零，避免负贡献拉低组合；
        两个都 <= 0 时退化为等权（保守兜底）。
        """
        w_lgb = max(lgb_ic, 0.0)
        w_xgb = max(xgb_ic, 0.0)
        total = w_lgb + w_xgb
        if total < 1e-9:
            w_lgb = w_xgb = 0.5
        else:
            w_lgb /= total
            w_xgb /= total
        return cls(lgb_model=lgb_model, xgb_model=xgb_model,
                   feat_cols=feat_cols, lgb_weight=w_lgb, xgb_weight=w_xgb)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(
    train_df:      pd.DataFrame,
    val_df:        pd.DataFrame,
    feat_cols:     list[str],
    lgb_weight:    float = 0.5,
    half_life_days: int  = 120,
    use_decay:     bool  = True,
    dynamic_weight: bool = True,   # ← 新增：是否用 IC 动态决定混合权重
) -> EnsembleModel:
    """
    Fit one LightGBM and one XGBoost from scratch; blend predictions.

    Parameters
    ----------
    train_df       : training rows (feat_cols + TARGET_COLUMN, NaN-free, sanitized)
    val_df         : held-out rows for early stopping (sanitized)
    feat_cols      : feature column names
    lgb_weight     : static blend weight for LGB (only used when dynamic_weight=False)
    half_life_days : time-decay half-life in trading days
    use_decay      : if False, uniform sample weights
    dynamic_weight : if True, blend weights are proportional to each model's inner_val IC
    """
    cols = [c for c in feat_cols if c in train_df.columns]

    train_weights = (
        get_time_decay_weights(train_df["date"], half_life_days)
        if use_decay else None
    )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        train_df[cols],
        train_df[TARGET_COLUMN],
        sample_weight=train_weights,
        eval_set=[(val_df[cols], val_df[TARGET_COLUMN])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    # FIX：early_stopping_rounds 从 XGB_PARAMS 里移出，在 fit() 里传入更大的值。
    # 原来 30 轮对于 40 天的 inner_val 太激进（信噪比低，模型还没收敛就被停掉）。
    # 改为 80 轮：给 XGBoost 足够时间翻越早期震荡找到真正的改善方向。
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(
        train_df[cols],
        train_df[TARGET_COLUMN],
        sample_weight=train_weights,
        eval_set=[(val_df[cols], val_df[TARGET_COLUMN])],
        early_stopping_rounds=80,   # ← 从 30 → 80
        verbose=False,
    )

    # ── 动态混合权重（按 inner_val IC 加权）────────────────────────────────────
    if dynamic_weight:
        p_lgb = lgb_model.predict(val_df[cols])
        p_xgb = xgb_model.predict(val_df[cols])
        ic_lgb = rank_ic(val_df[TARGET_COLUMN].to_numpy(), p_lgb, val_df["date"].to_numpy())
        ic_xgb = rank_ic(val_df[TARGET_COLUMN].to_numpy(), p_xgb, val_df["date"].to_numpy())
        return EnsembleModel.with_ic_weights(lgb_model, xgb_model, cols, ic_lgb, ic_xgb)

    return EnsembleModel(
        lgb_model  = lgb_model,
        xgb_model  = xgb_model,
        feat_cols  = cols,
        lgb_weight = lgb_weight,
        xgb_weight = 1.0 - lgb_weight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def rank_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates:  np.ndarray,
) -> float:
    """Mean daily cross-sectional Spearman rank IC."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def icir(ic_series: list[float]) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    arr = np.array([x for x in ic_series if not np.isnan(x)])
    if len(arr) < 2:
        return float("nan")
    return float(arr.mean() / (arr.std() + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Purged Walk-Forward Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFCVResult:
    fold_ics:   list[float] = field(default_factory=list)
    fold_dates: list[str]   = field(default_factory=list)
    mean_ic:    float       = float("nan")
    ic_ir:      float       = float("nan")
    n_folds:    int         = 0

    def summary(self) -> str:
        per_fold = "  ".join(f"{x:.4f}" for x in self.fold_ics)
        return (
            "\n" + "=" * 60 + "\n"
            "  Purged Walk-Forward CV — Results\n"
            + "=" * 60 + "\n"
            f"  Folds      : {self.n_folds}\n"
            f"  Mean IC    : {self.mean_ic:.4f}\n"
            f"  IC IR      : {self.ic_ir:.4f}\n"
            f"  IC / fold  : {per_fold}\n"
            + "=" * 60
        )


def walk_forward_cv(
    train_pool:     pd.DataFrame,
    n_folds:        int   = 5,
    val_days:       int   = TEST_VAL_DAYS,
    embargo_days:   int   = EMBARGO_DAYS,
    min_train_days: int   = 120,
    lgb_weight:     float = 0.5,
    half_life_days: int   = 120,
    use_decay:      bool  = True,
    dynamic_weight: bool  = True,  # ← 新增
    verbose:        bool  = True,
) -> WFCVResult:
    """
    Purged Walk-Forward CV for the self-test section (25% grade).

    FIX（严重）：每折现在都会在训练集上 fit sanitizer，
    并将其应用到 inner_val 和 val 集，防止极端值跨折泄露。
    """
    feat_cols  = _available_feat_cols(train_pool)
    all_dates  = np.sort(train_pool["date"].unique())
    total_days = len(all_dates)

    if verbose:
        decay_desc = f"ON (half_life={half_life_days}d)" if use_decay else "OFF"
        print(f"\n>> Running {n_folds}-Fold Purged Walk-Forward CV "
              f"[time_decay={decay_desc}]...")

    # Build fold pairs
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

        train_end = pd.Timestamp(all_dates[train_end_idx])

        inner_start_idx = max(0, train_end_idx - INNER_VAL_DAYS)
        inner_start     = pd.Timestamp(all_dates[inner_start_idx])

        train_df   = train_pool[train_pool["date"] <  inner_start].copy()
        inner_val  = train_pool[
            (train_pool["date"] >= inner_start) &
            (train_pool["date"] <= train_end)
        ].copy()
        val_df     = train_pool[
            (train_pool["date"] >= val_start) &
            (train_pool["date"] <= val_end)
        ].copy()

        if len(train_df) < 500 or inner_val.empty or val_df.empty:
            if verbose:
                print(f"  Fold {k+1}: skipped (too few rows)")
            continue

        # ── FIX（严重）：每折独立 fit sanitizer，仅用本折训练集 ──────────────
        stats     = fit_sanitizer(train_df, feat_cols)
        train_df  = apply_sanitizer(train_df, feat_cols, stats)
        inner_val = apply_sanitizer(inner_val, feat_cols, stats)
        val_df    = apply_sanitizer(val_df, feat_cols, stats)
        # ─────────────────────────────────────────────────────────────────────

        model  = train_ensemble(
            train_df, inner_val, feat_cols,
            lgb_weight=lgb_weight,
            half_life_days=half_life_days,
            use_decay=use_decay,
            dynamic_weight=dynamic_weight,  # ← 透传
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
                f"  LGB={model.lgb_weight:.2f}({model.lgb_model.best_iteration_}iter)"
                f"  XGB={model.xgb_weight:.2f}({model.xgb_model.best_iteration}iter)"
            )

    result.n_folds = len(result.fold_ics)
    result.mean_ic = float(np.nanmean(result.fold_ics)) if result.fold_ics else float("nan")
    result.ic_ir   = icir(result.fold_ics)

    if verbose:
        print(result.summary())

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 投资组合构建
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(scores: pd.Series, top_k: int = 30) -> pd.DataFrame:
    """信号加权（Signal-weighted）。"""
    top_stocks = scores.nlargest(top_k)
    raw_w = top_stocks - top_stocks.min() + 1e-5
    w     = raw_w / raw_w.sum()
    w     = w.clip(upper=MAX_WEIGHT)
    w     = w / w.sum()
    return pd.DataFrame({"stock_code": w.index, "weight": w.values})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",       type=str,  default="submission_ensemble.csv")
    parser.add_argument("--top-k",     type=int,  default=30)
    parser.add_argument("--cv",        action="store_true", help="先执行 Walk-Forward CV")
    parser.add_argument("--cv-folds",  type=int,  default=5)
    parser.add_argument(
        "--half-life", type=int, default=120,
        help="时间衰减半衰期（交易日，默认 120≈半年）。建议测试 [60,90,120,180]。"
    )
    parser.add_argument(
        "--no-decay", action="store_true",
        help="禁用时间衰减权重（等权对照实验）"
    )
    args = parser.parse_args()

    use_decay  = not args.no_decay
    data_dir   = Path(__file__).parent / "data"

    print(">> Loading data...")
    raw_df = pd.read_parquet(data_dir / "prices.parquet")

    print(">> Building features...")
    panel = build_features(raw_df)

    feat_cols  = _available_feat_cols(panel)
    train_pool = training_frame(panel)

    print(f">> Feature columns ({len(feat_cols)}): {feat_cols}")
    if use_decay:
        print(f">> Time-decay weights: ON  (half_life={args.half_life} days)")
    else:
        print(">> Time-decay weights: OFF (uniform)")

    if args.cv:
        walk_forward_cv(
            train_pool,
            n_folds=args.cv_folds,
            half_life_days=args.half_life,
            use_decay=use_decay,
            dynamic_weight=True,
        )

    # ── 生成最终 submission ───────────────────────────────────────────────────
    print("\n>> Generating final submission (Ensemble)...")

    all_dates = np.sort(train_pool["date"].unique())
    total     = len(all_dates)

    # FIX（严重）：as_of_date 改为动态取数据最新日期，不再硬编码 ──────────────
    as_of_date = pd.Timestamp(panel["date"].max())
    print(f">> as_of_date: {as_of_date.date()}  (auto-detected from data)")
    # ─────────────────────────────────────────────────────────────────────────

    # 用最近 TEST_VAL_DAYS 之前的数据训练（与 CV fold 逻辑对齐）
    inner_val_end_idx   = -(EMBARGO_DAYS + 1)
    inner_val_start_idx = inner_val_end_idx - INNER_VAL_DAYS + 1
    train_end_idx       = inner_val_start_idx - EMBARGO_DAYS - 1

    inner_val_start = pd.Timestamp(all_dates[inner_val_start_idx])
    inner_val_end   = pd.Timestamp(all_dates[inner_val_end_idx])
    train_end       = pd.Timestamp(all_dates[train_end_idx])

    final_train = train_pool[train_pool["date"] <  inner_val_start].copy()
    final_val   = train_pool[
        (train_pool["date"] >= inner_val_start) &
        (train_pool["date"] <= inner_val_end)
    ].copy()

    # FIX（严重）：最终提交也走 sanitizer ─────────────────────────────────────
    stats       = fit_sanitizer(final_train, feat_cols)
    final_train = apply_sanitizer(final_train, feat_cols, stats)
    final_val   = apply_sanitizer(final_val, feat_cols, stats)
    # ─────────────────────────────────────────────────────────────────────────

    final_model = train_ensemble(
        final_train, final_val, feat_cols,
        half_life_days=args.half_life,
        use_decay=use_decay,
        dynamic_weight=True,
    )

    pred_df = prediction_frame(panel, as_of=as_of_date)
    if pred_df.empty:
        raise ValueError(
            f"找不到 {as_of_date.date()} 的特征，"
            "请检查价格数据是否包含这一天。"
        )

    # FIX（严重）：推断时也用相同的 sanitizer ─────────────────────────────────
    pred_df = apply_sanitizer(pred_df, feat_cols, stats)
    # ─────────────────────────────────────────────────────────────────────────

    pred_df["score"] = final_model.predict(pred_df)
    scores           = pd.Series(
        pred_df["score"].values,
        index=pred_df["stock_code"]
    )
    scores = (scores - scores.mean()) / (scores.std() + 1e-6)

    weights_df = build_portfolio(scores, top_k=args.top_k)
    weights_df.to_csv(args.out, index=False)
    print(f">> Submission saved → {args.out}  ({len(weights_df)} stocks)")