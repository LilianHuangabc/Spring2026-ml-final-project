"""Feature engineering for the CSI500 stock-selection competition — T+5 edition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from functools import partial

# ---------------------------------------------------------------------------
# Target & horizon
# ---------------------------------------------------------------------------
TARGET_COLUMN   = "target_excess_5d"   # stock fwd return minus index fwd return
FORWARD_HORIZON = 5

# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # ── Core reversal (strong) ───────────────────────────────────────────
    "rev_10d",
    "rev_20d",
    "rev_ma20",
    "rev_5d",
    "rev_ma60",

    # ── Short-term reversal ──────────────────────────────────────────────
    "rev_1d",
    "rev_3d",

    # ── Price-volume divergence (strongest signals) ──────────────────────
    "rev_pvd_10d",
    "rev_pvd_5d",
    "rev_pvt",

    # ── Order flow / microstructure ──────────────────────────────────────
    "rev_ofi_ma5",
    "rev_ofi_ma3",
    "rev_ibp_5d",
    "vol_lead_ret_1d",
    "vol_lead_ret_2d",
    "upper_shadow_ratio",

    # ── Volume reversal / regime ─────────────────────────────────────────
    "rev_vol_z_20d",
    "rev_vs_3d",
    "rev_ta_3d",
    "vol_regime",

    # ── RSI reversal ─────────────────────────────────────────────────────
    "rev_rsi",
]

_RANK_TARGETS = [
    "rev_10d", "rev_20d",
    "rev_ma20", "rev_5d", "rev_ma60",
    "rev_1d", "rev_3d",
    "rev_pvd_10d", "rev_pvd_5d", "rev_pvt",
    "rev_ofi_ma5", "rev_ofi_ma3", "rev_ibp_5d",
    "vol_lead_ret_1d", "vol_lead_ret_2d",
    "upper_shadow_ratio",
    "rev_vol_z_20d", "rev_vs_3d", "rev_ta_3d",
    "vol_regime", "rev_rsi",
]


# ---------------------------------------------------------------------------
# Pandas-version-safe groupby helper
# ---------------------------------------------------------------------------
def _safe_groupby_apply(df: pd.DataFrame, group_col: str, func) -> pd.DataFrame:
    parts = [func(grp) for _, grp in df.groupby(group_col, sort=False)]
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-stock time-series features
# ---------------------------------------------------------------------------
def _per_stock_features(df: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return df

    stock = df["stock_code"].iloc[0]

    # Reindex to full trading calendar — prevents lookahead from gaps
    df = df.set_index("date").reindex(all_dates)
    df["stock_code"] = stock

    # Suspension handling: freeze price, zero volume
    close = df["close"].ffill().astype(float)
    high  = df["high"].ffill().astype(float)
    low   = df["low"].ffill().astype(float)
    vol   = df["volume"].fillna(0).astype(float)

    if "turnover" in df.columns:
        turnover = df["turnover"].fillna(0).astype(float)
    else:
        turnover = pd.Series(0.0, index=df.index)

    # ── Returns ───────────────────────────────────────────────────────────
    ret_1d = close.pct_change(1, fill_method=None)
    df["ret_1d"]  = ret_1d
    df["ret_3d"]  = close.pct_change(3, fill_method=None)
    df["ret_5d"]  = close.pct_change(5, fill_method=None)
    df["ret_10d"] = close.pct_change(10, fill_method=None)
    df["ret_20d"] = close.pct_change(20, fill_method=None)
    df["rev_1d"]  = -ret_1d

    # ── Target: 5-day forward return (no lookahead — shifted BEFORE feature shift) ──
    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0

    # ── Volatility ────────────────────────────────────────────────────────
    df["vol_5d"]  = ret_1d.rolling(5).std()
    df["vol_10d"] = ret_1d.rolling(10).std()
    df["vol_20d"] = ret_1d.rolling(20).std()

    # ── Volume Z-scores ───────────────────────────────────────────────────
    vol_mean20 = vol.rolling(20).mean()
    vol_std20  = vol.rolling(20).std().replace(0, np.nan)
    vol_mean5  = vol.rolling(5).mean()
    vol_std5   = vol.rolling(5).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean20) / vol_std20
    df["volume_z_5d"]  = (vol - vol_mean5)  / vol_std5

    # ── Turnover ──────────────────────────────────────────────────────────
    df["turnover_ma_5d"]  = turnover.rolling(5).mean()
    df["turnover_ma_20d"] = turnover.rolling(20).mean()

    # ── MA deviation ──────────────────────────────────────────────────────
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    # ── RSI(14) ───────────────────────────────────────────────────────────
    delta = close.diff()
    up    = delta.clip(lower=0).rolling(14).mean()
    down  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + up / down)

    # ── Shadow ratios (microstructure) ────────────────────────────────────
    hl = (high - low).replace(0, np.nan)
    df["upper_shadow_ratio"] = (high - close.clip(upper=high)) / hl
    df["lower_shadow_ratio"] = (close.clip(lower=low) - low)   / hl

    # ── OFI: Order-Flow Imbalance ─────────────────────────────────────────
    vol_weight = vol / vol.rolling(20).mean().replace(0, np.nan)
    ofi = (df["lower_shadow_ratio"] - df["upper_shadow_ratio"]) * vol_weight
    df["ofi_ma3"]  = ofi.rolling(3).mean()
    df["ofi_ma5"]  = ofi.rolling(5).mean()
    ofi_ma3  = ofi.rolling(3).mean()
    ofi_ma10 = ofi.rolling(10).mean().replace(0, np.nan)
    df["ofi_accel"] = ofi_ma3 / ofi_ma10 - 1.0

    # ── Intraday buy pressure ─────────────────────────────────────────────
    buy_pressure = ((close - low) / hl).fillna(0.5)
    df["intraday_buy_pressure_3d"] = buy_pressure.rolling(3).mean()
    df["intraday_buy_pressure_5d"] = buy_pressure.rolling(5).mean()

    # ── Amplitude / turnover ──────────────────────────────────────────────
    amplitude = (high - low) / close.replace(0, np.nan)
    to_5d     = turnover.rolling(5).mean().replace(0, np.nan)
    df["amplitude_per_turnover"] = amplitude.rolling(5).mean() / to_5d

    # ── Volume-price lead-lag ─────────────────────────────────────────────
    vol_z = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-6)
    df["vol_lead_ret_1d"] = vol_z.shift(1).rolling(10).corr(ret_1d)
    df["vol_lead_ret_2d"] = vol_z.shift(2).rolling(10).corr(ret_1d)

    # ── Volume spikes ─────────────────────────────────────────────────────
    vol_ma3  = vol.rolling(3).mean()
    vol_ma5  = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean().replace(0, np.nan)
    df["volume_spike_3d"] = vol_ma3 / vol_ma20 - 1.0
    df["volume_spike_5d"] = vol_ma5 / vol_ma20 - 1.0

    # ── Price-volume divergence ───────────────────────────────────────────
    vol_chg = vol.replace(0, np.nan).pct_change(fill_method=None)
    df["price_vol_divergence_5d"]  = ret_1d.rolling(5).corr(vol_chg)
    df["price_vol_divergence_10d"] = ret_1d.rolling(10).corr(vol_chg)

    # ── Price-volume trend ────────────────────────────────────────────────
    vol_avg = vol.rolling(20).mean().replace(0, np.nan)
    df["price_volume_trend"] = (ret_1d * (vol / vol_avg)).rolling(10).mean()

    # ── Residual momentum ─────────────────────────────────────────────────
    ma20     = close.rolling(20).mean()
    residual = close - ma20
    df["res_mom_5d"]  = residual.pct_change(5,  fill_method=None)
    df["res_mom_10d"] = residual.pct_change(10, fill_method=None)

    # ── Volatility skew ───────────────────────────────────────────────────
    df["vol_skew_20d"] = ret_1d.rolling(20).skew()

    # ── Turnover acceleration ─────────────────────────────────────────────
    to_ma3  = turnover.rolling(3).mean()
    to_ma5  = turnover.rolling(5).mean()
    to_ma20 = turnover.rolling(20).mean().replace(0, np.nan)
    df["turnover_accel_3d"] = to_ma3 / to_ma20 - 1.0
    df["turnover_accel"]    = to_ma5 / to_ma20 - 1.0

    # ── Reversal features (negated momentum — dominant regime) ───────────
    df["rev_3d"]  = -close.pct_change(3,  fill_method=None)
    df["rev_5d"]  = -close.pct_change(5,  fill_method=None)
    df["rev_10d"] = -close.pct_change(10, fill_method=None)
    df["rev_20d"] = -close.pct_change(20, fill_method=None)
    df["rev_ma20"] = -(close / close.rolling(20).mean() - 1.0)
    df["rev_ma60"] = -(close / close.rolling(60).mean() - 1.0)
    df["rev_rsi"]  = 100.0 - df["rsi_14"]
    df["rev_ofi_ma3"] = -ofi.rolling(3).mean()
    df["rev_ofi_ma5"] = -ofi.rolling(5).mean()
    df["rev_ibp_5d"]  = -buy_pressure.rolling(5).mean()
    df["rev_vol_z_20d"] = -(vol - vol_mean20) / vol_std20
    df["rev_vs_3d"] = -(vol_ma3 / vol_ma20 - 1.0)
    df["rev_vs_5d"] = -(vol_ma5 / vol_ma20 - 1.0)
    df["rev_pvd_5d"]  = -ret_1d.rolling(5).corr(vol_chg)
    df["rev_pvd_10d"] = -ret_1d.rolling(10).corr(vol_chg)
    df["rev_pvt"]  = -(ret_1d * (vol / vol_avg)).rolling(10).mean()
    df["rev_ta_3d"] = -(to_ma3 / to_ma20 - 1.0)
    df["rev_ta"]    = -(to_ma5 / to_ma20 - 1.0)

    # ── Trend persistence (T+5 optimized) ────────────────────────────────
    # Linear regression slope of close over last N days, normalized by close
    def _linreg_slope(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window, dtype=float)
        x -= x.mean()
        def _slope(y):
            if y.isna().any():
                return np.nan
            y_arr = y.to_numpy(dtype=float)
            return np.dot(x, y_arr - y_arr.mean()) / (np.dot(x, x) + 1e-12)
        return series.rolling(window).apply(_slope, raw=False)

    df["trend_5d"]  = _linreg_slope(close, 5)  / close.replace(0, np.nan)
    df["trend_10d"] = _linreg_slope(close, 10) / close.replace(0, np.nan)

    # Fraction of last 5 days with positive daily returns (0–1)
    df["trend_consistency"] = (ret_1d > 0).rolling(5).mean()

    # Skip-1 momentum: 5d momentum minus 1d (avoids microstructure noise)
    df["mom_5d_minus_1d"] = df["ret_5d"] - ret_1d

    # ── Volatility-adjusted momentum ──────────────────────────────────────
    vol_5d_safe  = df["vol_5d"].replace(0, np.nan)
    vol_10d_safe = df["vol_10d"].replace(0, np.nan)
    vol_20d_safe = df["vol_20d"].replace(0, np.nan)
    df["vol_adj_mom_5d"]  = df["ret_5d"]  / vol_5d_safe
    df["vol_adj_mom_10d"] = df["ret_10d"] / vol_20d_safe
    df["vol_regime"]      = vol_5d_safe / vol_20d_safe

    # ── Volatility-adjusted returns (cleaner, using matched windows) ──────
    df["ret_5d_vol_adj"]  = df["ret_5d"]  / vol_10d_safe
    df["ret_10d_vol_adj"] = df["ret_10d"] / vol_20d_safe

    # ── Smoothed momentum (EMA) ───────────────────────────────────────────
    df["ret_ema_5d"]  = ret_1d.ewm(span=5, min_periods=3).mean()
    df["ret_ema_10d"] = ret_1d.ewm(span=10, min_periods=5).mean()

    # ── Interaction features ──────────────────────────────────────────────
    df["ret_5d_x_vol_5d"] = df["ret_5d"] * vol_5d_safe
    df["rev_10d_x_vol_z"] = -close.pct_change(10, fill_method=None) * (
        (vol - vol_mean20) / vol_std20
    )

    # ── Mean reversion vs trend regime ────────────────────────────────────
    # Hurst proxy: >1 trending, <1 mean-reverting
    ret_1d_roll5_std = ret_1d.rolling(5).std().replace(0, np.nan)
    df["hurst_proxy"] = (df["ret_5d"] ** 2) / (5 * ret_1d_roll5_std ** 2)

    # Momentum acceleration: recent 3d return minus 3d return from 2 days ago
    df["price_accel"] = df["ret_3d"] - df["ret_5d"].shift(2)

    # ── Cross-sectional relative strength (computed post-panel in _cross_sectional_ranks) ──
    # Store raw values here; ranks are added by _cross_sectional_ranks via _RANK_TARGETS
    # rs_5d and rs_10d are the pct-rank of ret_5d and ret_10d respectively
    df["rs_5d"]  = df["ret_5d"]
    df["rs_10d"] = df["ret_10d"]

    # ── Clean infinities ──────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)

    # ── Shift features by 1 day to prevent lookahead (target is NOT shifted) ──
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].shift(1)

    df.index.name = "date"
    df = df.reset_index()
    return df


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------
def _winsorize(panel: pd.DataFrame) -> pd.DataFrame:
    for col in _RANK_TARGETS:
        if col not in panel.columns:
            continue
        panel[col] = panel.groupby("date")[col].transform(
            lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
        )
    return panel


def _robust_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    for col in _RANK_TARGETS:
        if col not in panel.columns:
            continue
        def _z(x):
            med = x.median()
            mad = (x - med).abs().median()
            return (x - med) / (mad + 1e-6)
        panel[f"{col}_z"] = panel.groupby("date")[col].transform(_z)
    return panel


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    for base in _RANK_TARGETS:
        if base in panel.columns:
            panel[f"{base}_rank"] = panel.groupby("date")[base].rank(
                method="average", pct=True, na_option="keep"
            )
    return panel


_CS_ZSCORE_TARGETS = []


def _cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    csz_cols = {}
    for col in _CS_ZSCORE_TARGETS:
        if col not in panel.columns:
            continue
        g = panel.groupby("date")[col]
        mean = g.transform("mean")
        std = g.transform("std") + 1e-8
        csz_cols[f"{col}_csz"] = (panel[col] - mean) / std
    return pd.concat([panel, pd.DataFrame(csz_cols, index=panel.index)], axis=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _available_feat_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c in panel.columns]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_features(
    prices: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"date", "stock_code", "close", "high", "low", "volume"}
    missing  = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])

    all_dates = pd.DatetimeIndex(sorted(prices["date"].unique()), name="date")

    func  = partial(_per_stock_features, all_dates=all_dates)
    panel = _safe_groupby_apply(prices, "stock_code", func)

    # Subtract index forward return → target becomes true excess return (alpha)
    # With beta-adjustment: target = raw_fwd_ret - beta * idx_fwd_ret
    # beta is computed per-stock using 60d rolling regression of daily returns.
    if index_df is not None and TARGET_COLUMN in panel.columns:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx_close = idx.set_index("date")["close"].sort_index().astype(float)
        idx_ret_1d = idx_close.pct_change(fill_method=None).rename("_idx_ret_1d")
        idx_fwd = (idx_close.shift(-FORWARD_HORIZON) / idx_close - 1.0).rename("_idx_fwd")

        panel = panel.merge(idx_fwd.reset_index(), on="date", how="left")
        panel = panel.merge(idx_ret_1d.reset_index(), on="date", how="left")

        # Rolling beta per stock: cov(ret_1d, idx_ret_1d) / var(idx_ret_1d), 60d window
        def _rolling_beta(grp: pd.DataFrame) -> pd.Series:
            grp = grp.sort_values("date")
            cov = grp["ret_1d"].rolling(60, min_periods=30).cov(grp["_idx_ret_1d"])
            var = grp["_idx_ret_1d"].rolling(60, min_periods=30).var()
            return (cov / (var + 1e-9)).clip(lower=0.3, upper=2.0)

        panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)
        panel["_beta"] = (
            panel.groupby("stock_code", group_keys=False)
                 .apply(_rolling_beta)
                 .reset_index(level=0, drop=True)
        )
        panel["_beta"] = panel["_beta"].fillna(1.0)

        # Beta-adjusted excess target
        panel[TARGET_COLUMN] = panel[TARGET_COLUMN] - panel["_beta"] * panel["_idx_fwd"]
        panel = panel.drop(columns=["_idx_fwd", "_idx_ret_1d", "_beta"])

    panel = _winsorize(panel)
    panel = _robust_zscore(panel)
    panel = _cross_sectional_ranks(panel)
    panel = _cross_sectional_zscore(panel)

    # rs_5d / rs_10d = cross-sectional pct-rank of ret_5d / ret_10d
    # _cross_sectional_ranks already produced ret_5d_rank and ret_10d_rank;
    # expose them under the names used in FEATURE_COLUMNS
    if "ret_5d_rank" in panel.columns:
        panel["rs_5d"] = panel["ret_5d_rank"]
    if "ret_10d_rank" in panel.columns:
        panel["rs_10d"] = panel["ret_10d_rank"]

    # Cross-sectional z-score the excess-return target to stabilize training
    if TARGET_COLUMN in panel.columns:
        grouped = panel.groupby("date")[TARGET_COLUMN]
        panel[TARGET_COLUMN] = (
            (panel[TARGET_COLUMN] - grouped.transform("mean"))
            / (grouped.transform("std") + 1e-8)
        )

    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    feat_cols = _available_feat_cols(panel)
    df = panel.dropna(subset=feat_cols + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of     = pd.Timestamp(as_of)
    feat_cols = _available_feat_cols(panel)
    df        = panel[panel["date"] == as_of].dropna(subset=feat_cols).copy()
    df        = df.reset_index(drop=True)
    if "stock_code" not in df.columns:
        raise RuntimeError("prediction_frame: stock_code not found.")
    return df
