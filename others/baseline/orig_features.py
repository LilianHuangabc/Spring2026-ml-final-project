"""
Feature engineering for the CSI500 stock-selection baseline.
[UPGRADED VERSION: 100% Cross-Sectional Features & Warning Fixed]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 我们只让 XGBoost 看到 Rank（排名）特征，彻底抹除大盘暴涨暴跌带来的干扰！
FEATURE_COLUMNS = [
    "ret_1d_rank", "ret_5d_rank", "ret_10d_rank", "ret_20d_rank", 
    "vol_20d_rank", "rsi_14_rank",
    "buy_pressure_rank", "amplitude_rank", "vol_accel_rank", "pv_corr_rank"
]
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算单只股票的绝对指标（随后在全市场中进行排名）"""
    df = df.sort_values("date").copy()
    close = df["close"]
    vol = df["volume"].astype(float)

    # 1. 动量/反转 (使用 fill_method=None 消除 Pandas 警告)
    df["ret_1d"] = close.pct_change(1, fill_method=None)
    df["ret_5d"] = close.pct_change(5, fill_method=None)
    df["ret_10d"] = close.pct_change(10, fill_method=None)
    df["ret_20d"] = close.pct_change(20, fill_method=None)

    # 2. 波动率
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # 3. 经典 RSI
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    rs = up / down
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ==========================================
    # 🚀 高阶量价因子 (Alpha Factors)
    # ==========================================
    
    # 4. 日内买盘压力 (Intraday Buy Pressure)
    # 逻辑：收盘价越接近最高价，说明日内多头力量越强
    high_low_range = df["high"] - df["low"]
    df["buy_pressure"] = (df["close"] - df["open"]) / high_low_range.replace(0, np.nan)
    
    # 5. 真实振幅 (Amplitude)
    # 逻辑：振幅大的股票往往更容易存在机构博弈或资金异动
    df["amplitude"] = high_low_range / close.shift(1)
    
    # 6. 成交量加速率 (Volume Acceleration)
    # 逻辑：近5天平均成交量对比近20天平均成交量，捕捉放量信号
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean().replace(0, np.nan)
    df["vol_accel"] = vol_ma5 / vol_ma20 - 1.0
    
    # 7. 量价相关性 (Price-Volume Correlation)
    # 逻辑：计算过去10天每日涨跌幅与成交量变化的相关性。
    # "量价齐升"时该值为正，A股中往往是强势特征。
    vol_chg = vol.pct_change(1, fill_method=None)
    df["pv_corr"] = df["ret_1d"].rolling(10).corr(vol_chg)

    # ------------------------------------------
    # 计算目标 (未来 5 天收益率)
    # ------------------------------------------
    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    
    return df


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """横截面排名：将所有绝对数值转换为 0 到 1 之间的排名"""
    # 提取需要被排名的基础列名 (也就是去掉 FEATURE_COLUMNS 里的 _rank 后缀)
    base_cols = [col.replace("_rank", "") for col in FEATURE_COLUMNS]
    
    # 按日期分组，在每一天内部对所有的股票特征进行排序
    for base in base_cols:
        # 如果有些天缺失某些指标（比如停牌），rank会自动忽略并处理
        panel[f"{base}_rank"] = panel.groupby("date")[base].rank(method="average", pct=True)
        
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    # 确保包含了 open, high, low 等新加入的列
    required = {"date", "stock_code", "open", "close", "high", "low", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    
    # 避免 groupby.apply 的 FutureWarning，加入 include_groups=False
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features, include_groups=False)
        .reset_index(drop=True)
    )
    
    # 为避免旧版 groupby 丢失 stock_code 的问题，如果丢失强制补回
    if "stock_code" not in panel.columns:
        panel["stock_code"] = prices["stock_code"].values
        
    panel = _cross_sectional_ranks(panel)
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df