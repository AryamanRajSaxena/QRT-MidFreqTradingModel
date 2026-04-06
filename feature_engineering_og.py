import pandas as pd
import numpy as np

EPS = 1e-8


def _broadcast_series_to_frame(series: pd.Series, columns) -> pd.DataFrame:
    return pd.DataFrame(
        np.repeat(series.to_numpy()[:, None], len(columns), axis=1),
        index=series.index,
        columns=columns,
    )


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Smaller feature set with only:
      - HAR volatility features
      - market regime
      - trend_vol_adj
      - macd_vol
      - rsi_vol
      - meta_momentum
    """

    base_keep = [
        "relative_strength_index", 
        "macd", 
        "trend_5_20", 
        "trend_20_60", 
        "volatility_20", 
        "volatility_60", 
        "average_true_range", 
        "on_balance_volume", 
        "chaikin_money_flow", 
        "accumulation_distribution_index",
        "accumulation_distribution_index",
        "trend_1_3", 
        "volume",
        "ease_of_movement",
        "chande_momentum_oscillator"
    ]

    available_feats = features.columns.get_level_values(0).unique().tolist()
    keep_feats = [f for f in base_keep if f in available_feats]


    base = features.loc[:, (keep_feats, slice(None))].copy()

    rsi = base["relative_strength_index"]
    macd = base["macd"]
    trend_20_60 = base["trend_20_60"]
    vol_20 = base["volatility_20"]
    vol_60 = base["volatility_60"]
    cmo = base["chande_momentum_oscillator"]
    trend_5_20 = base["trend_5_20"]

    # Volatility ratio
    vol_ratio = vol_20 / (vol_60 + EPS)

    # HAR-style volatility ladder
    vol_daily = vol_20
    vol_weekly = vol_daily.rolling(5, min_periods=3).mean()
    vol_monthly = vol_daily.rolling(22, min_periods=5).mean()

    # Soft market regime signal
    market_vol_sr = vol_20.mean(axis=1)
    market_vol_mean = market_vol_sr.rolling(252, min_periods=20).mean()
    market_vol_std = market_vol_sr.rolling(252, min_periods=20).std()
    market_vol_z = (market_vol_sr - market_vol_mean) / (market_vol_std + EPS)
    market_regime_soft = 1.0 / (1.0 + np.exp(-market_vol_z.fillna(0.0)))
    regime_df = _broadcast_series_to_frame(market_regime_soft, vol_20.columns)

    # Requested interactions
    trend_vol_adj = trend_20_60 / (vol_60 + EPS)
    macd_vol = macd * vol_ratio
    rsi_vol = rsi / (vol_20 + EPS)

    # Meta momentum
    meta_momentum = (
        rsi.rank(axis=1, pct=True)
        + macd.rank(axis=1, pct=True)
        + cmo.rank(axis=1, pct=True)
    ) / 3.0

    # ─── Requested Advanced Alpha Formulas ───
    # We use CSRank (.rank(axis=1, pct=True)) on the continuous components as specified

    # Alpha 1
    macd_mean_10 = macd.rolling(10, min_periods=5).mean()
    alpha1_core = (macd - macd_mean_10) / (vol_20 + EPS)
    alpha1 = alpha1_core.rank(axis=1, pct=True) * (rsi < 30).astype(float)

    # Alpha 2
    alpha2_core = (30 - rsi) / (vol_20 + EPS)
    alpha2 = alpha2_core.rank(axis=1, pct=True) * (rsi < 30).astype(float) * (vol_20 > vol_60).astype(float)

    # Alpha 4
    alpha4_core = trend_5_20 / (vol_20 + EPS)
    alpha4 = alpha4_core.rank(axis=1, pct=True) * (trend_20_60 > 0).astype(float)

    # Alpha 3 and 5 conditionally (requiring williams_r and close if available)
    alpha3 = pd.DataFrame(0.0, index=rsi.index, columns=rsi.columns)
    alpha5 = pd.DataFrame(0.0, index=rsi.index, columns=rsi.columns)

    if "williams_r" in available_feats:
        wr = features.loc[:, ("williams_r", slice(None))]
        wr.columns = wr.columns.get_level_values(1)  # flatten to match
        alpha3_core = (100 + wr) / (vol_20 + EPS)
        alpha3 = alpha3_core.rank(axis=1, pct=True) * (wr < -80).astype(float)

    engineered = {
        #"har_daily": vol_daily,
        #"har_weekly": vol_weekly,
        #"har_monthly": vol_monthly,
        "market_regime": regime_df,
        "trend_vol_adj": trend_vol_adj,
        #"macd_vol": macd_vol,
        "rsi_vol": rsi_vol,
        "meta_momentum": meta_momentum,
        "alpha1": alpha1,
        "alpha2": alpha2,
        #"alpha3": alpha3,
        "alpha4": alpha4,
    }

    engineered_features = pd.concat(engineered, axis=1)
    final_features = pd.concat([base, engineered_features], axis=1)

    total = len(final_features.columns.get_level_values(0).unique())
    print(f"Feature engineering complete. Returning {total} total features.")
    return final_features