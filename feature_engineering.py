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
    Smaller, cleaner feature set for GP + XGBoost.

    Keeps:
      - one oscillator: RSI
      - two trend horizons: trend_5_20, trend_20_60
      - volatility context: vol_20, vol_60, ATR
      - volume confirmation: OBV, CMF, ADI
      - regime context: vol_ratio, vol_rank, market_vol, soft regime
      - a few strong interactions
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
        "volume",
    ]

    available_feats = features.columns.get_level_values(0).unique().tolist()
    keep_feats = [f for f in base_keep if f in available_feats]

    missing = [f for f in base_keep if f not in available_feats]
    if missing:
        print(f"Skipping missing features: {missing}")

    base = features.loc[:, (keep_feats, slice(None))].copy()

    rsi = base["relative_strength_index"]
    macd = base["macd"]
    trend_5_20 = base["trend_5_20"]
    trend_20_60 = base["trend_20_60"]

    vol_20 = base["volatility_20"]
    vol_60 = base["volatility_60"]
    atr = base["average_true_range"]

    obv = base["on_balance_volume"]
    cmf = base["chaikin_money_flow"]
    adi = base["accumulation_distribution_index"]

    # Volatility regime context
    vol_ratio = vol_20 / (vol_60 + EPS)
    vol_rank = vol_20.rank(axis=1, pct=True)
    rel_vol_to_cross_section = vol_20 / (vol_20.mean(axis=1).to_frame().reindex(vol_20.index).values + EPS)

    market_vol_sr = vol_20.mean(axis=1)
    market_vol = _broadcast_series_to_frame(market_vol_sr, vol_20.columns)

    # Smooth regime signal instead of a hard threshold
    market_vol_mean = market_vol_sr.rolling(252, min_periods=20).mean()
    market_vol_std = market_vol_sr.rolling(252, min_periods=20).std()
    market_vol_z = (market_vol_sr - market_vol_mean) / (market_vol_std + EPS)
    market_regime_soft = 1.0 / (1.0 + np.exp(-market_vol_z.fillna(0.0)))
    market_regime_soft = _broadcast_series_to_frame(market_regime_soft, vol_20.columns)

    # HAR-style volatility ladder
    har_daily = vol_20
    har_weekly = vol_20.rolling(5, min_periods=3).mean()
    har_monthly = vol_20.rolling(22, min_periods=5).mean()

    har_short_vs_medium = har_daily / (har_weekly + EPS)
    har_medium_vs_long = har_weekly / (har_monthly + EPS)
    har_spread = har_daily - har_monthly

    # Strong, limited interactions
    trend_20_60_vol_ratio = trend_20_60 * vol_ratio
    macd_vol_ratio = macd * vol_ratio
    trend_20_60_over_vol = trend_20_60 / (vol_60 + EPS)
    rsi_over_vol = rsi / (vol_20 + EPS)

    trend_cmf = trend_20_60 * cmf
    macd_obv = macd * obv
    rsi_adi = rsi * adi

    engineered = {
        "vol_ratio": vol_ratio,
        "vol_rank": vol_rank,
        "rel_vol_to_cross_section": rel_vol_to_cross_section,
        "market_vol": market_vol,
        "market_regime_soft": market_regime_soft,
        "har_daily": har_daily,
        "har_weekly": har_weekly,
        "har_monthly": har_monthly,
        "har_short_vs_medium": har_short_vs_medium,
        "har_medium_vs_long": har_medium_vs_long,
        "har_spread": har_spread,
        "trend_20_60_vol_ratio": trend_20_60_vol_ratio,
        "macd_vol_ratio": macd_vol_ratio,
        "trend_20_60_over_vol": trend_20_60_over_vol,
        "rsi_over_vol": rsi_over_vol,
        "trend_cmf": trend_cmf,
        "macd_obv": macd_obv,
        "rsi_adi": rsi_adi,
    }

    engineered_features = pd.concat(engineered, axis=1)
    final_features = pd.concat([base, engineered_features], axis=1)

    total = len(final_features.columns.get_level_values(0).unique())
    print(f"Feature engineering complete. Returning {total} total features.")
    return final_features