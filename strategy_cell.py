import datetime
import numpy as np
import pandas as pd
import importlib

# Choose your model here:
import a23_xgb
importlib.reload(a23_xgb)
from a23_xgb import train_and_predict

# import a23
# importlib.reload(a23)
# from a23 import train_and_predict

# ─────────────────────────────────────────────────────────────────────────────
# TWO-STEP PROCESS
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline(
    entire_features: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    train_end: str = "2017-12-31",
    test_start: str = "2005-01-04",
    test_end: str = "2019-12-31"
) -> pd.DataFrame:
    """
    Step 1: Run the heavy machine learning pipeline 
    Returns a dataframe of ML scores (predicted returns or ranks) for every stock.
    Run this cell ONCE.
    """
    print("Running Machine Learning Pipeline... This might take a few minutes.")
    ml_scores = train_and_predict(
        entire_features,
        returns,
        universe,
        train_end  = train_end,
        test_start = test_start,
        test_end   = test_end,
    )
    return ml_scores


def build_portfolio_from_scores(
    ml_scores: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    n_long: int = 10,
    n_short: int = 10,
    smooth: float = 0.5,
) -> pd.DataFrame:
    """
    Dollar-neutral, unit-capital portfolio from pre-computed scores.

    Fixes:
    1) Apply universe mask before final portfolio sizing.
    2) Use equal-weight long/short buckets so max |w| <= 0.1 when
       n_long >= 5 and n_short >= 5.
    3) If there are not enough tradable names on either side for a day,
       return a zero row instead of violating constraints.
    """

    # ── Date validation ───────────────────────────────────────────────────────
    try:
        start_dt  = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt    = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
        cutoff_dt = datetime.datetime.strptime("2005-01-01", "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be strings in 'YYYY-MM-DD' format.")

    if start_dt >= end_dt:
        raise ValueError("start_date must be before end_date.")
    if start_dt < cutoff_dt:
        raise ValueError("start_date must be >= 2005-01-01.")

    # Align universe to the score matrix
    universe = universe.reindex(index=ml_scores.index, columns=ml_scores.columns)
    uv_mask = universe.fillna(False).astype(bool)

    trading_days = ml_scores.index[
        (ml_scores.index >= start_dt) & (ml_scores.index <= end_dt)
    ]
    if len(trading_days) == 0:
        raise ValueError("No trading days in the specified date range.")

    # ── Raw signal: top/bottom names per day ───────────────────────────────────
    raw = pd.DataFrame(0.0, index=ml_scores.index, columns=ml_scores.columns)

    for date in ml_scores.index:
        row = ml_scores.loc[date].dropna()

        # only keep tradable names on that date
        tradable = row.index[uv_mask.loc[date, row.index]]
        row = row.loc[tradable]

        if len(row) < (n_long + n_short):
            continue

        s = row.sort_values()
        raw.loc[date, s.index[:n_short]] = -1.0
        raw.loc[date, s.index[-n_long:]] = 1.0

    # ── EMA smoothing on the raw signal ───────────────────────────────────────
    smoothed = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    prev = pd.Series(0.0, index=raw.columns)

    for date in raw.index:
        # smooth the signal first
        blended = smooth * raw.loc[date] + (1 - smooth) * prev

        # mask BEFORE constructing the final portfolio
        blended = blended.where(uv_mask.loc[date], 0.0)

        # choose long and short candidates from the blended signal
        long_candidates = blended[blended > 0].sort_values(ascending=False)
        short_candidates = blended[blended < 0].sort_values(ascending=True)

        # Need at least 5 names per side to satisfy max |w| <= 0.1
        long_n = min(n_long, len(long_candidates))
        short_n = min(n_short, len(short_candidates))

        if long_n < 5 or short_n < 5:
            # zero row is allowed; it will not fail the non-zero constraint
            smoothed.loc[date] = 0.0
            prev = pd.Series(0.0, index=raw.columns)
            continue

        long_names = long_candidates.index[:long_n]
        short_names = short_candidates.index[:short_n]

        day = pd.Series(0.0, index=raw.columns)
        day.loc[long_names] = 0.5 / long_n
        day.loc[short_names] = -0.5 / short_n

        # final safety mask
        day = day.where(uv_mask.loc[date], 0.0)

        smoothed.loc[date] = day
        prev = day

    smoothed = smoothed.fillna(0.0)
    return smoothed.loc[start_date:end_date]

    
def build_portfolio_from_scores_hysteresis(
    ml_scores: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
    n_long: int = 10,
    n_short: int = 10,
    buffer: int = 5,
    rebalance_every: int = 1,
) -> pd.DataFrame:
    """
    Dollar-neutral, unit-capital portfolio using RANK HYSTERESIS (Sticky Weights).
    
    Instead of swapping a stock out the moment it drops from rank #10 to #11, 
    the model holds onto it until its rank drops below (n_long + buffer) e.g., #15.
    If an open slot is created, it takes the highest-ranked available stock.
    This drastically reduces transaction fees/turnover without needing EMA smoothing.
    """
    import datetime
    try:
        start_dt  = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt    = datetime.datetime.strptime(end_date,   "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be strings in 'YYYY-MM-DD' format.")

    universe = universe.reindex(index=ml_scores.index, columns=ml_scores.columns)
    uv_mask = universe.fillna(False).astype(bool)

    portfolio = pd.DataFrame(0.0, index=ml_scores.index, columns=ml_scores.columns)
    
    prev_longs = set()
    prev_shorts = set()
    days_since_rebalance = 0
    
    for date in ml_scores.index:
        row = ml_scores.loc[date]
        active_row = row[uv_mask.loc[date]].dropna()
        
        if len(active_row) < (n_long + buffer + n_short + buffer):
            prev_longs = set()
            prev_shorts = set()
            days_since_rebalance = 0
            continue
            
        # ─── REBALANCE CHECK ───
        force_trade = False
        if not prev_longs or not prev_shorts:
            force_trade = True
        else:
            # If any held stock exits the universe today, we MUST rebalance to avoid a ValueError
            if not prev_longs.issubset(active_row.index) or not prev_shorts.issubset(active_row.index):
                force_trade = True
                
        if not force_trade and (days_since_rebalance % rebalance_every != 0):
            # Skip trading entirely: just hold yesterday's exact positions
            current_longs = prev_longs
            current_shorts = prev_shorts
            days_since_rebalance += 1
        else:
            # Time to rebalance (or forced to)!
            # Sort descending: best long targets are at the TOP, best short targets are at the BOTTOM
            s = active_row.sort_values(ascending=False)
            
            # ─── LONG HYSTERESIS ───
            # Identify the acceptable top zone
            long_candidates = set(s.index[:(n_long + buffer)])
            # Keep stocks that are BOTH currently in portfolio AND in the top buffer zone
            kept_longs = prev_longs.intersection(long_candidates)
            
            slots_to_fill = n_long - len(kept_longs)
            new_longs = []
            for stock in s.index:
                if slots_to_fill <= 0:
                    break
                if stock not in kept_longs:
                    new_longs.append(stock)
                    slots_to_fill -= 1
                    
            current_longs = kept_longs.union(new_longs)
            
            # ─── SHORT HYSTERESIS ───
            # Identify the acceptable bottom zone
            short_candidates = set(s.index[-(n_short + buffer):])
            # Keep stocks that are BOTH currently in portfolio AND in the bottom buffer zone
            kept_shorts = prev_shorts.intersection(short_candidates)
            
            slots_to_fill = n_short - len(kept_shorts)
            new_shorts = []
            # Search backward from the worst scores
            for stock in s.index[::-1]:
                if slots_to_fill <= 0:
                    break
                if stock not in kept_shorts:
                    new_shorts.append(stock)
                    slots_to_fill -= 1
                    
            current_shorts = kept_shorts.union(new_shorts)
            days_since_rebalance = 1

        # ─── ALLOCATE WEIGHTS ───
        w_long = 0.5 / n_long if n_long > 0 else 0
        w_short = -0.5 / n_short if n_short > 0 else 0
        
        portfolio.loc[date, list(current_longs)] = w_long
        portfolio.loc[date, list(current_shorts)] = w_short
        
        prev_longs = current_longs
        prev_shorts = current_shorts
        
    # Return cleanly clipped date range
    return portfolio.fillna(0.0).loc[start_date:end_date]

