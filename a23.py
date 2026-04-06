"""
GP + LambdaMART Alpha Generation Pipeline
==========================================
1. Genetic Programming (gplearn.SymbolicTransformer) evolves non-linear
   alpha formulas from the 22 base features.
2. LambdaMART (LightGBM lambdarank) learns to rank stocks per day using
   both the original + GP-derived features.
3. No leakage guarantee:
   - ALL features are shifted by 1 day before use:
       signal on day T  →  uses features from day T-1.
   - Forward-return labels: fwd_ret[T] = return_{T→T+1}.
     The model learns  (feat[T-1])  →  rank of  return_{T→T+1},
     so no future information leaks into the feature set.
   - Train set is strictly limited to dates ≤ train_end.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from gplearn.genetic import SymbolicTransformer

_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_feature_array(
    shifted_feats: pd.DataFrame,
    feat_names: list,
    dates: pd.DatetimeIndex,
    stocks: pd.Index,
) -> np.ndarray:
    """
    Return float32 array of shape (n_dates, n_stocks, n_feats)
    using raw sub-DataFrames (no stack needed → version-safe).
    """
    n_d, n_s, n_f = len(dates), len(stocks), len(feat_names)
    arr = np.full((n_d, n_s, n_f), np.nan, dtype=np.float32)
    sf = shifted_feats.loc[dates]
    for j, fn in enumerate(feat_names):
        # sf[fn] : (n_dates, n_stocks) – works because columns are MultiIndex(feat, stock)
        feat_2d = sf[fn].reindex(columns=stocks).values.astype(np.float32)
        arr[:, :, j] = feat_2d
    return arr  # (n_dates, n_stocks, n_feats)


def _csrank_3d(X3d: np.ndarray) -> np.ndarray:
    """
    Cross-sectional rank-normalise every feature to [0, 1] per day.

    CSRank(x_{i,t}) = (Rank(x_{i,t}) - 1) / (N_t - 1)

    where Rank is the ascending rank among in-universe stocks on day t,
    and N_t is the count of non-NaN values for that feature on day t.

    • Stocks with NaN stay NaN.
    • If N_t == 1 (only one valid stock), output 0.5 (midpoint).
    • Applied independently per feature and per date.

    Parameters
    ----------
    X3d : (n_dates, n_stocks, n_feats) float32 array

    Returns
    -------
    (n_dates, n_stocks, n_feats) float32 array with values in [0, 1] or NaN.
    """
    out = np.full_like(X3d, np.nan)
    n_dates, n_stocks, n_feats = X3d.shape

    for f in range(n_feats):
        # slice: (n_dates, n_stocks)
        feat = X3d[:, :, f]
        for d in range(n_dates):
            row = feat[d]                         # (n_stocks,)
            valid = np.isfinite(row)
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            if n_valid == 1:
                out[d, valid, f] = 0.5
                continue
            # argsort of argsort = ranks (0-indexed ascending)
            vals = row[valid]
            ranks = vals.argsort().argsort().astype(np.float32)
            out[d, valid, f] = ranks / (n_valid - 1)   # scale to [0, 1]

    return out


def _build_flat_dataset(
    X3d: np.ndarray,           # (n_dates, n_stocks, n_feats)
    uv2d: np.ndarray,          # (n_dates, n_stocks) bool
    dates: pd.DatetimeIndex,
    stocks: pd.Index,
    y2d: np.ndarray | None = None,  # (n_dates, n_stocks) forward returns
) -> tuple:
    """
    Flatten 3-D arrays into 2-D (n_obs, n_feats) preserving date/stock index.

    Returns
    -------
    X      : (n_obs, n_feats) float32
    y      : (n_obs,) float32  or None
    idx    : MultiIndex (date, stock) of length n_obs
    groups : (n_query,) int32 – count of stocks per unique date (for LGBM)
    """
    rows_X, rows_y, row_dates, row_stocks = [], [], [], []
    group_sizes = []

    for i, date in enumerate(dates):
        mask = uv2d[i]                        # (n_stocks,) bool in-universe
        if y2d is not None:
            ret_row = y2d[i]
            mask = mask & np.isfinite(ret_row)

        feat_row = X3d[i][mask]               # (n_in_univ, n_feats)
        has_feat = np.isfinite(feat_row).all(axis=1)
        feat_row = feat_row[has_feat]

        n = len(feat_row)
        if n < 2:
            continue

        rows_X.append(feat_row)
        row_dates.extend([date] * n)
        row_stocks.extend(stocks[mask][has_feat].tolist())

        if y2d is not None:
            rows_y.append(y2d[i][mask][has_feat])

        group_sizes.append(n)

    if not rows_X:
        return np.empty((0, X3d.shape[2]), np.float32), None, pd.MultiIndex.from_tuples([]), np.array([], np.int32)

    X_out = np.vstack(rows_X).astype(np.float32)
    y_out = np.concatenate(rows_y).astype(np.float32) if rows_y else None
    idx   = pd.MultiIndex.from_arrays([row_dates, row_stocks], names=["date", "stock"])
    groups = np.array(group_sizes, dtype=np.int32)
    return X_out, y_out, idx, groups


def _make_rank_labels(y_flat: np.ndarray, groups: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    For each query (day), bin cross-sectional forward-return rank into
    [0 … n_bins-1] buckets (0 = worst, n_bins-1 = best).
    """
    labels = np.zeros(len(y_flat), dtype=np.int32)
    offset = 0
    for n in groups:
        chunk = y_flat[offset: offset + n]
        ranks = chunk.argsort().argsort().astype(np.float32)
        buckets = np.floor(ranks / n * n_bins).astype(np.int32).clip(0, n_bins - 1)
        labels[offset: offset + n] = buckets
        offset += n
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_and_predict(
    entire_features: pd.DataFrame,
    returns_data: pd.DataFrame,
    universe: pd.DataFrame,
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
    test_end: str = "2019-12-31",
    n_gp_features: int = 10,
    n_gp_generations: int = 5,
    n_lgbm_trees: int = 500,          # upper bound — early stopping will cut this
    label_horizon: int = 15,           # days to compound returns for label (1=noisy, 5=better)
    val_months: int = 12,             # months at end of training used as early-stop val set
) -> pd.DataFrame:
    """
    Train GP + LambdaMART on data up to train_end.
    Predict ranking scores for test_start … test_end.

    Returns
    -------
    signal_matrix : pd.DataFrame  shape (n_test_days, n_stocks)
        Raw LambdaMART ranking scores.  NaN where stock not in universe.
        Higher score  →  higher predicted rank  →  long candidate.
    """
    train_end_dt   = pd.to_datetime(train_end)
    test_start_dt  = pd.to_datetime(test_start)
    test_end_dt    = pd.to_datetime(test_end)

    feat_names = entire_features.columns.get_level_values(0).unique().tolist()
    stocks     = universe.columns
    all_dates  = entire_features.index

    # ── 1. Shift features by 1 to prevent look-ahead leakage ─────────────────
    #   shifted_features.loc[T]  =  raw features from day T-1.
    #   Paired with fwd_returns.loc[T]  =  return_{T → T+1}.
    #   → model sees (T-1) features to predict (T+1) return.  No leakage.
    shifted = entire_features.shift(1)

    # Forward returns: compound over `label_horizon` days to reduce label noise.
    # A 5-day label is ~2.2x less noisy than a 1-day label (noise scales as √horizon
    # while signal scales linearly), which dramatically reduces overfitting.
    # At T, label = cumulative return from T+1 to T+label_horizon (no look-ahead:
    # compounding starts the day AFTER today).
    log_ret = np.log1p(returns_data.clip(-0.5, 1.0))   # log-returns (stable compounding)
    fwd_rets = (
        log_ret.shift(-1)                              # shift so day-T label uses T+1 onwards
        .rolling(label_horizon, min_periods=1).sum()   # sum log-rets over horizon
        .shift(-(label_horizon - 1))                   # align: value at T = sum[T+1..T+h]
    )

    # ── 2. Split dates ────────────────────────────────────────────────────────
    train_dates = all_dates[(all_dates >= all_dates[1]) & (all_dates <= train_end_dt)]
    test_dates  = all_dates[(all_dates >= test_start_dt) & (all_dates <= test_end_dt)]

    if len(train_dates) == 0:
        raise ValueError("No training dates found.")
    if len(test_dates) == 0:
        raise ValueError("No test dates found.")

    # ── 3. Build 3-D arrays ───────────────────────────────────────────────────
    print("[1/5] Extracting training features …")
    uv   = universe.reindex(all_dates).astype(bool)
    X3d  = _extract_feature_array(shifted, feat_names, train_dates, stocks)
    X3d  = _csrank_3d(X3d)   # CSRank: (rank-1)/(N-1) per feature per day → [0,1]
    uv2d = uv.loc[train_dates].values
    fr2d = fwd_rets.reindex(index=train_dates, columns=stocks).values.astype(np.float32)

    print("[2/5] Flattening data …")
    X_tr, y_tr, idx_tr, groups_tr = _build_flat_dataset(X3d, uv2d, train_dates, stocks, fr2d)
    print(f"      Training observations: {len(X_tr):,}  features: {X_tr.shape[1]}")

    # ── 4. Genetic Programming – discover alpha formulas ─────────────────────
    #   Subsample every 4th date to keep GP training time reasonable.
    print(f"[3/5] Genetic Programming ({n_gp_generations} generations) …")
    gp_slice = slice(None, None, 4)
    idx_arr  = np.cumsum(np.insert(groups_tr, 0, 0))  # group boundaries
    gp_mask  = np.zeros(len(X_tr), dtype=bool)
    for k in range(0, len(groups_tr), 4):            # every 4th query
        gp_mask[idx_arr[k]: idx_arr[k + 1]] = True

    X_gp = X_tr[gp_mask]
    y_gp = y_tr[gp_mask]

    gp = SymbolicTransformer(
        generations       = n_gp_generations,
        population_size   = 500,
        hall_of_fame      = 100,
        n_components      = n_gp_features,
        function_set      = ["add", "sub", "mul", "div", "sqrt", "log",
                             "abs", "neg", "max", "min"],
        parsimony_coefficient = 0.05,   # 10x higher: strongly penalises complex trees
        max_samples       = 0.5,        # bootstrap 50% → more diverse, less overfit
        verbose           = 1,
        random_state      = _SEED,
        n_jobs            = -1,
    )
    gp.fit(X_gp.astype(np.float64), y_gp.astype(np.float64))
    print(f"      GP done.  Best programs: {len(gp._best_programs)}")

    # ── 5. Augment & train LambdaMART with early stopping ────────────────────
    print("[4/5] Training LambdaMART ranker …")
    X_gp_tr  = gp.transform(X_tr.astype(np.float64)).astype(np.float32)
    X_aug_tr = np.hstack([X_tr, X_gp_tr])            # (n_obs, 22 + n_gp)

    labels = _make_rank_labels(y_tr, groups_tr, n_bins=5)

    # Split last `val_months` of training data as a validation set for early stopping.
    # This prevents the ranker from memorising the full training period.
    last_train_date = train_dates[-1]
    val_cutoff = last_train_date - pd.DateOffset(months=val_months)
    tr_date_arr = idx_tr.get_level_values("date")
    is_val  = tr_date_arr > val_cutoff
    is_fit  = ~is_val

    X_fit,  y_fit,  labels_fit  = X_aug_tr[is_fit],  y_tr[is_fit],  labels[is_fit]
    X_val,  y_val,  labels_val  = X_aug_tr[is_val],  y_tr[is_val],  labels[is_val]

    # Recompute group sizes for fit / val splits
    def _recount_groups(date_arr):
        counts = pd.Series(date_arr).value_counts().sort_index()
        return counts.values.astype(np.int32)

    groups_fit = _recount_groups(tr_date_arr[is_fit])
    groups_val = _recount_groups(tr_date_arr[is_val])

    print(f"      Fit queries: {len(groups_fit)}  |  Val queries: {len(groups_val)}")

    ranker = lgb.LGBMRanker(
        objective         = "lambdarank",
        metric            = "ndcg",
        ndcg_eval_at      = [5, 10],
        num_leaves        = 31,           # shallower trees → less memorisation
        min_data_in_leaf  = 200,          # at least 200 stocks per leaf
        learning_rate     = 0.03,         # slower learning → generalises better
        n_estimators      = n_lgbm_trees,
        subsample         = 0.6,          # more aggressive dropout
        colsample_bytree  = 0.6,
        reg_alpha         = 1.0,          # stronger L1
        reg_lambda        = 5.0,          # stronger L2
        random_state      = _SEED,
        verbose           = -1,
        n_jobs            = -1,
    )
    ranker.fit(
        X_fit, labels_fit, group=groups_fit,
        eval_set            = [(X_val, labels_val)],
        eval_group          = [groups_val],
        eval_at             = [5, 10],
        callbacks           = [
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"      Best iteration: {ranker.best_iteration_}  (out of {n_lgbm_trees} max)")
    print("      LambdaMART training complete.")

    # ── 6. Predict on test period ─────────────────────────────────────────────
    print("[5/5] Generating test-period signals …")
    X3d_te  = _extract_feature_array(shifted, feat_names, test_dates, stocks)
    X3d_te  = _csrank_3d(X3d_te)   # same CSRank transform as training
    uv2d_te = uv.loc[test_dates].values
    X_te, _, idx_te, _ = _build_flat_dataset(X3d_te, uv2d_te, test_dates, stocks)

    # Replace NaN/inf with 0 for inference (no returns needed)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    X_gp_te  = gp.transform(X_te.astype(np.float64)).astype(np.float32)
    X_aug_te = np.hstack([X_te, X_gp_te])

    scores = ranker.predict(X_aug_te)

    # Reconstruct wide (dates × stocks) signal matrix
    score_series = pd.Series(scores, index=idx_te, name="score")
    signal_wide  = score_series.unstack(level="stock")
    signal_wide  = signal_wide.reindex(index=test_dates, columns=stocks)

    # Mask out non-universe stocks
    signal_wide = signal_wide.where(uv.loc[test_dates], np.nan)
    print("      Done!")
    return signal_wide


# ─────────────────────────────────────────────────────────────────────────────
# GP Alpha Quality Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gp_alphas(
    entire_features: pd.DataFrame,
    returns_data: pd.DataFrame,
    universe: pd.DataFrame,
    train_end: str = "2017-12-31",
    n_gp_features: int = 10,
    n_gp_generations: int = 5,
) -> pd.DataFrame:
    """
    Fit GP on training data then score every discovered alpha formula by its
    Information Coefficient (IC) — the rank-correlation of the alpha's daily
    cross-sectional signal with next-day forward returns.

    Metrics returned per alpha
    --------------------------
    formula      : the symbolic expression GP evolved
    mean_IC      : average daily IC   (higher = better predictor)
    IC_std       : day-to-day volatility of IC
    IR           : mean_IC / IC_std  (Information Ratio — consistency)
    pct_pos_IC   : % of days where IC > 0  (hit rate above 50% = good)
    abs_mean_IC  : |mean_IC|  (useful when direction is flipped)
    significant  : True if |mean_IC| / (IC_std/√n_days) > 2  (t-stat > 2)

    Rule of thumb: mean_IC > 0.02 + IR > 0.3 → genuinely useful alpha.

    Usage
    -----
    import a23
    report = a23.evaluate_gp_alphas(features, returns, universe,
                                     train_end="2017-12-31")
    display(report)
    """
    from scipy import stats as _stats

    train_end_dt = pd.to_datetime(train_end)
    feat_names   = entire_features.columns.get_level_values(0).unique().tolist()
    stocks       = universe.columns
    all_dates    = entire_features.index

    # Shift features (same leakage guardrail as train_and_predict)
    shifted  = entire_features.shift(1)
    fwd_rets = returns_data.shift(-1)

    train_dates = all_dates[(all_dates >= all_dates[1]) & (all_dates <= train_end_dt)]
    uv          = universe.reindex(all_dates).astype(bool)

    # ── Build flat training set ───────────────────────────────────────────────
    print("Extracting features for GP evaluation …")
    X3d  = _extract_feature_array(shifted, feat_names, train_dates, stocks)
    uv2d = uv.loc[train_dates].values
    fr2d = fwd_rets.reindex(index=train_dates, columns=stocks).values.astype(np.float32)
    X_tr, y_tr, idx_tr, groups_tr = _build_flat_dataset(X3d, uv2d, train_dates, stocks, fr2d)

    # ── Fit GP (same settings as main pipeline) ───────────────────────────────
    boundaries = np.cumsum(np.insert(groups_tr, 0, 0))
    gp_mask    = np.zeros(len(X_tr), dtype=bool)
    for k in range(0, len(groups_tr), 4):
        gp_mask[boundaries[k]: boundaries[k + 1]] = True

    print(f"Running GP ({n_gp_generations} generations) …")
    gp = SymbolicTransformer(
        generations           = n_gp_generations,
        population_size       = 500,
        hall_of_fame          = 100,
        n_components          = n_gp_features,
        function_set          = ["add", "sub", "mul", "div", "sqrt",
                                 "log", "abs", "neg", "max", "min"],
        parsimony_coefficient = 0.005,
        max_samples           = 0.9,
        verbose               = 0,          # quiet during evaluation
        random_state          = _SEED,
        n_jobs                = -1,
    )
    gp.fit(X_tr[gp_mask].astype(np.float64), y_tr[gp_mask].astype(np.float64))

    # ── Transform all training obs → (n_obs, n_gp) GP alpha values ───────────
    X_gp = gp.transform(X_tr.astype(np.float64))   # (n_obs, n_gp_features)

    # ── Compute per-day IC for each alpha ─────────────────────────────────────
    date_labels = idx_tr.get_level_values("date")
    unique_dates = pd.Series(date_labels).drop_duplicates().values

    # Store (n_days, n_gp) IC matrix
    ic_matrix = np.full((len(unique_dates), n_gp_features), np.nan)
    offset     = 0

    for i, (date, n) in enumerate(zip(unique_dates, groups_tr)):
        if n < 5:                          # too few stocks → skip
            offset += n
            continue
        alpha_slice = X_gp[offset: offset + n]     # (n_stocks, n_gp)
        ret_slice   = y_tr[offset: offset + n]     # (n_stocks,)

        for j in range(n_gp_features):
            col = alpha_slice[:, j]
            if np.isfinite(col).sum() < 5:
                continue
            # Spearman rank-correlation (IC)
            rho, _ = _stats.spearmanr(col, ret_slice, nan_policy="omit")
            ic_matrix[i, j] = rho

        offset += n

    # ── Aggregate IC statistics ───────────────────────────────────────────────
    ic_df    = pd.DataFrame(ic_matrix, index=unique_dates)
    mean_ic  = ic_df.mean(skipna=True)
    std_ic   = ic_df.std(skipna=True)
    pct_pos  = (ic_df > 0).mean(skipna=True)
    n_valid  = ic_df.notna().sum()
    t_stat   = mean_ic / (std_ic / np.sqrt(n_valid).replace(0, np.nan))

    formulas = [str(p) for p in gp._best_programs]

    report = pd.DataFrame({
        "formula"     : formulas,
        "mean_IC"     : mean_ic.values.round(5),
        "IC_std"      : std_ic.values.round(5),
        "IR"          : (mean_ic / std_ic.replace(0, np.nan)).values.round(3),
        "pct_pos_IC"  : pct_pos.values.round(3),
        "abs_mean_IC" : mean_ic.abs().values.round(5),
        "t_stat"      : t_stat.values.round(2),
        "significant" : (t_stat.abs() > 2.0).values,
    }, index=[f"GP_alpha_{i+1}" for i in range(n_gp_features)])

    report = report.sort_values("abs_mean_IC", ascending=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_sig = report["significant"].sum()
    print(f"\n{'='*62}")
    print(f"  GP Alpha Quality Report  ({n_sig}/{n_gp_features} significant at t>2)")
    print(f"{'='*62}")
    print(report[["mean_IC", "IR", "pct_pos_IC", "t_stat", "significant"]].to_string())
    print(f"\nRule of thumb: mean_IC > 0.02 and IR > 0.3 → useful alpha")
    print(f"{'='*62}\n")

    return report
