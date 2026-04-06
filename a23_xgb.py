"""
GP + XGBoost Alpha Generation Pipeline
=======================================
Identical data pipeline to a23.py (GP-discovered features, same leakage
guardrails) but replaces LambdaMART with an XGBoost regressor that
directly predicts next-day forward returns.

Key differences vs LambdaMART (a23.py):
  • XGBRegressor targets raw returns, not rank labels.
  • No "group" array needed — samples are i.i.d. from the regressor's POV.
  • Output scores are predicted return values; still ranked cross-sectionally
    to pick long/short.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from gplearn.genetic import SymbolicTransformer

# Re-use the shared helpers from a23.py
from a23 import _extract_feature_array, _build_flat_dataset, _csrank_3d

_SEED = 42


def train_and_predict(
    entire_features: pd.DataFrame,
    returns_data: pd.DataFrame,
    universe: pd.DataFrame,
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
    test_end: str = "2019-12-31",
    n_gp_features: int = 6, 
    n_gp_generations: int = 5, 
    n_xgb_trees: int = 200, 
) -> pd.DataFrame:
    """
    Train GP + XGBoost on data up to train_end.
    Predict forward-return scores for test_start … test_end.

    Returns
    -------
    signal_matrix : pd.DataFrame (n_test_days × n_stocks)
        Predicted next-day return for each stock.  NaN = not in universe.
        Higher = model expects this stock to outperform.
    """
    train_end_dt  = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt   = pd.to_datetime(test_end)

    feat_names = entire_features.columns.get_level_values(0).unique().tolist()
    stocks     = universe.columns
    all_dates  = entire_features.index

    # ── 1. Shift features by 1 day (no look-ahead) ───────────────────────────
    shifted  = entire_features.shift(1)
    fwd_rets = returns_data.shift(-1)          # label at T = return_{T→T+1}

    # ── 2. Date splits ────────────────────────────────────────────────────────
    train_dates = all_dates[(all_dates >= all_dates[1]) & (all_dates <= train_end_dt)]
    test_dates  = all_dates[(all_dates >= test_start_dt) & (all_dates <= test_end_dt)]

    uv = universe.reindex(all_dates).astype(bool)

    # ── 3. Build training arrays ──────────────────────────────────────────────
    print("[1/5] Extracting training features …")
    X3d  = _extract_feature_array(shifted, feat_names, train_dates, stocks)
    X3d  = _csrank_3d(X3d)   # CSRank: cross-sectional normalise to [0,1]
    uv2d = uv.loc[train_dates].values
    fr2d = fwd_rets.reindex(index=train_dates, columns=stocks).values.astype(np.float32)

    print("[2/5] Flattening …")
    X_tr, y_tr, idx_tr, _ = _build_flat_dataset(X3d, uv2d, train_dates, stocks, fr2d)
    print(f"      Training obs: {len(X_tr):,}  |  features: {X_tr.shape[1]}")

    # ── 4. Genetic Programming ────────────────────────────────────────────────
    print(f"[3/5] Genetic Programming ({n_gp_generations} generations) …")
    # Subsample every 4th query (date) to keep GP fast
    groups_approx = np.array([
        (idx_tr.get_level_values(0) == d).sum()
        for d in pd.Series(idx_tr.get_level_values(0)).drop_duplicates()
    ], dtype=np.int32)
    boundaries = np.cumsum(np.insert(groups_approx, 0, 0))
    gp_mask = np.zeros(len(X_tr), dtype=bool)
    for k in range(0, len(groups_approx), 4):
        gp_mask[boundaries[k]: boundaries[k + 1]] = True

    gp = SymbolicTransformer(
        generations           = n_gp_generations,
        population_size       = 500,
        hall_of_fame          = 100,
        n_components          = n_gp_features,
        function_set          = ["add", "sub", "mul", "div", "sqrt",
                                 "log", "abs", "neg", "max", "min"],
        parsimony_coefficient = 0.005,
        max_samples           = 0.9,
        verbose               = 1,
        random_state          = _SEED,
        n_jobs                = -1,
    )

    gp.fit(X_tr[gp_mask].astype(np.float64), y_tr[gp_mask].astype(np.float64))
    print(f"      GP done.  Best programs: {len(gp._best_programs)}")

    # ── 5. Augment & train XGBoost ────────────────────────────────────────────
    print("[4/5] Training XGBoost regressor …")
    X_gp_tr  = gp.transform(X_tr.astype(np.float64)).astype(np.float32)
    X_aug_tr = np.hstack([X_tr, X_gp_tr])                     # (n_obs, 22+n_gp)

    # Winsorise targets: clip at 1st / 99th percentile to reduce label noise
    lo, hi = np.percentile(y_tr, [1, 99])
    y_clipped = np.clip(y_tr, lo, hi).astype(np.float32)

    regressor = xgb.XGBRegressor(
        objective        = "reg:squarederror",
        n_estimators     = n_xgb_trees,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.8,
        reg_lambda       = 1.8,
        random_state     = _SEED,
        verbosity        = 0,
        n_jobs           = -1,
    )

    regressor.fit(X_aug_tr, y_clipped)
    print("      XGBoost training complete.")

    import shap
    print("      Calculating SHAP importance values…")
    explainer = shap.Explainer(regressor)
    shap_values = explainer(X_aug_tr)

    importance = np.abs(shap_values.values).mean(axis=0)
    
    # Optional: Attach feature names to the importance array for readability
    base_feats = entire_features.columns.get_level_values(0).unique().tolist()
    gp_feats = [f"gp_{i}" for i in range(n_gp_features)]
    feature_names = base_feats + gp_feats
    
    imp_series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 Most Important Features (SHAP):")
    print(imp_series.head(15).to_string())
    print("\n")
    # ── 6. Predict on test period ─────────────────────────────────────────────
    print("[5/5] Generating test-period signals …")
    X3d_te  = _extract_feature_array(shifted, feat_names, test_dates, stocks)
    X3d_te  = _csrank_3d(X3d_te)   # same CSRank transform as training
    uv2d_te = uv.loc[test_dates].values
    X_te, _, idx_te, _ = _build_flat_dataset(X3d_te, uv2d_te, test_dates, stocks)

    X_te     = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    X_gp_te  = gp.transform(X_te.astype(np.float64)).astype(np.float32)
    X_aug_te = np.hstack([X_te, X_gp_te])

    scores = regressor.predict(X_aug_te).astype(np.float32)

    score_series = pd.Series(scores, index=idx_te, name="score")
    signal_wide  = score_series.unstack(level="stock")
    signal_wide  = signal_wide.reindex(index=test_dates, columns=stocks)
    signal_wide  = signal_wide.where(uv.loc[test_dates], np.nan)

    print("      Done!")
    return signal_wide
