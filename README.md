# Machine Learning Quantitative Trading Strategy Pipeline

## Overview

This pipeline implements a robust, multi-stage quantitative trading strategy that combines **feature engineering**, **genetic programming-based alpha discovery**, and **gradient boosted decision trees (XGBoost)**.

The design emphasizes:

* Cross-sectional robustness
* Prevention of overfitting
* Discovery of nonlinear predictive signals
* Interpretability of model outputs

---

## 1. Feature Engineering (Baseline Signal Processing)

**File:** `feature_engineering_ogog.py`

This stage constructs a stable and expressive feature set from a curated group of ~15 technical indicators, including:

* MACD
* RSI
* CMO (Chande Momentum Oscillator)
* Trend channels
* ATR (Average True Range)
* OBV (On-Balance Volume)

### Key Transformations

#### Volatility-Adjusted Signals

All major signals are normalized by their respective trailing volatility:

* `trend_vol_adj = trend_20_60 / vol_60`
* `macd_vol`, `rsi_vol`, etc.

This ensures:

* Cross-sectional comparability
* Stability across varying market regimes

---

#### Meta Momentum Signal

A composite momentum feature is constructed by:

* Cross-sectionally ranking multiple indicators (RSI, MACD, CMO)
* Averaging their ranks

This produces a **robust, noise-resistant momentum signal**.

---

#### Regime Identification

Market regimes are captured using:

* HAR-style volatility features (daily, weekly, monthly)
* A continuous regime score derived from:

  * 1-year rolling Z-score
  * Logistic sigmoid transformation

This replaces rigid binary regimes with **smooth probabilistic transitions**.

---

## 2. ML Alpha Discovery & Tree Boosting

**File:** `a23_xgb.py`

This stage combines **automated feature discovery** with **supervised learning**.

---

### Genetic Programming (GP)

Using `gplearn`, the pipeline evolves new features by:

* Generating random mathematical expressions from base features
* Iteratively selecting, mutating, and recombining them
* Retaining high-performing expressions

The resulting symbolic expressions are:

* Explicit (interpretable)
* Highly nonlinear
* Appended as new features

---

### XGBoost Prediction Engine

XGBoost is trained to predict a forward-looking target:

* `compound_target_5d` (e.g., 5-day forward return)

Key design choices:

* Inputs are cross-sectionally ranked → model focuses on **relative performance**
* Reduces sensitivity to:

  * scale differences
  * extreme outliers
  * market-wide shocks

---

### Model Explainability (SHAP)

The pipeline integrates SHAP to:

* Quantify feature importance
* Explain model decisions
* Identify dominant alpha drivers

---

## 3. Supporting Pipeline Infrastructure

**File:** `a23.py`

This module provides critical infrastructure for robustness and correctness.

---

### Leakage Prevention & Target Alignment

* Targets are forward-shifted (`compound_target_5d`)
* Ensures strict separation between:

  * input features (present)
  * prediction targets (future)

This eliminates **lookahead bias**.

---

### Cross-Sectional Ranking (CSRank)

The `_csrank_3d` transformation:

* Converts raw features into ranks within each time slice
* Normalizes values into `[0, 1]`

Benefits:

* Removes scale dependence
* Improves stability of tree-based models
* Forces model to learn **relative alpha signals**

## 4. Portfolio Construction & Benchmark Strategy

**Function:** `generate_portfolio_vectorized(...)`

This module implements a **stateful long-short portfolio construction strategy** that converts model predictions into tradable positions while minimizing turnover and instability.

---

### Core Concept

The function builds a **market-neutral portfolio**:

* Top-ranked stocks → **Long positions**
* Bottom-ranked stocks → **Short positions**

Using:

* `ml_scores` (model predictions)
* Daily tradable universe (`today_universe`)

---

### Portfolio Structure

* **40 long positions**
* **40 short positions**
* Equal-weight allocation:

  * +0.5 total capital (long side)
  * -0.5 total capital (short side)

---

### Rank-Based Selection

Stocks are ranked daily based on predicted returns:

* Highest scores → strongest expected outperformers
* Lowest scores → weakest expected performers

---

### Hysteresis (Sticky Weights)

To reduce unnecessary trading, the strategy uses **rank hysteresis**:

* Instead of reselecting top 40 stocks every day
* A **buffer zone** is introduced (e.g., top 67 candidates)

#### Mechanism:

* Previously held positions are **retained** if they remain within the buffer
* Only underperforming positions are replaced

This results in:

* Lower turnover
* Reduced transaction costs
* More stable portfolios

---

### Rebalancing Logic

* Portfolio is updated periodically (`rebalance_every`)
* Forced rebalancing occurs if:

  * Portfolio is uninitialized
  * Stocks exit the tradable universe

Otherwise:

* Positions are **carried forward unchanged**

---

### Safety Handling

* Handles missing predictions
* Skips invalid days
* Ensures no crashes due to universe mismatch

---

### Design Philosophy

This component emphasizes:

* **Stability over reactivity**
* **Relative ranking over absolute predictions**
* **Turnover minimization via hysteresis**

---


## Key Design Philosophy

The pipeline is built on three principles:

1. **Relative over Absolute**

   * Cross-sectional ranking ensures signals are comparable across assets

2. **Simplicity + Expressiveness**

   * Start with clean features
   * Expand using GP only where useful

3. **Robustness over Overfitting**

   * Volatility normalization
   * Rank transformations
   * Leakage prevention

---

## Summary Workflow

1. Generate base technical indicators + volatility indicators
2. Normalize and rank features cross-sectionally
3. Use Genetic Programming to create nonlinear features
4. Train XGBoost on engineered + GP features
5. Interpret results using SHAP
6. Convert predictions to portfolio weights using rank-hysteresis

---

## Potential Extensions

* Multi-horizon targets (1d, 10d, 20d)
* Ensemble of GP models
* Dynamic feature selection
* Integration with reinforcement learning for execution

---

## Final Note

This pipeline prioritizes **generalization, interpretability, and robustness**, making it suitable for real-world quantitative trading applications where stability and explainability are critical.
