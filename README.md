# SBER 5-Minute Intraday Research Pipeline

> **Research notebook:** XGBoost binary classifier with probability calibration on Finam OHLCV 5-minute candles for SBER (Sberbank). Pure research / backtesting environment — not production trading code.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Validation Scheme](#validation-scheme)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Known Limitations & Weaknesses](#known-limitations--weaknesses)
- [Roadmap](#roadmap)

---

## Overview

This pipeline researches the feasibility of predicting the **direction of the next 5-minute candle** for SBER (Sberbank, MOEX) using classical technical features and a gradient-boosted tree classifier.

**Target variable:**
```
target_is_green_next = 1  if next 5m candle closes above its open (bullish)
                      0  otherwise (bearish / doji)
```

**Key design constraint:** All features are computed using **only current-bar-close or past-bar data**. No look-ahead bias by design.

---

## Dataset

| Property | Value |
|---|---|
| Ticker | SBER (Sberbank, MOEX) |
| Timeframe | 5-minute candles |
| Source | Finam (CSV export) |
| Columns | TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL |
| Total rows | ~41 657 bars |
| Date range | From 2024-01-03 |
| Class balance | Red (0): 53% · Green (1): 47% |

The raw CSV is **not included** in this repository (proprietary Finam data). Place your file at `./yearresult.csv` before running.

---

## Feature Engineering

Features are grouped into four categories, all computed without look-ahead:

### 1. Domain / Price-Action Features
| Feature | Description |
|---|---|
| `candle_body` | Close − Open |
| `body_abs` | abs(Close − Open) |
| `upper_wick`, `lower_wick` | Wick sizes |
| `body_to_range` | Body / (High − Low) |
| `close_pos_in_range` | (Close − Low) / (High − Low) |
| `direction` | sign(candle_body) |
| `is_green`, `is_red`, `is_doji_like` | Binary candle-type flags |
| `ret1` | 1-bar return |
| `gap_from_prev_close` | Open / prev_close − 1 |
| `body_x_vol`, `signed_vol` | Volume-weighted body signals |
| `money_flow_proxy` | (Close−Low − High−Close) / Range × Volume |

### 2. Rolling Window Features (windows: 6, 12, 24 bars)
- Rolling returns, range statistics (mean / std / z-score)
- Rolling MA, close vs MA, close z-score
- Volume MA ratio

### 3. Calendar Features
- Hour, minute, day-of-week (raw + cyclical sin/cos encoding)
- Minutes from session open
- Flags: `is_opening_30m`, `is_first_hour`, `is_first_bar_of_day`
- Bar index within day (`bar_in_day`)

### 4. Technical Indicators (via `pandas-ta`)
- EMA 10/20, SMA 20
- RSI 14
- ATR 14, NATR 14
- MACD (12/26/9)
- Bollinger Bands 20 (+ `close_pos_in_bbands`)
- Stochastic (14/3/3)
- OBV
- Price vs EMA/SMA ratios

**Total feature columns after lag embedding (nin=5):** ~90+ features × 5 lags.

---

## Model Architecture

```
Raw CSV
  └─► prepare_ohlcv_dataframe()   # parse, sort, type-cast
        └─► add_domain_features()
              └─► add_rolling_features()   # windows [6, 12, 24]
                    └─► add_calendar_features()
                          └─► add_ta_features()   # pandas-ta
                                └─► add_target()   # shift(-1)
                                      └─► build_X_y_for_model()  # series_to_supervised(nin=5)
                                            └─► XGBClassifier  (baseline)
                                                  └─► CalibratedClassifierCV (Platt sigmoid)
```

**XGBoost hyperparameters (default research config):**
```python
XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
```

**Calibration:** Platt scaling via `CalibratedClassifierCV(method='sigmoid', cv='prefit')` on a dedicated calibration split.

---

## Validation Scheme

| Split | Purpose | Approx. share |
|---|---|---|
| Train | Fit XGBoost | 60% |
| Valid | Evaluate & tune | 20% |
| Calib | Fit Platt scaler | 20% |

> ⚠️ **Walk-forward / time-series cross-validation is not yet implemented.** The current split is a simple chronological hold-out. This is a known limitation — see [Known Limitations](#known-limitations--weaknesses).

---

## Results

### Metrics Summary (XGBoost vs Calibrated)

| Split | XGB Acc | XGB AUC | XGB LogLoss | XGB Brier | Cal Acc | Cal AUC | Cal LogLoss | Cal Brier |
|---|---|---|---|---|---|---|---|---|
| Train | 0.5895 | 0.6276 | 0.6825 | 0.2447 | 0.5729 | 0.6276 | 0.6786 | 0.2428 |
| Valid | 0.5197 | 0.5282 | 0.6909 | 0.2489 | 0.5343 | 0.5282 | 0.6886 | 0.2478 |
| Calib | 0.5262 | 0.5326 | 0.6908 | 0.2488 | 0.5380 | 0.5326 | 0.6892 | 0.2481 |

**Key observations:**
- AUC on validation ≈ 0.528 — marginal edge above random (0.5), but modest
- Train/Valid gap indicates mild overfitting
- Platt calibration slightly reduces LogLoss and Brier score
- No transaction costs or slippage are modeled in these metrics

---

## Project Structure

```
sber-intraday-pipeline/
├── sber_intraday_pipeline-2.ipynb   # Main research notebook
├── README.md                        # This file
└── yearresult.csv                   # Raw data (NOT included, add manually)
```

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost pandas-ta matplotlib seaborn packaging
```

| Package | Version tested |
|---|---|
| Python | 3.8.5 |
| pandas | 1.4.2 |
| numpy | 1.22.3 |
| scikit-learn | 1.1.3 |
| xgboost | 1.6.1 |
| pandas-ta | latest |

---

## Usage

1. Export 5-minute OHLCV data for SBER from Finam and save as `yearresult.csv` in the repo root.
2. Open `sber_intraday_pipeline-2.ipynb` in Jupyter.
3. Run all cells sequentially (Cell 1 → Cell 11).

**Cell map:**

| Cell | Purpose |
|---|---|
| 1 | Imports & display settings |
| 2 | Helper: `series_to_supervised()` |
| 3 | Feature engineering functions |
| 4 | Master pipeline builder |
| 5 | Load data & build feature DataFrame |
| 6 | EDA — feature distributions |
| 7 | Train/Valid/Calib split |
| 8 | XGBoost baseline training & evaluation |
| 9 | Probability calibration (Platt scaling) |
| 10 | Metrics summary table |
| 11 | Feature importance (permutation + gain) |

---

## Known Limitations & Weaknesses

This is a **research prototype**. Before treating any signal as tradeable, the following issues must be addressed:

### 1. 🚨 Single Chronological Split (No Walk-Forward CV)
The model is validated on one fixed hold-out period. AUC ~0.528 may be inflated or deflated relative to out-of-sample performance across different market regimes. **Recommendation:** implement `TimeSeriesSplit` or a rolling walk-forward backtest.

### 2. 🚨 No Transaction Cost / Slippage Modeling
All metrics are computed on raw predictions. Real MOEX trades incur brokerage commissions (~0.04–0.06%), exchange fees, and slippage. At 5-minute frequency, round-trip costs can easily consume a 50-52% win-rate edge. **Recommendation:** model net P&L with realistic cost assumptions before declaring any positive expectancy.

### 3. ⚠️ Single Ticker / Survivorship Bias
Only SBER is analyzed. Results may not generalize to other liquid MOEX names. SBER itself is a survivorship-bias-free choice (it has been continuously traded), but the strategy has not been tested on a broader universe.

### 4. ⚠️ No Regime Detection
The model treats the entire 2024 time series as one stationary regime. In reality, SBER exhibits distinct trend, range, and high-volatility regimes that may require separate models or filters.

### 5. ⚠️ Feature Collinearity
Many features (EMA10, EMA20, SMA20, close_vs_ma12, etc.) are highly correlated. XGBoost handles this implicitly, but permutation importance may be noisy. **Recommendation:** apply VIF analysis or a dimensionality reduction step.

### 6. ⚠️ `warnings.filterwarnings('ignore')`
All warnings are silenced globally. This can hide important deprecation notices (e.g., the `base_estimator` → `estimator` rename in scikit-learn 1.2). **Recommendation:** use targeted warning filters.

### 7. ⚠️ Hardcoded Data Path
`DATAPATH = './yearresult.csv'` is hardcoded. **Recommendation:** move to a config file or CLI argument.

### 8. ⚠️ No Reproducibility Guard on Data Shuffling
The pipeline relies on chronological ordering, but there is no explicit assertion that the DataFrame is sorted before splitting. **Recommendation:** add `assert df.index.is_monotonic_increasing` after datetime parsing.

### 9. ⚠️ `nin=5` Lag Embedding is Arbitrary
The `series_to_supervised(nin=5)` creates 5-step lag features without ablation. Optimal lag length is unknown. **Recommendation:** hyperparameter-search over `nin` ∈ {1, 3, 5, 10, 20}.

### 10. ℹ️ Python 3.8 / Old Library Versions
The notebook was developed on Python 3.8.5 with pandas 1.4.2. Newer versions (pandas 2.x, scikit-learn 1.4+) may require minor API updates.

---

## Roadmap

- [ ] Walk-forward cross-validation (`TimeSeriesSplit`)
- [ ] Net P&L backtest with commission + slippage model
- [ ] Regime detection (HMM or volatility-regime filter)
- [ ] Hyperparameter search (Optuna) with purged CV
- [ ] Multi-ticker universe test
- [ ] Feature selection (Boruta / SHAP-based)
- [ ] SHAP values for interpretability
- [ ] Modularize into Python package (src layout)
- [ ] CI/CD with GitHub Actions + pytest

---

## License

MIT License. Research use only — not financial advice.
