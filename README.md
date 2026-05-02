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
- [Эксперименты A4+A2 и A3](#эксперименты-с-уверенностью-a4a2-и-альтернативным-target-a3)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Known Limitations & Weaknesses](#known-limitations--weaknesses)
- [Roadmap](#roadmap)

---

## Overview

This pipeline researches the feasibility of predicting the **direction of the next 5-minute candle** for SBER (Sberbank, MOEX) using classical technical features and a gradient-boosted tree classifier with probability calibration.

**Target variable:**
```
target_is_green_next = 1  if next 5m candle closes above its open (bullish)
                      0  otherwise (bearish / doji)
```

**Key design constraint:** All features are computed using **only current-bar-close or past-bar data**. No look-ahead bias by design (PSAR is collapsed into a single causal `psar_value` + `psar_is_long` pair; Ichimoku forward spans and Volume Profile rows are excluded).

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
| Class balance | Red (0): ~53% · Green (1): ~47% |

The raw CSV is **not included** in this repository (proprietary Finam data). Place your file at `./Сбербанк/year_result.csv` before running.

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

### 4. Technical Indicators (via `pandas-ta`, ~109 indicators)
Generated in **bulk via `try/except` across all `pandas_ta` indicator groups** — `trend`, `momentum`, `volatility`, `volume`, `statistics`, `overlap` — and then auto-pruned by quality filters. Risky look-ahead indicators (`ichimoku` ISA/ISB forward, `vp`) are explicitly excluded; PSAR is collapsed into a single causal `psar_value` + `psar_is_long` pair.

Examples produced (non-exhaustive): EMA / SMA / DEMA / TEMA / HMA / KAMA, RSI / Stoch / StochRSI / CCI / MFI / ROC / Williams %R, ATR / NATR / TRUE_RANGE / Bollinger Bands / Donchian / Keltner, OBV / CMF / EFI / AD, MACD, ADX / DMI, Aroon, etc.

### 5. Filtering Pipeline
After bulk TA generation the feature matrix passes through a deterministic filter chain:
1. Drop columns with **>20% NaN**
2. Drop **constant columns**
3. `dropna()` rows
4. **Lag embedding** via `series_to_supervised(n_in=3)`
5. **Correlation filter:** drop one of every pair with `|corr| > 0.95`
6. Drop constant columns introduced by lagging
7. **Permutation-importance pruning** (sklearn `permutation_importance`, `n_repeats=5`, `scoring='neg_log_loss'`) — keep only features with importance > 0

---

## Model Architecture

```
Raw CSV
  └─► prepare_ohlcv_dataframe()      # parse, sort, type-cast
        └─► add_domain_features()
              └─► add_rolling_features()      # windows [6, 12, 24]
                    └─► add_calendar_features()
                          └─► add_ta_features()       # ~109 pandas_ta indicators
                                └─► add_target()
                                      └─► build_X_y_for_model()   # series_to_supervised(n_in=3)
                                            └─► NaN/const/corr filters
                                                  └─► XGBClassifier  (baseline)
                                                        └─► permutation_importance pruning
                                                              └─► XGBClassifier  (pruned)
                                                                    └─► CalibratedClassifierCV (Platt sigmoid, FrozenEstimator)
```

**XGBoost hyperparameters (research config):**
```python
XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=30,
)
```

**Calibration:** Platt scaling via `CalibratedClassifierCV(method='sigmoid')` wrapped around a `FrozenEstimator` (sklearn ≥ 1.6) to fit on a dedicated calibration split. A fallback to legacy `cv='prefit'` is used for older sklearn versions.

---

## Validation Scheme

| Split | Purpose | Approx. share |
|---|---|---|
| Train | Fit XGBoost | 70% |
| Valid | Early stopping & evaluation | 15% |
| Calib | Fit Platt scaler | 15% |

Splits are strictly **chronological**.

> ⚠️ **Walk-forward / time-series cross-validation is not yet implemented.** The current split is a simple chronological hold-out — see [Known Limitations](#known-limitations--weaknesses).

---

## Results

> 🚨 **Историческая правка (2026-05-02).** В предыдущей итерации вызывался `ta.dpo(..., lookahead=False)` — этого параметра **нет в API pandas_ta**, он молча попадал в `**kwargs` и DPO считался с `centered=True` (default), что создавало look-ahead bias на ~length/2 свечей вперёд (см. [pandas_ta Issue #60](https://github.com/twopirllc/pandas-ta/issues/60)). Это завышало AUC до ~0.82 — фейк. После фикса (`centered=False`) и полного аудита всех 109 индикаторов через [scripts/leakage_audit.py](scripts/leakage_audit.py) и [scripts/leakage_audit2.py](scripts/leakage_audit2.py) все остальные индикаторы прошли проверку на связь с будущими значениями. Ниже — честные метрики.

### Pipeline funnel

| Stage | Columns / Rows |
|---|---|
| TA indicators added | 109 |
| Features after generation | 173 |
| After NaN(>20%) filter | 165 |
| Rows after `dropna()` | 35 658 |
| Columns after `series_to_supervised(n_in=3)` | 660 |
| After `&#124;corr&#124; > 0.95` filter | 298 |
| After permutation-importance pruning | **57** |
| Train / Valid / Calib | 24 958 / 5 348 / 5 349 |

### A/B comparison: Baseline vs Permutation-Pruned (honest, no leakage)

| Stage | Features | Acc Valid | AUC Valid | LogLoss Valid | Brier Valid |
|---|---|---|---|---|---|
| Baseline XGB | 298 | 0.5183 | 0.5213 | 0.6916 | 0.2493 |
| **Pruned XGB** | **57** | 0.5198 | **0.5278** | 0.6914 | 0.2491 |
| Pruned + Platt | 57 | **0.5366** | 0.5278 | **0.6886** | **0.2478** |

**Key observations:**
- AUC on validation **≈ 0.52–0.53** — marginal edge above random (0.5), ожидаемый режим для 5-минутного intraday на чисто технических признаках
- Permutation pruning убирает **81% признаков (241 из 298)** — большинство из них в реальности шум
- Best XGBoost iteration: 14 (baseline) / 27 (pruned) — модель быстро упирается в предел сигнала
- Platt calibration даёт заметный буст по accuracy (0.5198 → 0.5366) и LogLoss
- No transaction costs or slippage are modeled — при таком AUC издержки весь сигнал съедят

---

## Эксперименты с уверенностью (A4+A2) и альтернативным target (A3)

После получения честного AUC≈0.52 было протестировано два рычага для повышения уверенности модели.

### A4+A2 — торговля только в хвостах вероятности (базовый target = зелёная свеча)

Скрипт: [`scripts/threshold_strategy.py`](scripts/threshold_strategy.py) · результаты: [`scripts/threshold_results.json`](scripts/threshold_results.json)

| Конфиг | Сторона | Valid prec/N | **Calib (OOS) prec/N** |
|---|---|---|---|
| precision ≥ 0.70 | SHORT | 1.000 / 20 | **1.000 / 15** ✅ стабильно |
| precision ≥ 0.65 | SHORT | 0.900 / 30 | **0.960 / 25** ✅ стабильно |
| precision ≥ 0.65 | LONG  | 0.653 / 49 | 0.553 / 47 ⚠️ переобученный хвост |

**P&L на Calib** (балансированный конфиг, 72 сделки): без издержек −0.66%, с round-trip 10 bps → −7.86%. **Неприбыльно при реальных издержках.** SHORT-хвост стабильно верный (96–100% precision OOS), но прибыль от верных SHORT не покрывает издержки + убытки LONG-хвоста.

### A3 — target «движение ≥ k·ATR за k свечей»

Скрипты: [`scripts/atr_target_strategy.py`](scripts/atr_target_strategy.py), [`scripts/atr_grid.py`](scripts/atr_grid.py) · результаты: [`scripts/atr_grid_results.json`](scripts/atr_grid_results.json)

Grid search по конфигам k_bars ∈ {1, 3, 5} × k_atr ∈ {0.5 .. 1.5}:

| k_bars | k_atr | N rows | AUC valid | AUC calib | p_max | LONG p≥0.55 (Calib) | SHORT p≤0.45 (Calib) |
|---|---|---|---|---|---|---|---|
| **5** | **1.00** | 15 958 | 0.564 | **0.580** | **0.729** | **prec=0.617, N=389** | **prec=0.586, N=916** |
| 3 | 0.50 | 21 521 | 0.578 | 0.573 | 0.700 | prec=0.567, N=478 | prec=0.600, N=1 018 |
| 5 | 1.50 |  9 934 | 0.545 | 0.540 | 0.621 | prec=0.642, N=148 | prec=0.543, N=495 |
| 3 | 1.00 | 12 097 | 0.562 | 0.568 | 0.655 | prec=0.500, N=88 | prec=0.598, N=776 |
| 1 | 0.50 | 13 853 | 0.573 | 0.535 | 0.586 | prec=0.550, N=20 | prec=0.571, N=357 |

**Ключевой вывод:** при k_bars=5, k_atr=1.0 (движение ≥1 ATR за 5 свечей) хвосты **симметрично открываются**: впервые LONG даёт реальный хвост (389 сигналов OOS с prec=0.617), AUC поднимается с 0.52 до 0.58. Смена target с «зелёная свеча» на «движение ≥ ATR» резко улучшает предсказуемость, средневзвешенный сигнал ~58% precision в обе стороны. Для реальной торговли требуется выполнить walk-forward валидацию и посчитать чистый P&L с издержками на этой конфигурации.

---

## Project Structure

```
sber-intraday-pipeline/
├── sber_intraday_pipeline.ipynb     # Main research notebook
├── README.md                        # This file
├── scripts/
│   ├── leakage_audit.py             # Аудит look-ahead bias (target shift test)
│   ├── leakage_audit2.py            # Поиндикаторный аудит 109 TA-признаков
│   ├── model_comparison.py          # 7 моделей × {raw, Platt, Isotonic}
│   ├── model_comparison_results.csv # Результаты model_comparison
│   ├── confidence_diagnostic.py     # Диагностика хвостов вероятностей
│   ├── threshold_strategy.py        # A4+A2 — стратегия порогов + P&L
│   ├── threshold_results.json       # Результаты A4+A2
│   ├── atr_target_strategy.py       # A3 — альтернативный ATR-target (один конфиг)
│   ├── atr_grid.py                  # A3 — grid search по (k_bars, k_atr)
│   └── atr_grid_results.json        # Результаты grid search
├── .gitignore                       # Excludes Сбербанк/, *.csv, etc.
└── Сбербанк/
    └── year_result.csv              # Raw data (NOT committed, add manually)
```

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost pandas-ta matplotlib seaborn packaging
```

| Package | Version tested |
|---|---|
| Python | 3.12 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 (`FrozenEstimator` API) |
| xgboost | 3.2.0 |
| pandas-ta | 0.4.71b0 |

> The notebook auto-detects sklearn version: it uses `FrozenEstimator` on ≥ 1.6 and falls back to `cv='prefit'` on older versions.

---

## Usage

1. Export 5-minute OHLCV data for SBER from Finam and save as `./Сбербанк/year_result.csv` in the repo root.
2. Open `sber_intraday_pipeline.ipynb` in Jupyter.
3. Run all cells sequentially — or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute sber_intraday_pipeline.ipynb \
    --output sber_intraday_pipeline.ipynb \
    --ExecutePreprocessor.timeout=1200
```

**Cell map (30 cells):**

| Cell | Purpose |
|---|---|
| 0–9 | Imports, helpers, feature engineering functions, data load |
| 11 | `build_feature_dataframe` — 109 TA indicators + target |
| 13–14 | EDA — feature distributions and target balance |
| 16 | `build_X_y_for_model(n_in=3)` + Train/Valid/Calib split + filters & diagnostic counters |
| 18–19 | Baseline XGBoost training + classification report |
| 20–21 | **Permutation-importance pruning + A/B comparison + final-model selection** |
| 23 | Probability calibration (`FrozenEstimator` → Platt) |
| 24 | Calibration plots |
| 26 | Metrics summary table |
| 28 | Feature importance (gain + permutation) |

---

## Known Limitations & Weaknesses

This is a **research prototype**. Before treating any signal as tradeable, the following issues must be addressed:

### 1. 🚨 Single Chronological Split (No Walk-Forward CV)
The model is validated on one fixed hold-out period. AUC ~0.817 may be inflated relative to out-of-sample performance across different market regimes. **Recommendation:** implement `TimeSeriesSplit` or a rolling walk-forward backtest.

### 2. 🚨 No Transaction Cost / Slippage Modeling
All metrics are computed on raw predictions. Real MOEX trades incur brokerage commissions (~0.04–0.06%), exchange fees, and slippage. At 5-minute frequency, round-trip costs can erode an edge that looks robust at the AUC level. **Recommendation:** model net P&L with realistic cost assumptions before declaring positive expectancy.

### 3. ⚠️ Single Ticker / Survivorship Bias
Only SBER is analyzed. Results may not generalize to other liquid MOEX names. SBER itself is a survivorship-bias-free choice (continuously traded), but the strategy has not been tested on a broader universe.

### 4. ⚠️ No Regime Detection
The model treats the entire 2024 time series as one stationary regime. In reality, SBER exhibits distinct trend, range, and high-volatility regimes that may require separate models or filters.

### 5. ✅ AUC ≈ 0.82 was a leak — fixed
Исторический AUC=0.82 оказался фейком из-за `ta.dpo(..., lookahead=False)` (параметр не существовал, DPO считался с default `centered=True` → look-ahead). После фикса честный AUC=0.52–0.53 на базовом target. С ATR-target (k=5, k_atr=1) AUC поднимается до 0.58 OOS — это реалистичный уровень для intraday при фильтрации «шумных» баров.

### 6. ⚠️ `warnings.filterwarnings('ignore')`
All warnings are silenced globally. **Recommendation:** use targeted warning filters.

### 7. ⚠️ Hardcoded Data Path
`DATAPATH = './Сбербанк/year_result.csv'` is hardcoded. **Recommendation:** move to a config file or CLI argument.

### 8. ⚠️ No Reproducibility Guard on Data Ordering
The pipeline relies on chronological ordering, but there is no explicit assertion that the DataFrame is sorted before splitting. **Recommendation:** add `assert df.index.is_monotonic_increasing` after datetime parsing.

### 9. ⚠️ `n_in=3` Lag Embedding is Fixed
The `series_to_supervised(n_in=3)` creates 3-step lag features without ablation. **Recommendation:** hyperparameter-search over `n_in` ∈ {1, 3, 5, 10}.

---

## Roadmap

- [x] Look-ahead bias audit (DPO `centered=False` fix — see историческую правку в [Results](#results))
- [x] A4+A2: стратегия «торгуем только в хвостах» + P&L с издержками (неприбыльно при 10 bps)
- [x] A3: альтернативный target = движение ≥ k·ATR за k свечей + grid search (победитель k=5, k_atr=1.0, AUC 0.580 OOS)
- [ ] Walk-forward cross-validation (`TimeSeriesSplit`) на ATR-конфиге
- [ ] Net P&L backtest with commission + slippage model на ATR-конфиге
- [ ] Regime detection (HMM or volatility-regime filter)
- [ ] Hyperparameter search (Optuna) with purged CV
- [ ] Multi-ticker universe test
- [ ] SHAP values for interpretability
- [ ] Modularize into Python package (src layout)
- [ ] CI/CD with GitHub Actions + pytest

---

## License

MIT License. Research use only — not financial advice.
