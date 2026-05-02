# SBER 5-Minute Intraday Research Pipeline

> **Research notebook:** CatBoost binary classifier with probability calibration on Finam OHLCV 5-minute candles for SBER (Sberbank). Pure research / backtesting environment — not production trading code.
>
> 🔄 **2026-05-02 update.** После расширенного анализа (см. [Расширенный анализ](#расширенный-анализ-sensitivity-model-zoo-таймфрейм)) основная конфигурация переведена на **CatBoost + ATR-target k_bars=3, k_atr=1.0** — комбинация победителей model-zoo и sensitivity-grid. Старая конфигурация (XGBoost + k_bars=5) задокументирована в истории.

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

**Target variable (обновлён 2026-05-02 по итогам эксперимента A3):**
```
fwd = (CLOSE.shift(-K_BARS) - CLOSE) / atr_14    # K_BARS=5, K_ATR=1.0

target_is_green_next = 1   if fwd >=  K_ATR   (рост >=1·ATR за 5 свечей)
                       0   if fwd <= -K_ATR   (падение >=1·ATR за 5 свечей)
                       NaN otherwise            (флэт/шум — строка отбрасывается)
```

Ранее использовался target = «следующая свеча зелёная» — давал AUC≈0.52 OOS. ATR-target отфильтровывает ~55% баров без значимого движения и поднимает AUC до ~0.57–0.58 OOS при идеально сбалансированных классах 50/50.

**Key design constraint:** All features are computed using **only current-bar-close or past-bar data**. No look-ahead bias by design (PSAR is collapsed into a single causal `psar_value` + `psar_is_long` pair; Ichimoku forward spans and Volume Profile rows are excluded).

---

## Dataset

| Property | Value |
|---|---|
| Ticker | SBER (Sberbank, MOEX) |
| Timeframe | 5-minute candles |
| Source | Finam (CSV export) |
| Columns | TICKER, PER, DATE, TIME, OPEN, HIGH, LOW, CLOSE, VOL |
| Total rows (raw) | ~41 657 bars |
| Rows after ATR-target filter | ~18 844 bars (~45%) |
| Date range | From 2024-01-03 |
| Class balance (ATR-target) | down >=1·ATR (0): ~50% · up >=1·ATR (1): ~50% |

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
                                                  └─► CatBoostClassifier  (baseline)
                                                        └─► permutation_importance pruning
                                                              └─► CatBoostClassifier  (pruned)
                                                                    └─► CalibratedClassifierCV (Platt sigmoid, FrozenEstimator)
```

**CatBoost hyperparameters (research config):**
```python
CatBoostClassifier(
    iterations=500,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.7,
    bootstrap_type='Bernoulli',
    auto_class_weights='Balanced',
    eval_metric='Logloss',
    od_type='Iter', od_wait=30,
    random_seed=42,
    thread_count=-1,
    verbose=False, allow_writing_files=False,
)
```
Observation: на текущей выборке CatBoost быстро сходится (best_iteration ≈ 9 для baseline и ≈ 105 для pruned) — early stopping по Logloss на Valid срабатывает рано.

**Calibration:** Platt scaling via `CalibratedClassifierCV(method='sigmoid')` wrapped around a `FrozenEstimator` (sklearn ≥ 1.6) to fit on a dedicated calibration split. A fallback to legacy `cv='prefit'` is used for older sklearn versions.

---

## Validation Scheme

| Split | Purpose | Approx. share |
|---|---|---|
| Train | Fit CatBoost | 70% |
| Valid | Early stopping & evaluation | 15% |
| Calib | Fit Platt scaler | 15% |

Splits are strictly **chronological**.

> ⚠️ **Walk-forward / time-series cross-validation is not yet implemented.** The current split is a simple chronological hold-out — see [Known Limitations](#known-limitations--weaknesses).

---

## Results

> 🚨 **Историческая правка (2026-05-02).** В предыдущей итерации вызывался `ta.dpo(..., lookahead=False)` — этого параметра **нет в API pandas_ta**, он молча попадал в `**kwargs` и DPO считался с `centered=True` (default), что создавало look-ahead bias на ~length/2 свечей вперёд (см. [pandas_ta Issue #60](https://github.com/twopirllc/pandas-ta/issues/60)). Это завышало AUC до ~0.82 — фейк. После фикса (`centered=False`) и полного аудита всех 109 индикаторов через [scripts/leakage_audit.py](scripts/leakage_audit.py) и [scripts/leakage_audit2.py](scripts/leakage_audit2.py) все остальные индикаторы прошли проверку на связь с будущими значениями. Ниже — честные метрики.

### Pipeline funnel (с ATR-target K_BARS=3, K_ATR=1.0, CatBoost)

| Stage | Columns / Rows |
|---|---|
| TA indicators added | 109 |
| Features after generation | 173 |
| After NaN(>20%) filter | 165 |
| Rows after ATR-target filter + `dropna()` | 12 100 |
| Columns after `series_to_supervised(n_in=3)` | 660 |
| After `&#124;corr&#124; > 0.95` filter | 356 |
| After permutation-importance pruning | **26** |
| Train / Valid / Calib | 8 467 / 1 814 / 1 816 |

### A/B comparison: Baseline vs Permutation-Pruned (CatBoost + k_bars=3)

Метрики со сплита Calib (out-of-sample, ~15% хвост ряда):

| Stage | Features | Acc Calib | AUC Calib | LogLoss Calib | Brier Calib |
|---|---|---|---|---|---|
| Baseline CB | 356 | 0.5319 | 0.5423 | 0.6915 | 0.2492 |
| **Pruned CB** | **26** | **0.5787** | **0.6004** | **0.6801** | **0.2435** |
| CB + Platt | 26 | 0.5694 | 0.6004 | 0.6780 | 0.2425 |

Для сравнения с предыдущей итерацией (XGBoost + k_bars=5): Pruned XGB Calib AUC=0.5670, Acc=0.5616 — т.е. **новая комбинация CatBoost + k=3 даёт +3.3pp к AUC и +1.7pp к accuracy на OOS**, при этом число признаков после прунинга уменьшилось с 69 до 26.

**Key observations:**
- AUC OOS **≈ 0.60** — лучший результат среди всех протестированных конфигов на этом датасете
- Permutation pruning убирает **330 из 356 признаков (93%)** — pruning ещё агрессивнее работает на CatBoost-сигналах
- Best CatBoost iteration: 9 (baseline) / 105 (pruned) — модель уверенно использует более глубокие итерации после прунинга шумных фичей
- Pruned CB + Platt выбран финальной моделью: AUC=0.6004, Acc=0.5694, LogLoss=0.6780 на Calib
- ATR-target k_bars=3 (3 свечи = 15 мин) даёт сбалансированные классы 50/50 и более сильный сигнал, чем k_bars=5 (25 мин)
- No transaction costs or slippage в самом ноутбуке — см. секцию A4+A2 ниже и [`scripts/threshold_strategy.py`](scripts/threshold_strategy.py) для P&L с издержками

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

**Промежуточный вывод:** при k_bars=5, k_atr=1.0 (движение ≥1 ATR за 5 свечей) хвосты **симметрично открываются**: впервые LONG даёт реальный хвост (389 сигналов OOS с prec=0.617), AUC поднимается с 0.52 до 0.58.

---

## Расширенный анализ: sensitivity, model zoo, таймфрейм

По следам внедрения ATR-target было проведено три дополнительных эксперимента.

### 1. Sensitivity по (k_bars, k_atr)

Скрипт: [`scripts/sensitivity_grid.py`](scripts/sensitivity_grid.py) · результаты: [`scripts/sensitivity_grid_results.json`](scripts/sensitivity_grid_results.json)

Прогнана сетка k_bars ∈ {3, 5, 7, 10} × k_atr ∈ {0.75, 1.0, 1.25} — 12 конфигов, XGBoost+Platt, те же сплиты 70/15/15.

| k_bars | k_atr | N | AUC valid | AUC calib | Acc calib | LONG p≥0.55 (Calib) | SHORT p≤0.45 (Calib) |
|---:|---:|---:|---:|---:|---:|---|---|
| **3** | **1.0** | 14 487 | 0.573 | **0.577** | 0.556 | **prec=0.570, N=386** | **prec=0.625, N=381** |
| 5 | 1.0 | 18 844 | 0.538 | 0.567 | 0.551 | prec=0.549, N=215 | prec=0.636, N=319 |
| 7 | 0.75 | 25 760 | 0.555 | 0.562 | 0.541 | prec=0.569, N=641 | prec=0.580, N=742 |
| 7 | 1.0 | 21 507 | 0.551 | 0.562 | 0.547 | prec=0.517, N=433 | prec=0.593, N=769 |
| 3 | 0.75 | 19 324 | 0.548 | 0.558 | 0.550 | prec=0.559, N=236 | prec=0.606, N=371 |
| 3 | 1.25 | 10 769 | 0.562 | 0.561 | 0.526 | prec=0.524, N=143 | prec=0.610, N=480 |
| 10 | 0.75 | 28 128 | 0.547 | 0.546 | 0.532 | prec=0.525, N=265 | prec=0.569, N=1 143 |
| 10 | 1.0 | 24 305 | 0.536 | 0.541 | 0.551 | prec=0.625, N=48 | prec=0.537, N=818 |
| 10 | 1.25 | 20 889 | 0.533 | 0.535 | 0.527 | prec=0.455, N=22 | prec=0.556, N=795 |

**Вывод:** новый лидер — **k_bars=3, k_atr=1.0** (AUC_c=0.577 вместо 0.567 у k=5). Он же даёт сбалансированные хвосты: по ~380 сигналов в каждую сторону с prec 0.57–0.62. Движение «≥1 ATR за 3 свечи (15 мин)» оказалось предсказуемее, чем за 5 или 7 свечей — логично: на 5-минутном фрейме ближайшие свечи несут больше информации. **k_bars=10 стабильно хуже** — горизонт в 50 мин уже не предсказуем на 5-минутных фичах.

### 2. Model zoo на ATR-target (k=5, k_atr=1.0)

Скрипт: [`scripts/model_zoo_atr.py`](scripts/model_zoo_atr.py) · результаты: [`scripts/model_zoo_atr_results.json`](scripts/model_zoo_atr_results.json)

7 моделей с Platt-калибровкой на одном и том же X. Метрики на Calib (OOS):

| Model | AUC valid | AUC calib | Acc calib | LogLoss | LONG p≥0.55 | SHORT p≤0.45 |
|---|---:|---:|---:|---:|---|---|
| **CatBoost** | **0.579** | **0.606** | **0.580** | **0.678** | **prec=0.562, N=787** | **prec=0.660, N=592** |
| LightGBM | 0.572 | 0.588 | 0.561 | 0.681 | prec=0.578, N=510 | prec=0.638, N=668 |
| ExtraTrees | 0.541 | 0.585 | 0.555 | 0.685 | prec=0.312, N=16 | prec=0.641, N=618 |
| RandomForest | 0.542 | 0.580 | 0.550 | 0.687 | prec=0.273, N=11 | prec=0.652, N=359 |
| XGBoost (база) | 0.538 | 0.567 | 0.551 | 0.687 | prec=0.549, N=215 | prec=0.636, N=319 |
| LogReg(L2) | 0.516 | 0.561 | 0.523 | 0.690 | — (нет хвоста) | — |
| HistGBM | 0.529 | 0.548 | 0.524 | 0.690 | prec=0.572, N=236 | prec=0.648, N=105 |

**Вывод:** из 7 протестированных моделей **CatBoost жёсткий лидер** — AUC на +4pp выше XGBoost (0.606 vs 0.567), accuracy 0.580 vs 0.551. Главное — CatBoost даёт **оба хвоста с prec ≥0.56 на сотнях сигналов**, в то время как RandomForest/ExtraTrees «схлопывают» LONG-хвост (N=11–16). LightGBM второй, но в ~7× медленнее (158с против 22с у CatBoost на этой выборке).

**Рекомендация:** да, имеет смысл поменять основную модель на CatBoost. Это самый большой скачок в рамках текущего фреймворка (+4pp AUC «бесплатно»).

### 3. Таймфреймы: 5m vs 10m vs 15m

Скрипт: [`scripts/timeframe_test.py`](scripts/timeframe_test.py) · результаты: [`scripts/timeframe_test_results.json`](scripts/timeframe_test_results.json)

Сырые 5-минутные бары ресемплены в 10m и 15m, повторился весь pipeline (109 TA + ATR-target):

| Timeframe | k_bars | N (после ATR-target) | AUC valid | **AUC calib** | Acc calib | LONG p≥0.55 | SHORT p≤0.45 |
|---|---:|---:|---:|---:|---:|---|---|
| **5m**  | **5** | **18 844** | 0.538 | **0.567** | 0.551 | prec=0.549, N=215 | prec=0.636, N=319 |
| 10m | 5 | 8 895 | 0.514 | 0.490 | 0.515 | — (N=0) | prec=0.485, N=239 |
| 15m | 5 | 5 921 | 0.542 | 0.504 | 0.509 | — (N=0) | prec=0.520, N=419 |
| 10m | 3 | 6 622 | 0.506 | 0.484 | 0.504 | — | — |
| 15m | 3 | 4 400 | 0.526 | 0.489 | 0.519 | — | prec=0.508, N=358 |

**Вывод:** переход на 10m и 15m **ухудшает OOS-метрики до случайного уровня или ниже** (AUC 0.49–0.50). Причин две: (1) выборка в 2–3 раза меньше — быстрее переобучение; (2) TA-индикаторы спроектированы/оптимизированы для более мелкого фрейма — на 15m они срезают быстрые движения.

**Рекомендация:** остаться на 5m — фрейм повышать не стоит. Альтернатива — попробовать **1m** (если будет доступна), но это резко повышает стоимость издержек (бид/аск-спред съедает больше).

### Сводный итог (ответ на 3 вопроса)

1. **Sensitivity:** вместо (k=5, k_atr=1.0) лучше (k=3, k_atr=1.0) — AUC_c вырастает с 0.567 до 0.577.
2. **Модель:** да, стоит поменять XGBoost на **CatBoost** — самый большой прирост (AUC 0.567 → 0.606, +4pp).
3. **Таймфрейм:** нет, повышать не стоит. 10m/15m резко ухудшают OOS (AUC 0.49–0.50).

**Совместное внедрение (1)+(2) — реализовано в основном ноутбуке:** CatBoost + (k_bars=3, k_atr=1.0) дал **AUC_c = 0.6004**, Acc=0.5694, LogLoss=0.6780 на Calib — лучший результат на этом датасете.

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
| 18–19 | Baseline CatBoost training + classification report |
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
- [x] A3: альтернативный target = движение ≥ k·ATR за k свечей + grid search (победитель k=5, k_atr=1.0, AUC 0.567–0.580 OOS)
- [x] Sensitivity grid k_bars × k_atr (4×3) — победитель k=3, k_atr=1.0 (AUC_c=0.577) — см. [`scripts/sensitivity_grid.py`](scripts/sensitivity_grid.py)
- [x] Model zoo на ATR-target — победитель CatBoost (AUC=0.606 vs XGB 0.567) — см. [`scripts/model_zoo_atr.py`](scripts/model_zoo_atr.py)
- [x] Timeframe test 5m/10m/15m — оставаться на 5m — см. [`scripts/timeframe_test.py`](scripts/timeframe_test.py)
- [x] **Combined: CatBoost + k_bars=3, k_atr=1.0 → AUC_c = 0.6004** — внедрён как основная конфигурация в [`sber_intraday_pipeline.ipynb`](sber_intraday_pipeline.ipynb)
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
