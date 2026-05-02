"""
A4+A2: торговая стратегия по хвостам предсказаний.

Алгоритм:
  1. Обучаем XGB + Pruning + Platt (как в ноутбуке).
  2. На Valid находим пороги:
       thr_long  = min p, при котором precision_long  >= target
       thr_short = max p, при котором precision_short >= target
     где precision_long  = P(y=1 | p>=thr) — точность LONG-сигнала
         precision_short = P(y=0 | p<=thr) — точность SHORT-сигнала
  3. Замораживаем пороги, проверяем на Calib (out-of-sample).
  4. Считаем простую P&L: signed return следующей свечи (без издержек).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read("sber_intraday_pipeline.ipynb", as_version=4)

def clean_source(src):
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%","!")))

ns = {"__name__":"__not_main__"}
for idx, c in enumerate(nb.cells):
    if c.cell_type=="code" and idx in [2,4,6,8,10,11]:
        try: exec(clean_source(c.source), ns)
        except Exception as e: print(f"[warn] cell {idx}: {e}")

feature_df = ns["feature_df"]
build_X_y = ns["build_X_y_for_model"]

# --- 1. Pipeline до Pruning + Platt ---
X, y, _ = build_X_y(feature_df, n_in=3)
n = len(X); ntr=int(n*0.70); nva=int(n*0.15)
X_tr, y_tr = X.iloc[:ntr], y.iloc[:ntr]
X_va, y_va = X.iloc[ntr:ntr+nva], y.iloc[ntr:ntr+nva]
X_ca, y_ca = X.iloc[ntr+nva:], y.iloc[ntr+nva:]
idx_va = X_va.index; idx_ca = X_ca.index
print(f"Train={X_tr.shape}, Valid={X_va.shape}, Calib={X_ca.shape}")

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    USE_FROZEN = True
except ImportError:
    USE_FROZEN = False

scale = (y_tr==0).sum()/max((y_tr==1).sum(),1)
m_base = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
m_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

pi = permutation_importance(m_base, X_va, y_va, n_repeats=5,
    random_state=42, n_jobs=-1, scoring="neg_log_loss")
keep = X_tr.columns[pi.importances_mean > 0].tolist()
print(f"Keep features: {len(keep)}")

X_tr_p, X_va_p, X_ca_p = X_tr[keep], X_va[keep], X_ca[keep]
m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
m.fit(X_tr_p, y_tr, eval_set=[(X_va_p, y_va)], verbose=False)

if USE_FROZEN:
    m_cal = CalibratedClassifierCV(FrozenEstimator(m), method="sigmoid", cv=None).fit(X_ca_p, y_ca)
else:
    m_cal = CalibratedClassifierCV(m, method="sigmoid", cv="prefit").fit(X_ca_p, y_ca)

p_va = m_cal.predict_proba(X_va_p)[:,1]
p_ca = m_cal.predict_proba(X_ca_p)[:,1]

# --- 2. Подбор порогов на Valid ---
print("\n=== A4: подбор порогов на Valid ===")
print("Цель: для LONG  — precision_long  >= target_prec_long")
print("Цель: для SHORT — precision_short >= target_prec_short")
print("Минимум сигналов (min_n) — чтобы оценка не была шумом.\n")

def find_long_threshold(p, y, target_prec, min_n):
    """Найти минимальный thr (p>=thr → LONG), при котором precision>=target и N>=min_n."""
    candidates = np.unique(np.round(p, 3))
    best = None
    for thr in sorted(candidates, reverse=False):
        mask = p >= thr
        n = mask.sum()
        if n < min_n: continue
        prec = y[mask].mean()
        if prec >= target_prec:
            best = (thr, n, prec)
    return best

def find_short_threshold(p, y, target_prec, min_n):
    """Найти максимальный thr (p<=thr → SHORT), при котором precision>=target и N>=min_n."""
    candidates = np.unique(np.round(p, 3))
    best = None
    for thr in sorted(candidates, reverse=True):
        mask = p <= thr
        n = mask.sum()
        if n < min_n: continue
        prec = (1 - y[mask]).mean()  # precision_short = доля реальных red
        if prec >= target_prec:
            best = (thr, n, prec)
    return best

y_va_arr = y_va.values if hasattr(y_va, 'values') else np.asarray(y_va)
y_ca_arr = y_ca.values if hasattr(y_ca, 'values') else np.asarray(y_ca)

# Подбираем для двух целевых уровней precision: 0.65 и 0.70
configs = [
    ("conservative (precision >= 0.70)", 0.70, 20),
    ("balanced     (precision >= 0.65)", 0.65, 30),
    ("aggressive   (precision >= 0.60)", 0.60, 50),
]

results = []
for name, tp, mn in configs:
    print(f"--- {name} ---")
    long_thr = find_long_threshold(p_va, y_va_arr, tp, mn)
    short_thr = find_short_threshold(p_va, y_va_arr, tp, mn)
    print(f"  LONG : ", "не найден" if long_thr is None else f"thr_long >= {long_thr[0]:.3f}, N={long_thr[1]}, prec={long_thr[2]:.3f}")
    print(f"  SHORT: ", "не найден" if short_thr is None else f"thr_short <= {short_thr[0]:.3f}, N={short_thr[1]}, prec={short_thr[2]:.3f}")

    # --- Out-of-sample проверка на Calib ---
    if long_thr is not None:
        thr_l = long_thr[0]
        mask_l_ca = p_ca >= thr_l
        n_l_ca = mask_l_ca.sum()
        prec_l_ca = y_ca_arr[mask_l_ca].mean() if n_l_ca > 0 else float("nan")
    else:
        thr_l = None; n_l_ca = 0; prec_l_ca = float("nan")

    if short_thr is not None:
        thr_s = short_thr[0]
        mask_s_ca = p_ca <= thr_s
        n_s_ca = mask_s_ca.sum()
        prec_s_ca = (1 - y_ca_arr[mask_s_ca]).mean() if n_s_ca > 0 else float("nan")
    else:
        thr_s = None; n_s_ca = 0; prec_s_ca = float("nan")

    print(f"  → CALIB (out-of-sample):")
    print(f"     LONG : N={n_l_ca}, precision={prec_l_ca:.3f}" if thr_l else "     LONG : нет порога")
    print(f"     SHORT: N={n_s_ca}, precision={prec_s_ca:.3f}" if thr_s else "     SHORT: нет порога")
    print()

    results.append({
        "config": name, "target_prec": tp,
        "thr_long": thr_l, "n_long_va": long_thr[1] if long_thr else 0, "prec_long_va": long_thr[2] if long_thr else None,
        "thr_short": thr_s, "n_short_va": short_thr[1] if short_thr else 0, "prec_short_va": short_thr[2] if short_thr else None,
        "n_long_ca": n_l_ca, "prec_long_ca": prec_l_ca,
        "n_short_ca": n_s_ca, "prec_short_ca": prec_s_ca,
    })

# --- 3. P&L на Calib (без издержек) ---
# Возьмём конфиг "balanced" (precision>=0.65) для P&L расчёта
print("=== A2: P&L на Calib (балансированный конфиг, без издержек) ===")
cfg = next((r for r in results if "balanced" in r["config"]), None)
if cfg and (cfg["thr_long"] is not None or cfg["thr_short"] is not None):
    # Сигнал: +1 при LONG, -1 при SHORT, 0 иначе
    signal = np.zeros(len(p_ca))
    if cfg["thr_long"] is not None:
        signal[p_ca >= cfg["thr_long"]] = 1
    if cfg["thr_short"] is not None:
        signal[p_ca <= cfg["thr_short"]] = -1

    # Доходность следующей свечи: (CLOSE[t+1] - OPEN[t+1]) / OPEN[t+1]
    # Но в feature_df target=is_green_next, нам нужна сама доходность.
    # Восстановим её из feature_df через index alignment.
    fd = feature_df.copy()
    fd["next_ret"] = (fd["CLOSE"].shift(-1) - fd["OPEN"].shift(-1)) / fd["OPEN"].shift(-1)
    # X_ca.index — это индексы в feature_df после dropna и series_to_supervised.
    # Но build_X_y возвращает индексы X = индексы строк после всех фильтров.
    # Можно безопасно сделать через .reindex(X_ca.index).
    next_ret_ca = fd["next_ret"].reindex(idx_ca).values

    # P&L каждой сделки
    pnl = signal * next_ret_ca
    n_trades = (signal != 0).sum()
    n_long = (signal == 1).sum()
    n_short = (signal == -1).sum()
    n_total = len(signal)

    pnl_total = np.nansum(pnl)
    avg_per_trade = np.nansum(pnl) / max(n_trades, 1)
    avg_long = np.nansum(pnl[signal==1]) / max(n_long, 1)
    avg_short = np.nansum(pnl[signal==-1]) / max(n_short, 1)

    # Hit rate (доля прибыльных сделок)
    pnl_trade = pnl[signal != 0]
    pnl_trade = pnl_trade[~np.isnan(pnl_trade)]
    hit_rate = (pnl_trade > 0).mean() if len(pnl_trade) > 0 else float("nan")

    print(f"  Период Calib: {n_total} баров (~{n_total/72:.0f} торговых дней)")
    print(f"  Сделок всего: {n_trades} ({n_trades/n_total*100:.1f}% свечей)")
    print(f"    LONG : {n_long} сделок, средняя доходность {avg_long*100:+.4f}% за сделку")
    print(f"    SHORT: {n_short} сделок, средняя доходность {avg_short*100:+.4f}% за сделку")
    print(f"  Суммарный P&L (без издержек): {pnl_total*100:+.3f}%")
    print(f"  Средний P&L за сделку:        {avg_per_trade*100:+.4f}%")
    print(f"  Hit rate (доля прибыльных):   {hit_rate*100:.1f}%")

    # Сравнение с buy-and-hold за тот же период
    bh = fd["CLOSE"].reindex(idx_ca).pct_change().fillna(0).sum()
    print(f"  Buy-and-hold за период:       {bh*100:+.3f}%")

    # Сравнение с типичной комиссией: 0.05% round-trip = 0.05% за сделку
    # (предполагаем закрытие на следующей свече, поэтому commission_per_trade=0.05%)
    fee = 0.0005  # 5 bps в одну сторону, итого 10 bps round-trip
    pnl_after_fees = np.nansum(pnl - fee * 2 * (signal != 0))
    avg_after_fees = pnl_after_fees / max(n_trades, 1)
    print(f"\n  С комиссией 10 bps round-trip:")
    print(f"    P&L после комиссий:  {pnl_after_fees*100:+.3f}%")
    print(f"    Средний P&L/сделку:  {avg_after_fees*100:+.4f}%")
    print(f"    → {'ПРИБЫЛЬНО' if avg_after_fees > 0 else 'УБЫТОЧНО'} после комиссий")
else:
    print("  Не нашли валидных порогов для P&L расчёта.")

# Сохраним сводную таблицу результатов
import json
with open("scripts/threshold_results.json", "w") as f:
    json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()} for r in results], f, indent=2, default=str)
print("\nСохранено: scripts/threshold_results.json")
