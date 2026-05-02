"""
A3: новый target по движению в единицах ATR.

Идея: вместо "is_green_next" (закрытие выше открытия даже на 0.001%) использовать
"движение за k_bars свечей в единицах ATR". Это:
  - убирает шумные свечи около нуля (фильтрация неоднозначных примеров),
  - даёт симметричные хвосты для LONG и SHORT,
  - учит модель именно тому, что важно для торговли — направленному движению.

Определение target_atr (тройное: long / short / neutral):
  ret_fwd = (CLOSE[t+k_bars] - CLOSE[t]) / ATR[t]
  target =  1, если ret_fwd >=  k_atr   (long-движение)
            0, если ret_fwd <= -k_atr   (short-движение)
            NaN иначе                   (отбрасываем при обучении)

Так задача из бинарной "зелёная/красная" превращается в "куда пошло, ЕСЛИ
вообще движение есть". Бинарка остаётся, но учим только на informative примерах.
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

# --- 1. Создаём ATR-target ---
def make_atr_target(df, k_bars=3, k_atr=0.5):
    """target = +1/-1/NaN в зависимости от движения за k_bars в единицах ATR.

    Использует уже существующую колонку atr_14 (рассчитана через pandas_ta
    в add_ta_features). Возвращает копию df с колонкой 'target_atr'.
    """
    out = df.copy()
    if "atr_14" not in out.columns:
        raise KeyError("Нужна колонка atr_14 (должна появиться после add_ta_features)")
    fwd_ret = (out["CLOSE"].shift(-k_bars) - out["CLOSE"]) / out["atr_14"]
    out["target_atr_ret"] = fwd_ret  # для диагностики
    # Бинарный target: только определённые movement-и
    target = pd.Series(np.nan, index=out.index, name="target_atr")
    target[fwd_ret >=  k_atr] = 1
    target[fwd_ret <= -k_atr] = 0
    out["target_atr"] = target
    return out

# Проверим распределение для нескольких конфигов
print("=== Подбор k_bars и k_atr: какая доля строк остаётся informative? ===")
print(f"{'k_bars':>8s} {'k_atr':>6s} {'N_total':>10s} {'N_long':>10s} {'N_short':>10s} {'N_NaN':>10s} {'%info':>8s}")
configs = [
    (1, 0.30), (1, 0.50), (1, 0.75), (1, 1.00),
    (3, 0.50), (3, 0.75), (3, 1.00),
    (5, 0.75), (5, 1.00), (5, 1.50),
]
for k_bars, k_atr in configs:
    fd = make_atr_target(feature_df, k_bars=k_bars, k_atr=k_atr)
    n_long = (fd["target_atr"]==1).sum()
    n_short = (fd["target_atr"]==0).sum()
    n_nan = fd["target_atr"].isna().sum()
    total = len(fd)
    info_pct = (n_long + n_short) / total * 100
    print(f"{k_bars:>8d} {k_atr:>6.2f} {total:>10d} {n_long:>10d} {n_short:>10d} {n_nan:>10d} {info_pct:>7.1f}%")

# Берём конфиг (k_bars=3, k_atr=0.5): достаточно строк (~70%) и движение значимое
KBARS, KATR = 3, 0.5
print(f"\n=== Используем k_bars={KBARS}, k_atr={KATR} ===\n")

fd = make_atr_target(feature_df, k_bars=KBARS, k_atr=KATR)
print(f"target_atr balance: long={int((fd['target_atr']==1).sum())}, "
      f"short={int((fd['target_atr']==0).sum())}, "
      f"NaN={int(fd['target_atr'].isna().sum())}")

# --- 2. Адаптируем build_X_y_for_model под новый target ---
# Простейший способ: подменим target_is_green_next на target_atr и запустим
# существующую функцию (она ищет 'target_*' колонки).
fd2 = fd.copy()
fd2 = fd2.dropna(subset=["target_atr"])  # отбрасываем неоднозначные строки
fd2["target_is_green_next"] = fd2["target_atr"].astype(int)  # подмена для совместимости с build_X_y
print(f"После dropna по target_atr: {len(fd2)} строк")

X, y, _ = build_X_y(fd2, n_in=3)
n = len(X); ntr=int(n*0.70); nva=int(n*0.15)
X_tr, y_tr = X.iloc[:ntr], y.iloc[:ntr]
X_va, y_va = X.iloc[ntr:ntr+nva], y.iloc[ntr:ntr+nva]
X_ca, y_ca = X.iloc[ntr+nva:], y.iloc[ntr+nva:]
print(f"Train={X_tr.shape}, Valid={X_va.shape}, Calib={X_ca.shape}")
print(f"Class balance Train: long={y_tr.mean():.3f}, short={1-y_tr.mean():.3f}")

# --- 3. Обучаем XGB + Pruning + Platt на новом target ---
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
try:
    from sklearn.frozen import FrozenEstimator
    USE_FROZEN = True
except ImportError:
    USE_FROZEN = False

scale = (y_tr==0).sum()/max((y_tr==1).sum(),1)
print(f"scale_pos_weight={scale:.4f}")

print("\nОбучаем baseline XGB на target_atr...")
m_base = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
m_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p_va_base = m_base.predict_proba(X_va)[:,1]
print(f"  Baseline AUC_v={roc_auc_score(y_va, p_va_base):.4f}  "
      f"Acc_v={accuracy_score(y_va, p_va_base>=0.5):.4f}  "
      f"Brier_v={brier_score_loss(y_va, p_va_base):.4f}  "
      f"best_iter={m_base.best_iteration}")

print("Permutation pruning...")
pi = permutation_importance(m_base, X_va, y_va, n_repeats=5,
    random_state=42, n_jobs=-1, scoring="neg_log_loss")
keep = X_tr.columns[pi.importances_mean > 0].tolist()
print(f"  keep features: {len(keep)} / {X_tr.shape[1]}")

X_tr_p, X_va_p, X_ca_p = X_tr[keep], X_va[keep], X_ca[keep]
m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
m.fit(X_tr_p, y_tr, eval_set=[(X_va_p, y_va)], verbose=False)
p_va_pr = m.predict_proba(X_va_p)[:,1]
print(f"  Pruned   AUC_v={roc_auc_score(y_va, p_va_pr):.4f}  "
      f"Acc_v={accuracy_score(y_va, p_va_pr>=0.5):.4f}  "
      f"Brier_v={brier_score_loss(y_va, p_va_pr):.4f}  "
      f"best_iter={m.best_iteration}")

if USE_FROZEN:
    m_cal = CalibratedClassifierCV(FrozenEstimator(m), method="sigmoid", cv=None).fit(X_ca_p, y_ca)
else:
    m_cal = CalibratedClassifierCV(m, method="sigmoid", cv="prefit").fit(X_ca_p, y_ca)
p_va = m_cal.predict_proba(X_va_p)[:,1]
p_ca = m_cal.predict_proba(X_ca_p)[:,1]
print(f"  Cal+Platt AUC_v={roc_auc_score(y_va, p_va):.4f}  "
      f"Acc_v={accuracy_score(y_va, p_va>=0.5):.4f}  "
      f"Brier_v={brier_score_loss(y_va, p_va):.4f}")

# --- 4. Распределение predict_proba и хвосты ---
print("\n=== Распределение predict_proba (Valid, target_atr+Pruned+Platt) ===")
print(f"  min={p_va.min():.4f}  max={p_va.max():.4f}  "
      f"mean={p_va.mean():.4f}  std={p_va.std():.4f}")
quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print(f"  Quantiles: " + ", ".join(f"q{int(q*100)}={np.quantile(p_va,q):.3f}" for q in quantiles))

y_va_arr = y_va.values if hasattr(y_va, 'values') else np.asarray(y_va)
y_ca_arr = y_ca.values if hasattr(y_ca, 'values') else np.asarray(y_ca)

print("\n=== ХВОСТ p_long >= threshold (LONG-сигнал) ===")
print(f"  {'Thr':>6s} {'N_va':>7s} {'%':>6s} {'Prec_VA':>8s} {'N_ca':>7s} {'Prec_CA':>8s}")
for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    nva = (p_va>=thr).sum(); nca = (p_ca>=thr).sum()
    pv = y_va_arr[p_va>=thr].mean() if nva>0 else float("nan")
    pc = y_ca_arr[p_ca>=thr].mean() if nca>0 else float("nan")
    print(f"  {thr:>6.2f} {nva:>7d} {nva/len(p_va)*100:>5.1f}% {pv:>8.3f} {nca:>7d} {pc:>8.3f}")

print("\n=== ХВОСТ p_long <= threshold (SHORT-сигнал) ===")
print(f"  {'Thr':>6s} {'N_va':>7s} {'%':>6s} {'Prec_VA':>8s} {'N_ca':>7s} {'Prec_CA':>8s}")
for thr in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    nva = (p_va<=thr).sum(); nca = (p_ca<=thr).sum()
    pv = (1-y_va_arr[p_va<=thr]).mean() if nva>0 else float("nan")
    pc = (1-y_ca_arr[p_ca<=thr]).mean() if nca>0 else float("nan")
    print(f"  {thr:>6.2f} {nva:>7d} {nva/len(p_va)*100:>5.1f}% {pv:>8.3f} {nca:>7d} {pc:>8.3f}")

# --- 5. P&L: реальное движение в свечах с сигналом ---
# Восстановим forward return по индексам Calib
idx_ca = X_ca.index
fwd = (feature_df["CLOSE"].shift(-KBARS) - feature_df["CLOSE"]).reindex(idx_ca).values

# Подбор порога на Valid (precision>=0.65 для LONG, >=0.65 для SHORT)
def find_long(p, y, target_prec, min_n):
    cand = np.unique(np.round(p, 3)); best=None
    for thr in sorted(cand):
        m = p >= thr; n = m.sum()
        if n < min_n: continue
        pr = y[m].mean()
        if pr >= target_prec: best=(thr, n, pr)
    return best

def find_short(p, y, target_prec, min_n):
    cand = np.unique(np.round(p, 3)); best=None
    for thr in sorted(cand, reverse=True):
        m = p <= thr; n = m.sum()
        if n < min_n: continue
        pr = (1-y[m]).mean()
        if pr >= target_prec: best=(thr, n, pr)
    return best

print("\n=== A4 на target_atr: пороги под precision >= 0.65 на Valid ===")
ll = find_long(p_va, y_va_arr, 0.65, 30)
ss = find_short(p_va, y_va_arr, 0.65, 30)
print(f"  LONG : thr_long  >= {ll[0]:.3f}, N_va={ll[1]}, prec_va={ll[2]:.3f}" if ll else "  LONG : не найден")
print(f"  SHORT: thr_short <= {ss[0]:.3f}, N_va={ss[1]}, prec_va={ss[2]:.3f}" if ss else "  SHORT: не найден")

print("\n  → Out-of-sample на Calib:")
if ll:
    mca = p_ca>=ll[0]; n = mca.sum()
    pr = y_ca_arr[mca].mean() if n>0 else float("nan")
    print(f"     LONG : N={n}, precision={pr:.3f}")
if ss:
    mca = p_ca<=ss[0]; n = mca.sum()
    pr = (1-y_ca_arr[mca]).mean() if n>0 else float("nan")
    print(f"     SHORT: N={n}, precision={pr:.3f}")

# --- 6. Простая P&L на Calib (без издержек, цена входа = CLOSE[t], выход = CLOSE[t+k_bars]) ---
print("\n=== A2 на target_atr: P&L на Calib ===")
signal = np.zeros(len(p_ca))
if ll: signal[p_ca >= ll[0]] = 1
if ss: signal[p_ca <= ss[0]] = -1

# fwd[i] — движение в РУБЛЯХ за k_bars свечей. Нормируем на CLOSE
close_ca = feature_df["CLOSE"].reindex(idx_ca).values
ret_pct = fwd / close_ca  # % ход
pnl = signal * ret_pct
n_trades = (signal != 0).sum()
n_l = (signal==1).sum(); n_s = (signal==-1).sum()
pnl_t = pnl[signal!=0]; pnl_t = pnl_t[~np.isnan(pnl_t)]
hit = (pnl_t > 0).mean() if len(pnl_t)>0 else float("nan")

print(f"  Сделок: {n_trades} ({n_trades/len(signal)*100:.1f}% свечей)")
print(f"    LONG : {n_l}, SHORT: {n_s}")
print(f"  Hit rate: {hit*100:.1f}%")
print(f"  Сумма P&L: {np.nansum(pnl)*100:+.3f}%")
print(f"  Средний P&L/сделку: {np.nansum(pnl)/max(n_trades,1)*100:+.4f}%")

fee = 0.0005  # 5 bps в одну сторону
pnl_after = np.nansum(pnl) - fee*2*n_trades
print(f"\n  С комиссией 10 bps round-trip:")
print(f"    P&L после: {pnl_after*100:+.3f}%")
print(f"    Средний P&L/сделку: {pnl_after/max(n_trades,1)*100:+.4f}%")
print(f"    → {'ПРИБЫЛЬНО' if pnl_after>0 else 'УБЫТОЧНО'}")

# Сравнение с прежним is_green_next подходом
print("\n=== Сводка: target_is_green_next vs target_atr ===")
print(f"  target_is_green_next | balanced (>=0.65): LONG_OOS=0.553/47, SHORT_OOS=0.960/25")
print(f"  target_atr (k_bars={KBARS}, k_atr={KATR}) | (>=0.65):", end=" ")
ll_str = f"LONG_OOS={(y_ca_arr[p_ca>=ll[0]].mean() if ll else 0):.3f}/{(p_ca>=ll[0]).sum() if ll else 0}"
ss_str = f"SHORT_OOS={((1-y_ca_arr[p_ca<=ss[0]]).mean() if ss else 0):.3f}/{(p_ca<=ss[0]).sum() if ss else 0}"
print(f"{ll_str}, {ss_str}")
