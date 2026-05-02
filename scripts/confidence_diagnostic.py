"""
Диагностика уверенности текущей модели.

Цель: понять, есть ли у нас "хвост" предсказаний с p>=0.7, и какая там реальная
точность. Это базовая линия для всех будущих улучшений.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read(os.path.join(REPO_ROOT, "sber_intraday_pipeline.ipynb"), as_version=4)

def clean_source(src):
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%","!")))

ns = {"__name__":"__not_main__"}
for idx, c in enumerate(nb.cells):
    if c.cell_type=="code" and idx in [2,4,6,8,10,11]:
        try: exec(clean_source(c.source), ns)
        except Exception as e: print(f"[warn] cell {idx}: {e}")

feature_df = ns["feature_df"]
build_X_y = ns["build_X_y_for_model"]

# Build full pipeline
X, y, _ = build_X_y(feature_df, n_in=3)
n = len(X); ntr=int(n*0.70); nva=int(n*0.15)
X_tr, y_tr = X.iloc[:ntr], y.iloc[:ntr]
X_va, y_va = X.iloc[ntr:ntr+nva], y.iloc[ntr:ntr+nva]
X_ca, y_ca = X.iloc[ntr+nva:], y.iloc[ntr+nva:]
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

# Pruning
pi = permutation_importance(m_base, X_va, y_va, n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_log_loss")
keep = X_tr.columns[pi.importances_mean > 0].tolist()
print(f"Keep features: {len(keep)}")

X_tr_p, X_va_p, X_ca_p = X_tr[keep], X_va[keep], X_ca[keep]
m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
m.fit(X_tr_p, y_tr, eval_set=[(X_va_p, y_va)], verbose=False)

# Platt calibration
if USE_FROZEN:
    m_cal = CalibratedClassifierCV(FrozenEstimator(m), method="sigmoid", cv=None).fit(X_ca_p, y_ca)
else:
    m_cal = CalibratedClassifierCV(m, method="sigmoid", cv="prefit").fit(X_ca_p, y_ca)

p_va_raw = m.predict_proba(X_va_p)[:,1]
p_va_cal = m_cal.predict_proba(X_va_p)[:,1]

print("\n=== Распределение predict_proba (Valid, Pruned + Platt) ===")
print(f"  min={p_va_cal.min():.4f}  max={p_va_cal.max():.4f}  mean={p_va_cal.mean():.4f}  std={p_va_cal.std():.4f}")
quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print(f"  Quantiles: " + ", ".join(f"q{int(q*100)}={np.quantile(p_va_cal,q):.3f}" for q in quantiles))

# Что бывает в хвостах ВПРАВО (predict green)
print("\n=== ХВОСТ p_green ≥ threshold (предсказываем GREEN) ===")
print(f"  {'Thr':>6s} {'N':>7s} {'%':>7s} {'P_real':>8s} {'p_mean':>8s}")
for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
    mask = p_va_cal >= thr
    n_sel = mask.sum()
    if n_sel < 5:
        print(f"  {thr:>6.2f} {n_sel:>7d} {n_sel/len(p_va_cal)*100:>6.1f}%      —          —")
        continue
    win = y_va[mask].mean()  # доля реальных green
    p_mean = p_va_cal[mask].mean()
    print(f"  {thr:>6.2f} {n_sel:>7d} {n_sel/len(p_va_cal)*100:>6.1f}% {win:>8.4f} {p_mean:>8.4f}")

# Что бывает в хвостах ВЛЕВО (predict red)
print("\n=== ХВОСТ p_green ≤ threshold (предсказываем RED, hit = 1 - p_green) ===")
print(f"  {'Thr':>6s} {'N':>7s} {'%':>7s} {'P_real_red':>10s} {'p_red_mean':>10s}")
for thr in [0.50, 0.48, 0.45, 0.42, 0.40, 0.35, 0.30, 0.25]:
    mask = p_va_cal <= thr
    n_sel = mask.sum()
    if n_sel < 5:
        print(f"  {thr:>6.2f} {n_sel:>7d} {n_sel/len(p_va_cal)*100:>6.1f}%       —             —")
        continue
    win_red = (1 - y_va[mask]).mean()  # доля реальных red
    p_red_mean = (1 - p_va_cal[mask]).mean()
    print(f"  {thr:>6.2f} {n_sel:>7d} {n_sel/len(p_va_cal)*100:>6.1f}% {win_red:>10.4f} {p_red_mean:>10.4f}")

# Reliability diagram (data only, no plotting)
print("\n=== Reliability bins (Valid, Pruned + Platt) ===")
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_va, p_va_cal, n_bins=10, strategy='quantile')
print(f"  {'p_predicted':>12s} {'p_actual':>10s}")
for pp, pt in zip(prob_pred, prob_true):
    print(f"  {pp:>12.4f} {pt:>10.4f}")

# Дополнительно: симметрия хвостов
print("\n=== Двухсторонняя стратегия: |p - 0.5| > delta ===")
print(f"  {'delta':>6s} {'N_sel':>7s} {'%':>7s} {'Acc':>8s}")
for delta in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    mask = np.abs(p_va_cal - 0.5) > delta
    n_sel = mask.sum()
    if n_sel < 5: continue
    pred = (p_va_cal[mask] >= 0.5).astype(int)
    acc = (pred.values == y_va[mask].values).mean() if hasattr(y_va, 'values') else (pred == y_va[mask]).mean()
    print(f"  {delta:>6.2f} {n_sel:>7d} {n_sel/len(p_va_cal)*100:>6.1f}% {acc:>8.4f}")
