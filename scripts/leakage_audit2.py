"""
Прицельная проверка: убрать DPO и PVR (подозрение на look-ahead) и сравнить.
"""
import os, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
NB_PATH = os.path.join(REPO_ROOT, "sber_intraday_pipeline.ipynb")

nb = nbformat.read(NB_PATH, as_version=4)
def clean_source(src):
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%","!")))

ns = {"__name__":"__not_main__"}
for idx, c in enumerate(nb.cells):
    if c.cell_type=="code" and idx in [2,4,6,8,10,11]:
        try: exec(clean_source(c.source), ns)
        except Exception as e: print(f"[warn] cell {idx}: {e}")

feature_df = ns["feature_df"]
build_X_y = ns["build_X_y_for_model"]

# ---- Прямая проверка DPO: формула должна быть с centered=False ----
import pandas_ta as ta
print(f"pandas_ta version: {ta.version if hasattr(ta,'version') else '?'}")

# Создадим тестовую серию: random walk, посмотрим как DPO ведёт себя
rng = np.random.default_rng(0)
prices = pd.Series(np.cumsum(rng.standard_normal(200)) + 100)
dpo_default = ta.dpo(prices, length=9)
dpo_no_center = ta.dpo(prices, length=9, centered=False)
# Сравним: если в момент i pandas_ta использует price[i+5], то dpo_default[i] != dpo_no_center[i]
print(f"\nDPO test on random walk (len=200, length=9):")
print(f"  default DPO   - первые валидные индексы: {dpo_default.first_valid_index()}, последние: {dpo_default.last_valid_index()}")
print(f"  centered=False - первые валидные индексы: {dpo_no_center.first_valid_index()}, последние: {dpo_no_center.last_valid_index()}")
diff = (dpo_default - dpo_no_center).dropna()
print(f"  Разница mean={diff.mean():.4f}, max_abs={diff.abs().max():.4f}, std={diff.std():.4f}")
print(f"  → Если разница большая — pandas_ta.dpo по умолчанию ИСПОЛЬЗУЕТ БУДУЩИЕ значения.")

# Также проверим ширину lookahead: сравним shift у dpo_default
print("\nПроверим, на сколько свечей сдвинут default DPO:")
for s in range(-10, 11):
    ser = dpo_default.shift(s)
    cor = ser.corr(dpo_no_center)
    if pd.notna(cor) and cor > 0.99:
        print(f"  default ≈ centered=False shifted by {s} (corr={cor:.4f})")

# ---- Эксперимент 1: убрать ВСЕ возможные look-ahead колонки и переучить ----
SUSPECTS = ["dpo", "pvr", "midpoint_2", "midprice_2"]

cols_to_drop = []
for c in feature_df.columns:
    cl = c.lower()
    for s in SUSPECTS:
        if cl.startswith(s) or cl.startswith(s+"_") or cl == s:
            cols_to_drop.append(c)
            break

print(f"\nКандидаты на удаление: {cols_to_drop}")

# Уберём их из feature_df и пересоберём
fd2 = feature_df.drop(columns=cols_to_drop, errors="ignore")
X, y, _ = build_X_y(fd2, n_in=3)
n = len(X); ntr=int(n*0.70); nva=int(n*0.15)
Xt, yt = X.iloc[:ntr], y.iloc[:ntr]
Xv, yv = X.iloc[ntr:ntr+nva], y.iloc[ntr:ntr+nva]
Xc, yc = X.iloc[ntr+nva:], y.iloc[ntr+nva:]

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, log_loss
scale = (yt==0).sum()/max((yt==1).sum(),1)
m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=20, eval_metric="logloss", random_state=42,
    scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
print(f"\nОбучаем без подозрительных признаков. X={X.shape}")
m.fit(Xt, yt, eval_set=[(Xv,yv)], verbose=False)
p = m.predict_proba(Xv)[:,1]
print(f"  AUC_valid={roc_auc_score(yv,p):.4f}  Acc={accuracy_score(yv,p>=0.5):.4f}  "
      f"LogLoss={log_loss(yv,p,labels=[0,1]):.4f}  Brier={brier_score_loss(yv,p):.4f}  "
      f"best_iter={m.best_iteration}")

# Топ-15 после удаления
booster = m.get_booster()
gain = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False).head(15)
print(f"\nТоп-15 признаков после удаления подозрительных:")
for k, v in gain.items():
    print(f"  {k:50s} gain={v:.2f}")

# Тест look-ahead на чистой модели
print("\n=== Повторный тест на сдвиге target после удаления подозрительных ===")
for shift in [1, 2, 5, 10, 20]:
    fd3 = fd2.copy()
    fd3["target_is_green_next"] = fd3["is_green"].shift(-shift).astype("float")
    fd3 = fd3.dropna(subset=["target_is_green_next"])
    fd3["target_is_green_next"] = fd3["target_is_green_next"].astype(int)
    Xs, ys, _ = build_X_y(fd3, n_in=3)
    if len(Xs) < 1000: continue
    ns_ = len(Xs); ntr=int(ns_*0.70); nva=int(ns_*0.15)
    Xtr, ytr = Xs.iloc[:ntr], ys.iloc[:ntr]
    Xva, yva = Xs.iloc[ntr:ntr+nva], ys.iloc[ntr:ntr+nva]
    sc = (ytr==0).sum()/max((ytr==1).sum(),1)
    mm = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=20, eval_metric="logloss", random_state=42,
        scale_pos_weight=sc, early_stopping_rounds=30, n_jobs=-1)
    mm.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False)
    pp = mm.predict_proba(Xva)[:,1]
    print(f"  shift=t+{shift:2d}: AUC_valid={roc_auc_score(yva,pp):.4f}  Acc={accuracy_score(yva,pp>=0.5):.4f}")
