"""
Аудит на look-ahead leakage.

Идея: если AUC модели объясняется реальным сигналом про "следующую свечу",
то при сдвиге target на N свечей вперёд (предсказываем не t+1, а t+5, t+20)
качество должно деградировать к ~0.5.

Если AUC остаётся высоким — где-то в признаках протекает будущая информация.
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
NB_PATH = os.path.join(REPO_ROOT, "sber_intraday_pipeline.ipynb")

nb = nbformat.read(NB_PATH, as_version=4)

def clean_source(src: str) -> str:
    out = []
    for line in src.split("\n"):
        s = line.lstrip()
        if s.startswith("%") or s.startswith("!"): continue
        out.append(line)
    return "\n".join(out)

ns = {"__name__": "__not_main__"}
target_indices = [2, 4, 6, 8, 10, 11]
code_cells = [(i, c.source) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
for idx, src in code_cells:
    if idx not in target_indices: continue
    try: exec(clean_source(src), ns)
    except Exception as e: print(f"  [warn] cell {idx}: {e}")

feature_df = ns["feature_df"]
print(f"feature_df: {feature_df.shape}")
print(f"Колонки target_*: {[c for c in feature_df.columns if c.startswith('target_')]}")

# Проверим определение target_is_green_next: должен быть shift(-1) от is_green
# В исходной формуле: target_is_green_next(t) = is_green(t+1)
# Сравним вручную:
if "target_is_green_next" in feature_df.columns and "is_green" in feature_df.columns:
    chk = (feature_df["target_is_green_next"].iloc[:-1].values ==
           feature_df["is_green"].shift(-1).iloc[:-1].values)
    print(f"target == is_green.shift(-1)? Совпадение на {chk.mean()*100:.2f}% строк")

# --- Соберём данные для модели как обычно ---
build_X_y_for_model = ns["build_X_y_for_model"]
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

# Базовый запуск (для контроля): target_is_green_next, n_in=3
X, y, _ = build_X_y_for_model(feature_df, n_in=3)
n = len(X); n_train = int(n*0.70); n_valid = int(n*0.15)
X_tr, y_tr = X.iloc[:n_train], y.iloc[:n_train]
X_va, y_va = X.iloc[n_train:n_train+n_valid], y.iloc[n_train:n_train+n_valid]
X_ca, y_ca = X.iloc[n_train+n_valid:], y.iloc[n_train+n_valid:]

scale = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
def make_xgb():
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=20, eval_metric="logloss", random_state=42,
        scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)

print("\n=== ТЕСТ 0: baseline (target = is_green(t+1)) ===")
m = make_xgb(); m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p = m.predict_proba(X_va)[:,1]
print(f"  AUC_valid={roc_auc_score(y_va, p):.4f}  Acc={accuracy_score(y_va, p>=0.5):.4f}  Brier={brier_score_loss(y_va,p):.4f}  best_iter={m.best_iteration}")

# --- ТЕСТ 1: сдвиг target на N свечей вперёд ---
# Создадим новый target: is_green(t+N) для разных N
print("\n=== ТЕСТ 1: проверка на look-ahead через сдвиг горизонта ===")
print("Если AUC остаётся высоким при увеличении горизонта — есть утечка.\n")

for shift in [1, 2, 3, 5, 10, 20]:
    # Берём is_green из feature_df (он на t), сдвигаем на -shift, чтобы получить is_green(t+shift)
    fd = feature_df.copy()
    fd["target_shifted"] = fd["is_green"].shift(-shift)
    fd = fd.dropna(subset=["target_shifted"])

    # Пересоберём X/y вручную через те же фильтры, но с новым target
    # Простейший способ — заменить target в feature_df и снова прогнать build_X_y_for_model
    fd2 = fd.copy()
    fd2["target_is_green_next"] = fd2["target_shifted"].astype(int)

    Xs, ys, _ = build_X_y_for_model(fd2, n_in=3)
    if len(Xs) < 1000: continue
    ns_ = len(Xs); ntr = int(ns_*0.70); nva = int(ns_*0.15)
    Xtr, ytr = Xs.iloc[:ntr], ys.iloc[:ntr]
    Xva, yva = Xs.iloc[ntr:ntr+nva], ys.iloc[ntr:ntr+nva]

    scale_s = (ytr==0).sum() / max((ytr==1).sum(), 1)
    mm = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=20, eval_metric="logloss", random_state=42,
        scale_pos_weight=scale_s, early_stopping_rounds=30, n_jobs=-1)
    mm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pp = mm.predict_proba(Xva)[:,1]
    print(f"  shift=t+{shift:2d}: AUC_valid={roc_auc_score(yva, pp):.4f}  "
          f"Acc={accuracy_score(yva, pp>=0.5):.4f}  Brier={brier_score_loss(yva,pp):.4f}  "
          f"pos_rate={ytr.mean():.3f}  best_iter={mm.best_iteration}")

# --- ТЕСТ 2: random target (sanity check) ---
print("\n=== ТЕСТ 2: random target (должен быть AUC≈0.5) ===")
rng = np.random.default_rng(42)
y_rand_full = pd.Series(rng.integers(0, 2, size=len(X)), index=X.index, name="rand")
y_rtr = y_rand_full.iloc[:n_train]; y_rva = y_rand_full.iloc[n_train:n_train+n_valid]
mr = make_xgb(); mr.fit(X_tr, y_rtr, eval_set=[(X_va, y_rva)], verbose=False)
pr = mr.predict_proba(X_va)[:,1]
print(f"  AUC_valid={roc_auc_score(y_rva, pr):.4f}  (ожидаем ~0.5)")

# --- ТЕСТ 3: топ-фичи по gain, не выглядят ли они как утечка ---
print("\n=== ТЕСТ 3: топ-15 признаков baseline-модели по gain ===")
booster = m.get_booster()
gain = booster.get_score(importance_type="gain")
gain_s = pd.Series(gain).sort_values(ascending=False).head(15)
for k, v in gain_s.items():
    print(f"  {k:50s}  gain={v:.2f}")

print("\nГотово.")
