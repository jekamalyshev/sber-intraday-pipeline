"""
Мини-исследование: сравнение моделей классификации с калибровкой
на тех же данных, что и финальный пайплайн (n_in=3, pruning по permutation importance).
"""
import os, sys, warnings, time, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
NB_PATH = os.path.join(REPO_ROOT, "sber_intraday_pipeline.ipynb")

# --- 1. Загружаем функции feature engineering, выполняя только нужные ячейки ноутбука ---
nb = nbformat.read(NB_PATH, as_version=4)

def clean_source(src: str) -> str:
    # Удаляем IPython-магию и shell-вызовы (%matplotlib, %time, !pip и т.п.)
    cleaned = []
    for line in src.split("\n"):
        s = line.lstrip()
        if s.startswith("%") or s.startswith("!"):
            continue
        if s.startswith("display(") or s == "display":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

ns = {"__name__": "__not_main__"}

# Выполняем code-ячейки 0..11 (определения функций + загрузка frame + build_feature_dataframe)
# Затем сами выполним build_X_y_for_model + split + pruning, чтобы не тащить в ns plt/sns.
target_indices = [2, 4, 6, 8, 10, 11]  # imports, helpers, FE funcs, builder, load CSV, build feature_df
for cell in nb.cells:
    pass

code_cells = [(i, c.source) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
print(f"Всего code-cells: {len(code_cells)}")

for idx, src in code_cells:
    if idx not in target_indices:
        continue
    src_clean = clean_source(src)
    print(f"  exec cell idx={idx} ...")
    try:
        exec(src_clean, ns)
    except Exception as e:
        print(f"    [warn] {type(e).__name__}: {e}")

assert "feature_df" in ns, "feature_df не получен"
assert "build_X_y_for_model" in ns
print(f"feature_df: {ns['feature_df'].shape}")

# --- 2. Делаем свой split + pruning ---
build_X_y_for_model = ns["build_X_y_for_model"]
feature_df = ns["feature_df"]

X, y, leakage_cols = build_X_y_for_model(feature_df, n_in=3)
print(f"X={X.shape}, y={y.shape}")

n = len(X)
n_train = int(n * 0.70); n_valid = int(n * 0.15)
X_train = X.iloc[:n_train];                    y_train = y.iloc[:n_train]
X_valid = X.iloc[n_train:n_train+n_valid];     y_valid = y.iloc[n_train:n_train+n_valid]
X_calib = X.iloc[n_train+n_valid:];            y_calib = y.iloc[n_train+n_valid:]
print(f"Train={X_train.shape}, Valid={X_valid.shape}, Calib={X_calib.shape}")

# --- 3. Обучаем XGB и делаем permutation pruning ---
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_for_pi = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=20,
    eval_metric="logloss", random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=30, n_jobs=-1,
)
print("Training XGB for PI...")
t0 = time.time()
xgb_for_pi.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(f"  best_iter={xgb_for_pi.best_iteration} ({time.time()-t0:.1f}s)")

print("Permutation importance...")
t0 = time.time()
pi = permutation_importance(
    xgb_for_pi, X_valid, y_valid,
    n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_log_loss",
)
print(f"  ({time.time()-t0:.1f}s)")
keep_cols = X_train.columns[pi.importances_mean > 0].tolist()
print(f"Оставляем {len(keep_cols)} из {X_train.shape[1]} признаков.")

X_tr = X_train[keep_cols]; X_va = X_valid[keep_cols]; X_ca = X_calib[keep_cols]
y_tr, y_va, y_ca = y_train, y_valid, y_calib

# --- 4. Стандартизированные версии для линейных моделей ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_tr)
X_tr_s = pd.DataFrame(scaler.transform(X_tr), index=X_tr.index, columns=X_tr.columns)
X_va_s = pd.DataFrame(scaler.transform(X_va), index=X_va.index, columns=X_va.columns)
X_ca_s = pd.DataFrame(scaler.transform(X_ca), index=X_ca.index, columns=X_ca.columns)

# --- 5. Метрики ---
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_prob); ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        acc = (y_true[m] == (y_prob[m] >= 0.5)).mean()
        conf = y_prob[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)

def eval_split(name, model, Xv, yv):
    p = model.predict_proba(Xv)[:, 1]
    return {
        f"{name}_Acc":     round(accuracy_score(yv, p >= 0.5), 4),
        f"{name}_AUC":     round(roc_auc_score(yv, p), 4),
        f"{name}_LogLoss": round(log_loss(yv, p, labels=[0, 1]), 4),
        f"{name}_Brier":   round(brier_score_loss(yv, p), 4),
        f"{name}_ECE":     round(expected_calibration_error(yv, p), 4),
    }

# --- 6. Кандидаты и калибровка ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
    USE_FROZEN = True
except ImportError:
    USE_FROZEN = False
print(f"USE_FROZEN={USE_FROZEN}")

def calibrate(base, Xc, yc, method):
    if USE_FROZEN:
        return CalibratedClassifierCV(FrozenEstimator(base), method=method, cv=None).fit(Xc, yc)
    return CalibratedClassifierCV(base, method=method, cv="prefit").fit(Xc, yc)

candidates = []

print("\n[1/7] LogReg L2"); t=time.time()
m = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("LogReg L2", m, X_tr_s, X_va_s, X_ca_s))

print("[2/7] LogReg L1"); t=time.time()
m = LogisticRegression(penalty="l1", C=0.5, solver="saga", max_iter=4000, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("LogReg L1", m, X_tr_s, X_va_s, X_ca_s))

print("[3/7] DecisionTree"); t=time.time()
m = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("DecisionTree", m, X_tr, X_va, X_ca))

print("[4/7] RandomForest"); t=time.time()
m = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1).fit(X_tr, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("RandomForest", m, X_tr, X_va, X_ca))

print("[5/7] ExtraTrees"); t=time.time()
m = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1).fit(X_tr, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("ExtraTrees", m, X_tr, X_va, X_ca))

print("[6/7] HistGradientBoosting"); t=time.time()
m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=42, early_stopping=True, validation_fraction=0.15).fit(X_tr, y_tr)
print(f"  {time.time()-t:.1f}s"); candidates.append(("HistGB", m, X_tr, X_va, X_ca))

print("[7/7] XGBoost (pruned)"); t=time.time()
m = XGBClassifier(
    n_estimators=600, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=20,
    eval_metric="logloss", random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=30, n_jobs=-1,
).fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"  {time.time()-t:.1f}s, best_iter={m.best_iteration}")
candidates.append(("XGBoost", m, X_tr, X_va, X_ca))

# --- 7. Замеряем raw / Platt / Isotonic ---
rows = []
for name, model, Xtr, Xva, Xca in candidates:
    print(f"\n=== {name} ===")
    raw_v = eval_split("Valid", model, Xva, y_va)
    raw_c = eval_split("Calib", model, Xca, y_ca)
    rows.append({"Model": name, "Variant": "raw", **raw_v, **raw_c})

    cal_p = calibrate(model, Xca, y_ca, method="sigmoid")
    pl_v = eval_split("Valid", cal_p, Xva, y_va); pl_c = eval_split("Calib", cal_p, Xca, y_ca)
    rows.append({"Model": name, "Variant": "Platt", **pl_v, **pl_c})

    cal_i = calibrate(model, Xca, y_ca, method="isotonic")
    is_v = eval_split("Valid", cal_i, Xva, y_va); is_c = eval_split("Calib", cal_i, Xca, y_ca)
    rows.append({"Model": name, "Variant": "Isotonic", **is_v, **is_c})

    print(f"  raw      AUC_v={raw_v['Valid_AUC']:.4f}  Brier_v={raw_v['Valid_Brier']:.4f}  ECE_v={raw_v['Valid_ECE']:.4f}")
    print(f"  Platt    AUC_v={pl_v['Valid_AUC']:.4f}  Brier_v={pl_v['Valid_Brier']:.4f}  ECE_v={pl_v['Valid_ECE']:.4f}")
    print(f"  Isotonic AUC_v={is_v['Valid_AUC']:.4f}  Brier_v={is_v['Valid_Brier']:.4f}  ECE_v={is_v['Valid_ECE']:.4f}")

df = pd.DataFrame(rows)
out_csv = os.path.join(REPO_ROOT, "scripts", "model_comparison_results.csv")
df.to_csv(out_csv, index=False)
print(f"\nСохранено: {out_csv}")
print("\n=== ИТОГОВАЯ ТАБЛИЦА (Valid) ===")
print(df[["Model","Variant","Valid_Acc","Valid_AUC","Valid_LogLoss","Valid_Brier","Valid_ECE"]].to_string(index=False))
