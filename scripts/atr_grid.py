"""
Гриф по конфигам k_bars, k_atr — где острее всего хвосты.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read("sber_intraday_pipeline.ipynb", as_version=4)
def cs(s): return "\n".join(l for l in s.split("\n") if not l.lstrip().startswith(("%","!")))
ns = {"__name__":"__not_main__"}
for idx, c in enumerate(nb.cells):
    if c.cell_type=="code" and idx in [2,4,6,8,10,11]:
        try: exec(cs(c.source), ns)
        except: pass
feature_df = ns["feature_df"]; build_X_y = ns["build_X_y_for_model"]

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
try:
    from sklearn.frozen import FrozenEstimator
    USE_FROZEN = True
except ImportError:
    USE_FROZEN = False

def make_atr_target(df, k_bars, k_atr):
    out = df.copy()
    fwd = (out["CLOSE"].shift(-k_bars) - out["CLOSE"]) / out["atr_14"]
    target = pd.Series(np.nan, index=out.index)
    target[fwd >= k_atr] = 1
    target[fwd <= -k_atr] = 0
    out["target_atr"] = target
    return out

def run_config(k_bars, k_atr, verbose=False):
    fd = make_atr_target(feature_df, k_bars, k_atr)
    fd = fd.dropna(subset=["target_atr"]).copy()
    fd["target_is_green_next"] = fd["target_atr"].astype(int)

    X, y, _ = build_X_y(fd, n_in=3)
    n = len(X); ntr=int(n*0.70); nva=int(n*0.15)
    X_tr, y_tr = X.iloc[:ntr], y.iloc[:ntr]
    X_va, y_va = X.iloc[ntr:ntr+nva], y.iloc[ntr:ntr+nva]
    X_ca, y_ca = X.iloc[ntr+nva:], y.iloc[ntr+nva:]

    scale = (y_tr==0).sum()/max((y_tr==1).sum(),1)
    m_b = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=20, eval_metric="logloss", random_state=42,
        scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
    m_b.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pi = permutation_importance(m_b, X_va, y_va, n_repeats=3, random_state=42, n_jobs=-1, scoring="neg_log_loss")
    keep = X_tr.columns[pi.importances_mean > 0].tolist()
    if len(keep) < 5: keep = X_tr.columns.tolist()
    X_tr_p, X_va_p, X_ca_p = X_tr[keep], X_va[keep], X_ca[keep]
    m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=20, eval_metric="logloss", random_state=42,
        scale_pos_weight=scale, early_stopping_rounds=30, n_jobs=-1)
    m.fit(X_tr_p, y_tr, eval_set=[(X_va_p, y_va)], verbose=False)
    if USE_FROZEN:
        cm = CalibratedClassifierCV(FrozenEstimator(m), method="sigmoid", cv=None).fit(X_ca_p, y_ca)
    else:
        cm = CalibratedClassifierCV(m, method="sigmoid", cv="prefit").fit(X_ca_p, y_ca)
    p_va = cm.predict_proba(X_va_p)[:,1]; p_ca = cm.predict_proba(X_ca_p)[:,1]
    yv = y_va.values; yc = y_ca.values

    auc_v = roc_auc_score(yv, p_va); auc_c = roc_auc_score(yc, p_ca)
    pmax_v = p_va.max(); pmin_v = p_va.min()

    # Хвосты на Valid
    n_long_55 = (p_va>=0.55).sum(); n_short_45 = (p_va<=0.45).sum()
    pr_long_55 = yv[p_va>=0.55].mean() if n_long_55>0 else float("nan")
    pr_short_45 = (1-yv[p_va<=0.45]).mean() if n_short_45>0 else float("nan")

    n_long_55_c = (p_ca>=0.55).sum(); n_short_45_c = (p_ca<=0.45).sum()
    pr_long_55_c = yc[p_ca>=0.55].mean() if n_long_55_c>0 else float("nan")
    pr_short_45_c = (1-yc[p_ca<=0.45]).mean() if n_short_45_c>0 else float("nan")

    return {
        "k_bars": k_bars, "k_atr": k_atr, "n_total": len(X),
        "AUC_v": auc_v, "AUC_c": auc_c, "p_min": pmin_v, "p_max": pmax_v,
        "n_L55_v": n_long_55, "prec_L55_v": pr_long_55,
        "n_S45_v": n_short_45, "prec_S45_v": pr_short_45,
        "n_L55_c": n_long_55_c, "prec_L55_c": pr_long_55_c,
        "n_S45_c": n_short_45_c, "prec_S45_c": pr_short_45_c,
    }

print(f"{'k_b':>4s} {'k_a':>4s} {'N_tot':>7s} {'AUC_v':>6s} {'AUC_c':>6s} {'p_max':>6s} "
      f"{'L55_v':>6s} {'prL_v':>6s} {'L55_c':>6s} {'prL_c':>6s} "
      f"{'S45_v':>6s} {'prS_v':>6s} {'S45_c':>6s} {'prS_c':>6s}")

configs = [
    (1, 0.5), (1, 0.75), (1, 1.0),
    (3, 0.5), (3, 0.75), (3, 1.0), (3, 1.25),
    (5, 0.75), (5, 1.0), (5, 1.5),
]

results = []
for kb, ka in configs:
    try:
        r = run_config(kb, ka)
        print(f"{r['k_bars']:>4d} {r['k_atr']:>4.2f} {r['n_total']:>7d} "
              f"{r['AUC_v']:>6.4f} {r['AUC_c']:>6.4f} {r['p_max']:>6.3f} "
              f"{r['n_L55_v']:>6d} {r['prec_L55_v']:>6.3f} {r['n_L55_c']:>6d} {r['prec_L55_c']:>6.3f} "
              f"{r['n_S45_v']:>6d} {r['prec_S45_v']:>6.3f} {r['n_S45_c']:>6d} {r['prec_S45_c']:>6.3f}")
        results.append(r)
    except Exception as e:
        print(f"{kb} {ka}: error {e}")

import json
with open("scripts/atr_grid_results.json", "w") as f:
    json.dump(results, f, default=str, indent=2)
print("\nСохранено: scripts/atr_grid_results.json")
