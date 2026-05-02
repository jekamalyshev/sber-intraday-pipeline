"""
Сравнение моделей на ATR-target (k_bars=5, k_atr=1.0).
Каждая модель — с Platt-калибровкой. Метрики на Calib (OOS).
"""
import os, warnings, json, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read("sber_intraday_pipeline.ipynb", as_version=4)
def cs(s): return "\n".join(l for l in s.split("\n") if not l.lstrip().startswith(("%","!")))
ns = {"__name__":"__not_main__"}
for idx in [2,4,6,8,10,11]:
    try: exec(cs(nb.cells[idx].source), ns)
    except Exception as e: pass
feature_df = ns["feature_df"]; build_X_y = ns["build_X_y_for_model"]

X, y, _ = build_X_y(feature_df, n_in=3)
n=len(X); ntr=int(n*0.70); nva=int(n*0.15)
Xtr,ytr=X.iloc[:ntr],y.iloc[:ntr]
Xva,yva=X.iloc[ntr:ntr+nva],y.iloc[ntr:ntr+nva]
Xca,yca=X.iloc[ntr+nva:],y.iloc[ntr+nva:]
print(f"Splits: train={len(Xtr)}  valid={len(Xva)}  calib={len(Xca)}")
print(f"Балансы: tr={ytr.mean():.3f} va={yva.mean():.3f} ca={yca.mean():.3f}")

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
try:
    from sklearn.frozen import FrozenEstimator; USE_FROZEN=True
except: USE_FROZEN=False
try:
    from lightgbm import LGBMClassifier; HAS_LGB=True
except: HAS_LGB=False
try:
    from catboost import CatBoostClassifier; HAS_CB=True
except: HAS_CB=False

# заполняем NaN перед линейными моделями
Xtr_f = Xtr.fillna(0); Xva_f = Xva.fillna(0); Xca_f = Xca.fillna(0)

models = {
    "XGBoost": XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                             colsample_bytree=0.8,random_state=42,n_jobs=-1,eval_metric="logloss"),
    "RandomForest": RandomForestClassifier(n_estimators=400,max_depth=8,min_samples_leaf=20,
                                           random_state=42,n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=400,max_depth=10,min_samples_leaf=20,
                                       random_state=42,n_jobs=-1),
    "HistGBM": HistGradientBoostingClassifier(max_iter=300,max_depth=5,learning_rate=0.05,random_state=42),
    "LogReg(L2)": Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(C=0.1,max_iter=2000,n_jobs=-1))]),
}
if HAS_LGB:
    models["LightGBM"] = LGBMClassifier(n_estimators=400,max_depth=-1,num_leaves=31,learning_rate=0.05,
                                        subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
if HAS_CB:
    models["CatBoost"] = CatBoostClassifier(iterations=400,depth=5,learning_rate=0.05,verbose=False,random_state=42)

results = []
for name, m in models.items():
    t0 = time.time()
    Xt, Xv, Xc = (Xtr_f, Xva_f, Xca_f) if "LogReg" in name else (Xtr, Xva, Xca)
    try:
        m.fit(Xt, ytr)
    except Exception as e:
        print(f"{name}: FIT FAIL {e}"); continue
    try:
        cal = CalibratedClassifierCV(FrozenEstimator(m),method="sigmoid") if USE_FROZEN \
              else CalibratedClassifierCV(m,method="sigmoid",cv="prefit")
        cal.fit(Xv, yva)
        p_v = cal.predict_proba(Xv)[:,1]; p_c = cal.predict_proba(Xc)[:,1]
        calib = "Platt"
    except Exception as e:
        p_v = m.predict_proba(Xv)[:,1]; p_c = m.predict_proba(Xc)[:,1]
        calib = "raw"
    auc_v = roc_auc_score(yva,p_v); auc_c = roc_auc_score(yca,p_c)
    acc_c = accuracy_score(yca,(p_c>=0.5).astype(int))
    ll_c  = log_loss(yca,p_c); br_c = brier_score_loss(yca,p_c)
    dt = time.time() - t0
    row = dict(model=name, calib=calib, time_s=round(dt,1),
               auc_valid=round(float(auc_v),4), auc_calib=round(float(auc_c),4),
               acc_calib=round(float(acc_c),4), logloss_calib=round(float(ll_c),4), brier_calib=round(float(br_c),4),
               p_min=round(float(p_c.min()),3), p_max=round(float(p_c.max()),3))
    # хвосты OOS
    def slc(p,y,lo,hi,side):
        msk=(p>=lo)&(p<=hi); n=int(msk.sum())
        if n==0: return 0,float("nan")
        return n, float((y[msk]==(1 if side=="L" else 0)).mean())
    nL,pL = slc(p_c,yca.values,0.55,1.0,"L")
    nS,pS = slc(p_c,yca.values,0.0,0.45,"S")
    row.update(long_p55_n=int(nL), long_p55_prec=round(pL,3) if nL else None,
               short_p45_n=int(nS), short_p45_prec=round(pS,3) if nS else None)
    results.append(row)
    print(f"{name:14s} ({dt:5.1f}s, {calib}): AUC_v={auc_v:.3f} AUC_c={auc_c:.3f} Acc_c={acc_c:.3f} LL={ll_c:.3f} L55={nL}@{pL if not np.isnan(pL) else 0:.3f} S45={nS}@{pS if not np.isnan(pS) else 0:.3f}")

results.sort(key=lambda r: r["auc_calib"], reverse=True)
with open("scripts/model_zoo_atr_results.json","w") as f:
    json.dump(results,f,indent=2,default=str)
print("\nSaved scripts/model_zoo_atr_results.json")
