"""
Расширенный sensitivity-анализ: k_bars × k_atr.
Замеряем AUC_v, AUC_c, баланс выборки, доля строк, prec в хвостах.
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read("sber_intraday_pipeline.ipynb", as_version=4)
def cs(s): return "\n".join(l for l in s.split("\n") if not l.lstrip().startswith(("%","!")))
ns = {"__name__":"__not_main__"}
# Загрузим определения функций (cells 2,4,6) и данные (10) — но не cell 11, чтобы не строить feature_df с фиксированным target
for idx in [2,4,6,8,10]:
    try: exec(cs(nb.cells[idx].source), ns)
    except Exception as e: print(f"cell {idx} skip: {e}")

prepare = ns["prepare_ohlcv_dataframe"]
add_dom = ns["add_domain_features"]
add_roll= ns["add_rolling_features"]
add_cal = ns["add_calendar_features"]
add_ta  = ns["add_ta_features"]
build_X_y = ns["build_X_y_for_model"]

frame = ns["frame"]
df = prepare(frame); df = add_dom(df); df = add_roll(df, windows=(6,12,24)); df = add_cal(df); df = add_ta(df)
print(f"Базовый feature_df (без target): {df.shape}")

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
try:
    from sklearn.frozen import FrozenEstimator; USE_FROZEN=True
except ImportError: USE_FROZEN=False

def make_target(d, k_bars, k_atr):
    out = d.copy()
    fwd = (out["CLOSE"].shift(-k_bars) - out["CLOSE"]) / out["atr_14"]
    t = pd.Series(np.nan, index=out.index)
    t[fwd >= k_atr] = 1.0
    t[fwd <= -k_atr] = 0.0
    out["target_is_green_next"] = t
    out = out.dropna(subset=["target_is_green_next"]).reset_index(drop=True)
    out["target_is_green_next"] = out["target_is_green_next"].astype("int8")
    return out

results = []
GRID_K_BARS = [3,5,7,10]
GRID_K_ATR  = [0.75,1.0,1.25]
for kb in GRID_K_BARS:
    for ka in GRID_K_ATR:
        d = make_target(df, kb, ka)
        if len(d) < 1500:
            results.append({"k_bars":kb,"k_atr":ka,"n":len(d),"skip":"too_small"})
            print(f"k={kb} k_atr={ka}: SKIP, only {len(d)} rows"); continue
        try:
            X, y, _ = build_X_y(d, n_in=3)
        except Exception as e:
            print(f"k={kb} k_atr={ka}: BUILD FAIL {e}"); continue
        n=len(X); ntr=int(n*0.70); nva=int(n*0.15)
        Xtr,ytr=X.iloc[:ntr],y.iloc[:ntr]
        Xva,yva=X.iloc[ntr:ntr+nva],y.iloc[ntr:ntr+nva]
        Xca,yca=X.iloc[ntr+nva:],y.iloc[ntr+nva:]
        if yva.nunique()<2 or yca.nunique()<2:
            print(f"k={kb} k_atr={ka}: degenerate split"); continue
        clf = XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,
                            subsample=0.8,colsample_bytree=0.8,random_state=42,
                            n_jobs=-1,eval_metric="logloss")
        clf.fit(Xtr,ytr)
        try:
            cal = CalibratedClassifierCV(FrozenEstimator(clf),method="sigmoid") if USE_FROZEN \
                  else CalibratedClassifierCV(clf,method="sigmoid",cv="prefit")
            cal.fit(Xva,yva)
            p_v = cal.predict_proba(Xva)[:,1]
            p_c = cal.predict_proba(Xca)[:,1]
        except Exception:
            p_v = clf.predict_proba(Xva)[:,1]
            p_c = clf.predict_proba(Xca)[:,1]
        auc_v = roc_auc_score(yva,p_v); auc_c = roc_auc_score(yca,p_c)
        acc_v = accuracy_score(yva,(p_v>=0.5).astype(int)); acc_c = accuracy_score(yca,(p_c>=0.5).astype(int))

        # Хвосты на Calib (OOS)
        def slc(p,y,lo,hi,side):
            m=(p>=lo)&(p<=hi)
            if m.sum()==0: return 0,float("nan")
            n=int(m.sum()); prec=float((y[m]==(1 if side=="L" else 0)).mean())
            return n,prec
        nL,pL = slc(p_c,yca.values,0.55,1.0,"L")
        nS,pS = slc(p_c,yca.values,0.0,0.45,"S")

        balance = float((y==1).mean())
        share = len(d)/len(df)
        row = dict(k_bars=kb,k_atr=ka,n_total=int(len(d)),share_of_raw=round(share,3),
                   balance_up=round(balance,3),
                   auc_valid=round(float(auc_v),4),auc_calib=round(float(auc_c),4),
                   acc_valid=round(float(acc_v),4),acc_calib=round(float(acc_c),4),
                   p_min=round(float(p_c.min()),3),p_max=round(float(p_c.max()),3),
                   long_p55_n=int(nL),long_p55_prec=round(pL,3) if nL else None,
                   short_p45_n=int(nS),short_p45_prec=round(pS,3) if nS else None)
        results.append(row)
        print(f"k={kb} k_atr={ka:>4}: N={len(d):>6}  AUC_v={auc_v:.3f} AUC_c={auc_c:.3f}  L55={nL}@{pL if not np.isnan(pL) else float('nan'):.3f}  S45={nS}@{pS if not np.isnan(pS) else float('nan'):.3f}")

with open("scripts/sensitivity_grid_results.json","w") as f:
    json.dump(results,f,indent=2,default=str)
print("\nSaved scripts/sensitivity_grid_results.json")
