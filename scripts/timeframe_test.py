"""
Тест на агрегированных 10m и 15m таймфреймах.
Ресемплируем сырой 5m → 10m/15m, прогоняем тот же pipeline с ATR-target (k=5, k_atr=1.0).
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, nbformat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
nb = nbformat.read("sber_intraday_pipeline.ipynb", as_version=4)
def cs(s): return "\n".join(l for l in s.split("\n") if not l.lstrip().startswith(("%","!")))
ns = {"__name__":"__not_main__"}
# нам нужны только функции и raw frame (cells 2,4,6,8,10), без cell 11 (он строит feature_df на 5m)
for idx in [2,4,6,8,10]:
    try: exec(cs(nb.cells[idx].source), ns)
    except: pass

frame = ns["frame"]; build_X_y = ns["build_X_y_for_model"]
prepare = ns["prepare_ohlcv_dataframe"]; add_dom=ns["add_domain_features"]
add_roll=ns["add_rolling_features"]; add_cal=ns["add_calendar_features"]; add_ta=ns["add_ta_features"]
add_target = ns["add_target"]

def resample_ohlcv(df_raw, rule):
    df = prepare(df_raw).copy()
    df = df.set_index("datetime")
    agg = df.resample(rule, label="right", closed="right").agg(
        TICKER=("TICKER","first"), PER=("PER","first"),
        DATE=("DATE","first"), TIME=("TIME","first"),
        OPEN=("OPEN","first"), HIGH=("HIGH","max"),
        LOW=("LOW","min"), CLOSE=("CLOSE","last"),
        VOL=("VOL","sum"),
    ).dropna(subset=["OPEN","CLOSE"]).reset_index()
    return agg

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
try:
    from sklearn.frozen import FrozenEstimator; USE_FROZEN=True
except: USE_FROZEN=False

results = []
configs = [
    ("5min",  "5min",  5, 1.0),   # baseline (на чистом frame, для сравнения)
    ("10min", "10min", 5, 1.0),
    ("15min", "15min", 5, 1.0),
    ("10min", "10min", 3, 1.0),   # на 10m k_bars=3 эквивалентно ~30 мин
    ("15min", "15min", 3, 1.0),   # ~45 мин
]
for label, rule, k_bars, k_atr in configs:
    print(f"\n=== {label}, k_bars={k_bars}, k_atr={k_atr} ===")
    if rule == "5min":
        # без ресемпла (исходный)
        df = prepare(frame)
    else:
        df = resample_ohlcv(frame, rule)
    print(f"  Bars after resample: {len(df):,}")
    df = add_dom(df); df = add_roll(df, windows=(6,12,24))
    df = add_cal(df); df = add_ta(df)
    df = add_target(df, k_bars=k_bars, k_atr=k_atr)
    print(f"  After ATR-target: {len(df):,} bars (balance up={df['target_is_green_next'].mean():.3f})")
    if len(df) < 1500:
        print(f"  SKIP, too small"); continue
    try:
        X, y, _ = build_X_y(df, n_in=3)
    except Exception as e:
        print(f"  build_X_y FAIL: {e}"); continue
    # чистка inf/-inf, появляются при ресемпле (деление на 0 в ratio-фичах)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    n=len(X); ntr=int(n*0.70); nva=int(n*0.15)
    Xtr,ytr=X.iloc[:ntr],y.iloc[:ntr]
    Xva,yva=X.iloc[ntr:ntr+nva],y.iloc[ntr:ntr+nva]
    Xca,yca=X.iloc[ntr+nva:],y.iloc[ntr+nva:]
    if yva.nunique()<2 or yca.nunique()<2: print("  degenerate split"); continue

    clf = XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                        colsample_bytree=0.8,random_state=42,n_jobs=-1,eval_metric="logloss")
    clf.fit(Xtr,ytr)
    try:
        cal = CalibratedClassifierCV(FrozenEstimator(clf),method="sigmoid") if USE_FROZEN \
              else CalibratedClassifierCV(clf,method="sigmoid",cv="prefit")
        cal.fit(Xva,yva)
        p_v = cal.predict_proba(Xva)[:,1]; p_c = cal.predict_proba(Xca)[:,1]
    except:
        p_v = clf.predict_proba(Xva)[:,1]; p_c = clf.predict_proba(Xca)[:,1]
    auc_v=roc_auc_score(yva,p_v); auc_c=roc_auc_score(yca,p_c)
    acc_c=accuracy_score(yca,(p_c>=0.5).astype(int)); ll_c=log_loss(yca,p_c)

    def slc(p,y,lo,hi,side):
        m=(p>=lo)&(p<=hi); n=int(m.sum())
        if n==0: return 0,float("nan")
        return n, float((y[m]==(1 if side=="L" else 0)).mean())
    nL,pL = slc(p_c,yca.values,0.55,1.0,"L"); nS,pS = slc(p_c,yca.values,0.0,0.45,"S")

    row = dict(timeframe=label, k_bars=k_bars, k_atr=k_atr,
               n_total=int(len(df)), n_features=int(X.shape[1]),
               train=int(len(Xtr)),valid=int(len(Xva)),calib=int(len(Xca)),
               balance_up=round(float(y.mean()),3),
               auc_valid=round(float(auc_v),4), auc_calib=round(float(auc_c),4),
               acc_calib=round(float(acc_c),4), logloss_calib=round(float(ll_c),4),
               p_min=round(float(p_c.min()),3), p_max=round(float(p_c.max()),3),
               long_p55_n=int(nL), long_p55_prec=round(pL,3) if nL else None,
               short_p45_n=int(nS), short_p45_prec=round(pS,3) if nS else None)
    results.append(row)
    print(f"  AUC_v={auc_v:.3f} AUC_c={auc_c:.3f} Acc_c={acc_c:.3f} L55={nL}@{pL if not np.isnan(pL) else 0:.3f} S45={nS}@{pS if not np.isnan(pS) else 0:.3f}")

with open("scripts/timeframe_test_results.json","w") as f:
    json.dump(results,f,indent=2,default=str)
print("\nSaved scripts/timeframe_test_results.json")
