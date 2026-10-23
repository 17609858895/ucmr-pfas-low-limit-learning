"""E3 (variant via argv[1]='nostate': drop State/Region categoricals, keep lat/lon). Spatial transfer: leave-states-out (10 folds grouped by State) and leave-one-EPA-region-out.

Models: LightGBM classifier at 4 ng/L (best common family in E1 is re-checked there), XGB-AFT normal,
CURE-N. Per-state metrics feed the transfer map (Fig. 3f).
Outputs: results/e3_oof.parquet, e3_overall.csv, e3_by_state.csv
"""
import pathlib, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common_data import load, events, HERE
from censored_models import XGBAFT, CureAFT, LGBClassifier

R = HERE / "results"
X, y_lo, y_hi, meta = load()
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
if TAG == "nostate":
    X = X.drop(columns=["State", "Region"])
SUF = f"_{TAG}" if TAG else ""
T4 = np.log10(4.0)
e4 = events(y_lo, y_hi, 4.0)
designs = {"PWS": GroupKFold(5).split(X, groups=meta.PWSID.values),
           "State": GroupKFold(10).split(X, groups=meta.State.astype(str).values),
           "Region": GroupKFold(meta.Region.nunique()).split(X, groups=meta.Region.astype(str).values)}
oof = meta.copy()
rows = []
for dname, sp in designs.items():
    P = {m: np.zeros(len(X)) for m in ["LGB", "AFT-N", "CURE-N"]}
    for tr, te in sp:
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        P["LGB"][te] = LGBClassifier(T4).fit(Xtr, y_lo[tr], y_hi[tr]).prob_exceed(Xte)
        P["AFT-N"][te] = XGBAFT("normal", 0.8).fit(Xtr, y_lo[tr], y_hi[tr]).predict_dist(Xte).sf(T4 - 1e-9)
        P["CURE-N"][te] = CureAFT("normal", 0.8, n_iter=5).fit(Xtr, y_lo[tr], y_hi[tr]).predict_dist(Xte).sf(T4 - 1e-9)
    for m, p in P.items():
        oof[f"{dname}_{m}"] = p
        nb = e4.mean() * (1 - e4.mean())
        rows.append(dict(design=dname, model=m, AUC=roc_auc_score(e4, p), PRAUC=average_precision_score(e4, p),
                         BSS=1 - brier_score_loss(e4, p) / nb, prev_ratio=p.mean() / e4.mean()))
    print(dname, "done", flush=True)
oof["e4"] = e4
oof.to_parquet(R / f"e3_oof{SUF}.parquet", index=False)
ov = pd.DataFrame(rows); ov.to_csv(R / f"e3_overall{SUF}.csv", index=False)
print(ov.round(4).to_string())
st = []
for s, d in oof.groupby("State", observed=True):
    r = dict(State=s, n=len(d), n_pws=d.PWSID.nunique(), obs_prev=d.e4.mean(), lat=d.lat.mean(), lon=d.lon.mean())
    for m in ["LGB", "CURE-N"]:
        p = d[f"State_{m}"]
        r[f"pred_prev_{m}"] = p.mean()
        r[f"AUC_{m}"] = roc_auc_score(d.e4, p) if 0 < d.e4.sum() < len(d) else np.nan
        r[f"Brier_{m}"] = brier_score_loss(d.e4, p)
    st.append(r)
pd.DataFrame(st).to_csv(R / f"e3_by_state{SUF}.csv", index=False)
