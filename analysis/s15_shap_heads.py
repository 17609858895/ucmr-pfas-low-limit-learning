"""E4. TreeSHAP for the two CURE-N heads fitted on all sampling points.

Incidence head: LightGBM classifier pi(x) (log-odds scale).
Intensity head: XGB-AFT mu(x) (natural-log concentration scale, via pred_contribs).
Outputs: results/e4_shap_features.csv (mean |SHAP| per feature and head), e4_shap_groups.csv,
         e4_shap_values.parquet (top features, for beeswarm), e4_rank_corr.json
"""
import json, pathlib, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import shap, xgboost as xgb
from scipy.stats import spearmanr
from common_data import load, feature_group, HERE
from censored_models import CureAFT

R = HERE / "results"
X, y_lo, y_hi, meta = load()
m = CureAFT("normal", 0.8, n_iter=6).fit(X, y_lo, y_hi)
sv_inc = shap.TreeExplainer(m.clf).shap_values(X)
sv_inc = sv_inc[1] if isinstance(sv_inc, list) else sv_inc
sv_int = m.aft.b.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)[:, :-1]
feat = pd.DataFrame({"feature": X.columns, "group": [feature_group(c) for c in X.columns],
                     "incidence": np.abs(sv_inc).mean(0), "intensity": np.abs(sv_int).mean(0)})
feat["incidence_share"] = feat.incidence / feat.incidence.sum()
feat["intensity_share"] = feat.intensity / feat.intensity.sum()
feat.to_csv(R / "e4_shap_features.csv", index=False)
grp = feat.groupby("group")[["incidence_share", "intensity_share"]].sum().reset_index()
grp.to_csv(R / "e4_shap_groups.csv", index=False)
rho = spearmanr(feat.incidence, feat.intensity).correlation
json.dump({"spearman_rank_heads": float(rho), "n": int(len(X))}, open(R / "e4_rank_corr.json", "w"), indent=1)
keep = []
for head, sv in [("incidence", sv_inc), ("intensity", sv_int)]:
    top = feat.sort_values(head, ascending=False).feature.head(12).tolist()
    for f in top:
        j = list(X.columns).index(f)
        x = X[f]
        xr = x.cat.codes.astype(float).replace(-1, np.nan) if str(x.dtype) == "category" else x.astype(float)
        xr = xr.rank(pct=True)
        keep.append(pd.DataFrame({"head": head, "feature": f, "shap": sv[:, j], "xrank": xr.values}))
pd.concat(keep).to_parquet(R / "e4_shap_values.parquet", index=False)
print(grp.round(3).to_string()); print("rank corr", rho)
