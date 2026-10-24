"""E2 follow-up. Evaluate every reporting-limit scenario on the same nonanchor cohort.

s11 scored unanchored scenarios on all 26,703 points and anchored scenarios on the 24,126
points in nonanchor PWSs. This script re-scores the stored out-of-fold predictions of s11
(no refitting) so that anchored and unanchored fits share one evaluation cohort, and reports
the fraction of the full-resolution (M = 4 ng/L) skill recovered by the anchors.
Outputs: results/e2_metrics_nonanchor.csv, results/e2_anchor_recovery.csv.
"""
import pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common_data import load, events, HERE

R = HERE / "results"
X, y_lo_true, y_hi_true, meta = load()
pred = pd.read_parquet(R / "e2_pred.parquet")
pred = pred[~pred.anchor]

met = []
for (M, a, n, t), d in pred.groupby(["M", "a", "model", "t"]):
    e = events(y_lo_true[d.idx.values], y_hi_true[d.idx.values], t)
    nb = e.mean() * (1 - e.mean())
    met.append(dict(M=M, a=a, model=n, t=t, n=len(d), AUC=roc_auc_score(e, d.p),
                    PRAUC=average_precision_score(e, d.p),
                    BSS=(1 - brier_score_loss(e, d.p) / nb) if n != "CLF_M" else np.nan,
                    prev_ratio=(d.p.mean() / e.mean()) if n != "CLF_M" else np.nan))
met = pd.DataFrame(met)
met.to_csv(R / "e2_metrics_nonanchor.csv", index=False)

# Share of the M = 4 ng/L skill lost by coarsening that the 10% anchors recover (CURE-N, t = 4).
c = met[(met.model == "CURE-N") & (met.t == 4)].set_index(["M", "a"])
ref = c.loc[(4, 0.0)]
rec = []
for M in (10, 20, 40):
    u, an = c.loc[(M, 0.0)], c.loc[(M, 0.1)]
    rec.append(dict(M=M, BSS_ref=ref.BSS, BSS_unanchored=u.BSS, BSS_anchored=an.BSS,
                    BSS_recovered=(an.BSS - u.BSS) / (ref.BSS - u.BSS),
                    AP_ref=ref.PRAUC, AP_unanchored=u.PRAUC, AP_anchored=an.PRAUC,
                    AP_recovered=(an.PRAUC - u.PRAUC) / (ref.PRAUC - u.PRAUC)))
rec = pd.DataFrame(rec)
rec.to_csv(R / "e2_anchor_recovery.csv", index=False)
print(rec.round(4).to_string(index=False))
