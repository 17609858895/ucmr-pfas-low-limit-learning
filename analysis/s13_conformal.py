"""E5. Label-conditional Mondrian conformal triage on out-of-fold recalibrated CURE probabilities (E6 = E1 CURE-N + cross-fitted distributional recalibration; final main model).

Strata: source type (groundwater vs surface/mixed). Cross-conformal: points in PWS fold k are
calibrated on the other folds. For each label y, include y if 1 - p_y <= q_y (finite-sample
quantile in the stratum). Outcomes: {0} certified compliant, {1} certified exceedance,
{0,1} or {} -> sampling needed.
Outputs: results/e5_triage.parquet, e5_summary.csv, e5_curve.csv
"""
import pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common_data import events, HERE

R = HERE / "results"
o = pd.read_parquet(R / "e6_recal_oof.parquet")
MODEL = "CURE-R@4"
p = o[MODEL].values
y = events(o.y_lo.values, o.y_hi.values, 4.0)
src = np.where(o.WaterType.astype(str) == "GW", "GW", "SW")


def qhat(s, alpha):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s)[k - 1]


def triage(alpha):
    lab = np.empty(len(p), object)
    for k in range(o.fold.max() + 1):
        te = o.fold.values == k
        for g in ("GW", "SW"):
            cal = (~te) & (src == g)
            q1 = qhat(1 - p[cal & (y == 1)], alpha)
            q0 = qhat(p[cal & (y == 0)], alpha)
            m = te & (src == g)
            in1 = (1 - p[m]) <= q1
            in0 = p[m] <= q0
            lab[m] = np.where(in1 & ~in0, "Certified exceedance",
                              np.where(in0 & ~in1, "Certified compliant", "Sampling needed"))
    return lab


rows, curve = [], []
for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    lab = triage(alpha)
    for g in ("GW", "SW", "All"):
        m = np.ones(len(p), bool) if g == "All" else src == g
        L, Y = lab[m], y[m]
        cov1 = np.mean(L[Y == 1] != "Certified compliant")      # exceedances not wrongly cleared
        cov0 = np.mean(L[Y == 0] != "Certified exceedance")
        rows.append(dict(alpha=alpha, source=g, n=int(m.sum()), coverage_exceed=cov1, coverage_compliant=cov0,
                         frac_certified_compliant=np.mean(L == "Certified compliant"),
                         frac_certified_exceed=np.mean(L == "Certified exceedance"),
                         frac_sampling=np.mean(L == "Sampling needed"),
                         missed_exceed_frac=1 - cov1))
    if alpha == 0.10:
        o["triage"] = lab
s = pd.DataFrame(rows)
s.to_csv(R / "e5_summary.csv", index=False)
o[["PWSID", "FacilityID", "SamplePointID", "State", "Region", "WaterType", "lat", "lon", MODEL, "triage"]].assign(
    e4=y).to_parquet(R / "e5_triage.parquet", index=False)
print(s.round(3).to_string())
