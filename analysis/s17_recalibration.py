"""E6. Cross-fitted distributional recalibration of CURE-N (fixes the ~16% prevalence under-prediction).

For each PWS fold k, three parameters are fitted on the out-of-fold predictions of the OTHER folds by
maximising the censored log-likelihood, then applied to fold k:
    logit pi' = logit pi + c,   mu' = mu + a,   sigma' = sigma * exp(b)
Because the whole predictive distribution is recalibrated, P(C >= t) stays monotone in t.
Outputs: results/e6_recal_oof.parquet (recalibrated pi/mu/sigma + P@t), e6_recal_metrics.csv, e6_recal_params.csv
"""
import pathlib, sys
import numpy as np, pandas as pd
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common_data import THRESHOLDS, events, HERE

R = HERE / "results"
O = pd.read_parquet(R / "e1_oof.parquet")
SIG = 0.8 / np.log(10)                      # CURE-N latency scale on log10 ng/L
y_lo, y_hi = O.y_lo.values, O.y_hi.values
det = np.isfinite(y_lo)
pi0 = np.clip(O.pi_CURE_N.values, 1e-6, 1 - 1e-6); mu0 = O.mu_CURE_N.values


def dist(theta, idx):
    a, b, c = theta
    pi = 1 / (1 + np.exp(-(np.log(pi0[idx] / (1 - pi0[idx])) + c)))
    return pi, mu0[idx] + a, SIG * np.exp(b)


def nll(theta, idx):
    pi, mu, s = dist(theta, idx)
    d = det[idx]
    ll_det = np.log(pi[d]) + stats.norm.logpdf((y_hi[idx][d] - mu[d]) / s) - np.log(s)
    ll_cen = np.log((1 - pi[~d]) + pi[~d] * stats.norm.cdf((y_hi[idx][~d] - mu[~d]) / s))
    return -(ll_det.sum() + ll_cen.sum()) / len(idx)


P = {t: np.zeros(len(O)) for t in THRESHOLDS}
PI, MU, S = np.zeros(len(O)), np.zeros(len(O)), np.zeros(len(O))
params = []
for k in sorted(O.fold.unique()):
    tr, te = np.where(O.fold != k)[0], np.where(O.fold == k)[0]
    r = optimize.minimize(nll, np.zeros(3), args=(tr,), method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-8))
    pi, mu, s = dist(r.x, te)
    PI[te], MU[te], S[te] = pi, mu, s
    for t in THRESHOLDS:
        P[t][te] = pi * stats.norm.sf((np.log10(t) - 1e-9 - mu) / s)
    params.append(dict(fold=int(k), a_mu=r.x[0], b_logsigma=r.x[1], c_logit_pi=r.x[2]))
pd.DataFrame(params).to_csv(R / "e6_recal_params.csv", index=False)

rows = []
for t in THRESHOLDS:
    e = events(y_lo, y_hi, t); nb = e.mean() * (1 - e.mean())
    for name, p in [("CURE-N", O[f"CURE-N@{t:g}"].values), ("CURE-N recal", P[t]), ("LGB", O[f"LGB@{t:g}"].values)]:
        rows.append(dict(model=name, t=t, AUC=roc_auc_score(e, p), PRAUC=average_precision_score(e, p),
                         BSS=1 - brier_score_loss(e, p) / nb, prev_ratio=p.mean() / e.mean()))
m = pd.DataFrame(rows); m.to_csv(R / "e6_recal_metrics.csv", index=False)
out = O[["PWSID", "FacilityID", "SamplePointID", "State", "Region", "WaterType", "lat", "lon", "fold", "y_lo", "y_hi"]].copy()
out["pi_recal"], out["mu_recal"], out["sigma_recal"] = PI, MU, S
for t in THRESHOLDS:
    out[f"CURE-R@{t:g}"] = P[t]
out.to_parquet(R / "e6_recal_oof.parquet", index=False)
print(pd.DataFrame(params).round(3).to_string())
print(m.pivot(index="t", columns="model", values="prev_ratio").round(3).to_string())
print(m.pivot(index="t", columns="model", values="BSS").round(4).to_string())
print(m.pivot(index="t", columns="model", values="PRAUC").round(4).to_string())
