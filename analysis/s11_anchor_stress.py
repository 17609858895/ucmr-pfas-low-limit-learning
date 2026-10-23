"""E2. MRL stress test with anchor-calibrated stacking and multi-threshold evaluation.

Training labels are re-censored at a coarse reporting limit M; a fraction `a` of training PWS
(the anchor, mimicking a low-MRL monitoring round such as UCMR5 pooled with legacy UCMR3 data)
keeps the true MRL of 4 ng/L. Test labels are always the true UCMR5 labels.
Stacking variants:
  STACK         weights maximise inner-OOF censored log score on all training rows
  ANCHOR-STACK  weights (and hence distribution family) chosen on anchor rows only
Outputs: results/e2_pred.parquet (P(C>=t) for every model, M, a, t) and e2_metrics.csv.
"""
import pathlib, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common_data import load, events, THRESHOLDS, HERE
from censored_models import XGBAFT, CureAFT, LGBSubstitution, LGBClassifier, Mixture, Dist, dist_sub, fit_stack_weights

R = HERE / "results"
X, y_lo_true, y_hi_true, meta = load()
groups = meta.PWSID.values
folds = list(GroupKFold(5).split(X, groups=groups))
rng = np.random.default_rng(11)
upws = np.unique(groups)
anchor_pws = set(rng.choice(upws, int(0.10 * len(upws)), replace=False))
is_anchor = np.isin(groups, list(anchor_pws))
LT = np.log10(THRESHOLDS)
BASE = {"AFT-N": lambda: XGBAFT("normal", 0.8, rounds=600), "AFT-L": lambda: XGBAFT("logistic", 0.5, rounds=600),
        "CURE-N": lambda: CureAFT("normal", 0.8, n_iter=4, rounds=400),
        "CURE-L": lambda: CureAFT("logistic", 0.5, n_iter=4, rounds=400)}


def censor(M, a):
    lm = np.log10(M)
    coarse = ~(is_anchor & (a > 0))
    cens = (~np.isfinite(y_lo_true)) | (coarse & (y_hi_true < lm - 1e-9))
    y_hi = np.where(cens & coarse, np.maximum(lm, y_hi_true), y_hi_true)
    y_hi = np.where(cens & ~coarse, y_hi_true, y_hi)
    return np.where(cens, -np.inf, y_hi_true), y_hi


def sfm(d):
    return np.column_stack([d.sf(t - 1e-9) for t in LT])


rows, t0 = [], time.time()
for M, a in [(4, 0.0), (10, 0.0), (10, 0.1), (20, 0.0), (20, 0.1), (40, 0.0), (40, 0.1)]:
    y_lo, y_hi = censor(M, a)
    out = {n: np.zeros((len(X), len(LT))) for n in ["CLF_M", "SUB"] + list(BASE) + ["STACK", "ANCHOR-STACK"]}
    for k, (tr, te) in enumerate(folds):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        clf = LGBClassifier(np.log10(M)).fit(Xtr, y_lo[tr], y_hi[tr]).prob_exceed(Xte)
        out["CLF_M"][te] = clf[:, None]
        out["SUB"][te] = sfm(LGBSubstitution().fit(Xtr, y_lo[tr], y_hi[tr]).predict_dist(Xte))
        inner = list(GroupKFold(2).split(Xtr, groups=groups[tr]))
        d_te, d_in = [], []
        for n, mk in BASE.items():
            d = mk().fit(Xtr, y_lo[tr], y_hi[tr]).predict_dist(Xte)
            out[n][te] = sfm(d); d_te.append(d)
            mu, pi, sg, fam = np.zeros(len(tr)), np.ones(len(tr)), None, None
            for itr, ite in inner:
                di = mk().fit(Xtr.iloc[itr], y_lo[tr][itr], y_hi[tr][itr]).predict_dist(Xtr.iloc[ite])
                mu[ite], pi[ite], sg, fam = di.mu, di.pi, di.sigma[0], di.fam
            dd = Dist.__new__(Dist); dd.mu, dd.pi, dd.sigma, dd.fam = mu, pi, np.full(len(tr), sg), fam
            d_in.append(dd)
        w = fit_stack_weights(d_in, y_lo[tr], y_hi[tr])
        out["STACK"][te] = sfm(Mixture(d_te, w))
        anc = is_anchor[tr] if a > 0 else np.zeros(len(tr), bool)
        if anc.sum() > 50:
            wa = fit_stack_weights([dist_sub(d, anc) for d in d_in], y_lo[tr][anc], y_hi[tr][anc])
        else:
            wa = w
        out["ANCHOR-STACK"][te] = sfm(Mixture(d_te, wa))
        print(f"M={M} a={a} fold {k} {time.time()-t0:.0f}s w={np.round(w,2)} wa={np.round(wa,2)}", flush=True)
    for n, arr in out.items():
        for j, t in enumerate(THRESHOLDS):
            rows.append(pd.DataFrame({"M": M, "a": a, "model": n, "t": t, "idx": np.arange(len(X)),
                                      "p": arr[:, j], "anchor": is_anchor}))
    pd.concat(rows).to_parquet(R / "e2_pred.parquet", index=False)

pred = pd.concat(rows)
met = []
for (M, a, n, t), d in pred.groupby(["M", "a", "model", "t"]):
    d = d[~d.anchor] if a > 0 else d          # evaluate on non-anchor systems only
    e = events(y_lo_true[d.idx.values], y_hi_true[d.idx.values], t)
    nb = e.mean() * (1 - e.mean())
    met.append(dict(M=M, a=a, model=n, t=t, AUC=roc_auc_score(e, d.p), PRAUC=average_precision_score(e, d.p),
                    BSS=(1 - brier_score_loss(e, d.p) / nb) if n != "CLF_M" else np.nan,
                    prev_ratio=(d.p.mean() / e.mean()) if n != "CLF_M" else np.nan))
met = pd.DataFrame(met)
met.to_csv(R / "e2_metrics.csv", index=False)
print(met[met.t == 4].round(3).to_string())
