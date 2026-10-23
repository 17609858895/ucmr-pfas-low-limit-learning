"""E1. Common algorithms vs censored survival learners, one PWS-grouped 5-fold split.

Classifiers (LR, RF, ExtraTrees, XGBoost, CatBoost, LightGBM, MLP) are trained separately for each
threshold t in {4,6,8,10,15,20} ng/L. Censored learners (MRL/2 substitution, XGB-AFT normal/logistic,
CURE normal/logistic, CURE-Stack) are trained once and evaluated at every threshold.
Outputs: results/e1_oof.parquet (all OOF probabilities + CURE-N pi/mu), e1_metrics.csv, e1_boot.csv.
"""
import json, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
from common_data import load, events, THRESHOLDS, CAT, HERE
from censored_models import XGBAFT, CureAFT, LGBSubstitution, Mixture, Dist, dist_sub, fit_stack_weights

R = HERE / "results"
X, y_lo, y_hi, meta = load()
groups = meta.PWSID.values
folds = list(GroupKFold(5).split(X, groups=groups))
num = [c for c in X.columns if c not in CAT]


def design(Xtr, Xte):
    A = pd.get_dummies(pd.concat([Xtr, Xte]), columns=CAT, dummy_na=False).astype(float)
    miss = A[num].isna().add_suffix("_na").astype(float)
    A = pd.concat([A, miss.loc[:, miss.iloc[:len(Xtr)].mean() > 0.01]], axis=1)
    A = A.fillna(A.iloc[:len(Xtr)].median()).fillna(0)
    sc = StandardScaler().fit(A.iloc[:len(Xtr)])
    Z = sc.transform(A)
    return Z[:len(Xtr)], Z[len(Xtr):]


def clf_models():
    return {
        "LR": ("Z", lambda: LogisticRegression(C=0.1, max_iter=3000)),
        "RF": ("Z", lambda: RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt", n_jobs=16, random_state=0)),
        "ET": ("Z", lambda: ExtraTreesClassifier(500, min_samples_leaf=5, max_features="sqrt", n_jobs=16, random_state=0)),
        "XGB": ("X", lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                                               colsample_bytree=0.7, enable_categorical=True, tree_method="hist", n_jobs=16)),
        "CatBoost": ("C", lambda: CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, verbose=0,
                                                     thread_count=16, random_seed=0)),
        "LGB": ("X", lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)),
        "MLP": ("Z", lambda: MLPClassifier((128, 64), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)),
    }


Xc = X.copy()
for c in CAT:
    Xc[c] = Xc[c].astype(str)
P = {}
dist_store = {n: {} for n in ["SUB", "AFT-N", "AFT-L", "CURE-N", "CURE-L"]}
fold_id = np.zeros(len(X), int)
t0 = time.time()
for k, (tr, te) in enumerate(folds):
    fold_id[te] = k
    Ztr, Zte = design(X.iloc[tr], X.iloc[te])
    for name, (kind, mk) in clf_models().items():
        for t in THRESHOLDS:
            y = events(y_lo[tr], y_hi[tr], t)
            m = mk()
            if kind == "Z":
                p = m.fit(Ztr, y).predict_proba(Zte)[:, 1]
            elif kind == "C":
                p = m.fit(Xc.iloc[tr], y, cat_features=CAT).predict_proba(Xc.iloc[te])[:, 1]
            else:
                p = m.fit(X.iloc[tr], y).predict_proba(X.iloc[te])[:, 1]
            P.setdefault(f"{name}@{t:g}", np.zeros(len(X)))[te] = p
        print(f"fold {k} {name} {time.time()-t0:.0f}s", flush=True)
    for name, mk in [("SUB", LGBSubstitution), ("AFT-N", lambda: XGBAFT("normal", 0.8)),
                     ("AFT-L", lambda: XGBAFT("logistic", 0.5)), ("CURE-N", lambda: CureAFT("normal", 0.8)),
                     ("CURE-L", lambda: CureAFT("logistic", 0.5))]:
        dist_store[name][k] = mk().fit(X.iloc[tr], y_lo[tr], y_hi[tr]).predict_dist(X.iloc[te])
    print(f"fold {k} censored done {time.time()-t0:.0f}s", flush=True)


def assemble(n):
    mu, sg, pi = np.zeros(len(X)), np.zeros(len(X)), np.ones(len(X)); fam = None
    for k, d in dist_store[n].items():
        i = fold_id == k
        mu[i], sg[i], pi[i], fam = d.mu, d.sigma, d.pi, d.fam
    o = Dist.__new__(Dist); o.mu, o.sigma, o.pi, o.fam = mu, sg, pi, fam
    return o


full = {n: assemble(n) for n in dist_store}
base = ["AFT-N", "AFT-L", "CURE-N", "CURE-L"]
W = {}
stack_p = {t: np.zeros(len(X)) for t in THRESHOLDS}
for k in range(5):
    tr, te = fold_id != k, fold_id == k
    w = fit_stack_weights([dist_sub(full[n], tr) for n in base], y_lo[tr], y_hi[tr])
    W[k] = dict(zip(base, map(float, w)))
    mix = Mixture([dist_sub(full[n], te) for n in base], w)
    for t in THRESHOLDS:
        stack_p[t][te] = mix.sf(np.log10(t) - 1e-9)
for n, d in full.items():
    for t in THRESHOLDS:
        P[f"{n}@{t:g}"] = d.sf(np.log10(t) - 1e-9)
for t in THRESHOLDS:
    P[f"STACK@{t:g}"] = stack_p[t]

oof = pd.DataFrame(P)
oof = pd.concat([meta.reset_index(drop=True), pd.DataFrame({"fold": fold_id, "y_lo": y_lo, "y_hi": y_hi,
                 "pi_CURE_N": full["CURE-N"].pi, "mu_CURE_N": full["CURE-N"].mu}), oof], axis=1)
oof.to_parquet(R / "e1_oof.parquet", index=False)
json.dump(W, open(R / "e1_stack_weights.json", "w"), indent=1)

# ---------- metrics ----------
models = list(clf_models()) + ["SUB", "AFT-N", "AFT-L", "CURE-N", "CURE-L", "STACK"]
rows = []
for m in models:
    for t in THRESHOLDS:
        e = events(y_lo, y_hi, t); p = P[f"{m}@{t:g}"]
        nb = e.mean() * (1 - e.mean())
        rows.append(dict(model=m, threshold=t, prevalence=e.mean(), AUC=roc_auc_score(e, p),
                         PRAUC=average_precision_score(e, p), Brier=brier_score_loss(e, p),
                         BSS=1 - brier_score_loss(e, p) / nb, pred_prev=p.mean()))
    if m in clf_models():   # monotonicity across thresholds (a single distribution is monotone by construction)
        M = np.column_stack([P[f"{m}@{t:g}"] for t in THRESHOLDS])
        rows[-1]["frac_nonmonotone"] = float((np.diff(M, axis=1) > 1e-9).any(axis=1).mean())
met = pd.DataFrame(rows)
met.to_csv(R / "e1_metrics.csv", index=False)
# PWS-cluster bootstrap CIs at 4 ng/L
rng = np.random.default_rng(0)
ug, gi = np.unique(groups, return_inverse=True)
e4 = events(y_lo, y_hi, 4.0)
B = []
for b in range(300):
    idx = np.repeat(np.arange(len(X)), np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi])
    for m in models:
        p = P[f"{m}@4"][idx]; e = e4[idx]
        B.append(dict(b=b, model=m, AUC=roc_auc_score(e, p), PRAUC=average_precision_score(e, p),
                      BSS=1 - brier_score_loss(e, p) / (e.mean() * (1 - e.mean()))))
pd.DataFrame(B).to_csv(R / "e1_boot.csv", index=False)
print(met[met.threshold == 4].round(4).to_string())
print(met.groupby("model").frac_nonmonotone.max())
