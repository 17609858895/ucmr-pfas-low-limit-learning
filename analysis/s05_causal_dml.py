"""Adjusted observational contrasts between treatment-report groups.

The original internal filename and ATT/ATE column names are retained.
These score summaries are associations and do not identify treatment effects.
All computational formulas and sample selection are unchanged.
"""
import json, pathlib, re, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from release_paths import WORK as HERE
R = HERE / "results"
df = pd.read_parquet(HERE / "data/model_table.parquet")
key = ["PWSID", "FacilityID", "SamplePointID"]
u5 = pd.read_parquet(HERE / "data/ucmr5_final.parquet")
li = u5[u5.Contaminant == "lithium"].groupby(key).val.max()
df = df.merge((li.notna()).rename("lithium_det").astype(float).reset_index(), on=key, how="left")

ad = pd.read_parquet(HERE / "data/ucmr5_addtl.parquet")
tr = ad[ad.AdditionalDataElement == "PFASTreatment"].groupby(key).Response.apply(set).rename("tset").reset_index()
df = df.merge(tr, on=key, how="inner")
cap = {"GAC", "PAC", "IEX", "NRO"}
df["T"] = df.tset.apply(lambda s: 1 if s & cap else (0 if s == {"NMT"} else -1))
df["T_gac"] = df.tset.apply(lambda s: 1 if s & {"GAC", "PAC"} and not s & {"IEX", "NRO"} else (0 if s == {"NMT"} else -1))
print("treatment sets:", df["T"].value_counts().to_dict())

cat = ["Size", "WaterType", "SPType", "State", "Region", "OWNER_TYPE_CODE"]
out_cols = [c for c in df.columns if c.endswith("_det") or c.endswith("_ng")]
Wd = df.drop(columns=out_cols + key + ["first_date", "n_events", "tset", "T", "T_gac", "treat_reported",
                                       "treat_any", "treat_gac", "treat_iex_ro"])
Wd.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in Wd.columns]
for c in cat:
    Wd[c] = Wd[c].astype("category")

OUTCOMES = {"PFOS": "PFOS_det", "PFOA": "PFOA_det", "PFHxS": "PFHxS_det", "PFHpA": "PFHpA_det",
            "PFHxA": "PFHxA_det", "PFBS": "PFBS_det", "PFPeA": "PFPeA_det", "PFBA": "PFBA_det",
            "Lithium (neg. control)": "lithium_det"}
CHAIN = {"PFOS": 8, "PFOA": 8, "PFHxS": 6, "PFHpA": 7, "PFHxA": 6, "PFBS": 4, "PFPeA": 5, "PFBA": 4}
kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.7, verbose=-1)


def aipw(tcol, ycol, W, sub, B=500, seed=0):
    d = sub[(sub[tcol] >= 0) & sub[ycol].notna()]
    Wx, T, Y, g = W.loc[d.index], d[tcol].values, d[ycol].values.astype(float), d.PWSID.values
    e, m1, m0 = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for trn, tst in GroupKFold(5).split(Wx, groups=g):
        e[tst] = lgb.LGBMClassifier(**kw).fit(Wx.iloc[trn], T[trn]).predict_proba(Wx.iloc[tst])[:, 1]
        for t, m in [(1, m1), (0, m0)]:
            s = trn[T[trn] == t]
            if len(np.unique(Y[s])) < 2:
                m[tst] = Y[s].mean(); continue
            m[tst] = lgb.LGBMClassifier(**kw).fit(Wx.iloc[s], Y[s]).predict_proba(Wx.iloc[tst])[:, 1]
    keep = (e > 0.02) & (e < 0.98)
    e, m1, m0, T, Y, g = e[keep], m1[keep], m0[keep], T[keep], Y[keep], g[keep]
    psi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
    # ATT score (treated population, which is the policy-relevant one)
    p = T.mean()
    psi_att = (T * (Y - m0) - (1 - T) * e * (Y - m0) / (1 - e)) / p
    rng = np.random.default_rng(seed)
    ug, gi = np.unique(g, return_inverse=True)
    bs, bt = [], []
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi]
        bs.append(np.average(psi, weights=c)); bt.append(np.average(psi_att, weights=c))
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    return dict(n=int(len(T)), n_treated=int(T.sum()), n_pws=int(len(ug)), kept_frac=float(keep.mean()),
                naive_diff=float(naive), ATE=float(psi.mean()), ATE_lo=float(np.percentile(bs, 2.5)),
                ATE_hi=float(np.percentile(bs, 97.5)), ATT=float(psi_att.mean()),
                ATT_lo=float(np.percentile(bt, 2.5)), ATT_hi=float(np.percentile(bt, 97.5)),
                base_rate_treated_if_untreated=float(np.mean(m0[T == 1])))


rows = []
for tcol, label in [("T", "GAC/IEX/RO"), ("T_gac", "GAC/PAC only")]:
    for name, ycol in OUTCOMES.items():
        r = aipw(tcol, ycol, Wd, df)
        r.update(treatment=label, outcome=name, chain=CHAIN.get(name, np.nan))
        rows.append(r)
        print(f"{label:12s} {name:24s} n={r['n']} T={r['n_treated']} naive={r['naive_diff']:+.3f} "
              f"ATT={r['ATT']:+.3f} [{r['ATT_lo']:+.3f},{r['ATT_hi']:+.3f}] "
              f"ATE={r['ATE']:+.3f} [{r['ATE_lo']:+.3f},{r['ATE_hi']:+.3f}] m0|T=1={r['base_rate_treated_if_untreated']:.3f}",
              flush=True)
res = pd.DataFrame(rows)
res.to_csv(R / "causal_aipw_treatment.csv", index=False)
