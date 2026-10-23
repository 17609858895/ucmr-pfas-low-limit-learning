"""Within-PWS fixed-effects contrast: treated vs untreated entry points of the same system
(removes all system-level confounding; well-level contamination differences remain)."""
import pathlib
import numpy as np, pandas as pd
from release_paths import WORK as HERE
key = ["PWSID", "FacilityID", "SamplePointID"]
df = pd.read_parquet(HERE / "data/model_table.parquet")
u5 = pd.read_parquet(HERE / "data/ucmr5_final.parquet")
li = u5[u5.Contaminant == "lithium"].groupby(key).val.max()
df = df.merge(li.notna().astype(float).rename("lithium_det").reset_index(), on=key, how="left")
ad = pd.read_parquet(HERE / "data/ucmr5_addtl.parquet")
tr = ad[ad.AdditionalDataElement == "PFASTreatment"].groupby(key).Response.apply(set).rename("t").reset_index()
d = df.merge(tr, on=key)
cap = {"GAC", "PAC", "IEX", "NRO"}
d["T"] = d.t.apply(lambda s: 1 if s & cap else (0 if s == {"NMT"} else -1)); d = d[d["T"] >= 0]
g = d.groupby("PWSID")["T"].agg(["min", "max"]); m = d[d.PWSID.isin(g[(g["min"] == 0) & (g["max"] == 1)].index)]
grp = {k: v for k, v in m.groupby("PWSID")}; ids = np.array(list(grp))
def fe(s, y):
    gi = s.PWSID.values if "PWSID" in s else s.index.get_level_values(0)
    dy = s[y] - s.groupby(gi)[y].transform("mean"); dt = s["T"] - s.groupby(gi)["T"].transform("mean")
    return (dy * dt).sum() / (dt ** 2).sum()
rows, rng = [], np.random.default_rng(0)
for c in ["PFOS", "PFOA", "PFHxS", "PFHpA", "PFBS", "PFHxA", "PFPeA", "PFBA", "lithium"]:
    y = c + "_det"; mm = m[m[y].notna()]
    b = fe(mm, y); bs = []
    for _ in range(1000):
        s = pd.concat([grp[i] for i in rng.choice(ids, len(ids))], keys=range(len(ids)))
        s = s[s[y].notna()].droplevel(1); s.index.name = "bid"; s = s.reset_index(drop=False).rename(columns={"bid": "B"})
        s["PWSID"] = s["B"]; bs.append(fe(s, y))
    rows.append(dict(outcome=c, n_sp=len(mm), n_pws=mm.PWSID.nunique(), n_treated=int(mm["T"].sum()), FE=b,
                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / "results/causal_within_pws_FE.csv", index=False)
