"""Build the sample-point modelling table.

Unit: UCMR5 sample point (PWSID x FacilityID x SamplePointID).
Targets (left-censored, ng/L): max over UCMR5 events of each PFAS; nondetect -> censored at MRL.
Predictors combine pre-2022 chemistry with retrospective registry descriptors:
  - SDWIS system attributes, ZIP-centroid geography, sample point/source type
  - PWS-level legacy fingerprints from UCMR1-4 (incl. UCMR3 PFAS at high MRLs)
Optional "system knowledge" block from UCMR5 additional data elements (kept separate).
"""
import pathlib
import numpy as np
import pandas as pd

from release_paths import WORK as HERE
from release_paths import REPO as ROOT, RAW
D = HERE / "data"
key = ["PWSID", "FacilityID", "SamplePointID"]
PFAS = ["PFOA", "PFOS", "PFHxS", "PFNA", "PFBS", "PFHpA", "HFPO-DA", "PFBA", "PFPeA", "PFHxA"]

u5 = pd.read_parquet(D / "ucmr5_final.parquet")
u5 = u5[u5.SamplePointType.isin(["EP", "MR"])]

# ---------- targets ----------
tg = []
for c in PFAS:
    d = u5[u5.Contaminant == c]
    g = d.groupby(key).agg(val=("val", "max"), mrl=("MRL", "max"), n=("val", "size"),
                           first=("date", "min"))
    g[c + "_det"] = g.val.notna().astype(int)
    g[c + "_ng"] = np.where(g.val.notna(), g.val * 1000, g.mrl * 1000)  # value or censoring point
    tg.append(g[[c + "_det", c + "_ng"]])
T = pd.concat(tg, axis=1)
ev = u5.groupby(key).agg(n_events=("SampleEventCode", "nunique"), first_date=("date", "min"),
                         Size=("Size", "first"), WaterType=("FacilityWaterType", "first"),
                         SPType=("SamplePointType", "first"), State=("State", "first"),
                         Region=("Region", "first"))
T = ev.join(T).reset_index()

# ---------- SDWIS system attributes ----------
pws = pd.read_csv(RAW / "sdwa/SDWA_PUB_WATER_SYSTEMS.csv", dtype=str,
                  usecols=["PWSID", "PWS_TYPE_CODE", "OWNER_TYPE_CODE", "POPULATION_SERVED_COUNT",
                           "PRIMARY_SOURCE_CODE", "IS_WHOLESALER_IND", "SERVICE_CONNECTIONS_COUNT",
                           "FIRST_REPORTED_DATE", "SOURCE_WATER_PROTECTION_CODE"]).drop_duplicates("PWSID")
pws["pop_log"] = np.log10(pd.to_numeric(pws.POPULATION_SERVED_COUNT, errors="coerce").clip(lower=1))
pws["conn_log"] = np.log10(pd.to_numeric(pws.SERVICE_CONNECTIONS_COUNT, errors="coerce").clip(lower=1))
pws["sys_age"] = 2023 - pd.to_datetime(pws.FIRST_REPORTED_DATE, format="%m/%d/%Y", errors="coerce").dt.year
pws["purchased"] = pws.PRIMARY_SOURCE_CODE.str.endswith("P").astype(float)
pws["wholesaler"] = (pws.IS_WHOLESALER_IND == "Y").astype(float)
pws = pws[["PWSID", "pop_log", "conn_log", "sys_age", "purchased", "wholesaler", "OWNER_TYPE_CODE"]]

# ---------- ZIP centroid geography ----------
z = pd.read_csv(RAW / "ucmr5/UCMR5_ZIPCodes.txt", sep="\t", dtype=str)
gz = pd.read_csv(RAW / "gazetteer/2023_Gaz_zcta_national.txt", sep="\t", dtype=str)
gz.columns = [c.strip() for c in gz.columns]
gz["lat"] = pd.to_numeric(gz.INTPTLAT); gz["lon"] = pd.to_numeric(gz.INTPTLONG)
gz["aland"] = pd.to_numeric(gz.ALAND_SQMI)
z = z.merge(gz[["GEOID", "lat", "lon", "aland"]], left_on="ZIPCODE", right_on="GEOID", how="left")
geo = z.groupby("PWSID").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_zip=("ZIPCODE", "nunique"),
                             zip_area_log=("aland", lambda s: np.log10(s.sum() + 1)))

# ---------- PWS structure in UCMR5 ----------
st = u5.drop_duplicates(key).groupby("PWSID").agg(
    n_sp=("SamplePointID", "size"), frac_sw=("FacilityWaterType", lambda s: s.isin(["SW", "GU"]).mean()))

# ---------- legacy UCMR1-4 fingerprints (PWS level) ----------
al = pd.read_parquet(D / "ucmr_all.parquet")
al["year"] = al.CollectionDate.dt.year
al = al[al.year <= 2021]
al["round"] = pd.cut(al.year, [1999, 2006, 2012, 2017, 2022], labels=["U1", "U2", "U3", "U4"])
al["v"] = pd.to_numeric(al.AnalyticalResultValue, errors="coerce")
al["det"] = (al.AnalyticalResultsSign == "=").astype(float)
summ = al.groupby("Contaminant", observed=True).agg(pws=("PWSID", "nunique"), det=("det", "mean"))
keep = summ[(summ.pws >= 1000) & (summ.det >= 0.004) & (summ.det < 0.999)].index.tolist()
keep = [k for k in keep if not k.startswith("HAA")]
print("legacy analytes kept:", len(keep), keep)
L = al[al.Contaminant.isin(keep)]
L = L.assign(Contaminant=L.Contaminant.astype(str))
fd = L.pivot_table(index="PWSID", columns="Contaminant", values="det", aggfunc="mean", observed=True)
fd.columns = ["L_detf_" + c for c in fd.columns]
lm = L.assign(lv=np.log10(L.v.where(L.v > 0))).pivot_table(index="PWSID", columns="Contaminant",
                                                           values="lv", aggfunc="max", observed=True)
lm.columns = ["L_lmax_" + c for c in lm.columns]
haa = al[al.Contaminant.isin(["HAA5", "HAA9", "HAA6Br"])].pivot_table(
    index="PWSID", columns="Contaminant", values="v", aggfunc="median", observed=True)
haa.columns = ["L_med_" + str(c) for c in haa.columns]
rounds = al.groupby("PWSID", observed=True)["round"].apply(lambda s: set(s.astype(str)))
rr = pd.DataFrame({f"in_{r}": rounds.apply(lambda s: float(r in s)) for r in ["U1", "U2", "U3", "U4"]})
leg = pd.concat([fd, lm, haa, rr], axis=1)
leg.index = leg.index.astype(str)
# UCMR3 PFAS summary (high MRLs): any detection of 6 PFAS and log max
p3 = al[(al["round"] == "U3") & al.Contaminant.isin(["PFOA", "PFOS", "PFHxS", "PFNA", "PFBS", "PFHpA"])]
leg["U3_pfas_anydet"] = p3.groupby("PWSID", observed=True).det.max().reindex(leg.index.astype(str)).values
leg["U3_pfas_ndet"] = p3[p3.det == 1].groupby("PWSID", observed=True).Contaminant.nunique().reindex(
    leg.index.astype(str)).fillna(0).values
leg.loc[leg.U3_pfas_anydet.isna(), "U3_pfas_ndet"] = np.nan

# ---------- optional system-knowledge block (UCMR5 additional data elements) ----------
ad = pd.read_parquet(D / "ucmr5_addtl.parquet")
def flag(elem, val):
    d = ad[ad.AdditionalDataElement == elem]
    return d.groupby(key).Response.apply(lambda s: float((s == val).any()))
K = pd.concat({"K_pfas_occ_yes": flag("PFASOccurrence", "Yes"),
               "K_pfas_occ_dk": flag("PFASOccurrence", "DK"),
               "K_pot_src_yes": flag("PotentialPFASSources", "Yes")}, axis=1)
tr = ad[ad.AdditionalDataElement == "PFASTreatment"].groupby(key).Response.apply(set)
K["treat_reported"] = tr.reindex(K.index).notna().astype(float)
K["treat_any"] = tr.apply(lambda s: float(bool(s & {"GAC", "IEX", "NRO", "PAC", "OTH", "BAC", "MFL"})))
K["treat_gac"] = tr.apply(lambda s: float("GAC" in s or "PAC" in s))
K["treat_iex_ro"] = tr.apply(lambda s: float(bool(s & {"IEX", "NRO"})))
K = K.reset_index()

# ---------- assemble ----------
X = (T.merge(pws, on="PWSID", how="left").merge(geo, left_on="PWSID", right_index=True, how="left")
       .merge(st, left_on="PWSID", right_index=True, how="left")
       .merge(leg, left_on="PWSID", right_index=True, how="left")
       .merge(K, on=key, how="left"))
X.to_parquet(D / "model_table.parquet", index=False)
print(X.shape)
print("legacy coverage (any UCMR1-4):", X[["in_U1", "in_U2", "in_U3", "in_U4"]].notna().any(axis=1).mean())
print("lat coverage", X.lat.notna().mean(), "pop coverage", X.pop_log.notna().mean())
y = (X[["PFOA_det", "PFOS_det"]].fillna(0).max(axis=1))
print("PFOA/PFOS >=MRL (4 ng/L) prevalence:", y.mean(), len(y))
