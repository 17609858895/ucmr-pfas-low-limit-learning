"""Data-support check: UCMR5 final, overlap with UCMR3 PFAS, additional data elements."""
import pandas as pd, numpy as np, pathlib
from release_paths import RAW, DATA
P = RAW / "ucmr5"
OUT = DATA
u5 = pd.read_csv(P/"UCMR5_All.txt", sep="\t", encoding="latin-1", dtype=str)
u5["val"] = pd.to_numeric(u5.AnalyticalResultValue, errors="coerce")
u5["MRL"] = pd.to_numeric(u5.MRL, errors="coerce")
u5["date"] = pd.to_datetime(u5.CollectionDate, format="%m/%d/%Y", errors="coerce")
u5.drop(columns=["PWSName","FacilityName","SamplePointName","AnalyticalResultValue","CollectionDate"]).to_parquet(OUT/"ucmr5_final.parquet", index=False)
print("UCMR5 rows", len(u5), "years", u5.date.dt.year.value_counts().sort_index().to_dict())
print(u5.Units.value_counts().to_dict())
key = ["PWSID","FacilityID","SamplePointID"]
pf = u5[u5.Contaminant.isin(["PFOA","PFOS","PFHxS","PFNA","HFPO-DA","PFBS"])]
sp = pf.pivot_table(index=key, columns="Contaminant", values="val", aggfunc="max")  # µg/L, NaN = all <MRL
print("sample points", len(sp), "PWS", sp.index.get_level_values(0).nunique())
ex = ((sp["PFOA"]>=0.004)|(sp["PFOS"]>=0.004))
print("SP with max PFOA or PFOS >=4 ng/L:", ex.mean().round(4), ex.sum())
pws_ex = ex.groupby(level=0).max(); print("PWS any SP exceed:", pws_ex.mean().round(4), pws_ex.sum(), "of", len(pws_ex))
print(u5.drop_duplicates(key).FacilityWaterType.value_counts().to_dict(), u5.drop_duplicates("PWSID").Size.value_counts().to_dict())
# UCMR3 PFAS
al = pd.read_parquet(OUT/"ucmr_all.parquet")
u3 = al[(al.CollectionDate.dt.year.between(2012,2016)) & al.Contaminant.isin(["PFOA","PFOS","PFHxS","PFNA","PFBS","PFHpA"])]
print("UCMR3 PFAS MRL by analyte:", u3.groupby("Contaminant",observed=True).MRL.agg(["min","median","max"]).to_dict("index"))
p3 = set(u3.PWSID.astype(str)); p5=set(u5.PWSID)
print("PWS in UCMR3 PFAS", len(p3), "in UCMR5", len(p5), "overlap", len(p3&p5))
k3 = set(map(tuple, u3[key].astype(str).drop_duplicates().values)); k5=set(sp.index)
print("SP-key overlap U3/U5:", len(k3&k5), "of U3", len(k3))
# legacy coverage UCMR1-4 per PWS
for lo,hi,nm in [(2000,2005,"UCMR1"),(2008,2011,"UCMR2"),(2013,2016,"UCMR3"),(2018,2021,"UCMR4")]:
    s=set(al[al.CollectionDate.dt.year.between(lo,hi)].PWSID.astype(str)); print(nm, "PWS", len(s), "overlap U5", len(s&p5))
ad = pd.read_csv(P/"UCMR5_AddtlDataElem.txt", sep="\t", encoding="latin-1", dtype=str)
print(ad.AdditionalDataElement.value_counts().to_dict())
for e in ["PFASTreatment","PFASOccurrence","PotentialPFASSources","TreatmentInformation","DisinfectantType"]:
    print(e, ad[ad.AdditionalDataElement==e].Response.value_counts().head(15).to_dict())
ad.to_parquet(OUT/"ucmr5_addtl.parquet", index=False)
