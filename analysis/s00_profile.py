"""Profile UCMR1-5 combined CSV and cache to parquet."""
import pandas as pd, numpy as np, pathlib, json
from release_paths import REPO as ROOT, RAW
from release_paths import DATA as OUT
OUT.mkdir(exist_ok=True)
cat_cols = ["PWSID","Size","FacilityID","FacilityWaterType","SamplePointID","SamplePointType",
            "Contaminant","Units","MethodID","AnalyticalResultsSign","SampleEventCode",
            "MonitoringRequirement","Region","State","UCMR1SampleType"]
df = pd.read_csv(RAW/"combined_ucmr_data.csv", usecols=cat_cols+["CollectionDate","MRL","AnalyticalResultValue"],
                 dtype={c:"category" for c in cat_cols}, low_memory=False, encoding_errors="replace")
df["CollectionDate"] = pd.to_datetime(df["CollectionDate"], errors="coerce")
df.to_parquet(OUT/"ucmr_all.parquet", index=False)
print(df.shape)
df["year"] = df.CollectionDate.dt.year
print(df.groupby("year").size().to_string())
c = df.groupby("Contaminant", observed=True).agg(n=("PWSID","size"), pws=("PWSID","nunique"),
      det=("AnalyticalResultsSign", lambda s:(s=="=").mean()), y0=("year","min"), y1=("year","max"),
      unit=("Units", lambda s: s.mode().iat[0] if len(s.mode()) else None))
pd.set_option("display.width",200)
print(c.sort_values("y0").to_string())
print(df.Size.value_counts(), df.FacilityWaterType.value_counts(), df.AnalyticalResultsSign.value_counts())
