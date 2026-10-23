"""Shared loader: PFOA/PFOS target (log10 ng/L, left-censored) and retrospective covariates."""
import pathlib, re
import numpy as np
import pandas as pd

from release_paths import WORK as HERE
CAT = ["Size", "WaterType", "SPType", "State", "Region", "OWNER_TYPE_CODE"]
THRESHOLDS = [4.0, 6.0, 8.0, 10.0, 15.0, 20.0]


def feature_group(c):
    """Interpretation groups used for grouped SHAP."""
    if c in ("lat", "lon", "State", "Region", "n_zip", "zip_area_log"):
        return "Geography"
    if c.startswith("U3_") or any(c.endswith(s) for s in ("_PFOA", "_PFOS", "_PFHxS", "_PFHpA")):
        return "Legacy PFAS (UCMR3)"
    if c.startswith("L_"):
        return "Legacy co-contaminants"
    if c in ("WaterType", "SPType", "Size", "frac_sw", "n_sp", "pop_log", "conn_log"):
        return "Source & scale"
    return "System attributes"


def load(with_knowledge=False):
    df = pd.read_parquet(HERE / "data/model_table.parquet")
    det = df[["PFOA_det", "PFOS_det"]].fillna(0).max(axis=1).astype(bool).values
    val = np.where(det, np.fmax(df.PFOA_ng.where(df.PFOA_det == 1), df.PFOS_ng.where(df.PFOS_det == 1)),
                   np.fmax(df.PFOA_ng, df.PFOS_ng))
    ok = np.isfinite(val)
    df, det, val = df[ok].reset_index(drop=True), det[ok], val[ok]
    y_hi = np.log10(val)
    y_lo = np.where(det, y_hi, -np.inf)
    drop = [c for c in df.columns if c.endswith("_det") or c.endswith("_ng")] + \
           ["PWSID", "FacilityID", "SamplePointID", "first_date", "n_events"]
    if not with_knowledge:
        drop += [c for c in df.columns if c.startswith("K_") or c.startswith("treat_")]
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in X.columns]
    for c in CAT:
        X[c] = X[c].astype("category")
    meta = df[["PWSID", "FacilityID", "SamplePointID", "State", "Region", "WaterType", "lat", "lon"]].copy()
    return X, y_lo, y_hi, meta


def events(y_lo, y_hi, t):
    """Observed exceedance 1{C >= t}; valid because UCMR5 MRL (4 ng/L) <= t."""
    return (np.isfinite(y_lo) & (y_hi >= np.log10(t) - 1e-9)).astype(int)
