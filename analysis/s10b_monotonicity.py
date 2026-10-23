"""E1 add-on: threshold-consistency of separately trained classifiers with a practical tolerance.
A point is inconsistent if P(C >= t_{k+1}) exceeds P(C >= t_k) by more than `tol` for some k.
Distribution models (substitution, AFT, CURE, CURE-Stack) are consistent by construction.
Output: results/e1_monotonicity.csv
"""
import pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common_data import THRESHOLDS, HERE
O = pd.read_parquet(HERE / "results/e1_oof.parquet")
rows = []
for m in ["LR", "RF", "ET", "XGB", "CatBoost", "LGB", "MLP", "CURE-N", "STACK"]:
    P = np.column_stack([O[f"{m}@{t:g}"] for t in THRESHOLDS])
    d = np.diff(P, axis=1)
    for tol in [0.0, 0.01, 0.02]:
        rows.append(dict(model=m, tol=tol, frac_inconsistent=float((d > tol + 1e-12).any(axis=1).mean()),
                         max_inversion=float(d.max())))
r = pd.DataFrame(rows); r.to_csv(HERE / "results/e1_monotonicity.csv", index=False)
print(r.pivot(index="model", columns="tol", values="frac_inconsistent").round(3))
