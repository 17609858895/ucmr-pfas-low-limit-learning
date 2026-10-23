"""Robustness of the AIPW treatment effects: (a) no system-knowledge confounders,
(b) contaminated-source subgroup (UCMR3 PFAS detected or system reports prior PFAS occurrence),
(c) placebo: treatment permuted within State (should give ~0)."""
import pathlib, sys, re
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("c", pathlib.Path(__file__).parent / "s05_causal_dml.py")
src = open(spec.origin, encoding="utf-8").read().split("rows = []")[0]   # reuse data prep + aipw()
exec(src)
OUT = ["PFOS", "PFOA", "PFHxS", "PFHpA", "PFBS", "PFHxA", "PFPeA", "PFBA", "Lithium (neg. control)"]
rows = []
noK = Wd.drop(columns=[c for c in Wd.columns if c.startswith("K_")])
sub_cont = df[(df.U3_pfas_anydet == 1) | (df.K_pfas_occ_yes == 1)]
rng = np.random.default_rng(3)
plc = df.copy()
plc["T_perm"] = plc.groupby("State", observed=True)["T"].transform(lambda s: rng.permutation(s.values))
for tag, W, data, tcol in [("no_system_knowledge", noK, df, "T"), ("contaminated_subgroup", Wd, sub_cont, "T"),
                           ("placebo_permuted", Wd, plc, "T_perm")]:
    for name in OUT:
        r = aipw(tcol, OUTCOMES[name], W, data, B=300)
        r.update(analysis=tag, outcome=name); rows.append(r)
        print(f"{tag:22s} {name:24s} n={r['n']} T={r['n_treated']} ATT={r['ATT']:+.3f} [{r['ATT_lo']:+.3f},{r['ATT_hi']:+.3f}]", flush=True)
pd.DataFrame(rows).to_csv(R / "causal_robustness.csv", index=False)
