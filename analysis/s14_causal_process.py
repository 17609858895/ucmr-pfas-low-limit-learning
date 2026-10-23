"""Treatment-process split: GAC/PAC-only vs any IEX/RO, each against 'no treatment' (NMT).
Mechanistic prediction: IEX/RO should lower short-chain acids more than GAC/PAC."""
import pathlib
import numpy as np, pandas as pd
src = open(pathlib.Path(__file__).parent / "s05_causal_dml.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)
df["T_iexro"] = df.tset.apply(lambda s: 1 if s & {"IEX", "NRO"} else (0 if s == {"NMT"} else -1))
rows = []
for tcol, label in [("T_gac", "GAC/PAC only"), ("T_iexro", "IEX/RO")]:
    for name, ycol in OUTCOMES.items():
        r = aipw(tcol, ycol, Wd, df, B=500); r.update(treatment=label, outcome=name, chain=CHAIN.get(name, np.nan))
        # E-value for the ATT on the risk-ratio scale (VanderWeele & Ding 2017)
        m0 = r["base_rate_treated_if_untreated"]; rr = max((m0 + r["ATT"]) / max(m0, 1e-6), 1e-6)
        rr = rr if rr >= 1 else 1 / rr
        r["E_value"] = rr + np.sqrt(rr * (rr - 1))
        rows.append(r)
        print(f"{label:12s} {name:24s} n={r['n']} T={r['n_treated']} ATT={r['ATT']:+.3f} [{r['ATT_lo']:+.3f},{r['ATT_hi']:+.3f}] E={r['E_value']:.2f}", flush=True)
pd.DataFrame(rows).to_csv(R / "causal_process_split.csv", index=False)
