"""Recompute the nine historical-marker associations used in Table S12.

Derived from the adopted summary in 02/scripts/prepare_support.py. Uses an
explicit identifier join to the eligible PFOA/PFOS response cohort. This is an
observational association, not a causal or independent external validation.
"""
import numpy as np
import pandas as pd
from common_data import load, events, HERE

def main():
    _, lo, hi, meta = load()
    mt = pd.read_parquet(HERE/'data/model_table.parquet')
    keys = ['PWSID', 'FacilityID', 'SamplePointID']
    cols = [c for c in mt if c.startswith('L_detf_')]
    d = meta[keys].merge(mt[keys+cols], on=keys, how='left', validate='one_to_one')
    d['event'] = events(lo, hi, 4)
    pw = d.groupby('PWSID').agg(event=('event','max'), **{c:(c,'first') for c in cols})
    names = {'1,4-dioxane':'1,4-Dioxane', '1,1-dichloroethane':'1,1-DCA',
             'HCFC-22':'HCFC-22', 'chlorate':'Chlorate', '1,2,3-trichloropropane':'1,2,3-TCP',
             'NDMA':'NDMA', 'perchlorate':'Perchlorate', 'germanium':'Germanium', 'cobalt':'Cobalt'}
    rng = np.random.default_rng(0)
    rows = []
    for analyte, label in names.items():
        z = pw.dropna(subset=['L_detf_'+analyte])
        g = (z['L_detf_'+analyte]>0).to_numpy()
        y = z.event.to_numpy()
        bs = []
        for _ in range(2000):
            idx = rng.integers(0,len(z),len(z)); gg, yy = g[idx], y[idx]
            bs.append(yy[gg].mean()/yy[~gg].mean())
        rows.append(dict(analyte=label, n_det=int(g.sum()), n_nodet=int((~g).sum()),
                         rate_det=y[g].mean(), rate_nodet=y[~g].mean(),
                         RR=y[g].mean()/y[~g].mean(), RR_lo=np.percentile(bs,2.5),
                         RR_hi=np.percentile(bs,97.5)))
    out = pd.DataFrame(rows).sort_values('RR')
    out.to_csv(HERE/'results/legacy_associations.csv', index=False)
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
