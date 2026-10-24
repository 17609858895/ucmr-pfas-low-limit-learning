"""Check published aggregate results with the Python standard library only.

This verifies arithmetic and manuscript values from retained summaries. It does
not refit models or independently validate predictions against observations.
"""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'retained_results'

def read(name):
    with (R/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def select(name, **keys):
    rows = [r for r in read(name) if all(str(r[k])==str(v) for k,v in keys.items())]
    assert len(rows)==1, (name,keys,len(rows))
    return rows[0]

def main():
    recal = [r for r in read('e6_recal_metrics.csv') if r['model']=='CURE-N recal']
    assert len(recal)==6
    ratios = [float(r['prev_ratio']) for r in recal]
    assert round(min(ratios),3)==.965 and round(max(ratios),3)==1.005
    row = select('e6_recal_metrics.csv', model='CURE-N recal', t='4.0')
    assert [round(float(row[k]),3) for k in ['AUC','PRAUC','BSS']]==[.835,.491,.229]
    no_anchor = select('e2_metrics.csv', M='20', a='0.0', model='CURE-N', t='4.0')
    anchor = select('e2_metrics.csv', M='20', a='0.1', model='CURE-N', t='4.0')
    assert round(float(no_anchor['BSS']),3)==.060
    assert round(float(anchor['BSS']),3)==.120
    same = {(r['M'],r['a']):r for r in read('e2_metrics_nonanchor.csv') if r['model']=='CURE-N' and r['t']=='4.0'}
    assert int(same[('20','0.0')]['n'])==24126
    assert round(float(same[('20','0.0')]['BSS']),3)==.063 and round(float(same[('20','0.1')]['BSS']),3)==.120
    assert round(float(same[('40','0.0')]['BSS']),3)==.012 and round(float(same[('40','0.1')]['BSS']),3)==.088
    rec = {r['M']:r for r in read('e2_anchor_recovery.csv')}
    for M in ['20','40']:
        ref,u,a = (float(rec[M][k]) for k in ['BSS_ref','BSS_unanchored','BSS_anchored'])
        assert math.isclose((a-u)/(ref-u), float(rec[M]['BSS_recovered']), rel_tol=1e-9)
    assert [round(100*float(rec[M]['BSS_recovered'])) for M in ['20','40']]==[35,36]
    assert [round(100*float(rec[M]['AP_recovered'])) for M in ['20','40']]==[33,39]
    mono = [r for r in read('e1_monotonicity.csv') if r['model'] in ['CURE-N','STACK']]
    assert len(mono)==6 and all(float(r['frac_inconsistent'])==0 for r in mono)
    marker = select('legacy_associations.csv', analyte='1,4-Dioxane')
    assert round(float(marker['RR']),2)==3.39
    assert math.isclose(float(marker['rate_det'])/float(marker['rate_nodet']),float(marker['RR']),rel_tol=1e-12)
    features = read('e4_shap_features.csv')
    assert len(features)==78
    for col in ['incidence_share','intensity_share']:
        assert abs(sum(float(r[col]) for r in features)-1)<1e-6
    legacy = select('e4_shap_groups.csv', group='Legacy co-contaminants')
    assert round(100*float(legacy['incidence_share']))==47
    assert round(100*float(legacy['intensity_share']))==41
    triage = select('e5_summary.csv', alpha='0.1', source='All')
    assert int(triage['n'])==26703
    assert round(100*float(triage['frac_certified_compliant']),1)==45.9
    assert round(100*float(triage['frac_sampling']),1)==38.7
    assert abs(sum(float(triage[k]) for k in ['frac_certified_compliant','frac_certified_exceed','frac_sampling'])-1)<1e-12
    historical = json.loads((R/'historical_analyte_observed.json').read_text())
    assert all(v['n_pws']==4435 for v in historical.values())
    joint = read('historical_joint_by_area.csv')
    assert sum(int(r['n']) for r in joint)==4439
    assert sum(int(r['n'])>=20 for r in joint)==39
    report = {'status':'PASS', 'checks':['six recalibration ratios', '4 ng/L discrimination and Brier skill',
              '20 ng/L anchor and unanchored Brier skill', 'same-cohort anchor comparison and recovered skill', 'probability consistency', 'dioxane rate-ratio arithmetic',
              '78 feature shares', 'co-contaminant group shares', 'prediction-set proportions',
              'distinct historical cohort sizes and 39 reporting areas'],
              'scope':'Aggregate-summary verification; no model refitting and no independent data validation.'}
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
