"""Summarize frozen methods; proxy maxima are never converted to official Anon."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/runs/20260925-G20-001'
METRICS=[('mia','all','tpr_at_fpr_0.01','mia_all'),('mia','true_member_rare_detected_control','tpr_at_fpr_0.01','mia_rare'),
 ('aia','all','r_raw','aia_all'),('aia','stratified_all','r_raw','aia_stratified'),('aia','rare_proxy','r_se_adjusted','aia_rare_se')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['exploration','confirmation'],required=True);args=ap.parse_args()
    files=([OUT/f'{prefix}_fold{f}_matrix.csv' for f in [2701,2702,2703] for prefix in ['shortlist','geography']]
           if args.stage=='exploration' else [OUT/f'confirmation_fold{f}_matrix.csv' for f in [2801,2802,2803]])
    m=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    assert not m.duplicated(['candidate','fold','version','channel','method','group']).any()
    m.to_csv(OUT/f'{args.stage}_all_methods.csv',index=False,lineterminator='\n')
    rows=[]
    for candidate,g in m.groupby('candidate'):
        c=ROOT/candidate;score=c.with_name(c.stem.removeprefix('C_')+'_score.json')
        row={'candidate':candidate,'U':json.loads(score.read_text())['utility'] if score.exists() else None}
        for channel,group,metric,label in METRICS:
            v=g[g.channel.eq(channel)&g.group.eq(group)]
            assert v[metric].notna().all()
            per=v.groupby('fold')[metric].max()
            row[label+'_max']=per.max();row[label+'_foldmax_mean']=per.mean();row[label+'_foldmax_min']=per.min()
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('U',ascending=False)
    table.to_csv(OUT/f'{args.stage}_summary.csv',index=False,lineterminator='\n')
    # Counts and binomial uncertainty describe fixed predictions, not selection-adjusted inference.
    counts=m[m.channel.eq('aia')&m.group.isin(['all','rare_proxy'])].copy()
    for tag in ['member','control']:
        n=counts['n_'+tag];p=counts['p_'+tag[0]];h=np.rint(n*p)
        assert np.allclose(h/n,p)
        counts[tag+'_hits']=h
        center=(p+1.96**2/(2*n))/(1+1.96**2/n)
        half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/(1+1.96**2/n)
        counts[tag+'_wilson_low']=np.maximum(0,center-half);counts[tag+'_wilson_high']=np.minimum(1,center+half)
    counts.to_csv(OUT/f'{args.stage}_aia_counts.csv',index=False,lineterminator='\n')
    scenarios=[]
    for r in rows:
        if r['U'] is None:continue
        for anon in [.4,.6,.676312,.7,.8,.9,1.]:
            scenarios.append({'candidate':r['candidate'],'assumed_Anon':anon,'hypothetical_score':min(r['U'],anon),
              'baseline_absolute_ceiling':.676311999928737,'label':'assumption, not estimate'})
    pd.DataFrame(scenarios).to_csv(OUT/f'{args.stage}_hypothetical_scores.csv',index=False,lineterminator='\n')
    cols=['candidate','U']+[x[3]+'_max' for x in METRICS]
    print(table[cols].to_string(index=False))

if __name__=='__main__':main()
