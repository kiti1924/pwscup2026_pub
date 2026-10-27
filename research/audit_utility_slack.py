"""Verify G22 perturbations, exact paired MIA metrics and control sensitivity."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from research.utility_slack import BASE,OUT,transform
from research.adaptive_privacy import dump,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--confirmation',action='store_true');a=ap.parse_args()
    prefix='confirmation_' if a.confirmation else ''
    m=pd.read_csv(OUT/(prefix+'all_methods.csv'));base=pd.read_csv(BASE)
    ref=m[m.candidate.eq(BASE.relative_to(ROOT).as_posix())&m.channel.eq('mia')]
    keys=['fold','version','method','group'];cols=['tpr_at_fpr_0.01','auc','n_member','n_control']
    ref=ref.set_index(keys).sort_index();checks=[]
    for mp in sorted(OUT.glob('jitter_*_manifest.json')):
        r=json.loads(mp.read_text());path=ROOT/r['candidate'];c=pd.read_csv(path)
        pd.testing.assert_frame_equal(c,transform(base,r['width']))
        present=m.candidate.eq(r['candidate']).any()
        if present:
            test=m[m.candidate.eq(r['candidate'])&m.channel.eq('mia')].set_index(keys).sort_index()
            pd.testing.assert_frame_equal(test[cols],ref[cols],check_exact=True)
        checks.append(dict(candidate=r['candidate'],sha256=digest(path),reproduced=True,evaluated=bool(present)))
    controls=m[m.candidate.str.endswith('C_positive_raw.csv')&m.group.eq('all')].copy()
    controls['observed']=np.where(controls.channel.eq('mia'),controls['tpr_at_fpr_0.01'],controls.r_raw)
    controls['threshold']=np.where(controls.channel.eq('mia'),.9,.8)
    controls['pass']=controls.observed>=controls.threshold
    controls.to_csv(OUT/(prefix+'raw_positive_checks.csv'),index=False,lineterminator='\n')
    dump(OUT/(prefix+'invariants.json'),dict(checks=checks,all_evaluated_mia_metrics_exact=True,
         positive_failures=controls.loc[~controls['pass'],['fold','method','observed','threshold']].to_dict('records'),
         limitation='Failed positive methods are not safety evidence; proxy risk is not official Anon'))
    print(f'{prefix}Reproduced {len(checks)} candidates; all evaluated MIA equal; positive failures {sum(~controls["pass"])}')

if __name__=='__main__':main()
