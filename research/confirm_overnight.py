"""Evaluate a preregistered frozen single CSV against incumbent and controls."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import research.background_partition as bp
from research.adaptive_privacy import dump,digest,DIST
from research.geography_free_attacks import evaluate as evaluate_geo
from research.evaluate_background_transport import anchors
from research.overnight_refinement import versioned_matrix
from research.summarize_tradeoff import METRICS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--fold',type=int);ap.add_argument('--summarize',action='store_true');a=ap.parse_args()
    out=a.out.resolve();frozen=json.loads((out/'frozen_selection.json').read_text())
    if a.summarize:
        m=pd.concat([pd.read_csv(out/f'confirmation_fold{f}_matrix.csv') for f in frozen['confirmation_folds']],ignore_index=True)
        m.to_csv(out/'confirmation_all_methods.csv',index=False,lineterminator='\n');rows=[]
        for candidate,g in m.groupby('candidate'):
            p=ROOT/candidate;score=p.with_name(p.stem.removeprefix('C_')+'_score.json')
            row={'candidate':candidate,'U':json.loads(score.read_text())['utility'] if score.exists() else None}
            for channel,group,metric,label in METRICS:
                per=g[g.channel.eq(channel)&g.group.eq(group)].groupby('fold')[metric].max()
                assert len(per)==len(frozen['confirmation_folds'])
                row[label+'_max']=per.max();row[label+'_mean']=per.mean();row[label+'_min']=per.min()
                for f,v in per.items():row[label+f'_fold{f}']=v
            rows.append(row)
        table=pd.DataFrame(rows);table.to_csv(out/'confirmation_summary.csv',index=False,lineterminator='\n')
        print(table[['candidate','U']+[x[3]+'_max' for x in METRICS]].to_string(index=False));return
    assert a.fold in frozen['confirmation_folds']
    candidate=ROOT/frozen['candidate'];incumbent=ROOT/frozen['incumbent']
    assert digest(candidate)==frozen['candidate_sha256']
    assert digest(incumbent)==frozen['incumbent_sha256']
    bp.OUT=out/'confirmation_partitions';bp.OUT.mkdir(exist_ok=True)
    split=bp.partition_frame(pd.read_csv(DIST/'A_bg.csv'),a.fold);p=bp.OUT/f'fold{a.fold}_split.csv'
    if p.exists():pd.testing.assert_frame_equal(split,pd.read_csv(p),check_exact=True)
    else:split.to_csv(p,index=False,lineterminator='\n')
    fold=bp.SeparatedFold(a.fold)
    paths=[candidate,incumbent]+[p for k,p in anchors().items() if k.startswith('positive_') or k.startswith('negative_')]
    folder=out/'confirmation_evaluation';folder.mkdir(exist_ok=True)
    geo=out/'confirmation_geography';geo.mkdir(exist_ok=True);results=[]
    for p in paths:
        results += [fold.evaluate(p,folder),evaluate_geo(fold,p,geo)]
        versioned_matrix(results).to_csv(out/f'confirmation_fold{a.fold}_partial.csv',index=False,lineterminator='\n')
    frame=versioned_matrix(results)
    assert not frame.duplicated(['candidate','fold','version','channel','method','group']).any()
    frame.to_csv(out/f'confirmation_fold{a.fold}_matrix.csv',index=False,lineterminator='\n')
    dump(out/f'confirmation_fold{a.fold}_checkpoint.json',{'fold':a.fold,'candidates':len(paths),
         'split_sha256':fold.split_hash,'frozen_selection_sha256':digest(out/'frozen_selection.json'),'complete':True})
    print(f'Confirmation fold {a.fold} complete',flush=True)


if __name__=='__main__':main()
