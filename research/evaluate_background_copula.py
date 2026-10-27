"""Evaluate G17d on the frozen G17c separated folds and retained anchors."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.background_copula import OUT,PARTITION_OUT,BASE,dump,digest
from research.background_partition import SeparatedFold,PARTITION_VERSION
from research.evaluate_background_transport import maxima,anchors
from research.repair_sex_correlation import risk_gates
from research.adaptive_privacy import matrix


def historical_result(path,fold):
    matches=list((PARTITION_OUT/'evaluation').glob(path.stem+f'_*_fold{fold}.json'))
    assert len(matches)==1
    saved=json.loads(matches[0].read_text(encoding='utf-8'))
    assert saved['candidate_sha256']==digest(path)
    assert saved['evaluation_partition']==PARTITION_VERSION
    assert saved['split_sha256']==digest(PARTITION_OUT/f'fold{fold}_split.csv')
    assert saved['script_sha256']==digest(ROOT/'research/adaptive_privacy.py')
    assert saved['partition_script_sha256']==digest(ROOT/'research/background_partition.py')
    return saved,matches[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,required=True,choices=[0,1,2]);args=ap.parse_args()
    records=json.loads((OUT/f'seed{args.seed}/utility_summary.json').read_text(encoding='utf-8'))['candidates']
    paths={r['key']:ROOT/r['candidate'] for r in records if not r['pruned']}
    output=OUT/'evaluation';output.mkdir(exist_ok=True)
    anchor_paths=anchors();comparisons=[];results=[];inputs={}
    for seed in [2701,2702,2703]:
        checkpoint=PARTITION_OUT/f'anchors_fold{seed}_checkpoint.json'
        assert all(x['pass'] for x in json.loads(checkpoint.read_text())['positive_gates'])
        base,bpath=historical_result(anchor_paths['incumbent'],seed)
        parent,ppath=historical_result(anchor_paths['high_parent'],seed)
        for p in [checkpoint,bpath,ppath]:inputs[p.relative_to(ROOT).as_posix()]=digest(p)
        fold=SeparatedFold(seed)
        for key,path in paths.items():
            result=fold.evaluate(path,output);results.append(result)
            delta={k:v-maxima(parent)[k] for k,v in maxima(result).items()}
            comparisons.append({'key':key,'fold':seed,'maxima':maxima(result),'parent_max_deltas':delta,
                'baseline_paired_deltas':risk_gates(result,base),'parent_paired_deltas':risk_gates(result,parent),
                'max_nonworsening':all(delta[k]<=.01 for k in ['mia_all','mia_rare','aia_stratified']),
                'rare_improves':delta['mia_rare']<=-1/24+1e-12})
    frame=matrix(results);frame['evaluation_partition']=PARTITION_VERSION
    frame.to_csv(OUT/f'seed{args.seed}_attack_matrix.csv',index=False,lineterminator='\n')
    dump(OUT/f'seed{args.seed}_risk_summary.json',{'seed':args.seed,'n_evaluations':len(results),
        'partition_version':PARTITION_VERSION,'comparisons':comparisons,'reference_inputs_sha256':inputs,
        'pruned_unmeasured':[r['key'] for r in records if r['pruned']],
        'candidate_sha256':{k:digest(p) for k,p in paths.items()},'label':'Exploration proxies; not final Anon'})


if __name__=='__main__':main()
