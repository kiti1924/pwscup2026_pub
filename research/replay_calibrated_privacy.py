"""Rescore saved G17e predictions, without rerunning or selecting attacks."""
from pathlib import Path
import argparse
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.calibrated_copula import OUT,dump,digest
from research.background_partition import SeparatedFold
from research.privacy_suite import mia_metrics,success_metrics,standardized_aia
import numpy as np
import pandas as pd


def same(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:same(actual[key],expected[key])
    elif isinstance(actual,(int,float)):
        assert math.isclose(actual,expected,abs_tol=1e-12,rel_tol=1e-12) or (math.isnan(actual) and math.isnan(expected)),(actual,expected)
    else:assert actual==expected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args();checked=[]
    for seed in [2701,2702,2703]:
        fold=SeparatedFold(seed)
        for path in sorted((OUT/'evaluation').glob(f'C_calcopula_s{args.seed}_*_fold{seed}.json')):
            saved=json.loads(path.read_text());pred_path=path.with_name(path.stem+'_predictions.csv')
            pred=pd.read_csv(pred_path,float_precision='round_trip')
            np.testing.assert_array_equal(pred.record_id.to_numpy(),fold.query.record_id.to_numpy())
            np.testing.assert_array_equal(pred.source.to_numpy(),np.where(fold.labels==1,'B14_member','background_control'))
            masks={'all':np.ones(len(fold.query),bool),'true_member_rare_detected_control':np.r_[fold.mr,fold.cr],'detected_both':np.r_[fold.dr,fold.cr]}
            for name,expected in saved['mia'].items():
                same({k:mia_metrics(pred['mia_'+name].to_numpy(),fold.labels,m) for k,m in masks.items()},expected)
            for name,expected in saved['aia'].items():
                hit=np.abs(pred['time_hat_'+name].to_numpy()-fold.query.time.to_numpy())<=.5
                mh,ch=hit[:len(fold.b)],hit[len(fold.b):]
                same({'all':success_metrics(mh,ch),'rare_proxy':success_metrics(mh[fold.mr],ch[fold.cr]),
                      'stratified_all':standardized_aia(fold.b,fold.controls,mh,ch)},expected)
            checked.append({'result':path.relative_to(ROOT).as_posix(),'predictions_sha256':digest(pred_path),'result_sha256':digest(path)})
    expected=json.loads((OUT/f'seed{args.seed}_risk_summary.json').read_text())['n_evaluations']
    assert len(checked)==expected
    dump(OUT/f'seed{args.seed}_privacy_replay.json',{'n_evaluations':len(checked),'n_method_evaluations':12*len(checked),
        'verification':'Saved prediction scores rescored within 1e-12; query keys/source order matched','files':checked})
    print('Saved predictions replayed:',len(checked),'candidate-folds,',12*len(checked),'methods')


if __name__=='__main__':main()
