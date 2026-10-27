"""G17c generation/reference/control separation for existing v2 attack algorithms."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.adaptive_privacy import (Fold, DIST, NUMERIC, STATIC, _load_inputs,
    _member_rare_mask, _rare_mask_from_reference, static_distances, conditional_min, dump, digest)
import numpy as np
import pandas as pd

OUT=ROOT/'research/runs/20260924-G17c-001'
PARTITION_VERSION='G17c-bg-separated-v1'


def partition_indices(n, fold):
    if n<=3500: raise ValueError('Need 2000 generation, 1500 reference and nonempty control')
    order=np.random.default_rng(2026092401).permutation(n)
    generation=order[:2000]
    evaluation=np.random.default_rng(fold).permutation(order[2000:])
    reference,control=evaluation[:1500],evaluation[1500:]
    assert len(np.unique(np.r_[generation,reference,control]))==n
    return generation,reference,control


def partition_frame(bg,fold):
    if bg.record_id.duplicated().any(): raise ValueError('Background record_id must be unique')
    generation,reference,control=partition_indices(len(bg),fold)
    parts=[('generation',generation),('reference',reference),('control',control)]
    result=pd.concat([pd.DataFrame({'background_row':idx,'record_id':bg.iloc[idx].record_id.to_numpy(),
                                   'partition':name}) for name,idx in parts],ignore_index=True)
    assert result.background_row.nunique()==len(bg) and result.record_id.nunique()==len(bg)
    return result


class SeparatedFold(Fold):
    """Only initialization changes; all twelve v2 inference methods are inherited."""
    def __init__(self,seed):
        self.seed=seed
        self.b,bs,self.bg,self.cfg=_load_inputs(DIST)
        pd.testing.assert_frame_equal(self.b,bs.drop(columns='is_rare'))
        self.split=partition_frame(self.bg,seed)
        split_path=OUT/f'fold{seed}_split.csv'
        pd.testing.assert_frame_equal(self.split,pd.read_csv(split_path),check_exact=True)
        self.split_hash=digest(split_path)
        self.generation_indices=self.split.loc[self.split.partition=='generation','background_row'].to_numpy()
        reference=self.split.loc[self.split.partition=='reference','background_row'].to_numpy()
        control=self.split.loc[self.split.partition=='control','background_row'].to_numpy()
        self.ref=self.bg.iloc[reference].reset_index(drop=True)
        self.controls=self.bg.iloc[control].reset_index(drop=True)
        self.query=pd.concat([self.b,self.controls],ignore_index=True)
        self.labels=np.r_[np.ones(len(self.b),int),np.zeros(len(self.controls),int)]
        self.mr=_member_rare_mask(self.b,bs)
        self.cr=_rare_mask_from_reference(self.controls,DIST)
        self.dr=_rare_mask_from_reference(self.b,DIST)
        self.scale=self.ref[NUMERIC].std(ddof=0).to_numpy()
        self.scale=np.where(self.scale>0,self.scale,1.)
        self.qstatic=self.query[STATIC].copy();self.qstatic['detected_rare']=np.r_[self.dr,self.cr]
        rstatic=self.ref[STATIC].copy();rstatic['detected_rare']=_rare_mask_from_reference(self.ref,DIST)
        rd=static_distances(self.qstatic,rstatic,self.scale)
        self.rmin,self.r5=rd.min(axis=1),np.partition(rd,4,axis=1)[:,4]
        self.rcond=conditional_min(rd,self.qstatic,rstatic);self.cache={}

    def evaluate(self,path,out):
        rel=path.relative_to(ROOT).as_posix()
        tag=path.stem+'_'+hashlib.sha256(rel.encode()).hexdigest()[:8]+f'_fold{self.seed}'
        result_path=out/(tag+'.json')
        if result_path.exists():
            saved=json.loads(result_path.read_text(encoding='utf-8'))
            assert saved['evaluation_partition']==PARTITION_VERSION
            assert saved['split_sha256']==self.split_hash
            assert saved['partition_script_sha256']==digest(Path(__file__))
        result=super().evaluate(path,out)
        result.update(evaluation_partition=PARTITION_VERSION,split_sha256=self.split_hash,
                      partition_script_sha256=digest(Path(__file__)))
        dump(result_path,result)
        return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'partition_manifest.json').exists(): raise FileExistsError('Partition checkpoint already exists')
    bg=pd.read_csv(DIST/'A_bg.csv'); records=[]
    generation=None
    for fold in [2701,2702,2703]:
        frame=partition_frame(bg,fold);path=OUT/f'fold{fold}_split.csv';frame.to_csv(path,index=False)
        g=frame.loc[frame.partition=='generation','background_row'].to_numpy()
        if generation is None:generation=g
        else:np.testing.assert_array_equal(generation,g)
        counts={}
        for name in ['generation','reference','control']:
            idx=frame.loc[frame.partition==name,'background_row'].to_numpy()
            counts[name]={'n':len(idx),'detected_rare':int(_rare_mask_from_reference(bg.iloc[idx],DIST).sum())}
        records.append({'fold':fold,'split':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'counts':counts})
    dump(OUT/'partition_manifest.json',{'partition_version':PARTITION_VERSION,'generation_seed':2026092401,
        'background':(DIST/'A_bg.csv').relative_to(ROOT).as_posix(),'background_sha256':digest(DIST/'A_bg.csv'),
        'folds':records,'invariants':'Every row/id appears once per fold; same generation subset across all folds',
        'limitations':'Evaluation row reuse across folds and historical exploration; not independent held-out population'})
    print(json.dumps(records))


if __name__=='__main__':main()
