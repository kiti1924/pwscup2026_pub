"""FA114 single unmeasured main candidate; no upload/score entry point."""
from pathlib import Path
import argparse, datetime, io, json, sys, zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'research')]
import numpy as np
import pandas as pd
from rare_fixed_calibration import priority,reassign
from verify_rare_fixed_calibration import independent_order
from package_public_attack import run as package
from run_rare_fixed_calibration import load,read,sha,dump,hashes


def run(out):
    plan=json.loads((out/'plan.json').read_text());assert not (out/'candidate').exists()
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(plan['deadline_jst'])
    hashes(out,'input_hashes.json',ROOT)
    accessed=set()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            s=str(args[0])
            if any(v in s for v in ['scoring_truth','/sources/truth','/sources/attacks','parsed_details']):
                raise RuntimeError('Forbidden inference input')
            p=Path(args[0]).resolve()
            if p.is_relative_to(ROOT):accessed.add(str(p.relative_to(ROOT)))
    sys.addaudithook(guard)
    p106=ROOT/'research/runs/20261010-FA106-cl06-main-transfer-001/execution/run-001'
    raw=load(p106/'raw_public_arrays.npz');cal=load(p106/'calibration_public_arrays.npz');old=load(p106/'predictions.npz')
    ids,targets,r,x=(raw[k] for k in ['record_id','targets','rare','excluded'])
    base=old['baseline_S3'];np.testing.assert_array_equal(ids,old['record_id']);np.testing.assert_array_equal(targets,old['targets'])
    public=ROOT/'_private_team14/main_attack_20260928/files'
    declared=[int(k) for k in json.loads((public/'mia_columns.json').read_text())['target_columns'] if int(k)!=14]
    assert set(targets)==set(declared) and len(targets)==22 and len(ids)==len(np.unique(ids))==1049
    own=pd.read_csv(ROOT/'_private_team14/B_self.csv',usecols=['record_id','is_rare','prefecture']).set_index('record_id').loc[ids]
    np.testing.assert_array_equal(own.is_rare.to_numpy(bool),r)
    np.testing.assert_array_equal(own.prefecture.eq(own.prefecture.mode().iloc[0]).to_numpy(),raw['home'])
    exclusion=load(ROOT/'research/runs/20261010-FA37-known-private-001/main/matches.npz')
    np.testing.assert_array_equal(exclusion['record_id'],ids);np.testing.assert_array_equal(exclusion['strict_mask'],x)
    with zipfile.ZipFile(ROOT/'research/submissions/974967/submitted_server_original.zip') as z:
        aia=z.read('F_aia.csv');original_mia=z.read('F_mia.csv');token=z.read('token.txt')
    assert original_mia==(ROOT/'research/submissions/four-measured-backup-20261010/candidate/F_mia.csv').read_bytes()
    original=read(io.BytesIO(original_mia)).set_index('record_id');np.testing.assert_array_equal(original.loc[ids,list(map(str,targets))].to_numpy(),base)
    cp=priority(cal['blend_e0'],raw['home']);pred,reports=reassign(base,cp,r,x)
    global_ctrl,_=reassign(base,priority(cal['control_e0'],raw['home']),r,x);np.testing.assert_array_equal(global_ctrl,base)
    for j in range(len(targets)):
        expect,_=independent_order(base[:,j],cp[:,j],r,x);np.testing.assert_array_equal(expect,pred[:,j])
    perm=np.random.default_rng(20261011114).permutation(len(ids));inv=np.argsort(perm)
    shuffled,_=reassign(base[perm],cp[perm],r[perm],x[perm]);np.testing.assert_array_equal(shuffled[inv],pred)
    np.savez_compressed(out/'inference_arrays.npz',record_id=ids,targets=targets,rare=r,excluded=x,baseline=base,priority=cp,predictions=pred)
    df=pd.DataFrame(pred,columns=list(map(str,targets)));df.insert(0,'record_id',ids);df.to_csv(out/'F_mia.csv',index=False,lineterminator='\n')
    (out/'F_aia.csv').write_bytes(aia);saved=read(out/'F_mia.csv').set_index('record_id').loc[ids,list(map(str,targets))].to_numpy()
    np.testing.assert_array_equal(saved.view(np.uint64),pred.view(np.uint64))
    np.testing.assert_array_equal(saved[r|x].view(np.uint64),base[r|x].view(np.uint64))
    np.testing.assert_array_equal(np.sort(saved,axis=0),np.sort(base,axis=0))
    dump(out/'tie_reports.json',[dict(target=int(k),**v) for k,v in zip(targets,reports)])
    package(out)
    with zipfile.ZipFile(out/'candidate/candidate.zip') as z:
        assert z.read('F_aia.csv')==aia and z.read('token.txt')==token
        assert z.read('F_mia.csv')==(out/'F_mia.csv').read_bytes()
    manifest=json.loads((out/'manifest.json').read_text());manifest.pop('FA98_screen',None)
    manifest.update(experiment='FA114',new_predictions=1,main_measured=False,score=None,
                    MIA_protected_source=972302,AIA_source_submission=974967,rare_excluded_bit_exact=True,
                    all_value_multisets_equal=True,baseline_ties_preserved=True,independent_order_columns=22,
                    shuffle_invariance=True,original_AIA_bytes_exact=True,rare_mask_server_agreement='conditional on public B_self alignment; not independently observed')
    dump(out/'manifest.json',manifest)
    dump(out/'source_binding.json',dict(source_zip_sha256=sha(ROOT/'research/submissions/974967/submitted_server_original.zip'),
         rare_rows=int(r.sum()),excluded_rows=int(x.sum()),eligible_rows=int((~r & ~x).sum()),
         changed_cells=int((pred!=base).sum()),changed_targets=int((pred!=base).any(axis=0).sum()),
         AIA_bytes_identical=True,MIA_fixed_scores_identical=True,main_score=None,new_submissions=0))
    dump(out/'builder_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/rare_fixed_calibration.py',ROOT/'research/verify_rare_fixed_calibration.py']})
    dump(out/'access_audit.json',dict(repository_paths=sorted(accessed),membership_truth_reads=0,official_score_reads=0))
    print(json.dumps(json.loads((out/'source_binding.json').read_text())))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out.resolve())
