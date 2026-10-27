"""FA103 frozen CSV packaging only; never predicts, scores or uploads."""
from pathlib import Path
import argparse, hashlib, io, json, subprocess, sys, time, zipfile
import pandas as pd
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p):return pd.read_csv(p,float_precision='round_trip')


def build(parts):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in ['F_mia.csv','F_aia.csv','token.txt']:
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.create_system=3
            entry.external_attr=0o100644<<16;entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,parts[name],compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return stream.getvalue()


def run(out):
    started=time.monotonic();plan=json.loads((out/'plan.json').read_text())
    assert not (out/'complete.json').exists()
    for p,h in json.loads((out/'input_hashes.json').read_text()).items():assert sha(ROOT/p)==h,p
    parts={'F_aia.csv':(ROOT/plan['AIA_source']).read_bytes(),'F_mia.csv':(ROOT/plan['MIA_source']).read_bytes(),'token.txt':(ROOT/'_private_team14/token.txt').read_bytes()}
    folder=out/'candidate';folder.mkdir()
    for name,value in parts.items():(folder/name).write_bytes(value)
    first=build(parts);second=build(parts);assert first==second
    (folder/'candidate.zip').write_bytes(first);(out/'determinism_rebuild.zip').write_bytes(second)
    with zipfile.ZipFile(folder/'candidate.zip') as z:
        assert set(z.namelist())==set(parts)
        for n,v in parts.items():assert z.read(n)==v and z.getinfo(n).date_time==tuple(plan['zip_timestamp'])
    public=ROOT/'_private_team14/main_attack_20260928/files'
    targets=[int(x) for x in json.loads((public/'mia_columns.json').read_text())['target_columns'] if int(x)!=14]
    a=read(folder/'F_aia.csv');m=read(folder/'F_mia.csv')
    assert len(a)==7172 and not a.duplicated(['target_k','challenge_row_id']).any()
    assert sorted(a.target_k.unique())==targets and a.groupby('target_k').size().eq(326).all()
    for k in targets:assert set(a[a.target_k.eq(k)].challenge_row_id)==set(read(public/f'aia_challenge_{k}.csv').challenge_row_id)
    assert len(m)==1049 and m.record_id.is_unique and set(m)==set(['record_id']+[str(k) for k in targets])
    for n,v in [('F_aia.csv',a),('F_mia.csv',m)]:
        roundtrip=read(io.BytesIO(v.to_csv(index=False,lineterminator='\n').encode()))
        for col in v.select_dtypes(include='number'):np.testing.assert_array_equal(v[col],roundtrip[col])
    sys.path.insert(0,str(ROOT/'pwscup2026-kit/starter/src'))
    from pwscup2026.kit.validate import validate_submission_zip
    cfg=yaml.safe_load((ROOT/'_private_team14/scoring_config.yaml').read_text())
    validation=validate_submission_zip(folder/'candidate.zip',cfg,public,'attack')
    assert validation.ok,'Official package validation failed (details suppressed to protect token)'
    manifest={'candidate':plan['candidate'],'main_measured':False,'score':None,'FA98_screen':'FAIL preserved','aia_source':plan['AIA_source'],'mia_source':plan['MIA_source'],'source_csv_bytes_exact':True,'aia_targets':targets,'aia_keys':len(a),'mia_rows':len(m),'official_ZIP_includes_MIA_AIA':True,'official_ZIP_pass':True,'deterministic_two_builds_equal':True,'float_roundtrip_exact':True,'zip_timestamp':plan['zip_timestamp'],'zip_sha256':sha(folder/'candidate.zip'),'zip_bytes':len(first),'column_hashes':{n:sha(folder/n) for n in ['F_aia.csv','F_mia.csv']},'new_predictions':0,'new_scores':0,'new_submissions':0}
    dump(out/'manifest.json',manifest)
    dump(out/'code_hashes.json',{'research/package_public_attack.py':sha(Path(__file__))})
    dump(out/'complete.json',{'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'seconds':time.monotonic()-started,'running_processes_after_return':0,'new_submissions':0})
    assert time.monotonic()-started<plan['budget_seconds']
    print('FA103 package pass',manifest['zip_sha256'],manifest['aia_keys'],manifest['mia_rows'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
