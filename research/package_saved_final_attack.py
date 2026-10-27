"""Build a validated final candidate from saved keyed predictions; never submit."""
import argparse
import json
import zipfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.build_final_attack_baseline import ROOT, read, sha, write_json, config
from research.final_attack_api import VERSION, keyed_values, MIA_METHODS
from pwscup2026.kit.validate import validate_submission_zip, validate_mia_submission, validate_aia_submission
import pandas as pd


def normalize_mia_overrides(targets, overrides):
    result = {}
    for key, method in (overrides or {}).items():
        if isinstance(key, bool) or not str(key).isdigit():
            raise ValueError("Override target must be an integer")
        target = int(key)
        if target not in targets or target in result:
            raise ValueError("Unknown, self, or duplicate override target")
        if method not in MIA_METHODS:
            raise ValueError("Unknown override method")
        result[target] = method
    return result


def package(out, mia_method, aia_method, extra=None, mia_overrides=None):
    out=out.resolve()
    if extra is not None:extra=extra.resolve()
    base=ROOT/'research/runs/20261006-G15b-001'
    manifest=json.loads((ROOT/'_private_team14/main_attack_20260928/IMPORT_MANIFEST.json').read_text())
    dist=ROOT/manifest['files_root'];cfg=config()
    ids=read(ROOT/'_private_team14/B.csv').record_id
    targets=[int(k) for k in manifest['targets_including_self'] if int(k)!=14]
    overrides=normalize_mia_overrides(targets,mia_overrides)
    out.mkdir(parents=True, exist_ok=False)
    fm=pd.DataFrame({'record_id':ids});rows=[];routes=[];challenge={}
    for k in targets:
        selected_mia=overrides.get(k,mia_method)
        source=base/'predictions'
        marker=json.loads((source/f'target_{k}.json').read_text())
        assert sha(source/f'mia_{k}.csv')==marker['mia_sha256']
        assert sha(source/f'aia_{k}.csv')==marker['aia_sha256']
        mp=source/f'mia_{k}.csv';ap=source/f'aia_{k}.csv'
        if extra:
            if (extra/f'mia_{k}.csv').exists() and selected_mia in read(extra/f'mia_{k}.csv'):mp=extra/f'mia_{k}.csv'
            if (extra/f'aia_{k}.csv').exists() and aia_method in read(extra/f'aia_{k}.csv'):ap=extra/f'aia_{k}.csv'
            saved=json.loads((extra/f'target_{k}.json').read_text())
            if mp.parent==extra:assert sha(mp)==saved['mia_sha256']
            if ap.parent==extra:assert sha(ap)==saved['aia_sha256']
        m,a=read(mp),read(ap);q=read(dist/f'aia_challenge_{k}.csv')
        fm[str(k)]=keyed_values(m,'record_id',ids,selected_mia)
        rows.append(pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id,
            'time_hat':keyed_values(a,'challenge_row_id',q.challenge_row_id,aia_method)}))
        challenge[k]=set(q.challenge_row_id)
        routes.append({'target_k':k,'mia_method':selected_mia,'aia_method':aia_method,
            'source_mia':str(mp.relative_to(ROOT)),'source_aia':str(ap.relative_to(ROOT)),
            'source_mia_sha256':sha(mp),'source_aia_sha256':sha(ap),'version':VERSION,
            'mia_version':'FA03-lawful-v1' if extra and mp.parent==extra else VERSION,
            'aia_version':'FA03-lawful-v1' if extra and ap.parent==extra else VERSION,
            'selection':'local comparison candidate; unknown final performance'})
    fa=pd.concat(rows,ignore_index=True)
    strict={'team_id':14,'mia_columns':{'target_columns':[str(k) for k in manifest['targets_including_self']]},
        'challenge':challenge,'attacker_record_ids':set(ids)}
    for r in [validate_mia_submission(fm,cfg,strict),validate_aia_submission(fa,cfg,strict)]:
        assert r.ok,r.errors
    fm.to_csv(out/'F_mia.csv',index=False);fa.to_csv(out/'F_aia.csv',index=False)
    zpath=out/'candidate.zip'
    with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in [('F_mia.csv',(out/'F_mia.csv').read_bytes()),('F_aia.csv',(out/'F_aia.csv').read_bytes()),
                            ('token.txt',(ROOT/'_private_team14/token.txt').read_bytes())]:
            info=zipfile.ZipInfo(name,date_time=(2026,10,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
    r=validate_submission_zip(zpath,cfg,dist,'attack');assert r.ok,r.errors
    with zipfile.ZipFile(zpath) as z:assert z.read('token.txt')==(ROOT/'_private_team14/token.txt').read_bytes()
    write_json(out/'routing_manifest.json',routes)
    write_json(out/'validation.json',{'ok':True,'strict_keys_ok':True,'token_bytes_ok':True,
        'mia_rows':len(fm),'aia_rows':len(fa),'targets':targets,'sha256':sha(zpath),
        'submitted':False,'official_score':None,'mia_method':mia_method,'aia_method':aia_method,
        'mia_overrides':overrides})
    return zpath


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mia',required=True);ap.add_argument('--aia',default='official_1nn')
    ap.add_argument('--extra',type=Path);ap.add_argument('--mia-overrides',type=Path);a=ap.parse_args()
    overrides=json.loads(a.mia_overrides.read_text()) if a.mia_overrides else None
    print(package(a.out,a.mia,a.aia,a.extra,overrides).relative_to(ROOT))
