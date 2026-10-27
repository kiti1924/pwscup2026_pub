"""Generate and validate fixed final-round predictions; never submits them."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"pwscup2026-kit/starter/src")]
import numpy as np
import pandas as pd
import yaml
from research.final_attack_api import (VERSION,COLUMNS,STATIC,QI,MIA_METHODS,AIA_METHODS,predict_mia,predict_aia,keyed_values)
from pwscup2026.kit.validate import validate_submission_zip,validate_mia_submission,validate_aia_submission

RUN=ROOT/"research/runs/20261006-G15b-001"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def read(path): return pd.read_csv(path,float_precision="round_trip")
def config():
    cfg=yaml.safe_load((ROOT/"_private_team14/scoring_config.yaml").read_text())
    # Same public reference defaults as starter/reference/_common.py, not secret scoring values.
    cfg["aia"].update(qi_continuous=QI[:5],block_keys=["sex","prefecture"],k_nn_onset=5)
    return cfg

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=RUN);args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"_private_team14/main_attack_20260928/IMPORT_MANIFEST.json").read_text())
    dist=ROOT/manifest["files_root"]
    for name,digest in manifest["internal_hashes"].items():
        if sha(dist/name)!=digest: raise ValueError("Input hash mismatch: "+name)
    targets=[int(k) for k in json.loads((dist/"mia_columns.json").read_text())["target_columns"] if int(k)!=14]
    own=read(ROOT/"_private_team14/B.csv")[["record_id"]+STATIC]
    bg=read(ROOT/"_private_team14/A_bg.csv")[COLUMNS]
    detector=json.loads((ROOT/"_private_team14/schema.json").read_text())["rare_detector"]
    cfg=config();ids=own.record_id.to_numpy()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/"_private_team14/B.csv",ROOT/"_private_team14/A_bg.csv",ROOT/"_private_team14/schema.json",ROOT/"_private_team14/scoring_config.yaml",ROOT/"_private_team14/token.txt",ROOT/"research/final_attack_api.py",Path(__file__)]}
    identity={"version":VERSION,"inputs":inputs,"public_hashes":manifest["internal_hashes"],"configuration":cfg,"targets":targets,"own_team":14}
    checkpoint=out/"input_manifest.json"
    if checkpoint.exists():
        if json.loads(checkpoint.read_text())!=identity: raise ValueError("Checkpoint input/version mismatch: use a fresh output directory")
    else: write_json(checkpoint,identity)
    predictions=out/"predictions";predictions.mkdir(exist_ok=True)
    fm=pd.DataFrame({"record_id":ids});aia=[];routes=[];summaries=[];challenge={}
    for k in targets:
        c=read(dist/f"C_{k}.csv")[COLUMNS]
        q=read(dist/f"aia_challenge_{k}.csv")[["challenge_row_id"]+QI]
        mp=predictions/f"mia_{k}.csv";ap=predictions/f"aia_{k}.csv"
        marker=predictions/f"target_{k}.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved!={"mia_sha256":sha(mp),"aia_sha256":sha(ap)}: raise ValueError("Saved prediction hash mismatch")
            m,a=read(mp),read(ap)
        else:
            if mp.exists() or ap.exists(): raise ValueError("Incomplete target checkpoint: retain and use a fresh output directory")
            m=predict_mia(c,own,bg,cfg,detector);a=predict_aia(c,q,bg,cfg)
            m.to_csv(mp,index=False,lineterminator="\n");a.to_csv(ap,index=False,lineterminator="\n")
            write_json(marker,{"mia_sha256":sha(mp),"aia_sha256":sha(ap)})
        fm[str(k)]=keyed_values(m,"record_id",ids,"kde_ratio")
        challenge[k]=set(q.challenge_row_id)
        aia.append(pd.DataFrame({"target_k":k,"challenge_row_id":q.challenge_row_id.to_numpy(),"time_hat":keyed_values(a,"challenge_row_id",q.challenge_row_id,"official_1nn")}))
        routes.append({"target_k":k,"mia_method":"kde_ratio","aia_method":"official_1nn","version":VERSION,"seed":None,"fallback":"reference KDE singular->0.5; AIA sex/prefecture->sex->all; tie mean sorted time","C_sha256":sha(dist/f"C_{k}.csv"),"query_sha256":sha(dist/f"aia_challenge_{k}.csv"),"prediction_hashes":json.loads(marker.read_text())})
        summaries.append({"target_k":k,"C_rows":len(c),"mia_rows":len(m),"aia_rows":len(a),"mia_methods":len(MIA_METHODS),"aia_methods":len(AIA_METHODS),"mia_unique":{n:int(m[n].nunique()) for n in MIA_METHODS}})
        print(f"target {k}: complete, {len(m)} MIA / {len(a)} AIA",flush=True)
    fa=pd.concat(aia,ignore_index=True)
    strict={"team_id":14,"mia_columns":{"target_columns":[str(k) for k in manifest["targets_including_self"]]},"challenge":challenge,"attacker_record_ids":set(ids)}
    validations=[validate_mia_submission(fm,cfg,strict),validate_aia_submission(fa,cfg,strict)]
    errors=[e for v in validations for e in v.errors]
    if errors: raise ValueError(errors)
    if len(fm)!=1049 or len(fa)!=7172: raise ValueError("Unexpected prediction dimensions")
    fm.to_csv(out/"F_mia.csv",index=False,lineterminator="\n");fa.to_csv(out/"F_aia.csv",index=False,lineterminator="\n")
    package=out/"baseline_official.zip"
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,content in [("F_mia.csv",(out/"F_mia.csv").read_bytes()),("F_aia.csv",(out/"F_aia.csv").read_bytes()),("token.txt",(ROOT/"_private_team14/token.txt").read_bytes())]:
            info=zipfile.ZipInfo(name,date_time=(2026,10,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
    validation=validate_submission_zip(package,cfg,dist,"attack")
    if not validation.ok: raise ValueError(validation.errors)
    with zipfile.ZipFile(package) as z:
        if z.read("token.txt")!=(ROOT/"_private_team14/token.txt").read_bytes():raise ValueError("Token byte mismatch")
    write_json(out/"routing_manifest.json",routes)
    write_json(out/"validation.json",{"package_ok":validation.ok,"strict_team_and_keys_ok":True,"mia_rows":len(fm),"aia_rows":len(fa),"targets":targets,"version":VERSION,"submitted":False,"official_score":None,"truth_read":False,"zip_sha256":sha(package),"target_summaries":summaries,"environment":{"python":platform.python_version(),**{n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","pyyaml"]}}})
    print("Validated baseline package; not submitted.",flush=True)

if __name__=="__main__":main()
