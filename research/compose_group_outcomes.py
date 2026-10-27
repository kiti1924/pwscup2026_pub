"""G21-002 reuse calibrated synthetic outcomes on fixed repaired static rows."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from research.adaptive_privacy import dump,digest
from research.repair_sex_correlation import validate_score,DIST

OUT=ROOT/'research/runs/20260925-G21-002'
BASE=ROOT/'research/runs/20260925-G21-001/C_sex_a0.67_s3301.csv'
DONORS=ROOT/'research/runs/20260924-G17e-001/seed2'
OUTCOMES=['time','onset','death']


def main():
    base=pd.read_csv(BASE);assert digest(BASE)=='3ffe9935855b712579250abc1b7e888eab2f8ab32be38b0e0279255d1bfdd87b'
    for i in [2,4,6]:
        donor=DONORS/f'C_calcopula_s2_w1.0_g1_i{i}.csv'
        dm=json.loads(donor.with_name(donor.stem.removeprefix('C_')+'_manifest.json').read_text())
        assert digest(donor)==dm['candidate_sha256']
        d=pd.read_csv(donor)
        pd.testing.assert_frame_equal(base.drop(columns=OUTCOMES+['sex','prefecture']),d.drop(columns=OUTCOMES+['sex','prefecture']))
        frame=base.copy();frame[OUTCOMES]=d[OUTCOMES]
        pd.testing.assert_frame_equal(frame.drop(columns=OUTCOMES),base.drop(columns=OUTCOMES),check_exact=True)
        pd.testing.assert_frame_equal(frame[OUTCOMES],d[OUTCOMES],check_exact=True)
        key=f'sex_group_i{i}';csv=OUT/f'C_{key}.csv';archive=OUT/f'{key}.zip';mp=OUT/f'{key}_manifest.json'
        inputs={p.relative_to(ROOT).as_posix():digest(p) for p in [BASE,donor,Path(__file__),DIST/'utility_ref.json',DIST/'scoring_config.yaml']}
        if mp.exists():
            r=json.loads(mp.read_text());assert r['inputs_sha256']==inputs and digest(csv)==r['candidate_sha256'];continue
        if csv.exists():pd.testing.assert_frame_equal(pd.read_csv(csv),frame)
        else:frame.to_csv(csv,index=False,lineterminator='\n')
        pd.testing.assert_frame_equal(pd.read_csv(csv).drop(columns=OUTCOMES),base.drop(columns=OUTCOMES))
        with zipfile.ZipFile(archive,'w') as z:
            for name,p in [('C.csv',csv),('token.txt',DIST/'token.txt')]:
                info=zipfile.ZipInfo(name,date_time=(2026,9,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,p.read_bytes())
        score=validate_score(csv,archive,OUT,key)
        record={'key':key,'iteration':i,'candidate':csv.relative_to(ROOT).as_posix(),'candidate_sha256':digest(csv),
                'zip':archive.relative_to(ROOT).as_posix(),'zip_sha256':digest(archive),'inputs_sha256':inputs,
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'utility':score['utility'],'facets':score['facets'],'validator':'OK',
                'replay':'All static columns preserved; outcomes exactly from recorded synthetic donor; CSV readback checked'}
        dump(mp,record);print(json.dumps(record['facets']),flush=True)


if __name__=='__main__':main()
