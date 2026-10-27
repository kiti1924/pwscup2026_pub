"""G21 bounded refinement, preserving all candidates and same-version comparisons."""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.repair_sex_correlation import repair, validate_score, DIST
from research.adaptive_privacy import dump, digest, matrix
from research.geography_free_attacks import evaluate as evaluate_geo
import research.background_partition as bp
from research.summarize_tradeoff import METRICS
import numpy as np
import pandas as pd

OUT = ROOT / 'research/runs/20260925-G21-001'
BASE = ROOT / 'research/runs/20260925-G17g-001/seed2/C_geo_s2_g0_G.csv'
G20 = ROOT / 'research/runs/20260925-G20-001'
ORIGINAL = BASE
REUSE_MATRIX = G20/'exploration_all_methods.csv'


def versioned_matrix(results):
    return pd.concat([matrix([r]).assign(version=r['attack_version']) for r in results], ignore_index=True)


def verify_reuse(frame, fold):
    history = pd.read_csv(G20/'historical_fold_maxima.csv')
    checked = []
    for rel in frame.candidate.unique():
        old = history[history.candidate.eq(rel) & history.fold.eq(fold.seed) & history.partition.eq(bp.PARTITION_VERSION)]
        tag = Path(rel).stem+'_'+hashlib.sha256(rel.encode()).hexdigest()[:8]+f'_fold{fold.seed}.json'
        local_v2 = ROOT/rel
        local_v2 = local_v2.parent/'evaluation'/tag
        v2 = local_v2 if local_v2.exists() else (ROOT/old.iloc[0].source if len(old) == 1 else G20/'shortlist_evaluation'/tag)
        local_geo = (ROOT/rel).parent/'geography'/tag
        geo = local_geo if local_geo.exists() else G20/'geography_evaluation'/tag
        for source, script in [(v2, ROOT/'research/adaptive_privacy.py'),
                               (geo, ROOT/'research/geography_free_attacks.py')]:
            r=json.loads(source.read_text())
            assert r['candidate_sha256'] == digest(ROOT/rel)
            assert r['split_sha256'] == fold.split_hash
            assert r['script_sha256'] in {digest(script), hashlib.sha256(script.read_bytes().replace(b'\r\n',b'\n')).hexdigest()}
            saved=versioned_matrix([r])
            expected=frame[frame.candidate.eq(rel)&frame.version.eq(r['attack_version'])]
            cols=['channel','method','group']
            for key,g in saved.groupby(cols):
                match=expected
                for col,val in zip(cols,key):match=match[match[col].eq(val)]
                assert len(match)==1
                for col in g.columns:
                    if col in ['candidate','fold','version']+cols:continue
                    if pd.isna(g.iloc[0][col]):continue
                    assert np.isclose(g.iloc[0][col],match.iloc[0][col],atol=1e-12)
            checked.append({'source':source.relative_to(ROOT).as_posix(),'sha256':digest(source)})
    return checked


def generate():
    base, b = pd.read_csv(BASE), pd.read_csv(DIST / 'B.csv')
    assert digest(BASE) == '98175bc965d6a28126a9cdd8f3d783880c325ae1cda9dfa31d683df1eaeec72a'
    inputs = [BASE, DIST/'B.csv', DIST/'utility_ref.json', DIST/'scoring_config.yaml',
              Path(__file__), ROOT/'research/repair_sex_correlation.py']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    for alpha in [.33, .67, 1.]:
        key = f'sex_a{alpha:.2f}_s3301'
        checkpoint = OUT / (key+'_manifest.json')
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            assert saved['inputs_sha256'] == hashes
            assert digest(ROOT/saved['candidate']) == saved['candidate_sha256']
            continue
        start = time.monotonic()
        frame, trace = repair(base, b, alpha, 3301)
        repeat, repeat_trace = repair(base, b, alpha, 3301)
        pd.testing.assert_frame_equal(frame, repeat)
        assert trace == repeat_trace
        csv, archive = OUT/f'C_{key}.csv', OUT/f'{key}.zip'
        if csv.exists():
            pd.testing.assert_frame_equal(pd.read_csv(csv), frame, check_dtype=False)
        else:
            frame.to_csv(csv, index=False, lineterminator='\n')
        dump(OUT/f'{key}_swaps.json', trace)
        with zipfile.ZipFile(archive, 'w') as z:
            for name, p in [('C.csv', csv), ('token.txt', DIST/'token.txt')]:
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, p.read_bytes())
        score = validate_score(csv, archive, OUT, key)
        with zipfile.ZipFile(archive) as z:
            assert z.read('C.csv') == csv.read_bytes()
            assert z.read('token.txt') == (DIST/'token.txt').read_bytes()
        record = dict(key=key, candidate=csv.relative_to(ROOT).as_posix(),
                      candidate_sha256=digest(csv), zip=archive.relative_to(ROOT).as_posix(),
                      zip_sha256=digest(archive), inputs_sha256=hashes,
                      code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      alpha=alpha, seed=3301, swaps=len(trace['swaps']), utility=score['utility'],
                      facets=score['facets'], validator='OK', regeneration='exact frame and trace',
                      seconds=time.monotonic()-start)
        dump(checkpoint, record)
        print(json.dumps({k: record[k] for k in ['key','utility','facets','swaps','seconds']}), flush=True)


def evaluate(fold_seed):
    assert fold_seed in [2701, 2702, 2703]
    fold = bp.SeparatedFold(fold_seed)
    folder = OUT / 'evaluation'; folder.mkdir(exist_ok=True)
    geo = OUT / 'geography'; geo.mkdir(exist_ok=True)
    results = []
    for manifest in sorted(OUT.glob('sex_*_manifest.json')):
        r = json.loads(manifest.read_text()); path = ROOT/r['candidate']
        assert digest(path) == r['candidate_sha256']
        results += [fold.evaluate(path, folder), evaluate_geo(fold, path, geo)]
        versioned_matrix(results).to_csv(OUT/f'fold{fold_seed}_partial.csv',index=False,lineterminator='\n')
    assert len(results) == 6
    baseline_matrix = REUSE_MATRIX
    old = pd.read_csv(baseline_matrix)
    controls = old.candidate.str.contains('positive_|negative_')
    reused = old[(old.candidate.isin([BASE.relative_to(ROOT).as_posix(),ORIGINAL.relative_to(ROOT).as_posix()]) | controls) & old.fold.eq(fold_seed)]
    assert reused.candidate.nunique() == len({BASE,ORIGINAL})+3
    evidence = verify_reuse(reused, fold)
    combined = pd.concat([reused, versioned_matrix(results)], ignore_index=True)
    assert not combined.duplicated(['candidate','fold','version','channel','method','group']).any()
    combined.to_csv(OUT/f'fold{fold_seed}_matrix.csv',index=False,lineterminator='\n')
    dump(OUT/f'fold{fold_seed}_checkpoint.json', dict(fold=fold_seed, completed=True,
         reuse_source=baseline_matrix.relative_to(ROOT).as_posix(), reuse_sha256=digest(baseline_matrix),
         split_sha256=fold.split_hash, checked_sources=evidence,
         source_note='G20 incumbent and controls, same v2/geography version and exploration fold'))
    print(f'fold {fold_seed} complete', flush=True)


def summarize():
    m = pd.concat([pd.read_csv(OUT/f'fold{f}_matrix.csv') for f in [2701,2702,2703]],ignore_index=True)
    m.to_csv(OUT/'all_methods.csv',index=False,lineterminator='\n')
    rows=[]
    for candidate, g in m.groupby('candidate'):
        p=ROOT/candidate;sp=p.with_name(p.stem.removeprefix('C_')+'_score.json')
        row={'candidate':candidate,'U':json.loads(sp.read_text())['utility'] if sp.exists() else None}
        for channel,group,metric,label in METRICS:
            v=g[g.channel.eq(channel)&g.group.eq(group)].groupby('fold')[metric].max()
            row[label+'_max']=v.max();row[label+'_mean']=v.mean()
            for f,value in v.items():row[label+f'_fold{f}']=value
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv',index=False,lineterminator='\n')
    print(summary[['candidate','U']+[x[3]+'_max' for x in METRICS]].to_string(index=False))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['generate','evaluate','summarize'],required=True)
    ap.add_argument('--fold',type=int);ap.add_argument('--out',type=Path);ap.add_argument('--base',type=Path)
    ap.add_argument('--reuse-matrix',type=Path);args=ap.parse_args()
    if args.out:OUT=args.out.resolve()
    if args.base:BASE=args.base.resolve()
    if args.reuse_matrix:REUSE_MATRIX=args.reuse_matrix.resolve()
    if args.stage=='generate':generate()
    elif args.stage=='evaluate':evaluate(args.fold)
    else:summarize()
