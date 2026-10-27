"""Shared IO for bounded diagnostic studies; inference remains in pure modules."""
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from research.compare_final_attacks import ROOT, PUBLIC, TRUTH, config, dump
from research.build_final_attack_baseline import read, sha
from research.compare_attack_portfolios import score
from research.final_attack_api import COLUMNS, STATIC, QI


def plan_inputs(out):
    plan = json.loads((out/'plan.json').read_text())
    for path, digest in plan['input_hashes'].items():
        assert sha(ROOT/path) == digest, path
    assert subprocess.check_output(['git', '-C', str(ROOT/'pwscup2026-kit'), 'rev-parse', 'HEAD'], text=True).strip() == plan['kit']
    return plan


def preliminary(k):
    old = ROOT/f'research/runs/20261009-FA10-rule-001/prelim/target{k}'
    return (read(PUBLIC/f'C_{k}.csv')[COLUMNS],
            read(ROOT/'_private_team10/B_self.csv')[['record_id']+STATIC],
            read(PUBLIC/f'aia_challenge_{k}.csv')[['challenge_row_id']+QI],
            read(ROOT/'_private_team10/A_bg.csv')[COLUMNS],
            read(old/'mia_truth.csv'), read(old/'aia_truth.csv'))


def save_case(folder, c, m, a, mt, at, k, mechanism, verify=False):
    rows = score(m, a, mt, at, k, 'exploratory_mechanism', mechanism=mechanism)
    if verify or (folder/'checkpoint.json').exists():
        for path, digest in json.loads((folder/'checkpoint.json').read_text())['files'].items():
            assert sha(folder/path) == digest
        for name, fresh in [('C.csv', c), ('mia.csv', m), ('aia.csv', a), ('metrics.csv', rows)]:
            saved = read(folder/name)
            pd.testing.assert_frame_equal(saved, fresh, check_dtype=False, atol=1e-12, rtol=0)
    else:
        folder.mkdir(parents=True)
        for name, frame in [('C.csv', c), ('mia.csv', m), ('aia.csv', a),
                            ('mia_truth.csv', mt), ('aia_truth.csv', at), ('metrics.csv', rows)]:
            frame.to_csv(folder/name, index=False)
        dump(folder/'checkpoint.json', {'files': {p.name: sha(p) for p in folder.glob('*.csv')}})
    return rows


def commit_engine(out, paths):
    info = {p: sha(ROOT/p) for p in paths}
    dest = out/'engine_manifest.json'
    if dest.exists():
        assert json.loads(dest.read_text()) == info
    else:
        dump(dest, info)


def summary_deltas(frame, keys, value='delta'):
    rows = []
    for key, part in frame.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        values = part[value]
        rows.append(dict(zip(keys, key), mean=float(values.mean()),
                         without_best=float(values.drop(values.idxmax()).mean()),
                         no_easy=float(part.loc[~part.target.isin([1, 2]), value].mean()),
                         wins=int((values > 1e-12).sum()), losses=int((values < -1e-12).sum())))
    return pd.DataFrame(rows)
