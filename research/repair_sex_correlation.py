"""G04 constrained aggregate-correlation repair; no attack labels in optimizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.adaptive_privacy import Fold, dump
from research.privacy_suite import digest
from research.run_processing import KIT
from experiments.local_privacy_audit import _rare_mask_from_reference
import numpy as np
import pandas as pd

BASE = ROOT / 'research/runs/20260924-G02-001/C_rare_aligned_s0.csv'
BASE_SCORE = BASE.parent / 'rare_aligned_s0_score.json'
DIST = ROOT / '_private_team14'
FEATURES = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG', 'smoking']


def repair(base, b, alpha, seed):
    if set(base.sex.unique()) != {0, 1}:
        raise ValueError('Expected binary sex coding')
    rng = np.random.default_rng(seed)
    ranks = base[FEATURES].rank(method='average').to_numpy()
    z = (ranks - ranks.mean(axis=0)) / ranks.std(axis=0)
    sex = base.sex.to_numpy().copy()
    denom = len(base) * sex.std()
    current = sex @ z / denom
    observed = b[FEATURES + ['sex']].corr(method='spearman').loc['sex', FEATURES].to_numpy()
    target = current + alpha * (observed - current)
    rare = _rare_mask_from_reference(base, DIST)
    pref = base.prefecture.to_numpy()
    trace = []
    initial = current.copy()
    for step in range(64):
        left = rng.choice(np.flatnonzero(sex == 0), 2000)
        right = rng.choice(np.flatnonzero(sex == 1), 2000)
        valid = (pref[left] == pref[right]) & (rare[left] == rare[right])
        left, right = left[valid], right[valid]
        if not len(left):
            break
        changes = (z[left] - z[right]) / denom
        error = current - target
        gains = 2 * (changes @ error) + (changes ** 2).sum(axis=1)
        best = int(np.argmin(gains))
        if gains[best] >= -1e-10:
            break
        i, j = int(left[best]), int(right[best])
        sex[i], sex[j] = 1, 0
        current += changes[best]
        trace.append({'step': step, 'row_0_to_1': i, 'row_1_to_0': j, 'loss': float(((current - target) ** 2).sum()),
                      'improvement': float(-gains[best])})
    result = base.copy()
    result['sex'] = sex
    pd.testing.assert_frame_equal(result.drop(columns='sex'), base.drop(columns='sex'))
    for frame in [base, result]:
        frame.attrs['group_counts'] = frame.assign(detected_rare=rare).groupby(['prefecture', 'detected_rare']).sex.sum()
    pd.testing.assert_series_equal(base.attrs['group_counts'], result.attrs['group_counts'])
    measured = result[FEATURES + ['sex']].corr(method='spearman').loc['sex', FEATURES].to_numpy()
    np.testing.assert_allclose(current, measured, atol=1e-12)
    return result, {'alpha': alpha, 'seed': seed, 'features': FEATURES, 'B_correlations': observed.tolist(),
                    'initial_correlations': initial.tolist(), 'target_correlations': target.tolist(),
                    'final_correlations': measured.tolist(), 'swaps': trace,
                    'changed_sex_rows': int((base.sex != result.sex).sum()),
                    'loss_before': float(((initial - target) ** 2).sum()), 'loss_after': float(((measured - target) ** 2).sum())}


def validate_score(csv, archive, out, key):
    env = dict(os.environ, PYTHONPATH=str(KIT / 'src'), PYTHONUTF8='1')
    for name, command in [
        ('validate', [sys.executable, '-m', 'pwscup2026.kit.validate', str(archive), '--kind', 'defense', '--dist', str(DIST)]),
        ('selfscore', [sys.executable, '-m', 'pwscup2026.kit.selfscore', str(csv), '--dist', str(DIST), '--json'])]:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8', timeout=1800)
        (out / f'{key}_{name}.log').write_text((proc.stdout + proc.stderr).replace(str(ROOT), '<repo>'), encoding='utf-8')
        if proc.returncode:
            raise RuntimeError(f'{key} {name} failed; log retained')
        if name == 'selfscore':
            payload = json.loads(proc.stdout)
            dump(out / f'{key}_score.json', payload)
    return payload


def risk_gates(result, baseline):
    deltas = {}
    for channel, group, metric, label in [
        ('mia', 'all', 'tpr_at_fpr_0.01', 'mia_all'),
        ('mia', 'true_member_rare_detected_control', 'tpr_at_fpr_0.01', 'mia_rare'),
        ('aia', 'all', 'r_raw', 'aia_all'), ('aia', 'stratified_all', 'r_raw', 'aia_stratified'),
        ('aia', 'rare_proxy', 'r_se_adjusted', 'aia_rare_se')]:
        deltas[label] = max(v[group][metric] - baseline[channel][method][group][metric] for method, v in result[channel].items())
    return deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out = args.out.resolve() / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise FileExistsError('Completed seed exists')
    base, b = pd.read_csv(BASE), pd.read_csv(DIST / 'B.csv')
    bs = json.loads(BASE_SCORE.read_text(encoding='utf-8'))
    folds = {seed: Fold(seed) for seed in [2401, 2402, 2403]}
    reference = {}
    for seed in folds:
        paths = list((ROOT / 'research/runs/20260924-G03b-001/evaluation').glob(f'C_rare_aligned_s0_*_fold{seed}.json'))
        assert len(paths) == 1
        reference[seed] = json.loads(paths[0].read_text(encoding='utf-8'))
        assert reference[seed]['candidate_sha256'] == digest(BASE)
    results = []
    for alpha in [.33, .67, 1.0]:
        started = time.monotonic()
        key = f'sexrepair_s{args.seed}_a{alpha:.2f}'
        checkpoint = out / f'{key}_manifest.json'
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text(encoding='utf-8'))
            assert record['candidate_sha256'] == digest(ROOT / record['candidate'])
            results.append(record)
            continue
        frame, trace = repair(base, b, alpha, args.seed)
        csv, archive = out / f'C_{key}.csv', out / f'{key}.zip'
        if csv.exists():
            pd.testing.assert_frame_equal(pd.read_csv(csv), frame, check_dtype=False)
        else:
            frame.to_csv(csv, index=False)
        dump(out / f'{key}_swaps.json', trace)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name, path in [('C.csv', csv), ('token.txt', DIST / 'token.txt')]:
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 24, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, path.read_bytes())
        score_path = out / f'{key}_score.json'
        score = json.loads(score_path.read_text(encoding='utf-8')) if score_path.exists() else validate_score(csv, archive, out, key)
        risk = {seed: risk_gates(fold.evaluate(csv, out), reference[seed]) for seed, fold in folds.items()}
        corr_delta = score['detail']['U_gen']['corr_score'] - bs['detail']['U_gen']['corr_score']
        utility_delta = score['utility'] - bs['utility']
        facets = {k: score['facets'][k] - v for k, v in bs['facets'].items()}
        gates = {'corr': corr_delta >= .01, 'utility': utility_delta >= -1e-10,
                 'facets': min(facets.values()) >= -.005, 'risk': max(v for fold in risk.values() for v in fold.values()) <= .01}
        record = {'key': key, 'candidate': csv.relative_to(ROOT).as_posix(), 'candidate_sha256': digest(csv),
                  'zip': archive.relative_to(ROOT).as_posix(), 'zip_sha256': digest(archive), 'seed': args.seed, 'alpha': alpha,
                  'base_sha256': digest(BASE), 'script_sha256': digest(Path(__file__)),
                  'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'input_sha256': {p.name: digest(p) for p in [DIST / n for n in ['B.csv', 'B_self.csv', 'A_bg.csv', 'utility_ref.json', 'scoring_config.yaml']]},
                  'utility': score['utility'], 'utility_delta': utility_delta, 'corr_delta': corr_delta,
                  'facets': score['facets'], 'facet_deltas': facets, 'risk_deltas': risk,
                  'gates': gates, 'pass': all(gates.values()), 'swaps': len(trace['swaps']), 'seconds': time.monotonic() - started,
                  'validator': 'OK', 'label': 'Exploratory processing trial; no server submission or confirmed adoption'}
        dump(checkpoint, record)
        results.append(record)
        print(json.dumps({k: record[k] for k in ['key', 'utility', 'utility_delta', 'corr_delta', 'gates', 'swaps', 'seconds']}), flush=True)
    dump(out / 'summary.json', {'seed': args.seed, 'candidates': results})


if __name__ == '__main__':
    main()
