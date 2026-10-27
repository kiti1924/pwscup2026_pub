"""Version 2 local attack exploration: all inference queries exclude secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pwscup2026-kit/starter/src')]
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor
from research.privacy_suite import (QI, digest, split_background, mia_metrics, success_metrics,
    standardized_aia, log_qi_predict)
from experiments.local_privacy_audit import _load_inputs, _member_rare_mask, _rare_mask_from_reference
from experiments.full_static_nearest_audit import NUMERIC, CATEGORICAL, EXACT_COLUMNS
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict

STATIC = NUMERIC + CATEGORICAL
DIST = ROOT / '_private_team14'
VERSION = 'G03b-v2'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def block_indices(frame, sex, pref, rare=None):
    mask = frame.sex.eq(sex).to_numpy() & frame.prefecture.eq(pref).to_numpy()
    if rare is not None:
        strict = mask & frame.detected_rare.eq(rare).to_numpy()
        if strict.any():
            return np.flatnonzero(strict)
    if not mask.any():
        mask = frame.sex.eq(sex).to_numpy()
    return np.flatnonzero(mask) if mask.any() else np.arange(len(frame))


def static_distances(query, donors, scale):
    d = cdist(query[NUMERIC].to_numpy() / scale, donors[NUMERIC].to_numpy() / scale, 'sqeuclidean')
    for col in CATEGORICAL:
        d += 2 * (query[col].to_numpy()[:, None] != donors[col].to_numpy()[None, :])
    return np.sqrt(d)


def conditional_min(distances, query, donors):
    return np.array([distances[i, block_indices(donors, row.sex, row.prefecture, row.detected_rare)].min()
                     for i, row in enumerate(query.itertuples(index=False))])


def qi_transform(frame):
    x = frame[QI[:5]].to_numpy(float).copy()
    x[:, [2, 4]] = np.log1p(x[:, [2, 4]])
    return x


def additional_aia(c, query, ref):
    if list(query.columns) != QI:
        raise ValueError('AIA inference accepts only public QI')
    scale = np.maximum(qi_transform(ref).std(axis=0, ddof=1), 1e-12)
    qc = cdist(qi_transform(query) / scale, qi_transform(c) / scale, 'sqeuclidean')
    qr = cdist(qi_transform(query) / scale, qi_transform(ref) / scale, 'sqeuclidean')
    ct, rt = c.time.to_numpy(), ref.time.to_numpy()
    sentinel = float(max(ct.max(), rt.max()) + 2)
    gated, contrast = [], []
    for i, row in enumerate(query.itertuples(index=False)):
        ci = block_indices(c, row.sex, row.prefecture)
        ri = block_indices(ref, row.sex, row.prefecture)
        cd, rd = qc[i, ci], qr[i, ri]
        gate = np.log(np.sqrt(rd.min()) + 1e-9) - np.log(np.sqrt(cd.min()) + 1e-9)
        gated.append(float(np.sort(ct[ci[cd <= cd.min()]]).mean()) if gate > .5 else sentinel)
        ck, rk = min(20, len(ci)), min(50, len(ri))
        ts = np.sort(ct[ci[cd <= np.partition(cd, ck - 1)[ck - 1]]])
        bs = rt[ri[rd <= np.partition(rd, rk - 1)[rk - 1]]]
        ends = np.searchsorted(ts, ts + 1., side='right') - 1
        centers = (ts + ts[ends]) / 2
        contrasts = np.mean(np.abs(centers[:, None] - ts) <= .5, axis=1) - np.mean(np.abs(centers[:, None] - bs) <= .5, axis=1)
        best = int(np.argmax(contrasts))
        contrast.append(float(centers[best]) if contrasts[best] > 0 else sentinel)
    # Fit only C(QI,time); query secrets and scoring labels never enter the fit.
    design = pd.get_dummies(pd.concat([c[QI], query], ignore_index=True), columns=['sex', 'prefecture'], dtype=float)
    model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=5, random_state=20260924, n_jobs=1)
    model.fit(design.iloc[:len(c)], c.time)
    return {'gated_log_1nn': np.asarray(gated), 'contrast_window20': np.asarray(contrast),
            'extra_trees_time': model.predict(design.iloc[len(c):])}


class Fold:
    def __init__(self, seed):
        self.seed = seed
        self.b, bs, self.bg, self.cfg = _load_inputs(DIST)
        pd.testing.assert_frame_equal(self.b, bs.drop(columns='is_rare'))
        tr, te = split_background(len(self.bg), seed)
        self.split = pd.DataFrame({'background_row': np.r_[tr, te], 'partition': ['reference'] * len(tr) + ['control'] * len(te)})
        self.ref, self.controls = self.bg.iloc[tr].reset_index(drop=True), self.bg.iloc[te].reset_index(drop=True)
        self.query = pd.concat([self.b, self.controls], ignore_index=True)
        self.labels = np.r_[np.ones(len(self.b), int), np.zeros(len(self.controls), int)]
        self.mr = _member_rare_mask(self.b, bs)
        self.cr = _rare_mask_from_reference(self.controls, DIST)
        self.dr = _rare_mask_from_reference(self.b, DIST)
        self.scale = self.ref[NUMERIC].std(ddof=0).to_numpy()
        self.scale = np.where(self.scale > 0, self.scale, 1.)
        self.qstatic = self.query[STATIC].copy()
        self.qstatic['detected_rare'] = np.r_[self.dr, self.cr]
        rstatic = self.ref[STATIC].copy()
        rstatic['detected_rare'] = _rare_mask_from_reference(self.ref, DIST)
        rd = static_distances(self.qstatic, rstatic, self.scale)
        self.rmin, self.r5 = rd.min(axis=1), np.partition(rd, 4, axis=1)[:, 4]
        self.rcond = conditional_min(rd, self.qstatic, rstatic)
        self.cache = {}

    def mia(self, c):
        key = hashlib.sha256(pd.util.hash_pandas_object(c[STATIC], index=False).to_numpy().tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        cstatic = c[STATIC].copy()
        cstatic['detected_rare'] = _rare_mask_from_reference(c, DIST)
        d = static_distances(self.qstatic, cstatic, self.scale)
        dm, d5 = d.min(axis=1), np.partition(d, 4, axis=1)[:, 4]
        dc = conditional_min(d, self.qstatic, cstatic)
        dn = cdist(self.qstatic[NUMERIC].to_numpy() / self.scale, c[NUMERIC].to_numpy() / self.scale).min(axis=1)
        values = {'kde_ratio': overlap_mia(c, self.ref, self.qstatic[STATIC], self.cfg),
                  'continuous_nn': -dn, 'full_static_nn': -dm,
                  'relative_static_nn': np.log(self.rmin + 1e-9) - np.log(dm + 1e-9),
                  'relative_static_k5': np.log(self.r5 + 1e-9) - np.log(d5 + 1e-9),
                  'conditional_relative_nn': np.log(self.rcond + 1e-9) - np.log(dc + 1e-9)}
        self.cache[key] = values
        return values

    def evaluate(self, path, out):
        started = time.monotonic()
        rel = path.relative_to(ROOT).as_posix()
        tag = path.stem + '_' + hashlib.sha256(rel.encode()).hexdigest()[:8] + f'_fold{self.seed}'
        result_path = out / (tag + '.json')
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            assert saved['candidate_sha256'] == digest(path)
            assert saved['attack_version'] == VERSION and saved['script_sha256'] == digest(Path(__file__))
            return saved
        c = pd.read_csv(path)
        result = {'candidate': rel, 'candidate_sha256': digest(path), 'fold': self.seed, 'attack_version': VERSION,
                  'script_sha256': digest(Path(__file__)), 'label': 'Exploratory local proxies, not official Anon', 'mia': {}, 'aia': {}}
        predictions = pd.DataFrame({'source': np.where(self.labels == 1, 'B14_member', 'background_control'),
                                    'record_id': self.query.record_id})
        masks = {'all': np.ones(len(self.query), bool), 'true_member_rare_detected_control': np.r_[self.mr, self.cr],
                 'detected_both': np.r_[self.dr, self.cr]}
        for name, values in self.mia(c).items():
            result['mia'][name] = {kind: mia_metrics(values, self.labels, mask) for kind, mask in masks.items()}
            predictions['mia_' + name] = values
        q = self.query[QI].copy()
        attacks = {'official_1nn': aia_predict(c, q, self.ref, self.cfg)[1],
                   'log_qi_1nn': log_qi_predict(c, q, self.ref, False),
                   'log_qi_window20': log_qi_predict(c, q, self.ref, True), **additional_aia(c, q, self.ref)}
        for name, values in attacks.items():
            hit = np.abs(values - self.query.time.to_numpy()) <= .5
            mh, ch = hit[:len(self.b)], hit[len(self.b):]
            result['aia'][name] = {'all': success_metrics(mh, ch), 'rare_proxy': success_metrics(mh[self.mr], ch[self.cr]),
                                    'stratified_all': standardized_aia(self.b, self.controls, mh, ch)}
            predictions['time_hat_' + name] = values
        result['seconds'] = time.monotonic() - started
        predictions.to_csv(out / (tag + '_predictions.csv'), index=False)
        dump(result_path, result)
        print(json.dumps({'candidate': path.name, 'fold': self.seed, 'seconds': round(result['seconds'], 2),
                          'mia_max': max(v['all']['tpr_at_fpr_0.01'] for v in result['mia'].values()),
                          'aia_max': max(v['stratified_all']['r_raw'] for v in result['aia'].values())}), flush=True)
        return result


def positive_controls(out):
    b = pd.read_csv(DIST / 'B.csv')[EXACT_COLUMNS]
    base = pd.read_csv(ROOT / 'research/runs/20260924-G02-001/C_rare_aligned_s0.csv')[EXACT_COLUMNS]
    idx = np.sort(np.random.default_rng(3101).choice(len(b), len(b) // 2, replace=False))
    partial = base.copy()
    partial.iloc[idx] = b.iloc[idx].to_numpy()
    paths = []
    for name, frame in [('positive_raw', b), ('positive_halfcopy', partial)]:
        path = out / f'C_{name}.csv'
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False)
        else:
            frame.to_csv(path, index=False)
        paths.append(path)
    index = out / 'positive_halfcopy_indices.csv'
    if not index.exists():
        pd.DataFrame({'B14_row_index': idx}).to_csv(index, index=False)
    return paths


def candidates(stage, out):
    paths = sorted((ROOT / 'research/runs/20260924-G02-001').glob('C_*.csv'))
    if stage == 'pilot':
        paths += [ROOT / 'research/runs/20260924-G02b-001/seed0/C_u0_h1.000_lr0.500_i2.csv']
    else:
        paths += sorted(p for p in (ROOT / 'research/runs/20260924-G02b-001').rglob('C_*.csv') if '_fold' not in p.name)
        assert len(paths) == 65
    return paths + positive_controls(out)


def matrix(results):
    rows = []
    for r in results:
        for channel in ['mia', 'aia']:
            for method, groups in r[channel].items():
                for group, metrics in groups.items():
                    rows.append({'candidate': r['candidate'], 'fold': r['fold'], 'version': VERSION,
                                 'channel': channel, 'method': method, 'group': group, **metrics})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['pilot', 'all'], required=True)
    ap.add_argument('--folds', required=True)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = candidates(args.stage, out)
    result_dir = out / 'evaluation'
    result_dir.mkdir(exist_ok=True)
    for seed in map(int, args.folds.split(',')):
        fold = Fold(seed)
        fold.split.to_csv(out / f'fold{seed}_split.csv', index=False)
        results = [fold.evaluate(p, result_dir) for p in paths]
        matrix(results).to_csv(out / f'{args.stage}_fold{seed}_matrix.csv', index=False)
        gates = []
        for r, (mt, at) in zip(results[-2:], [(.9, .8), (.2, .1)]):
            m = max(v['all']['tpr_at_fpr_0.01'] for v in r['mia'].values())
            a = max(v['all']['r_raw'] for v in r['aia'].values())
            gates.append({'candidate': r['candidate'], 'mia': m, 'aia': a, 'pass': bool(m >= mt and a >= at)})
        dump(out / f'{args.stage}_fold{seed}_checkpoint.json', {'fold': seed, 'n_candidates': len(results), 'positive_gates': gates,
             'script_sha256': digest(Path(__file__)), 'candidate_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
             'input_hashes': {p.name: digest(p) for p in [DIST / n for n in ['B.csv', 'B_self.csv', 'A_bg.csv', 'utility_ref.json', 'scoring_config.yaml']]},
             'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
        assert all(g['pass'] for g in gates), gates
        print('POSITIVE GATES ' + json.dumps(gates), flush=True)


if __name__ == '__main__':
    main()
