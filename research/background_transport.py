"""G17c preregistered background-rank transport; utility stage only."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.residual_rare import (DIST, BASE, _load_detector, _detector_mask,
    _fit_cause_cox, _run_path, load_cfg, write_c, dump, digest, validate_score)
from research.background_partition import OUT, partition_frame
from experiments.run_aligned_synth import _format_continuous, _sample_categorical
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

NUMERIC = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']


def transport(sample, source_groups, b, target_groups, alpha):
    """Only numeric group marginals enter; ties use their mean rank."""
    result = sample[NUMERIC].copy().reset_index(drop=True)
    trace = []
    for group in [False, True]:
        rows = np.flatnonzero(np.asarray(source_groups) == group)
        target = b.loc[np.asarray(target_groups) == group, NUMERIC]
        if not len(rows) or target.empty:
            raise ValueError('Both source and target groups must be nonempty')
        for col in NUMERIC:
            original = result.loc[rows, col].to_numpy(float)
            q = (pd.Series(original).rank(method='average').to_numpy() - .5) / len(rows)
            quantile = np.quantile(target[col].to_numpy(float), q)
            weight = 1. if col == 'FPG' else alpha
            value = (1-weight)*original + weight*quantile
            result.loc[rows, col] = value
            trace.append(pd.DataFrame({'synthetic_row': rows, 'source_rare': group,
                'column': col, 'source_value': original, 'rank_quantile': q,
                'target_quantile': quantile, 'weight': weight, 'transport_value': value}))
    return result, pd.concat(trace, ignore_index=True)


def generate(b, bg, cfg, detector, alpha, seed):
    split = partition_frame(bg, 2701)
    pd.testing.assert_frame_equal(split, pd.read_csv(OUT/'fold2701_split.csv'), check_exact=True)
    generation = split.loc[split.partition == 'generation', 'background_row'].to_numpy()
    gmask = _detector_mask(bg.iloc[generation], *detector)
    bmask = _detector_mask(b, *detector)
    assert len(b) == 1049 and bmask.sum() == 59
    rng = np.random.default_rng(19000+seed)
    # The group draw order is fixed before any evaluation.
    rare = rng.choice(generation[gmask], 59, replace=False)
    common = rng.choice(generation[~gmask], 990, replace=False)
    selected = np.r_[rare, common]
    groups = np.r_[np.ones(59, bool), np.zeros(990, bool)]
    assert len(np.unique(selected)) == 1049 and np.isin(selected, generation).all()
    continuous, trace = transport(bg.iloc[selected], groups, b, bmask, alpha)
    formatted = _format_continuous(continuous, cfg)
    frame = _sample_categorical(b, formatted, rng)
    frame.insert(0, 'record_id', np.arange(len(frame)))
    selection = pd.DataFrame({'synthetic_row': np.arange(len(frame)), 'background_row': selected,
        'background_record_id': bg.iloc[selected].record_id.to_numpy(), 'source_rare': groups,
        'transported_rare': _detector_mask(frame, *detector)})
    return frame, selection, trace


def numeric_distances(frame, b, bg):
    scale = b[NUMERIC].std(ddof=0).to_numpy()
    scale = np.maximum(scale, 1e-12)
    out = {}
    for name, ref in [('B', b), ('background', bg)]:
        d = cdist(frame[NUMERIC].to_numpy(float)/scale,
                  ref[NUMERIC].to_numpy(float)/scale).min(axis=1)
        out[name] = {'min': float(d.min()), 'median': float(np.median(d)),
                     'q01': float(np.quantile(d, .01)), 'near_1e-6': int((d < 1e-6).sum())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
    args = ap.parse_args()
    out = OUT/f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'utility_summary.json').exists():
        raise FileExistsError('Completed utility seed must not be rerun')
    if args.seed:
        pilot = json.loads((OUT/'seed0/utility_summary.json').read_text(encoding='utf-8'))
        assert pilot['expand_seeds'], 'Pilot did not meet preregistered expansion criterion'
    b = pd.read_csv(DIST/'B.csv'); bg = pd.read_csv(DIST/'A_bg.csv')
    cfg = load_cfg(); detector = _load_detector(DIST)
    ob, db = _fit_cause_cox(b, 'onset'), _fit_cause_cox(b, 'death')
    inputs = [DIST/'B.csv', DIST/'A_bg.csv', DIST/'utility_ref.json', DIST/'scoring_config.yaml',
              OUT/'fold2701_split.csv', Path(__file__), ROOT/'experiments/run_aligned_synth.py',
              ROOT/'experiments/run_calibrated_outcomes.py']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    records = []
    for alpha in [.5, 1.]:
        static, selection, trace = generate(b, bg, cfg, detector, alpha, args.seed)
        for name, frame in [('static', static), ('selection', selection), ('transport', trace)]:
            path = out/f'{name}_a{alpha:.1f}.csv'
            if path.exists():
                pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False, rtol=1e-13, atol=1e-13)
            else:
                frame.to_csv(path, index=False, lineterminator='\n')
        distances = numeric_distances(static, b, bg)
        trials = _run_path(b, static, cfg, ob, db, args.seed, 1., .5, 6)
        if len(trials) != 7:
            dump(out/f'cox_incomplete_a{alpha:.1f}.json', {'expected': 7, 'actual': len(trials),
                'reason': 'Cox convergence stop; do not replace failed iterations'})
        for trial in trials:
            started = time.monotonic()
            key = f'bgtransport_s{args.seed}_a{alpha:.1f}_i{trial.iteration}'
            checkpoint = out/f'{key}_manifest.json'
            if checkpoint.exists():
                r = json.loads(checkpoint.read_text(encoding='utf-8'))
                assert r['inputs_sha256'] == hashes
                assert digest(ROOT/r['candidate']) == r['candidate_sha256']
                records.append(r)
                continue
            csv, archive = out/f'C_{key}.csv', out/f'{key}.zip'
            write_c(trial.frame, csv)
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for name, path in [('C.csv', csv), ('token.txt', DIST/'token.txt')]:
                    info = zipfile.ZipInfo(name, date_time=(2026, 9, 24, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, path.read_bytes())
            score = validate_score(csv, archive, out, key)
            r = {'key': key, 'seed': args.seed, 'alpha': alpha, 'iteration': trial.iteration,
                'primary_setting': trial.iteration in [2, 4, 6],
                'candidate': csv.relative_to(ROOT).as_posix(), 'candidate_sha256': digest(csv),
                'zip': archive.relative_to(ROOT).as_posix(), 'zip_sha256': digest(archive),
                'utility': score['utility'], 'facets': score['facets'], 'validator': 'OK',
                'pruned': score['utility'] < .85, 'privacy_state': 'unmeasured',
                'detected_rare': int(selection.transported_rare.sum()),
                'novel': score['detail']['U_valid']['novel'], 'numeric_distances': distances,
                'beta_generator': trial.beta_generator.tolist(), 'code_commit': commit,
                'inputs_sha256': hashes, 'seconds': time.monotonic()-started,
                'label': 'Exploratory local utility only; no official Anon or adoption'}
            dump(checkpoint, r); records.append(r)
            print(json.dumps({k: r[k] for k in ['key', 'utility', 'facets', 'detected_rare', 'novel', 'pruned']}), flush=True)
    expand = any(r['primary_setting'] and r['utility'] >= .85 for r in records)
    dump(out/'utility_summary.json', {'seed': args.seed, 'n_candidates': len(records),
        'expand_seeds': expand, 'candidates': records})


if __name__ == '__main__':
    main()
