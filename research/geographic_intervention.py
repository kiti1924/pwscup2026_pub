"""G17g: paired county-distribution intervention with frozen clinical columns."""
from pathlib import Path
import argparse
import csv
import io
import json
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research.background_transport import DIST, dump, digest, validate_score
from research.background_partition import SeparatedFold, PARTITION_VERSION, partition_frame
from research.evaluate_background_copula import historical_result
from research.evaluate_background_transport import maxima, anchors
from research.repair_sex_correlation import risk_gates
from research.adaptive_privacy import matrix
from research.replay_calibrated_privacy import same
from research.privacy_suite import mia_metrics, success_metrics, standardized_aia

OUT = ROOT/'research/runs/20260925-G17g-001'
OLD = ROOT/'research/runs/20260924-G17e-001'
PART = ROOT/'research/runs/20260924-G17c-001'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked_inputs():
    data = read(OUT/'plan_inputs.json')
    for name, expected in data['inputs_sha256'].items():
        assert digest(ROOT/name) == expected, name
    return data


def inverse_cdf(values, uniforms):
    codes, counts = np.unique(np.asarray(values), return_counts=True)
    if not len(codes) or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError('Nonempty empirical support and uniforms in [0,1) required')
    cdf = np.cumsum(counts)/counts.sum()
    cdf[-1] = 1.
    return codes[np.searchsorted(cdf, uniforms, side='right')]


def replace_prefecture(parent, assignments):
    """Keep every other CSV field lexically identical, not just float-close."""
    rows = list(csv.reader(io.StringIO(parent.read_text(encoding='utf-8'))))
    column = rows[0].index('prefecture')
    assert len(assignments) == len(rows)-1
    for row, value in zip(rows[1:], assignments):
        row[column] = str(value)
    output = io.StringIO(newline='')
    csv.writer(output, lineterminator='\n').writerows(rows)
    return output.getvalue()


def parent_path(seed, gamma):
    return OLD/f'seed{seed}/C_calcopula_s{seed}_w1.0_g{gamma}_i4.csv'


def assignments(seed):
    b, bg = pd.read_csv(DIST/'B.csv'), pd.read_csv(DIST/'A_bg.csv')
    split = partition_frame(bg, 2701)
    pd.testing.assert_frame_equal(split, pd.read_csv(PART/'fold2701_split.csv'), check_exact=True)
    g = bg.iloc[split.loc[split.partition == 'generation', 'background_row'].to_numpy()]
    u = np.random.default_rng(22000+seed).random(len(b))
    return pd.DataFrame({'synthetic_row': np.arange(len(b)), 'uniform': u,
                         'B': inverse_cdf(b.prefecture, u), 'G': inverse_cdf(g.prefecture, u)})


def generate(seed):
    plan = checked_inputs()
    folder = OUT/f'seed{seed}'
    folder.mkdir(exist_ok=True)
    if (folder/'utility_summary.json').exists():
        raise FileExistsError('Completed seed; use replay, not regeneration')
    draws = assignments(seed)
    draws.to_csv(folder/'prefecture_draws.csv', index=False, lineterminator='\n')
    hashes = {**plan['inputs_sha256'], 'research/geographic_intervention.py': digest(Path(__file__))}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    records = []
    for gamma in [0, 1]:
        parent = parent_path(seed, gamma)
        parent_manifest = read(parent.with_name(parent.stem[2:]+'_manifest.json'))
        for source in ['B', 'G']:
            key = f'geo_s{seed}_g{gamma}_{source}'
            csvpath, archive = folder/f'C_{key}.csv', folder/f'{key}.zip'
            manifest = folder/f'{key}_manifest.json'
            content = replace_prefecture(parent, draws[source].to_numpy())
            if manifest.exists():
                record = read(manifest)
                assert record['inputs_sha256'] == hashes
                assert csvpath.read_text(encoding='utf-8') == content
                assert digest(csvpath) == record['candidate_sha256']
                records.append(record)
                continue
            csvpath.write_text(content, encoding='utf-8', newline='\n')
            original, changed = pd.read_csv(parent), pd.read_csv(csvpath)
            pd.testing.assert_frame_equal(original.drop(columns='prefecture'), changed.drop(columns='prefecture'), check_exact=True)
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for name, path in [('C.csv', csvpath), ('token.txt', DIST/'token.txt')]:
                    info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, path.read_bytes())
            score = validate_score(csvpath, archive, folder, key)
            record = {'key': key, 'seed': seed, 'gamma': gamma, 'source': source,
                'candidate': csvpath.relative_to(ROOT).as_posix(), 'candidate_sha256': digest(csvpath),
                'zip': archive.relative_to(ROOT).as_posix(), 'zip_sha256': digest(archive),
                'parent': parent.relative_to(ROOT).as_posix(), 'parent_sha256': digest(parent),
                'parent_utility': parent_manifest['utility'], 'parent_novel': parent_manifest['novel'],
                'utility': score['utility'], 'facets': score['facets'], 'novel': score['detail']['U_valid']['novel'],
                'pruned': False, 'validator': 'OK', 'unchanged_columns': original.columns.drop('prefecture').tolist(),
                'inputs_sha256': hashes, 'code_commit': commit, 'label': 'Exploratory mechanism comparison; no adoption'}
            dump(manifest, record)
            records.append(record)
            print(json.dumps({k: record[k] for k in ['key', 'utility', 'facets', 'novel']}), flush=True)
    dump(folder/'utility_summary.json', {'seed': seed, 'candidates': records})


def locate_result(path, fold, directory):
    found = list(directory.glob(path.stem+f'_*_fold{fold}.json'))
    if not found:
        return None
    assert len(found) == 1
    result = read(found[0])
    assert result['candidate_sha256'] == digest(path)
    assert result['evaluation_partition'] == PARTITION_VERSION
    assert result['split_sha256'] == digest(PART/f'fold{fold}_split.csv')
    assert result['script_sha256'] == digest(ROOT/'research/adaptive_privacy.py')
    assert result['partition_script_sha256'] == digest(ROOT/'research/background_partition.py')
    return result, found[0]


def predictions(path):
    return pd.read_csv(path.with_name(path.stem+'_predictions.csv'), float_precision='round_trip')


def rescore(result, path, fold):
    pred = predictions(path)
    np.testing.assert_array_equal(pred.record_id, fold.query.record_id)
    np.testing.assert_array_equal(pred.source, np.where(fold.labels == 1, 'B14_member', 'background_control'))
    masks = {'all': np.ones(len(fold.query), bool), 'true_member_rare_detected_control': np.r_[fold.mr, fold.cr],
             'detected_both': np.r_[fold.dr, fold.cr]}
    for method, expected in result['mia'].items():
        same({k: mia_metrics(pred['mia_'+method].to_numpy(), fold.labels, m) for k, m in masks.items()}, expected)
    for method, expected in result['aia'].items():
        hit = abs(pred['time_hat_'+method].to_numpy()-fold.query.time.to_numpy()) <= .5
        mh, ch = hit[:len(fold.b)], hit[len(fold.b):]
        same({'all': success_metrics(mh, ch), 'rare_proxy': success_metrics(mh[fold.mr], ch[fold.cr]),
              'stratified_all': standardized_aia(fold.b, fold.controls, mh, ch)}, expected)
    return pred


def evaluate(seed):
    checked_inputs()
    checkpoint = OUT/f'seed{seed}_risk_summary.json'
    if checkpoint.exists():
        raise FileExistsError('Completed evaluation seed')
    records = read(OUT/f'seed{seed}/utility_summary.json')['candidates']
    output = OUT/'evaluation'
    output.mkdir(exist_ok=True)
    results, paired, evidence = [], [], {}
    for fold_seed in [2701, 2702, 2703]:
        positive = read(PART/f'anchors_fold{fold_seed}_checkpoint.json')
        assert all(x['pass'] for x in positive['positive_gates'])
        base, basepath = historical_result(anchors()['incumbent'], fold_seed)
        high, highpath = historical_result(anchors()['high_parent'], fold_seed)
        fold = SeparatedFold(fold_seed)
        current, preds = {}, {}
        for record in records:
            path = ROOT/record['candidate']
            assert digest(path) == record['candidate_sha256']
            result = fold.evaluate(path, output)
            _, resultpath = locate_result(path, fold_seed, output)
            pred = rescore(result, resultpath, fold)
            parent = ROOT/record['parent']
            old = locate_result(parent, fold_seed, OLD/'evaluation')
            if old is None:
                fold.evaluate(parent, output)
                old = locate_result(parent, fold_seed, output)
            old_result, oldpath = old
            oldpred = rescore(old_result, oldpath, fold)
            for method in ['kde_ratio', 'continuous_nn']:
                np.testing.assert_array_equal(pred['mia_'+method], oldpred['mia_'+method])
            for p in [basepath, highpath, oldpath, resultpath]:
                evidence[p.relative_to(ROOT).as_posix()] = digest(p)
            key = (record['gamma'], record['source'])
            current[key], preds[key] = result, pred
            results.append(result)
            paired.append({'key': record['key'], 'seed': seed, 'gamma': record['gamma'], 'source': record['source'],
                'fold': fold_seed, 'maxima': maxima(result), 'parent_max_deltas': {k:v-maxima(old_result)[k] for k,v in maxima(result).items()},
                'baseline_paired_deltas': risk_gates(result, base), 'high_parent_paired_deltas': risk_gates(result, high)})
        for source in ['B', 'G']:
            columns = [c for c in preds[(0, source)].columns if c.startswith('mia_')]
            pd.testing.assert_frame_equal(preds[(0, source)][columns], preds[(1, source)][columns], check_exact=True)
        for gamma in [0, 1]:
            b, g = current[(gamma, 'B')], current[(gamma, 'G')]
            row = next(x for x in paired if x['fold'] == fold_seed and x['gamma'] == gamma and x['source'] == 'G')
            row['B_control_max_deltas'] = {k:v-maxima(b)[k] for k,v in maxima(g).items()}
            row['B_control_paired_deltas'] = risk_gates(g, b)
        print(f'seed{seed} fold{fold_seed}: 4 candidates, all predictions rescored, numeric and gamma MIA invariants pass', flush=True)
    matrix(results).to_csv(OUT/f'seed{seed}_attack_matrix.csv', index=False, lineterminator='\n')
    dump(checkpoint, {'seed': seed, 'n_evaluations': len(results), 'comparisons': paired, 'evidence_sha256': evidence,
                      'invariants': '12-column exact; numeric MIA and same-static MIA exact; all saved metrics replayed at 1e-12'})


def summary():
    checked_inputs()
    rows, folds, manifests, legacy = [], [], [], []
    baseline = read(ROOT/'research/runs/20260924-G02-001/rare_aligned_s0_score.json')
    for seed in range(3):
        records = read(OUT/f'seed{seed}/utility_summary.json')['candidates']
        riskpath = OUT/f'seed{seed}_risk_summary.json'
        risk = read(riskpath)
        for name, value in risk['evidence_sha256'].items():
            assert digest(ROOT/name) == value
        draws = assignments(seed)
        saved = pd.read_csv(OUT/f'seed{seed}/prefecture_draws.csv', float_precision='round_trip')
        pd.testing.assert_frame_equal(draws, saved, check_exact=True)
        for r in records:
            path = ROOT/r['candidate']
            assert digest(path) == r['candidate_sha256']
            assert path.read_text(encoding='utf-8') == replace_prefecture(ROOT/r['parent'], draws[r['source']].to_numpy())
            rr = [x for x in risk['comparisons'] if x['key'] == r['key']]
            assert len(rr) == 3
            strict = r['utility'] >= baseline['utility']+.01 and all(
                r['facets'][k]-baseline['facets'][k] >= -.01 for k in ['U_gen', 'U_rare', 'U_valid']) and max(
                v for x in rr for v in x['baseline_paired_deltas'].values()) <= .01
            high_deltas = []
            for x in rr:
                high, _ = historical_result(anchors()['high_parent'], x['fold'])
                high_deltas.append({k:v-maxima(high)[k] for k,v in x['maxima'].items()})
            eligible = r['utility'] >= .90 and r['novel'] >= .95 and all(
                d[k] <= .01 for d in high_deltas for k in ['mia_all', 'mia_rare', 'aia_stratified']) and sum(
                d['mia_rare'] <= -1/24+1e-12 for d in high_deltas) >= 2
            legacy.append({k:r[k] for k in ['key', 'seed', 'gamma', 'source', 'utility', 'novel']} |
                          {'strict_pass':strict, 'confirmation_entry':eligible})
            if r['source'] == 'B':
                continue
            control = next(x for x in records if x['gamma'] == r['gamma'] and x['source'] == 'B')
            comparisons = [x for x in risk['comparisons'] if x['key'] == r['key']]
            for x in comparisons:
                folds.append({k:x[k] for k in ['key', 'seed', 'gamma', 'fold']} | x['B_control_max_deltas'])
            deltas = [x['B_control_max_deltas'] for x in comparisons]
            nonworse = all(d[k] <= .01+1e-12 for d in deltas for k in ['mia_all', 'mia_rare', 'aia_stratified'])
            improve = sum(d['mia_rare'] <= -1/24+1e-12 for d in deltas)
            uok = r['utility'] >= control['utility']-.005-1e-12
            novelok = abs(r['novel']-control['novel']) <= 1e-12
            rows.append({'seed':seed, 'gamma':r['gamma'], 'key':r['key'], 'utility_G':r['utility'], 'utility_B':control['utility'],
                'utility_delta': r['utility']-control['utility'], 'novel':r['novel'], 'nonworsening':nonworse,
                'rare_improving_folds':improve, 'mechanism_pass':uok and novelok and nonworse and improve >= 2,
                **{k+'_max_delta': max(d[k] for d in deltas) for k in deltas[0]}})
        manifests.extend([riskpath, OUT/f'seed{seed}/utility_summary.json'])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'candidate_summary.csv', index=False, lineterminator='\n')
    pd.DataFrame(folds).to_csv(OUT/'paired_fold_deltas.csv', index=False, lineterminator='\n')
    legacy_frame = pd.DataFrame(legacy)
    legacy_frame.to_csv(OUT/'legacy_defense_gates.csv', index=False, lineterminator='\n')
    passed = [g for g in [0, 1] if frame.loc[frame.gamma == g, 'mechanism_pass'].all()]
    dump(OUT/'decision.json', {'mechanism_pass_gamma':passed, 'defense_adopted':False,
         'status':'mechanism_pass_requires_further_review' if passed else 'rejected', 'confirmation_used':False,
         'candidate_replay_count':12, 'n_candidate_folds':36, 'incumbent_submission':942802,
         'strict_pass_count':int(legacy_frame.strict_pass.sum()), 'confirmation_entry_count':int(legacy_frame.confirmation_entry.sum()),
         'inputs_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in manifests+[Path(__file__)]},
         'label':'Exploratory proxy results, not official Anon; old rejected experiments unchanged'})
    print(frame.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['generate', 'evaluate', 'summary'], required=True)
    ap.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = ap.parse_args()
    if args.stage != 'summary' and args.seed is None:
        ap.error('--seed required')
    if args.stage == 'generate': generate(args.seed)
    elif args.stage == 'evaluate': evaluate(args.seed)
    else: summary()
