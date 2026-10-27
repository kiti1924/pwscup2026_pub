"""G17c v2 attacks with disjoint generation, reference and control rows."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.background_partition import OUT, SeparatedFold, PARTITION_VERSION, partition_frame
from research.background_transport import BASE, DIST, dump, digest, write_c
from research.adaptive_privacy import matrix
from research.repair_sex_correlation import risk_gates
import numpy as np
import pandas as pd


def maxima(result):
    return {label: max(v[group][metric] for v in result[channel].values())
            for channel, group, metric, label in [
                ('mia', 'all', 'tpr_at_fpr_0.01', 'mia_all'),
                ('mia', 'true_member_rare_detected_control', 'tpr_at_fpr_0.01', 'mia_rare'),
                ('aia', 'all', 'r_raw', 'aia_all'),
                ('aia', 'stratified_all', 'r_raw', 'aia_stratified'),
                ('aia', 'rare_proxy', 'r_se_adjusted', 'aia_rare_se')]}


def anchors():
    negative = OUT/'C_negative_generation_copy.csv'
    bg = pd.read_csv(DIST/'A_bg.csv'); split = partition_frame(bg, 2701)
    g = split.loc[split.partition == 'generation', 'background_row'].to_numpy()
    indices = np.random.default_rng(19000).choice(g, 1049, replace=False)
    if not negative.exists():
        write_c(bg.iloc[indices].reset_index(drop=True), negative)
        pd.DataFrame({'background_row': indices, 'record_id': bg.iloc[indices].record_id.to_numpy()}).to_csv(
            OUT/'negative_generation_indices.csv', index=False, lineterminator='\n')
        dump(OUT/'negative_generation_manifest.json', {'seed': 19000,
            'background_sha256': digest(DIST/'A_bg.csv'), 'candidate_sha256': digest(negative),
            'note': 'Negative control only. Public background non-IID; not a defense candidate or utility claim.'})
    else:
        saved = pd.read_csv(OUT/'negative_generation_indices.csv')
        np.testing.assert_array_equal(indices, saved.background_row)
        assert digest(negative) == json.loads((OUT/'negative_generation_manifest.json').read_text())['candidate_sha256']
    return {'incumbent': BASE,
        'high_parent': ROOT/'research/runs/20260924-G05b-001/seed0/C_bounded_s0_b_group_i2.csv',
        'repelled_parent': ROOT/'research/runs/20260924-G17b-001/seed0/C_repel_s0_shift4_i2.csv',
        'positive_raw': ROOT/'research/runs/20260924-G03b-001/C_positive_raw.csv',
        'positive_halfcopy': ROOT/'research/runs/20260924-G03b-001/C_positive_halfcopy.csv',
        'negative_generation_copy': negative}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['anchors', 'seed'], required=True)
    ap.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = ap.parse_args()
    output = OUT/'evaluation'; output.mkdir(exist_ok=True)
    anchor_paths = anchors()
    if args.stage == 'seed':
        if args.seed is None: ap.error('--seed required')
        records = json.loads((OUT/f'seed{args.seed}/utility_summary.json').read_text(encoding='utf-8'))['candidates']
        paths = {r['key']: ROOT/r['candidate'] for r in records if not r['pruned']}
    else:
        records = []; paths = anchor_paths
    results, comparisons = [], []
    for seed in [2701, 2702, 2703]:
        fold = SeparatedFold(seed)
        if args.stage == 'seed':
            assert (OUT/f'anchors_fold{seed}_checkpoint.json').exists(), 'Evaluate controls first'
            base = fold.evaluate(anchor_paths['incumbent'], output)
            parent = fold.evaluate(anchor_paths['high_parent'], output)
        current = {}
        for key, path in paths.items():
            result = fold.evaluate(path, output)
            current[key] = result; results.append(result)
            if args.stage == 'seed':
                delta = {k: v-maxima(parent)[k] for k, v in maxima(result).items()}
                comparisons.append({'key': key, 'fold': seed, 'maxima': maxima(result),
                    'parent_max_deltas': delta, 'baseline_paired_deltas': risk_gates(result, base),
                    'parent_paired_deltas': risk_gates(result, parent),
                    'max_nonworsening': all(delta[k] <= .01 for k in ['mia_all', 'mia_rare', 'aia_stratified']),
                    'rare_improves': delta['mia_rare'] <= -1/24+1e-12})
        if args.stage == 'anchors':
            gates = []
            for name, mt, at in [('positive_raw', .9, .8), ('positive_halfcopy', .2, .1)]:
                m = maxima(current[name])
                gates.append({'candidate': name, 'mia': m['mia_all'], 'aia': m['aia_all'],
                              'pass': m['mia_all'] >= mt and m['aia_all'] >= at})
            dump(OUT/f'anchors_fold{seed}_checkpoint.json', {'fold': seed,
                'partition_version': PARTITION_VERSION, 'positive_gates': gates,
                'candidate_hashes': {k: digest(p) for k, p in paths.items()},
                'maxima': {k: maxima(v) for k, v in current.items()}})
            assert all(g['pass'] for g in gates), gates
    label = 'anchors' if args.stage == 'anchors' else f'seed{args.seed}'
    frame = matrix(results)
    frame['evaluation_partition'] = PARTITION_VERSION
    frame.to_csv(OUT/f'{label}_attack_matrix.csv', index=False, lineterminator='\n')
    dump(OUT/f'{label}_risk_summary.json', {'stage': label, 'n_evaluations': len(results),
        'partition_version': PARTITION_VERSION, 'comparisons': comparisons,
        'candidate_sha256': {k: digest(p) for k, p in paths.items()},
        'pruned_unmeasured': [r['key'] for r in records if r['pruned']],
        'note': 'Separate partition matrix; no direct metric deltas against old G03b folds; not official Anon.'})


if __name__ == '__main__':
    main()
