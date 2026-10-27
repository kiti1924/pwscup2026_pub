"""FA15 deterministic dual-channel comparison asset; never uploads submissions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'pwscup2026-kit/starter/src')]
import numpy as np
import pandas as pd
from research.mechanism_study_io import plan_inputs, commit_engine, dump, read
from research.build_final_attack_baseline import sha
from research.package_saved_final_attack import package


def deltas(out, new_m, new_a, before_folder, label):
    before_m, before_a = read(before_folder/'F_mia.csv'), read(before_folder/'F_aia.csv')
    assert set(before_m.record_id) == set(new_m.record_id)
    before_m = before_m.set_index('record_id').loc[new_m.record_id]
    assert set(before_m.columns) == set(new_m.columns.drop('record_id'))
    rows = []; mcells = 0
    for col in new_m.columns.drop('record_id'):
        delta = new_m[col].to_numpy()-before_m[col].to_numpy(); count = int((delta != 0).sum()); mcells += count
        rows.append(dict(reference=label, channel='mia', target=int(col), changed_rows=count, total_rows=len(delta), max_abs_delta=float(abs(delta).max())))
        if label == 'S1' and col == '9': assert np.array_equal(new_m[col], before_m[col])
    a = new_a.merge(before_a, on=['target_k', 'challenge_row_id'], suffixes=('_new', '_old'), validate='one_to_one')
    assert len(a) == len(new_a) == len(before_a) and not new_a.duplicated(['target_k', 'challenge_row_id']).any()
    a['delta'] = a.time_hat_new-a.time_hat_old; acells = int(a.delta.ne(0).sum())
    a.to_csv(out/(label+'_aia_deltas.csv'), index=False)
    for k, sub in a.groupby('target_k'):
        rows.append(dict(reference=label, channel='aia', target=int(k), changed_rows=int(sub.delta.ne(0).sum()),
            total_rows=len(sub), max_abs_delta=float(abs(sub.delta).max())))
    comparison = pd.DataFrame(rows); comparison.to_csv(out/(label+'_comparison.csv'), index=False)
    return dict(mia_changed_values=mcells, aia_changed_rows=acells,
        mia_changed_targets=int(((comparison.channel == 'mia') & comparison.changed_rows.gt(0)).sum()),
        aia_changed_targets=int(((comparison.channel == 'aia') & comparison.changed_rows.gt(0)).sum()))


def run(out):
    plan = plan_inputs(out)
    commit_engine(out, ['research/prepare_mechanism_comparison.py', 'research/package_saved_final_attack.py'])
    assert not subprocess.check_output(['git', '-C', str(ROOT/'pwscup2026-kit'), 'status', '--porcelain'], text=True).strip()
    # Values stay only in their intended file/archive and are never printed or hashed here.
    intended = (ROOT/'_private_team14/token.txt').read_bytes()
    assert intended == subprocess.check_output(['git', 'show', 'HEAD:_private_team14/token.txt'], cwd=ROOT)
    for name in ['candidate', 'regenerated']:
        package(out/name, plan['mia_method'], plan['aia_method'], ROOT/plan['predictions'], plan['mia_overrides'])
        p = out/name/'routing_manifest.json'; routes = json.loads(p.read_text())
        for route in routes:
            route.update(version='FA15-package-v1', source_merger_version='FA14-assets-v1',
                mia_version='G15b-lawful-v1' if route['mia_method'] == 'full_static_nn' else 'FA03-lawful-v1',
                aia_version='FA10-pref-only-v1',
                selection='Specialist official-comparison asset; final gain unknown. County robustness and feature-router hypotheses rejected; do not automatically replace S1.')
        dump(p, routes)
    for name in ['candidate.zip', 'F_mia.csv', 'F_aia.csv', 'routing_manifest.json', 'validation.json']:
        assert (out/'candidate'/name).read_bytes() == (out/'regenerated'/name).read_bytes(), name
    m, a = read(out/'candidate/F_mia.csv'), read(out/'candidate/F_aia.csv')
    s1 = deltas(out, m, a, ROOT/'research/runs/20261006-FA02-001/candidate_full_static', 'S1')
    older = deltas(out, m, a, ROOT/'research/runs/20261008-FA09-paired-001/candidate', 'unsubmitted_FA09')
    assert s1['mia_changed_values'] == plan['expected_mia_changed_values_vs_S1']
    assert s1['aia_changed_rows'] == plan['expected_aia_changed_rows_vs_S1']
    assert s1['mia_changed_targets'] == plan['expected_changed_targets']['mia']
    assert s1['aia_changed_targets'] == plan['expected_changed_targets']['aia']
    assert older['mia_changed_values'] == 0 and older['aia_changed_rows'] > 0
    result = dict(status='validated_comparison_asset_not_measured_final_gain',
        candidate=str((out/'candidate/candidate.zip').relative_to(ROOT)), sha256=sha(out/'candidate/candidate.zip'),
        source_hashes_checked=len(plan['input_hashes']), source_versions_correct=True, strict_keys_ok=True,
        official_validator_ok=True, deterministic_zip_csv_route_replay=True, token_tracked_bytes_ok=True,
        mia9_exact_S1=True, kit_clean=True, delta_vs_S1=s1, delta_vs_unsubmitted_FA09=older,
        submitted=False, submission_authorized=False, official_score=None, router_adopted=False,
        remaining_last_observed=7, remaining_live_verified=False, reserved_final_slots=2,
        next='Keep S1 listed. Additional explicit attack upload authorization then live phase/team/deadline/usage/duplicates check. One comparison upload, raw scores/details, exact floored AIA average and complete MIA comparison, integrate measured AIA target maxima before final validation/listing.')
    dump(out/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out.resolve())
