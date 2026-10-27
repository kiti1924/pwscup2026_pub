"""FA09: fixed mixes, scoring-only failure overlap, retrospective portfolio bounds."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from research import compare_final_attacks as base
from research.build_final_attack_baseline import read, sha
from research.final_attack_api import MIA_METHODS, AIA_METHODS, keyed_values, rank_unit
from research.attack_ensembles import combine, MIA_MIX, AIA_MIX, VERSION
from pwscup2026.scoring.attack_score import score_mia, score_aia, _tpr_at_fpr

OLD = ROOT / 'research/runs/20261006-FA02-001'
EXTRA = ROOT / 'research/runs/20261006-FA03-001'
MIA_SINGLE = MIA_METHODS + ['peer_bg_excluded']
AIA_SINGLE = AIA_METHODS + ['weighted_window5', 'contrast_window5']
BASELINES = {'mia': 'full_static_nn', 'aia': 'official_1nn'}


def join_predictions(first, second, key):
    result = first.copy()
    for method in second.columns.drop(key):
        if method in result:
            raise ValueError('Duplicate method')
        result[method] = keyed_values(second, key, result[key], method)
    return result


def roc_weights(confidence, truth, fpr=.01):
    """Diagnostic randomized operating-point weights reproduce official interpolation.

    Threshold is chosen with scoring truth; these are NOT inference predictions.
    Repeated-FPR vertices use the last vertex, matching numpy.interp.
    """
    y = np.asarray(truth, int)
    c = np.asarray(confidence, float)
    pos, neg = int(y.sum()), int((1-y).sum())
    if not pos or not neg:
        return np.zeros(len(y))
    order = np.argsort(-c, kind='mergesort')
    sc = c[order]
    ends = np.r_[np.flatnonzero(np.diff(sc) != 0), len(sc)-1]
    fp = np.r_[0., np.cumsum(1-y[order])[ends] / neg]
    counts = np.r_[0, ends+1]
    high = np.searchsorted(fp, fpr, side='right')
    weights = np.zeros(len(y))
    if high == len(fp):
        weights[:] = 1.
    else:
        low = high-1
        alpha = (fpr-fp[low]) / (fp[high]-fp[low])
        weights[order[:counts[low]]] = 1.
        weights[order[counts[low]:counts[high]]] = alpha
    expected, _ = _tpr_at_fpr(c, y, fpr)
    assert abs(weights[y == 1].mean()-expected) < 1e-12
    assert abs(weights[y == 0].mean()-fpr) < 1e-12
    return weights


def score(m, a, mt, at, target, group, seed=None, mechanism=None):
    cfg = base.config()
    assert mt.record_id.is_unique and set(mt.record_id) == set(m.record_id)
    assert at.challenge_row_id.is_unique and set(at.challenge_row_id) == set(a.challenge_row_id)
    mm = mt.set_index('record_id').loc[m.record_id]
    aa = at.set_index('challenge_row_id').loc[a.challenge_row_id]
    y = mm.member.to_numpy(int); rm = mm.is_rare.to_numpy(bool)
    am = aa.label.eq('member').to_numpy(); ar = aa.is_rare.to_numpy(bool)
    rows = []
    for method in m.columns.drop('record_id'):
        r = score_mia(m[method].to_numpy(), y, cfg, rare_mask=rm)
        rows.append(dict(channel='mia', method=method, score=r.score,
                         tpr_all=r.breakdown['tpr_at_low_fpr'], tpr_rare=r.breakdown['tpr_rare'],
                         n_member=int(y.sum()), n_control=int((1-y).sum()),
                         n_member_rare=int(y[rm].sum()), n_control_rare=int((1-y[rm]).sum())))
    for method in a.columns.drop('challenge_row_id'):
        p = a[method].to_numpy()
        r = score_aia(None, p[am], aa.onset.to_numpy()[am], aa.time.to_numpy()[am],
                      None, p[~am], aa.onset.to_numpy()[~am], aa.time.to_numpy()[~am], cfg,
                      member_rare_mask=ar[am], control_rare_mask=ar[~am])
        row = dict(channel='aia', method=method, score=r.score,
                   **{k: r.breakdown[k] for k in ['r_time_all', 'r_time_rare', 'r_time_rare_raw',
                       'succ_time_member', 'succ_time_control', 'succ_time_member_rare', 'succ_time_control_rare']},
                   n_member=int(am.sum()), n_control=int((~am).sum()),
                   n_member_rare=int((am & ar).sum()), n_control_rare=int((~am & ar).sum()))
        for phi in [0., .05, .1, .2]:
            row[f'floor_sensitivity_{phi:g}'] = max(0., r.score-phi)
        rows.append(row)
    for row in rows:
        row.update(target=target, group=group, seed=seed, mechanism=mechanism,
                   version=VERSION, official_final_score=False, aia_floor_known=False)
    return pd.DataFrame(rows)


def overlap(folder, m, a, mt, at, target):
    mm = mt.set_index('record_id').loc[m.record_id]
    aa = at.set_index('challenge_row_id').loc[a.challenge_row_id]
    y = mm.member.to_numpy(bool); rare = mm.is_rare.to_numpy(bool)
    mrows, pairs = [], []
    for stratum, mask in [('all', np.ones(len(y), bool)), ('rare', rare)]:
        truth = y[mask]
        weights = {method: roc_weights(m.loc[mask, method].to_numpy(), truth)
                   for method in m.columns.drop('record_id')}
        for method, w in weights.items():
            sub = pd.DataFrame({'record_id': m.loc[mask, 'record_id'], 'stratum': stratum,
                'method': method, 'member': truth, 'is_rare': rare[mask],
                'score': m.loc[mask, method], 'rank_unit': rank_unit(m.loc[mask, method].to_numpy()),
                'scoring_only_roc_weight': w})
            mrows.append(sub)
        for left, right in itertools.combinations(weights, 2):
            l, r = weights[left], weights[right]
            # min is shared randomized weight, not independence/causal probability.
            pairs.append(dict(target=target, stratum=stratum, left=left, right=right,
                n_member=int(truth.sum()), n_control=int((~truth).sum()),
                overlap_member_weight=float(np.minimum(l, r)[truth].sum()),
                overlap_falsepositive_weight=float(np.minimum(l, r)[~truth].sum()),
                right_new_member_weight=float(np.maximum(r-l, 0)[truth].sum()),
                right_lost_member_weight=float(np.maximum(l-r, 0)[truth].sum()),
                right_new_falsepositive_weight=float(np.maximum(r-l, 0)[~truth].sum())))
    pd.concat(mrows, ignore_index=True).to_csv(folder/'mia_operating_points.csv', index=False)
    rows = []
    am = aa.label.eq('member').to_numpy(); ar = aa.is_rare.to_numpy(bool)
    hits = {method: abs(a[method].to_numpy()-aa.time.to_numpy()) <= .5
            for method in a.columns.drop('challenge_row_id')}
    records = []
    for method, hit in hits.items():
        records.append(pd.DataFrame({'challenge_row_id': a.challenge_row_id, 'method': method,
            'member': am, 'is_rare': ar, 'time_hat': a[method], 'truth_time_scoring_only': aa.time.to_numpy(),
            'hit': hit, 'baseline_hit': hits[BASELINES['aia']]}))
    pd.concat(records, ignore_index=True).to_csv(folder/'aia_hits.csv', index=False)
    for left, right in itertools.combinations(hits, 2):
        for stratum, mask in [('all', np.ones(len(am), bool)), ('rare', ar)]:
            for label, sub in [('member', mask & am), ('control', mask & ~am)]:
                l, r = hits[left][sub], hits[right][sub]
                rows.append(dict(target=target, stratum=stratum, label=label, left=left, right=right,
                    n=int(sub.sum()), overlap_hits=int((l & r).sum()),
                    right_new_hits=int((~l & r).sum()), right_lost_hits=int((l & ~r).sum()),
                    both_miss=int((~l & ~r).sum())))
    # Rank correlations are independent of truth, complement gain is not.
    correlation = m.drop(columns='record_id').corr(method='spearman').rename_axis('method')
    correlation.to_csv(folder/'mia_rank_correlation.csv')
    return pd.DataFrame(pairs), pd.DataFrame(rows)


def analyze(metrics, out):
    summaries, selections, bounds, portfolios = [], [], [], []
    for channel, singles in [('mia', MIA_SINGLE), ('aia', AIA_SINGLE)]:
        v = metrics[metrics.channel.eq(channel)].pivot(index='target', columns='method', values='score')
        b = v[BASELINES[channel]]
        for method in v:
            d = v[method]-b; without_easy = d.drop(index=[1, 2])
            gain = np.maximum(d, 0)
            row = dict(channel=channel, method=method, mean_score=v[method].mean(), mean_delta=d.mean(),
                wins=int((d > 1e-12).sum()), losses=int((d < -1e-12).sum()),
                no_easy_mean_score=v[method].drop(index=[1, 2]).mean(), no_easy_mean_delta=without_easy.mean(),
                delta_without_best=d.drop(d.idxmax()).mean(),
                baseline_portfolio_marginal=gain.mean(), gains_gt005=int((d > .005).sum()),
                single=method in singles)
            if method in singles:
                other = v[[s for s in singles if s != method]].max(axis=1)
                row['unique_gain_against_other_singles'] = np.maximum(v[method]-other, 0).mean()
            row['specialist_gate'] = bool(row['single'] and row['gains_gt005'] >= 2 and gain.mean() > .005)
            row['whole_prelim_gate'] = bool(d.mean() > .005 and row['delta_without_best'] > 0 and without_easy.mean() > 0)
            summaries.append(row)
        for k in v.index:
            best_method = v.loc[k, singles].idxmax()
            bounds.append(dict(channel=channel, target=int(k), baseline_score=b.loc[k],
                best_single_method=best_method, best_single_score=v.loc[k, best_method],
                gain=v.loc[k, best_method]-b.loc[k], diagnostic_only=True))
        # Fixed size2/3 budget, baseline included; no inference router is fitted.
        others = [s for s in singles if s != BASELINES[channel]]
        for size in [2, 3]:
            for rest in itertools.combinations(others, size-1):
                methods = [BASELINES[channel], *rest]; best = v[methods].max(axis=1)
                portfolios.append(dict(channel=channel, size=size, methods='|'.join(methods),
                    mean_upper_bound=best.mean(), marginal= (best-b).mean(),
                    no_easy_upper_bound=best.drop(index=[1, 2]).mean(), diagnostic_only=True))
            p = max([r for r in portfolios if r['channel'] == channel and r['size'] == size],
                    key=lambda r: r['mean_upper_bound'])
            methods = p['methods'].split('|')
            for k in v.index:
                selections.append(dict(channel=channel, size=size, target=int(k),
                    selected_method=v.loc[k, methods].idxmax(), score=v.loc[k, methods].max(), diagnostic_only=True))
    pd.DataFrame(summaries).to_csv(out/'method_summary.csv', index=False)
    pd.DataFrame(bounds).to_csv(out/'per_target_upper_bound.csv', index=False)
    pd.DataFrame(portfolios).to_csv(out/'portfolio_upper_bounds.csv', index=False)
    pd.DataFrame(selections).to_csv(out/'retrospective_selection.csv', index=False)


def run(out):
    plan = json.loads((out/'plan.json').read_text())
    for path, digest in plan['input_hashes'].items():
        assert sha(ROOT/path) == digest, path
    all_prelim, all_shadow, mpairs, apairs, replay = [], [], [], [], []
    for group, seed, k, mechanism in (
        [('prelim', None, k, None) for k in plan['targets']] +
        [('shadow', s, k, mech) for s in plan['seeds'] for mech in base.MECHANISMS for k in range(3)]):
        sub = Path('prelim')/f'target{k}' if group == 'prelim' else Path('shadow')/f'seed{seed}'/f'{mechanism}_{k}'
        folder = out/sub
        old, extra = OLD/sub, EXTRA/sub
        base.load_checkpoint(old)
        m = join_predictions(read(old/'mia.csv'), read(extra/'mia.csv'), 'record_id')
        a = join_predictions(read(old/'aia.csv'), read(extra/'aia.csv'), 'challenge_row_id')
        em, ea = combine(m, a)
        m = join_predictions(m, em, 'record_id'); a = join_predictions(a, ea, 'challenge_row_id')
        if group == 'prelim':
            pool = read(base.TRUTH/'pool_ids_10.csv').set_index('record_id').loc[m.record_id]
            mt = pd.DataFrame({'record_id': m.record_id, 'member': pool.pool_id.isin(
                read(base.TRUTH/f'pool_ids_{k}.csv').pool_id).to_numpy(int), 'is_rare': pool.is_rare.to_numpy()})
            at = read(base.TRUTH/f'aia_truth_{k}.csv')
        else:
            mt, at = read(old/'mia_truth.csv'), read(old/'aia_truth.csv')
        result = score(m, a, mt, at, k, group, seed, mechanism)
        previous = pd.concat([read(old/'metrics.csv'), read(extra/'metrics.csv')], ignore_index=True)
        joined = result.merge(previous[['channel', 'method', 'score']], on=['channel', 'method'],
                              suffixes=('_new', '_old'), validate='one_to_one')
        error = abs(joined.score_new-joined.score_old).max(); assert error < 1e-12
        replay.append(dict(group=group, seed=seed, target=k, mechanism=mechanism,
                           pairs=len(joined), max_score_error=error))
        if folder.exists():
            # Completed boundaries must replay exactly; no historical overwrite.
            marker = json.loads((folder/'checkpoint.json').read_text())
            for name, digest in marker['files'].items():
                assert sha(folder/name) == digest
            saved = read(folder/'metrics.csv')
            assert len(saved) == len(result) and np.allclose(saved.score, result.score, atol=1e-12, rtol=0)
        else:
            folder.mkdir(parents=True)
            m.to_csv(folder/'mia.csv', index=False); a.to_csv(folder/'aia.csv', index=False)
            mt.to_csv(folder/'mia_truth.csv', index=False); at.to_csv(folder/'aia_truth.csv', index=False)
            result.to_csv(folder/'metrics.csv', index=False)
        if group == 'prelim':
            if (folder/'mia_pair_overlap.csv').exists():
                pm, pa = read(folder/'mia_pair_overlap.csv'), read(folder/'aia_pair_overlap.csv')
            else:
                pm, pa = overlap(folder, m, a, mt, at, k)
                pm.to_csv(folder/'mia_pair_overlap.csv', index=False)
                pa.to_csv(folder/'aia_pair_overlap.csv', index=False)
            mpairs.append(pm); apairs.append(pa); all_prelim.append(result)
        else:
            all_shadow.append(result)
        base.dump(folder/'checkpoint.json', {'files': {p.name: sha(p) for p in sorted(folder.glob('*.csv'))},
                                            'replayed_scores': len(joined), 'max_score_error': error})
        print(f'{group}/{seed}/{mechanism}/target{k}: 20 scores, replay {error:g}', flush=True)
    prelim = pd.concat(all_prelim, ignore_index=True); shadow = pd.concat(all_shadow, ignore_index=True)
    prelim.to_csv(out/'prelim_metrics.csv', index=False); shadow.to_csv(out/'shadow_metrics.csv', index=False)
    pd.concat(mpairs).to_csv(out/'mia_pair_overlap.csv', index=False)
    pd.concat(apairs).to_csv(out/'aia_pair_overlap.csv', index=False)
    pd.DataFrame(replay).to_csv(out/'single_replay.csv', index=False)
    analyze(prelim, out)
    rows = []
    for (channel, method), values in shadow.groupby(['channel', 'method']):
        b = shadow[(shadow.channel == channel) & (shadow.method == BASELINES[channel])]
        j = values.merge(b[['seed', 'target', 'mechanism', 'score']], on=['seed', 'target', 'mechanism'], suffixes=('', '_base'))
        for mechanism, sub in j.groupby('mechanism'):
            row = dict(channel=channel, method=method, mechanism=mechanism, mean_score=sub.score.mean(),
                       mean_delta=(sub.score-sub.score_base).mean(), n=len(sub))
            for col in ['tpr_all', 'tpr_rare', 'r_time_all', 'r_time_rare', 'succ_time_member', 'succ_time_control']:
                row[col] = sub[col].mean()
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'shadow_comparison.csv', index=False)
    base.dump(out/'verification.json', {'prelim_scores': len(prelim), 'shadow_scores': len(shadow),
        'existing_single_replay_pairs': sum(x['pairs'] for x in replay),
        'max_single_replay_error': max(x['max_score_error'] for x in replay),
        'new_ensemble_scores': 4*(len(plan['targets'])+36), 'truth_in_inference': False,
        'all_conditions_exploratory': True, 'official_final_score': None,
        'input_hashes_checked': len(plan['input_hashes']),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); run(args.out.resolve())
