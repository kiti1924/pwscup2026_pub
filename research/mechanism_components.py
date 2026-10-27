"""Fixed lawful FA12 algorithms and observable fingerprints; no file/truth access."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from research.final_attack_api import (COLUMNS, STATIC, NUM, CAT, QI, frame_check,
    checked_keys, rank_unit, predict_aia, log_transform, block, detected)
from research.final_attack_extensions import static_keys, conditional_distance, predict_local_aia
from research.prefecture_weight_attacks import aia as prefecture_aia

VERSION = 'FA12-components-v1'
PEER = ['peer_remove0_compare0', 'peer_remove1_compare0',
        'peer_remove0_compare1', 'peer_remove1_compare1']
NN = ['continuous_nn', 'full_static_nn', 'relative_static_nn', 'relative_static_k5']


def peer_factorial(cs, query, reference, own_team):
    frame_check(query, ['record_id']+STATIC); checked_keys(query, 'record_id')
    frame_check(reference, COLUMNS)
    scale = np.maximum(reference[NUM].std(ddof=0).to_numpy(), 1e-12)
    bgkeys = static_keys(reference); distances = {}; counts = []
    for k, c in sorted(cs.items()):
        frame_check(c, COLUMNS)
        if k == own_team: continue
        clean = c.loc[~np.isin(static_keys(c), bgkeys), STATIC]
        distances[k] = [conditional_distance(query, c, scale), conditional_distance(query, clean, scale)]
        counts.append(dict(target=k, clean_rows=len(clean), removed_rows=len(c)-len(clean)))
    fallback = conditional_distance(query, reference, scale); predictions = {}; traces = {}
    for k, values in distances.items():
        trace = pd.DataFrame({'record_id': query.record_id.to_numpy()}); m = trace.copy()
        for remove in [0, 1]:
            peers = [d[remove] for j, d in distances.items() if j != k and np.isfinite(d[remove]).any()]
            dp = np.median(np.column_stack(peers), axis=1) if peers else fallback
            dc = values[remove]; empty = not np.isfinite(dc).any()
            m[f'peer_remove{remove}_compare0'] = rank_unit(np.zeros(len(query)) if empty else -dc)
            m[f'peer_remove{remove}_compare1'] = rank_unit(np.zeros(len(query)) if empty else np.log(dp+1e-9)-np.log(dc+1e-9))
            trace[f'distance_remove{remove}'] = dc
            trace[f'peer_distance_remove{remove}'] = dp
            trace[f'peer_count_remove{remove}'] = len(peers)
            trace[f'self_in_peer_remove{remove}'] = False
        predictions[k] = m[['record_id']+PEER]; traces[k] = trace
    return predictions, pd.DataFrame(counts), traces


def nn_mia(c, q, bg):
    frame_check(c, COLUMNS); frame_check(bg, COLUMNS)
    frame_check(q, ['record_id']+STATIC); checked_keys(q, 'record_id')
    scale = np.maximum(bg[NUM].std(ddof=0).to_numpy(), 1e-12)
    mins, fifth, continuous = [], [], []
    for d in [c, bg]:
        distances = cdist(q[NUM].to_numpy()/scale, d[NUM].to_numpy()/scale)
        continuous.append(distances.min(axis=1))
        distances = distances**2
        for col in CAT: distances += 2*(q[col].to_numpy()[:, None] != d[col].to_numpy()[None, :])
        mins.append(np.sqrt(distances.min(axis=1)))
        fifth.append(np.sqrt(np.partition(distances, min(5, len(d))-1, axis=1)[:, min(5, len(d))-1]))
    return pd.DataFrame({'record_id': q.record_id.to_numpy(),
        'continuous_nn': rank_unit(-continuous[0]), 'full_static_nn': rank_unit(-mins[0]),
        'relative_static_nn': rank_unit(np.log(mins[1]+1e-9)-np.log(mins[0]+1e-9)),
        'relative_static_k5': rank_unit(np.log(fifth[1]+1e-9)-np.log(fifth[0]+1e-9))})


def aia_nine(c, q, bg, cfg):
    a = predict_aia(c, q, bg, cfg)
    extra = prefecture_aia(c, q, bg, cfg)
    for method in ['raw_soft_pref0', 'raw_soft_pref1']: a[method] = extra[method]
    local = predict_local_aia(c, q, bg)
    for method in ['weighted_window5', 'contrast_window5']: a[method] = local[method]
    return a


def features(c, aq, bg, detector):
    frame_check(c, COLUMNS); frame_check(bg, COLUMNS)
    frame_check(aq, ['challenge_row_id']+QI); checked_keys(aq, 'challenge_row_id')
    scale = np.maximum(bg[NUM].std(ddof=0).to_numpy(), 1e-12)
    ds = cdist(c[NUM].to_numpy()/scale, bg[NUM].to_numpy()/scale)
    first = ds.min(axis=1)
    qs = np.maximum(log_transform(bg).std(axis=0, ddof=1), 1e-12)
    qd = cdist(log_transform(aq)/qs, log_transform(c)/qs, 'sqeuclidean')
    nearest = []; gap = []; local_sd = []; empty = []
    for i, row in enumerate(aq[QI].itertuples(index=False)):
        strict = (c.sex.to_numpy() == row.sex) & (c.prefecture.to_numpy() == row.prefecture)
        empty.append(not strict.any())
        ix = block(row, c); d = np.sort(qd[i, ix]); nearest.append(np.sqrt(d[0]))
        gap.append(np.sqrt(d[min(1, len(d)-1)])-np.sqrt(d[0]))
        boundary = np.partition(qd[i, ix], min(20, len(ix))-1)[min(20, len(ix))-1]
        ts = c.time.to_numpy()[ix][qd[i, ix] <= boundary]
        local_sd.append(ts.std(ddof=0))
    times = c.time.to_numpy(); atom = float(c.time.mode().min()); nonatom = times[times != atom]
    hist = np.histogram(times, bins=np.arange(0., max(11., times.max()+1), .5))[0]
    modes = int(((hist[1:-1] > hist[:-2]) & (hist[1:-1] >= hist[2:])).sum())
    association = []; steps = []
    for j, col in enumerate(NUM):
        groups = c.groupby('prefecture')[col].agg(['mean', 'count'])
        association.append(float(((groups['mean']-c[col].mean())**2*groups['count']/len(c)).sum()/scale[j]**2))
        diff = np.diff(np.unique(c[col].to_numpy())); steps.append(np.median(diff)/scale[j] if len(diff) else 0.)
    marker = c[['BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']].to_numpy()
    return dict(bg_static_match=float(np.isin(static_keys(c), static_keys(bg)).mean()),
        static_duplicate=float(c[STATIC].duplicated().mean()), numeric_grid01=float(np.isclose(marker*10, np.round(marker*10), atol=1e-7, rtol=0).mean()),
        numeric_step=float(np.median(steps)), numeric_sd_ratio=float(np.mean(c[NUM].std(ddof=0).to_numpy()/scale)),
        detected_tail=float(detected(c, detector).mean()), prefecture_numeric_association=float(np.mean(association)),
        prefecture_empty_block=float(np.mean(empty)), reference_nn_p50=float(np.median(first)), reference_nn_p90=float(np.quantile(first, .9)),
        qi_nn_p10=float(np.quantile(nearest, .1)), qi_nn_p50=float(np.median(nearest)), qi_nn_p90=float(np.quantile(nearest, .9)),
        qi_nn_gap=float(np.median(gap)), local_time_sd=float(np.median(local_sd)), time_atom=float((times == atom).mean()),
        time_atom_value=atom, nonatom_time_sd=float(nonatom.std()) if len(nonatom) else 0., time_hist_modes=modes,
        time_grid01=float(np.isclose(times*10, np.round(times*10), atol=1e-7, rtol=0).mean()))
