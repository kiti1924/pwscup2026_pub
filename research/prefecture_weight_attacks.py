"""FA10 lawful fixed prefecture-only penalties; no truth or file access."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from research.final_attack_api import COLUMNS, STATIC, NUM, QI, frame_check, checked_keys, rank_unit
from pwscup2026.attack.aia import aia_predict

VERSION = 'FA10-pref-only-v1'
WEIGHTS = [0., .25, 1.]
MIA = [f'static_pref{w:g}' for w in WEIGHTS]
AIA = ['raw_hard_pref'] + [f'raw_soft_pref{w:g}' for w in WEIGHTS]


def mia(c, query, reference):
    frame_check(c, COLUMNS); frame_check(reference, COLUMNS)
    frame_check(query, ['record_id']+STATIC); checked_keys(query, 'record_id')
    scale = np.maximum(reference[NUM].std(ddof=0).to_numpy(), 1e-12)
    d = cdist(query[NUM].to_numpy()/scale, c[NUM].to_numpy()/scale)**2
    for col in ['sex', 'smoking']:
        d += 2*(query[col].to_numpy()[:, None] != c[col].to_numpy()[None, :])
    mismatch = query.prefecture.to_numpy()[:, None] != c.prefecture.to_numpy()[None, :]
    scores = pd.DataFrame({'record_id': query.record_id.to_numpy()})
    for w, name in zip(WEIGHTS, MIA):
        scores[name] = rank_unit(-np.sqrt(d+2*w*mismatch).min(axis=1))
    return scores


def aia(c, query, reference, cfg):
    frame_check(c, COLUMNS); frame_check(reference, COLUMNS)
    frame_check(query, ['challenge_row_id']+QI); checked_keys(query, 'challenge_row_id')
    scale = reference[QI[:5]].std(ddof=1).to_numpy()
    scale = np.where(scale > 0., scale, 1.)
    d = cdist(query[QI[:5]].to_numpy()/scale, c[QI[:5]].to_numpy()/scale, 'sqeuclidean')
    mismatch = query.prefecture.to_numpy()[:, None] != c.prefecture.to_numpy()[None, :]
    result = pd.DataFrame({'challenge_row_id': query.challenge_row_id.to_numpy(),
                          'raw_hard_pref': aia_predict(c.reset_index(drop=True), query[QI], reference, cfg)[1]})
    for w, name in zip(WEIGHTS, AIA[1:]):
        pred = []
        for i, row in enumerate(query[QI].itertuples(index=False)):
            mask = c.sex.to_numpy() == row.sex
            if not mask.any():
                mask = np.ones(len(c), bool)
            values = (d[i]+2*w*mismatch[i])[mask]
            # Preserve reference nearest-tie averaging instead of changing the tie policy.
            ts = c.time.to_numpy()[mask][values <= values.min()]
            pred.append(float(np.sort(ts).mean()))
        result[name] = pred
    return result
