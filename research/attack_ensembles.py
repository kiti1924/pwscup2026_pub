"""Fixed, truth-free FA09 prediction combinations (no fitted weights or router)."""
import numpy as np
import pandas as pd
from research.final_attack_api import rank_unit, keyed_values

VERSION = 'FA09-fixed-mix-v1'
MIA_MIX = ['rank_mean_full_peer', 'rank_max_full_peer']
AIA_MIX = ['mean_three', 'window_consensus_three']
AIA_SOURCES = ['official_1nn', 'log_qi_1nn', 'no_geo_sex_1nn']


def combine(mia, aia):
    """Inputs are keyed lawful point predictions, with no truth columns."""
    forbidden = {'member', 'label', 'is_rare', 'onset', 'time', 'pool_id'}
    if forbidden.intersection(mia.columns) or forbidden.intersection(aia.columns):
        raise ValueError('Truth is not a combination input')
    mv = np.column_stack([keyed_values(mia, 'record_id', mia.record_id, col)
                          for col in ['full_static_nn', 'peer_bg_excluded']])
    av = np.column_stack([keyed_values(aia, 'challenge_row_id', aia.challenge_row_id, col)
                          for col in AIA_SOURCES])
    if not np.isfinite(mv).all() or not np.isfinite(av).all():
        raise ValueError('Nonfinite predictions')
    ranks = np.column_stack([rank_unit(mv[:, i]) for i in range(2)])
    # Centred half-year support, not any use of actual outcome or membership.
    support = (np.abs(av[:, :, None] - av[:, None, :]) <= .5).sum(axis=2)
    chosen = support.argmax(axis=1)  # fixed source order breaks ties
    m = pd.DataFrame({'record_id': mia.record_id,
                      MIA_MIX[0]: ranks.mean(axis=1), MIA_MIX[1]: ranks.max(axis=1)})
    a = pd.DataFrame({'challenge_row_id': aia.challenge_row_id,
                      AIA_MIX[0]: av.mean(axis=1),
                      AIA_MIX[1]: av[np.arange(len(av)), chosen]})
    return m, a
