"""CL01 structural attack layer (Claude, 2026-10-09).

Design facts used (rulebook §3.1 + verified on released preliminary truth):
  * P_i (313 non-shared rows) is drawn from the team's home prefecture; every AIA
    challenge row of a target is in that target's home prefecture.  Preliminary
    truth: 315/319 home-prefecture rows of B10 are non-shared.  Main B14 has 343
    Hyogo rows for 313 P rows, so P(shared | home) ~= 30/343 = 0.087.
  * Every non-home row is shared with exactly one other cohort.
  * MIA power per target = max(TPR@1%FPR over all candidates, same over the 24
    rare candidates).  The rare channel is decided by the top of the 24-row
    rare ordering, which can be set independently of the all-channel slots.

MIA: keep an existing per-pair confidence matrix E0 for the all channel (with a
home-prefecture log prior), and reorder only the rare candidates inside their
existing all-channel slots by E0 + w * E1, where E1 is the record-wise
peer-normalised log 1-NN distance on the seven continuous columns.

Inputs are lawful only: own B (incl. our own is_rare), A_bg, released C_k and
challenge QIs.  Truth files are read only by the scoring helpers for the
preliminary world.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / 'pwscup2026-kit/starter/src'))
from pwscup2026.scoring import attack_score  # noqa: E402

CFG = {'scoring': {'low_fpr': 0.01, 'aia_rare_se_z': 1.64}, 'aia': {'tau_time': 0.5}}
CONT = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
LOGC = ['TG', 'ALT']
STATIC = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG']
N_COHORTS = 30


def load_world(world: str) -> dict:
    if world == 'prelim':
        D = MAIN / 'attack_phase/data/public/prelim_attack_20260825'
        T = MAIN / 'analysis/prelim_review_20260924/sources/truth'
        B = pd.read_csv(MAIN / '_private_team10/B_self.csv')
        bg = pd.read_csv(MAIN / '_private_team10/A_bg.csv')
        me = 10
        targets = sorted(int(re.findall(r'C_(\d+)', f)[0]) for f in glob.glob(str(D / 'C_*.csv')))
        targets = [t for t in targets if t != me]
    else:
        D = MAIN / '_private_team14/main_attack_20260928/files'
        T = None
        B = pd.read_csv(MAIN / '_private_team14/B_self.csv')
        bg = pd.read_csv(MAIN / '_private_team14/A_bg.csv')
        me = 14
        targets = [int(c) for c in json.load(open(D / 'mia_columns.json'))['target_columns'] if int(c) != me]
    W = dict(world=world, D=D, B=B, bg=bg, me=me, targets=targets,
             C={k: pd.read_csv(D / f'C_{k}.csv') for k in targets},
             Q={k: pd.read_csv(D / f'aia_challenge_{k}.csv') for k in targets})
    W['home_pref'] = B.prefecture.value_counts().index[0]
    W['home'] = B.prefecture.to_numpy(dtype=object) == W['home_pref']
    W['rare'] = B.is_rare.to_numpy() == 1
    if T is not None:
        P = {int(re.findall(r'_(\d+)\.csv', f)[0]): pd.read_csv(f) for f in glob.glob(str(T / 'pool_ids_*.csv'))}
        mine = P[me].set_index('record_id').loc[B.record_id]
        W['mem'] = {k: mine.pool_id.isin(set(P[k].pool_id)).to_numpy().astype(int) for k in targets}
        W['aia_truth'] = {k: pd.read_csv(T / f'aia_truth_{k}.csv') for k in targets}
    return W


# ---------------------------------------------------------------- MIA evidence
def _num(df, cols, bg):
    X = df[cols].astype(float).copy()
    Rf = bg[cols].astype(float).copy()
    for c in LOGC:
        if c in X:
            X[c] = np.log1p(X[c])
            Rf[c] = np.log1p(Rf[c])
    return ((X - Rf.mean()) / Rf.std()).to_numpy()


def _nnd(Xq, Xr):
    return NearestNeighbors(n_neighbors=1).fit(Xr).kneighbors(Xq)[0][:, 0]


def _bg_keys(bg):
    k = bg[STATIC].assign(**{c: bg[c].round(1) for c in CONT}).astype(str).to_numpy()
    return set(map(tuple, k))


def peer_evidence(W) -> np.ndarray:
    """E1[r, j] = median_m log d_m(r) - log d_j(r); C rows equal to a rounded A_bg row are dropped."""
    B, bg = W['B'], W['bg']
    keys = _bg_keys(bg)
    Xb = _num(B, CONT, bg)
    cols = []
    for k in W['targets']:
        C = W['C'][k]
        ck = C[STATIC].assign(**{c: C[c].round(1) for c in CONT}).astype(str).to_numpy()
        keep = np.array([tuple(r) not in keys for r in ck])
        if keep.sum() <= 50:
            keep[:] = True
        cols.append(_nnd(Xb, _num(C[keep], CONT, bg)))
    lD = np.log(np.column_stack(cols) + 1e-3)
    return np.median(lD, 1, keepdims=True) - lD


def _rlogit(S):
    U = np.column_stack([rankdata(S[:, j]) / (len(S) + 1) for j in range(S.shape[1])])
    return np.log(U) - np.log(1 - U)


def _zc(X):
    s = X.std(0)
    s[s == 0] = 1
    return (X - X.mean(0)) / s


def structural_mia(W, E0, E1, w_all=0.0, w_rare=1.0, p_home_all=0.087, p_home_rare=0.087):
    """Return confidences in (0,1]; column order = W['targets']."""
    home, rare = W['home'], W['rare']
    base = _zc(_rlogit(E0))
    A = base + w_all * _zc(E1) + np.log(p_home_all) * home[:, None]
    Rr = base + w_rare * _zc(E1) + np.log(p_home_rare) * home[:, None]
    out = A.copy()
    ri = np.where(rare)[0]
    for j in range(A.shape[1]):
        slots = np.sort(A[ri, j])[::-1]
        out[ri[np.argsort(-Rr[ri, j], kind='mergesort')], j] = slots
    return np.column_stack([rankdata(out[:, j], method='ordinal') / len(out) for j in range(out.shape[1])])


def score_mia(W, P):
    rows = {}
    for i, k in enumerate(W['targets']):
        res = attack_score.score_mia(P[:, i], W['mem'][k], CFG, rare_mask=W['rare'])
        rows[k] = (res.score, res.breakdown['tpr_at_low_fpr'], res.breakdown['tpr_rare'])
    return pd.DataFrame(rows, index=['score', 'all', 'rare']).T


def load_fmia(path, W) -> np.ndarray:
    F = pd.read_csv(path).set_index('record_id').loc[W['B'].record_id]
    return np.column_stack([F[str(k)].to_numpy(dtype=float) for k in W['targets']])


def write_fmia(path, W, P):
    df = pd.DataFrame({'record_id': W['B'].record_id.to_numpy()})
    for i, k in enumerate(W['targets']):
        df[str(k)] = P[:, i]
    df.to_csv(path, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', default='main')
    ap.add_argument('--base-fmia', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--w-all', type=float, default=0.0)
    ap.add_argument('--w-rare', type=float, default=1.0)
    ap.add_argument('--p-home-all', type=float, default=0.087)
    ap.add_argument('--p-home-rare', type=float, default=0.087)
    a = ap.parse_args()
    W = load_world(a.world)
    E0 = load_fmia(a.base_fmia, W)
    E1 = peer_evidence(W)
    P = structural_mia(W, E0, E1, a.w_all, a.w_rare, a.p_home_all, a.p_home_rare)
    write_fmia(a.out, W, P)
    print('wrote', a.out, P.shape)


# ---------------------------------------------------------------- AIA abstention
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT']
WRONG = 20.0  # time_hat >= 0 is the only validator bound; truth time <= 10, so 20 is a guaranteed miss


def _qnum(df, bg):
    X = df[QI].astype(float).copy()
    Rf = bg[QI].astype(float).copy()
    for c in LOGC:
        X[c] = np.log1p(X[c])
        Rf[c] = np.log1p(Rf[c])
    return ((X - Rf.mean()) / Rf.std()).to_numpy()


def _sex_nnd(Q, C, bg):
    Xq, Xc = _qnum(Q, bg), _qnum(C, bg)
    d = np.full(len(Q), np.inf)
    for s in (0, 1):
        qi = np.where(Q.sex.to_numpy() == s)[0]
        ci = np.where(C.sex.to_numpy() == s)[0]
        if len(qi) and len(ci):
            d[qi] = _nnd(Xq[qi], Xc[ci])
    return d


def aia_evidence(W, k, others_cache=None):
    """Membership evidence for challenge rows of target k (higher = more likely a target)."""
    Q, bg = W['Q'][k], W['bg']
    dk = _sex_nnd(Q, W['C'][k], bg)
    dbg = _sex_nnd(Q, bg, bg)
    dO = np.column_stack([_sex_nnd(Q, W['C'][m], bg) for m in W['targets'] if m != k])
    l = np.log(dk + 1e-3)
    return {'raw': -dk,
            'bg': np.log(dbg + 1e-3) - l,
            'peer': np.median(np.log(dO + 1e-3), 1) - l}


def abstain(base_time, e, frac_keep):
    thr = np.quantile(e, 1 - frac_keep)
    return np.where(e >= thr, base_time, WRONG)


def score_aia_target(W, k, time_hat):
    t = W['aia_truth'][k].set_index('challenge_row_id').loc[W['Q'][k].challenge_row_id]
    m = (t.label == 'member').to_numpy()
    c = (t.label == 'control').to_numpy()
    rr = t.is_rare.to_numpy().astype(bool)
    res = attack_score.score_aia(None, time_hat[m], t.onset.to_numpy()[m], t.time.to_numpy()[m],
                                 None, time_hat[c], t.onset.to_numpy()[c], t.time.to_numpy()[c], CFG,
                                 member_rare_mask=rr[m], control_rare_mask=rr[c])
    return res.score, res.breakdown
