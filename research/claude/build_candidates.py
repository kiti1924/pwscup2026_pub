"""Build CL01 exploratory candidates S3/S4/S5 for the main attack phase.

MIA: 971407 confidences as E0, structural layer from structural_attack.py.
AIA: per-target best of the two measured submissions (965017, 971407) as base
answers, then abstention (time_hat=20, a guaranteed miss) on low-evidence rows.
"""
import json, shutil, sys, zipfile, hashlib, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import structural_attack as sa

OUT = Path(sys.argv[1])
SUB = Path(sys.argv[2])  # dir containing sub971407/ and sub965017/ extracted ZIPs
W = sa.load_world('main')
REPO = sa.MAIN
det71 = json.load(open(REPO / 'research/submissions/971407/parsed_details.json'))['aia_per_target']
det65 = json.load(open(REPO / 'research/submissions/965017/parsed_details.json'))['aia_per_target']
A71 = pd.read_csv(SUB / 'sub971407/F_aia.csv')
A65 = pd.read_csv(SUB / 'sub965017/F_aia.csv')
F71 = pd.read_csv(SUB / 'sub971407/F_mia.csv')

# per-target best measured base
base_src = {k: ('965017' if det65[str(k)] > det71[str(k)] else '971407') for k in W['targets']}
def base_answers(k):
    A = A65 if base_src[k] == '965017' else A71
    a = A[A.target_k == k].set_index('challenge_row_id').loc[W['Q'][k].challenge_row_id]
    return a.time_hat.to_numpy(float)

E0 = sa.load_fmia(SUB / 'sub971407/F_mia.csv', W)
E1 = sa.peer_evidence(W)

def mia_variant(name):
    if name == 'A':
        return sa.structural_mia(W, E0, E1, 0.0, 1.0, 0.087, 0.087)
    if name == 'B':  # record-wise softmax (tau=2) before the rare ordering
        return structural_mia_softmax(2.0)
    if name == 'C':  # E1 also blended into the all channel
        return sa.structural_mia(W, E0, E1, 0.5, 1.0, 0.087, 0.087)
    raise ValueError(name)

def structural_mia_softmax(tau):
    home, rare = W['home'], W['rare']
    base = sa._zc(sa._rlogit(E0))
    A = base + np.log(0.087) * home[:, None]
    Rr = base + sa._zc(E1)
    nu = sa.N_COHORTS - 1 - len(W['targets'])
    Z = Rr / tau; mx = np.maximum(Z.max(1, keepdims=True), 0)
    X = np.exp(Z - mx); Rr = np.log(X / (X.sum(1, keepdims=True) + nu * np.exp(-mx)) + 1e-300)
    Rr = Rr + np.log(0.087) * home[:, None]
    out = A.copy(); ri = np.where(rare)[0]
    for j in range(A.shape[1]):
        out[ri[np.argsort(-Rr[ri, j], kind='mergesort')], j] = np.sort(A[ri, j])[::-1]
    from scipy.stats import rankdata
    return np.column_stack([rankdata(out[:, j], method='ordinal') / len(out) for j in range(out.shape[1])])

EV = {k: sa.aia_evidence(W, k) for k in W['targets']}
PLAN = {'S3': ('A', 'raw', 0.7), 'S4': ('B', 'bg', 0.5), 'S5': ('C', 'peer', 0.7)}
manifest = {'base_src': base_src}
for sname, (mv, ev, f) in PLAN.items():
    d = OUT / sname; d.mkdir(parents=True, exist_ok=True)
    P = mia_variant(mv)
    fm = pd.DataFrame({'record_id': W['B'].record_id.to_numpy()})
    for i, k in enumerate(W['targets']):
        fm[str(k)] = P[:, i]
    fm = fm[['record_id'] + [c for c in F71.columns if c != 'record_id']]
    fm.to_csv(d / 'F_mia.csv', index=False)
    rows = []
    for k in W['targets']:
        th = sa.abstain(base_answers(k), EV[k][ev], f)
        rows.append(pd.DataFrame({'target_k': k, 'challenge_row_id': W['Q'][k].challenge_row_id.to_numpy(), 'time_hat': th}))
    fa = pd.concat(rows, ignore_index=True)
    fa.to_csv(d / 'F_aia.csv', index=False)
    shutil.copy(REPO / '_private_team14/token.txt', d / 'token.txt')
    z = d / 'candidate.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in ['F_mia.csv', 'F_aia.csv', 'token.txt']:
            zf.write(d / fn, fn)
    manifest[sname] = {'mia_variant': mv, 'aia_evidence': ev, 'aia_keep_frac': f,
                       'zip_sha256': hashlib.sha256(z.read_bytes()).hexdigest(),
                       'n_abstain': int((fa.time_hat == sa.WRONG).sum())}
json.dump(manifest, open(OUT / 'manifest.json', 'w'), indent=1)
print(json.dumps(manifest, indent=1))
