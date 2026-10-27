"""FA106: one faithful CL06 main transfer; inference only, no scorer or truth."""
from pathlib import Path
import argparse, hashlib, io, json, sys, time, traceback, zipfile
START = time.monotonic()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(repo, out, plan_dir):
    assert repo.is_dir() and not out.exists(), "Require valid repo and fresh output directory"
    assert not out.is_relative_to(repo), "Output must be scratch, outside repository"
    out.mkdir(parents=True)
    accessed = set()
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        text = str(p)
        if any(x in text for x in ["/sources/truth", "/sources/attacks", "scoring_truth", "parsed_details", "token.txt"]):
            raise RuntimeError("Forbidden inference input")
        if p.is_relative_to(repo):
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            import os
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                raise RuntimeError("Repository write forbidden")
            accessed.add(p.relative_to(repo).as_posix())
    sys.addaudithook(guard)
    sys.path[:0] = [str(repo), str(repo/'research'), str(repo/'pwscup2026-kit/starter/src')]
    import numpy as np
    import pandas as pd
    import yaml
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata
    from research import conditional_null_mia as cn
    from research.final_attack_api import STATIC, NUM, CAT, rank_unit
    from research.mia_rare_slots import replay_s3, reorder
    from pwscup2026.kit.validate import validate_mia_submission
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def read(p, **kwargs): return pd.read_csv(p, float_precision='round_trip', **kwargs)
    def load(p):
        with np.load(p, allow_pickle=False) as z: return dict(z)
    def exact(a, b, name):
        ok = np.array_equal(a, b)
        checks[name] = bool(ok)
        dump(out/'checks.json', checks)
        if not ok: raise AssertionError(name + ': exact replay mismatch; stop, no fallback')
    def budget():
        if time.monotonic()-START >= 115: raise TimeoutError('Internal 115-second cap')
    checks = {}
    hashes = json.loads((plan_dir/'input_hashes.json').read_text())
    for name, h in hashes.items():
        assert sha(repo/name) == h, 'Preregistered input changed: '+name
    dump(out/'verified_inputs.json', hashes)
    cache = load(repo/'research/runs/20261010-FA101-main-public-contrasts-001/mia_public_arrays.npz')
    ids = cache['record_id']; targets = cache['targets'].astype(int)
    rare = cache['rare'].astype(bool); excluded = cache['excluded'].astype(bool)
    home = cache['home']; e1 = cache['e1']; baseline = cache['baseline']; hybrid = cache['hybrid']
    assert len(ids) == 1049 and len(np.unique(ids)) == 1049 and len(targets) == 22
    public = repo/'_private_team14/main_attack_20260928/files'
    declared = json.loads((public/'mia_columns.json').read_text())['target_columns']
    assert set(map(str, targets)) == set(declared)-{'14'} and len(np.unique(targets)) == 22
    baseline_path = repo/'research/submissions/four-measured-backup-20261010/candidate/F_mia.csv'
    bf = read(baseline_path).set_index('record_id')
    assert set(bf.index) == set(ids)
    exact(bf.loc[ids, [str(k) for k in targets]].to_numpy(), baseline, 'saved972302_source_exact')
    # Retain original baseline bytes verbatim; never replace them with serialization.
    (out/'972302_original_F_mia.csv').write_bytes(baseline_path.read_bytes())
    with zipfile.ZipFile(repo/'research/submissions/971407/submitted_server_original.zip') as z:
        ef = read(io.BytesIO(z.read('F_mia.csv'))).set_index('record_id')
    assert set(ef.index) == set(ids)
    expected_e0 = ef.loc[ids, [str(k) for k in targets]].to_numpy()
    own = read(repo/'_private_team14/B_self.csv', usecols=['record_id']+STATIC+['is_rare']).set_index('record_id').loc[ids]
    exact(own.is_rare.to_numpy(bool), rare, 'public_rare_flags_exact')
    exact(own.prefecture.eq(own.prefecture.mode().iloc[0]).to_numpy(), home, 'public_home_exact')
    previous = load(repo/'research/runs/20261010-FA37-known-private-001/main/matches.npz')
    exact(previous['record_id'], ids, 'excluded_record_order_exact')
    exact(previous['strict_mask'].astype(bool), excluded, 'saved_exclusion_exact')
    bg = read(repo/'_private_team14/A_bg.csv', usecols=STATIC)
    cs = {int(k): read(public/f'C_{k}.csv', usecols=STATIC) for k in targets}
    assert 14 not in cs and 9 in cs
    # FA03 raw peer score for all target distances: C9 stays in the rival pool.
    raw_own = cn.raw_peer(cs, own[STATIC], bg, 14); budget()
    raw_bg = cn.raw_peer(cs, bg, bg, 14); budget()
    # Exact full_static_nn raw branch from final_attack_api.predict_mia,
    # including cdist Euclidean then square (not sqeuclidean substitution).
    scale = np.maximum(bg[NUM].std(ddof=0).to_numpy(), 1e-12)
    def full_static(q):
        numeric = cdist(q[NUM].to_numpy()/scale, cs[9][NUM].to_numpy()/scale)
        full = numeric**2
        for col in CAT:
            full += 2*(q[col].to_numpy()[:, None] != cs[9][col].to_numpy()[None, :])
        return -np.sqrt(full).min(axis=1)
    raw_own[9] = full_static(own); raw_bg[9] = full_static(bg)
    so = np.column_stack([raw_own[int(k)] for k in targets])
    sb = np.column_stack([raw_bg[int(k)] for k in targets])
    assert np.isfinite(so).all() and np.isfinite(sb).all()
    e0 = np.column_stack([rank_unit(so[:, j]) for j in range(len(targets))])
    # Preserve raw inputs even if a historical exact replay fails below.
    np.savez_compressed(out/'raw_public_arrays.npz', record_id=ids, targets=targets,
                        raw_own=so, raw_bg=sb, e0=e0, expected_e0=expected_e0,
                        rare=rare, excluded=excluded, home=home, e1=e1)
    exact(e0, expected_e0, '971407_E0_exact')
    s3 = replay_s3(e0, e1, home, rare)
    exact(s3, baseline, '972302_S3_exact')
    evidence = e1+np.log(.087)*home[:, None]
    replay98 = reorder(s3, evidence, rare, excluded, ids)
    exact(replay98, hybrid, 'FA101_FA98_exact')
    knn = cn.knn_groups(own, bg, k=100); budget()
    blend = np.empty_like(e0); control = np.empty_like(e0)
    for j in range(len(targets)):
        s, z = so[:, j], sb[:, j]
        u = cn.upper_fraction(s, z, knn)
        blend[:, j] = cn.ordered(rankdata(s)+rankdata(-u), s)
        control[:, j] = cn.ordered(-cn.upper_fraction(s, z, None), s)
    cs3 = replay_s3(control, e1, home, rare)
    c98 = reorder(cs3, evidence, rare, excluded, ids)
    exact(cs3, baseline, 'all_BG_control_S3_identity')
    exact(c98, hybrid, 'all_BG_control_FA98_identity')
    candidate_s3 = replay_s3(blend, e1, home, rare)
    candidate = reorder(candidate_s3, evidence, rare, excluded, ids)
    exact(candidate[~rare], candidate_s3[~rare], 'stage_nonrare_exact')
    exact(candidate[excluded], candidate_s3[excluded], 'stage_excluded_exact')
    eligible = rare & ~excluded
    exact(np.sort(candidate[eligible], axis=0), np.sort(candidate_s3[eligible], axis=0), 'stage_eligible_rare_slots_exact')
    exact(np.sort(candidate, axis=0), np.sort(candidate_s3, axis=0), 'stage_full_value_multiset_exact')
    np.savez_compressed(out/'predictions.npz', record_id=ids, targets=targets,
                        baseline_S3=s3, baseline_FA98=replay98, control_FA98=c98,
                        candidate_S3=candidate_s3, fa98__blend=candidate)
    np.savez_compressed(out/'calibration_public_arrays.npz', knn=knn, blend_e0=blend, control_e0=control)
    frame = pd.DataFrame(candidate, columns=[str(k) for k in targets]); frame.insert(0, 'record_id', ids)
    frame.to_csv(out/'FA106_fa98_blend_F_mia.csv', index=False, lineterminator='\n')
    exact(read(out/'FA106_fa98_blend_F_mia.csv')[[str(k) for k in targets]].to_numpy(), candidate, 'candidate_CSV_roundtrip_exact')
    frozen = {p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npz', '.csv']}
    dump(out/'prediction_freeze.json', frozen)
    # Format validation only. No scorer or main outcome is imported/read.
    cfg = yaml.safe_load((repo/'_private_team14/scoring_config.yaml').read_text())
    strict = dict(team_id=14, mia_columns=json.loads((public/'mia_columns.json').read_text()), attacker_record_ids=set(ids))
    result = validate_mia_submission(frame, cfg, strict)
    assert result.ok, 'Official format validator failed'
    dump(out/'validation.json', dict(validator='validate_mia_submission', passed=True))
    rows = []
    for j, k in enumerate(targets):
        rows.append(dict(target=int(k), changed_vs_FA98=int((candidate[:, j] != hybrid[:, j]).sum()),
                         original_rare_values_changed=int((candidate[rare, j] != hybrid[rare, j]).sum()),
                         original_excluded_values_changed=int((candidate[excluded, j] != hybrid[excluded, j]).sum())))
    pd.DataFrame(rows).to_csv(out/'public_change_counts.csv', index=False)
    for name, h in frozen.items(): assert sha(out/name) == h
    budget()
    dump(out/'access_audit.json', dict(repository_paths=sorted(accessed), truth_reads=0, score_reads=0, external_shares=0))
    dump(out/'complete.json', dict(compute_seconds=time.monotonic()-START, new_candidates=1, rows=1049, targets=22,
                                  main_score=None, models=0, ZIPs=0, submissions=0,
                                  preservation='stage-relative; original rare/excluded values may change'))
    (out/'RESULT.md').write_text('One faithful fa98__blend main inference candidate frozen. All three baseline replays and all-BG identity checks passed. Stage-relative excluded/rare-slot preservation passed. Original FA98 excluded and rare values may change: see aggregate public_change_counts.csv. No main truth or score, ZIP, submission, model, or external share.\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--repo-root', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); out = args.out.resolve()
    try: run(args.repo_root.resolve(), out, Path(__file__).resolve().parent)
    except BaseException:
        if out.exists(): (out/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
