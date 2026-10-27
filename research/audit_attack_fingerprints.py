"""G10a: observable features are computed separately from retrospective scores."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/runs/20260924-G10a-001'
PUBLIC = ROOT / 'attack_phase/data/public/prelim_attack_20260825'
OLD = ROOT / 'analysis/prelim_review_20260924/outputs/fingerprints.csv'
SCORES = ROOT / 'research/runs/20260924-G03c-001/same_challenge_aia.csv'
CONT = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
STATIC = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG']
COLS = STATIC + ['time', 'onset', 'death']


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def keys(frame, columns):
    return frame[columns].astype(object).where(frame[columns].notna(), '<NA>').astype(str).agg('\x1f'.join, axis=1)


def observable_features(c, bg, detector):
    """No target identity, target truth, own B, or attack response arguments."""
    c, bg = c[COLS], bg[COLS]
    assert np.isfinite(c[CONT + ['time', 'onset', 'death']].to_numpy()).all()
    delta = c[detector['columns']].to_numpy() - np.asarray(detector['mu'])
    d = np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', delta, np.linalg.pinv(detector['sigma']), delta)))
    out = {
        'detected_rare': int((d >= detector['band']['lo']).sum()),
        'exact_duplicate_excess': float(c.duplicated().mean()),
        'background_static_match_rows': int(keys(c, STATIC).isin(set(keys(bg, STATIC))).sum()),
        'background_all13_match_rows': int(keys(c, COLS).isin(set(keys(bg, COLS))).sum()),
        'background_static_match_round10': int(pd.util.hash_pandas_object(c[STATIC].round(10), index=False).isin(
            pd.util.hash_pandas_object(bg[STATIC].round(10), index=False)).sum()),
        'onset_rate': c.onset.mean(), 'death_rate': c.death.mean(),
        'time_median': c.time.median(), 'unique_time': c.time.nunique(),
        'dominant_prefecture': c.prefecture.mode().iloc[0],
        'dominant_prefecture_rate': c.prefecture.value_counts(normalize=True).iloc[0],
        'prefecture_count': c.prefecture.nunique(),
        'spearman_mean_abs_delta_bg': np.abs(c[CONT].corr(method='spearman') - bg[CONT].corr(method='spearman')).to_numpy()[np.triu_indices(7, 1)].mean(),
    }
    out.update({col + '_mean': c[col].mean() for col in CONT})
    return out


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item()) + '\n', encoding='utf-8', newline='\n')


def main():
    bgpath, schema = ROOT/'_private_team10/A_bg.csv', ROOT/'_private_team10/schema.json'
    bg = read(bgpath)
    detector = json.loads(schema.read_text(encoding='utf-8'))['rare_detector']
    files = sorted(PUBLIC.glob('C_*.csv'), key=lambda p: int(p.stem.split('_')[-1]))
    assert len(files) == 21
    features = pd.DataFrame([{'target': int(p.stem.split('_')[-1]), **observable_features(read(p), bg, detector)} for p in files])
    assert features.target.is_unique
    old = read(OLD).set_index('target').loc[features.target]
    indexed = features.set_index('target'); checks = []
    for col in indexed:
        if col == 'dominant_prefecture':
            assert indexed[col].equals(old[col]); error = 0.
        else:
            error = float(np.max(np.abs(indexed[col].to_numpy() - old[col].to_numpy())))
            assert error <= 1e-10, (col, error)
        checks.append({'feature': col, 'max_abs_error': error, 'truth_required': False})
    features.to_csv(OUT/'observable_features.csv', index=False, lineterminator='\n')
    pd.DataFrame(checks).to_csv(OUT/'feature_replay.csv', index=False, lineterminator='\n')

    # Scores enter only after the feature file has been computed and written.
    scores = read(SCORES)
    assert not scores.duplicated(['target', 'method']).any() and len(scores) == 63
    assert set(scores.target) == set(features.target)
    assert set(scores.method) == {'official_1nn', 'log_qi_1nn', 'log_qi_window20'}
    assert scores.groupby('target').size().eq(3).all()
    assert scores[['n_member', 'n_control']].eq(163).all().all()
    assert scores[['n_member_rare', 'n_control_rare']].eq(13).all().all()
    scores['exclude_own_target'] = scores.target.ne(10)
    scores['round'] = 'preliminary'; scores['source_version'] = 'G03c-same-challenge-3AIA'
    scores.to_csv(OUT/'retrospective_labels.csv', index=False, lineterminator='\n')
    pivot = scores.pivot(index='target', columns='method', values='score')
    advantage = pivot.sub(pivot.official_1nn, axis=0).add_prefix('delta_vs_official__')
    advantage['posthoc_best_score'] = pivot.max(axis=1)
    advantage['posthoc_best_method'] = pivot.idxmax(axis=1)
    advantage['exclude_own_target'] = advantage.index != 10
    advantage.to_csv(OUT/'relative_advantage.csv', lineterminator='\n')
    # Key join validates coverage, without shipping truth into feature columns.
    assert len(features.merge(scores[['target','method']], on='target', validate='one_to_many')) == 63
    summaries = {}
    for name, frame in [('all21_diagnostic', pivot), ('exclude_own20', pivot.drop(index=10))]:
        summaries[name] = {'n_targets':len(frame), 'method_mean':frame.mean().to_dict(),
            'posthoc_best_mean':float(frame.max(axis=1).mean()),
            'strict_wins_vs_official':{col:int((frame[col] > frame.official_1nn + 1e-12).sum()) for col in frame},
            'posthoc_best_is_deployable':False}
    dump('summary.json', {'decision':'accept input foundation only', 'features':len(indexed.columns),
        'targets':21, 'score_rows':63, 'replay_max_abs_error':max(x['max_abs_error'] for x in checks),
        'summaries':summaries, 'router_evaluated':False, 'processing_candidate_changed':False})
    inputs = files + [bgpath, schema, OLD, SCORES, Path(__file__)]
    dump('manifest.json', {'inputs':{p.relative_to(ROOT).as_posix():digest(p) for p in inputs},
        'feature_inputs_only':[p.relative_to(ROOT).as_posix() for p in files+[bgpath,schema]],
        'excluded':['target ID as model feature','old_distance8_count','B_self diagnostics','published truth','attack scores as inference inputs'],
        'seed':None, 'budget_minutes':15, 'starting_commit':'b17fb68'})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
