from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parents[2]
targets = [1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
d = pd.read_csv(root/'attack_phase/outputs/20260906_mia_stage7_for_s3/public_rank_diagnostics.csv')
selected = d[d.method.eq('rare_kde_density_ratio')]
out = selected.pivot(index='record_id', columns='target', values='score').reindex(columns=targets).reset_index()
base = pd.read_csv(root/'attack_phase/outputs/20260828_mia_stage2_final/F_mia_candidate.csv').set_index('record_id')
out = out.set_index('record_id')
out.iloc[:, 0:2] = base.loc[out.index, ['1','2']].to_numpy()
out.reset_index().to_csv(root/'attack_phase/outputs/20260906_mia_stage7_for_s3/F_mia_rare_kde_weak_targets.csv', index=False)
print(out.shape)
