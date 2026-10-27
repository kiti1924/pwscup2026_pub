from pathlib import Path
import sys
import pandas as pd
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'attack_phase/src'))
from aia_stage7 import AIAStage7Engine

def main():
    td = root / 'attack_phase/data/public/prelim_attack_20260825'; team = root / '_private_team10'
    background = pd.read_csv(team / 'A_bg.csv')
    targets = [1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
    rows = []
    for k in targets:
        target = pd.read_csv(td / f'C_{k}.csv')
        challenge = pd.read_csv(td / f'aia_challenge_{k}.csv')
        engine = AIAStage7Engine(target, background, 'cell_modal_aggregate')
        values, _ = engine.predict(challenge)
        rows.append(pd.DataFrame({'target_k': k, 'challenge_row_id': challenge.challenge_row_id, 'time_hat': values}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(root / 'attack_phase/outputs/20260906_aia_stage7_for_s3/F_aia_cell_modal_aggregate.csv', index=False)
    print(out.shape)
if __name__ == '__main__': main()
