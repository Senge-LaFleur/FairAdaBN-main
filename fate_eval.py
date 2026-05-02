"""
FATE Metric Calculator
======================
FATE_FC = (ACC_m - ACC_b) / ACC_b  -  lambda * (FC_m - FC_b) / FC_b

Where:
  FC = one of EOpp0, EOpp1, EOdd (lower is fairer)
  b  = baseline  (main_baseline.py, exp_id=1)
  m  = mitigation model (FairAdaBN or FairAdaBN+Lmi)

Usage:
  python fate_eval.py --baseline_exp 1 --mitigation_exp 3 --rand_seed 0 --method best

Example full workflow:
  python main_baseline.py  --exp_id 1 --rand_seed 0 --max_epoch 400
  python main.py           --exp_id 3 --rand_seed 0 --max_epoch 400
  python main_Lmi.py       --exp_id 4 --rand_seed 0 --max_epoch 400

  python evaluation.py     --exp_id 1 --rand_seed 0 --method best
  python evaluation.py     --exp_id 3 --rand_seed 0 --method best
  python evaluation.py     --exp_id 4 --rand_seed 0 --method best

  python fate_eval.py --baseline_exp 1 --mitigation_exp 3 --rand_seed 0 --method best
  python fate_eval.py --baseline_exp 1 --mitigation_exp 4 --rand_seed 0 --method best
"""

import argparse
import pandas as pd

LAM = 1.0  # lambda weighting factor (paper default)

FC_LABELS = {
    'equal_opp_0': 'EOpp0 (TNR gap)',
    'equal_opp_1': 'EOpp1 (TPR gap)',
    'equal_odds':  'EOdd  (TPR+TNR gap)',
    'eom':         'EOM   (Opp. Margin)',
    'pqd':         'PQD   (Quality Disp.)',
    'dpm':         'DPM   (Parity Margin)',
}


def load_scores(exp_id, rand_seed, method):
    path = f'/kaggle/working/fair_scores/rand_seed={rand_seed}_exp_{exp_id}_{method}_fairness_scores.csv'
    try:
        return pd.read_csv(path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            f"Make sure you ran: python evaluation.py --exp_id {exp_id} "
            f"--rand_seed {rand_seed} --method {method}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute FATE metric')
    parser.add_argument('--baseline_exp',    type=str, required=True,
                        help='exp_id of the baseline model (main_baseline.py)')
    parser.add_argument('--mitigation_exp',  type=str, required=True,
                        help='exp_id of the mitigation model (main.py or main_Lmi.py)')
    parser.add_argument('--rand_seed',       type=str, default='0')
    parser.add_argument('--method',          choices=['best', 'final'], default='best')
    args = parser.parse_args()

    baseline   = load_scores(args.baseline_exp,   args.rand_seed, args.method)
    mitigation = load_scores(args.mitigation_exp, args.rand_seed, args.method)

    ACC_b = baseline['accuracy'].values[0]
    ACC_m = mitigation['accuracy'].values[0]

    print('\n' + '=' * 55)
    print(f'  FATE  (baseline exp={args.baseline_exp}  vs  mitigation exp={args.mitigation_exp})')
    print(f'  seed={args.rand_seed}  method={args.method}  lambda={LAM}')
    print('=' * 55)
    print(f"  Baseline   ACC: {ACC_b:.4f}")
    print(f"  Mitigation ACC: {ACC_m:.4f}")
    print(f"  Accuracy change: {(ACC_m - ACC_b) / ACC_b:+.4f}")
    print('-' * 55)

    for fc_key, fc_name in FC_LABELS.items():
        FC_b = baseline[fc_key].values[0]
        FC_m = mitigation[fc_key].values[0]
        fairness_change = (FC_m - FC_b) / FC_b if FC_b != 0 else float('nan')
        
        # For equal_opp_0, equal_opp_1, equal_odds, LOWER is fairer.
        # So a decrease in value (fairness_change < 0) means improved fairness.
        # FATE subtracts LAM * fairness_change.
        if fc_key in ['equal_opp_0', 'equal_opp_1', 'equal_odds']:
            FATE = (ACC_m - ACC_b) / ACC_b - LAM * fairness_change
        
        # For eom, pqd, dpm, HIGHER is fairer.
        # So an increase in value (fairness_change > 0) means improved fairness.
        # FATE adds LAM * fairness_change.
        else:
            FATE = (ACC_m - ACC_b) / ACC_b + LAM * fairness_change
            
        direction = 'IMPROVED' if FATE > 0 else 'DEGRADED'
        print(f"  FATE_{fc_key:<12}  {FATE:+.4f}   ({direction})")
        print(f"    Baseline {fc_name}: {FC_b:.4f}   Mitigation: {FC_m:.4f}   change: {fairness_change:+.4f}")

    print('=' * 55)
    print('\n  Interpretation:')
    print('    FATE > 0 : Model mitigates unfairness while maintaining accuracy  ✓')
    print('    FATE = 0 : No net benefit')
    print('    FATE < 0 : Accuracy loss outweighs fairness gain (or fairness worsened)')
    print()