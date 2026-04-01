from pathlib import Path
from subprocess import check_output

files = [
    'exp1_label_shift_synth.py','exp2_auc_pr_invariance.py','exp3_ess_vs_weight.py',
    'exp4_concept_drift.py','exp5_realdata_breast_cancer.py','exp6_calibration_label_shift.py',
    'exp7_drift_types.py','exp8_multimodal_label_shift.py','exp9_covtype_label_shift.py',
    'exp10_credit_card_fraud.py','exp11_high_variance_medical.py'
]

root = Path('.').resolve()
target_dir = root / 'src' / 'drift_or_shift' / 'experiments'
target_dir.mkdir(parents=True, exist_ok=True)

for f in files:
    source = f'src/drift_or_shift/experiments/experiments/{f}'
    try:
        out = check_output(['git', 'show', f'HEAD:{source}'], cwd=str(root))
        with open(target_dir / f, 'wb') as fw:
            fw.write(out)
    except Exception as e:
        print('cannot read', source, e)

print('done')
