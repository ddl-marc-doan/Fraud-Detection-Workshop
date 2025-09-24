# File: trainer_ada.py
from pathlib import Path
import json,sys,os
sys.path.append(os.environ["DOMINO_WORKING_DIR"])
from sklearn.ensemble import AdaBoostClassifier
from exercises.c_TrainingAndEvaluation.generic_trainer import train_fraud


# Load DataFrame from dataset 
try:
    transformed_df_filename = Path("/workflow/inputs/transformed_filename").read_text().strip()
    print('using workflow input: transformed_filename', transformed_df_filename)
except FileNotFoundError as e:
    print('file not found error', e)
    transformed_df_filename = 'transformed_cc_transactions.csv'

model_name = 'AdaBoost'
model_obj = AdaBoostClassifier(
            n_estimators=10,
            learning_rate=0.1,
            algorithm="SAMME",
        )

res = train_fraud(model_obj, model_name, transformed_df_filename)

DROP = {"threshold_scan", "curves"}
small = {k: v for k, v in res.items() if k not in DROP}

print(f"Training {model_name} completed successfully")
print(json.dumps({k: small.get(k) for k in ['roc_auc','f1_fraud','accuracy','log_loss']}, indent=2))

out_path = Path("/workflow/outputs/results")
if out_path.parent.exists():
    out_path.write_text(json.dumps(small))  # JSON, not str(dict)
