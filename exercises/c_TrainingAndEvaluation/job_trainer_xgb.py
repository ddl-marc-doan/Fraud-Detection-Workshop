import os
from domino import Domino
project = os.environ['DOMINO_PROJECT_NAME']
user = os.environ['DOMINO_PROJECT_OWNER']
domino = Domino(
    f"{user}/{project}",
    api_key = os.environ["DOMINO_USER_API_KEY"],
    host = os.environ["DOMINO_API_HOST"],
)
hwtier = 'Small'
execution = 'exercises/c_TrainingAndEvaluation/trainer_xgb.py'
title = f'Train XGBoost classifier'
domino.job_start(execution, title=title, hardware_tier_name=hwtier)
#assuming that each line of the file input_data.txt lists the path to a dataset you'd like to run your job on
# sobol_cases = list(range(1,11))
# # running the jobs
# for sobol_case in sobol_cases:
#     domino.job_start(f"julia --project=/mnt/ -e 'using Pkg; Pkg.instantiate();' && julia --project=/mnt/ calib_sobol.jl -c {sobol_case} -n 6 --job-name case_{sobol_case} --sobol-size 1000 --datadir /domino/datasets/local/DK/results", title=f'case {sobol_case}', hardware_tier_name='Laptop')