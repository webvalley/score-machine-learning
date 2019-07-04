import papermill as pm
import os

data_dir = "/datadrive/random_forests_clinical_data"
for exp_dir in os.listdir(data_dir):
    if "bz2" in exp_dir:
        continue
    pm.execute_notebook('/home/webvalley/notebooks/RandomForestsGraphEverything.ipynb', '/home/webvalley/notebooks/'+ exp_dir +'-Graphs.ipynb', parameters = dict(exp_dir = exp_dir))
