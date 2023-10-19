
import os
import collections
import pandas as pd
from tqdm import tqdm
import joblib

paths = ["../models/convlstm/output/",
         "../models/earthformer/output/",
         "../models/linear/output/"]

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


log_df = pd.DataFrame()
i = 0

for path in paths:
    if not os.path.exists(path):
        continue
    
    configs = os.listdir(path)

    for conf in tqdm(configs):
        if not os.path.exists(path + conf + "/config.pkl"):
            continue
        
        log = joblib.load(path + conf + "/config.pkl")
        
        for k, v in flatten_dict(log).items():
            if "_data" not in k and "_path" not in k:
                try:
                    log_df.loc[i, k] = v
                except:
                    log_df.loc[i, k] = str(v)
        
        log_df.loc[i, "config_path"] = os.path.abspath(path + conf + "/config.pkl")
        i += 1
        
log_df.to_csv('../output/run_summary.csv', index=False)
