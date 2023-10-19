import os
import joblib

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import accuracy_score
import numpy as np

from tqdm.auto import tqdm


paths = [
    "../models/convlstm/output/",
         "../models/earthformer/output/",
         "../models/linear/output/"
         ]

def calc_metrics(log, true, pred, prefix):
    metrics = {}
    if log["task"] == "reg":
        true, pred = true.flatten(), pred.flatten()
        metrics["r2" + "_" + prefix] = r2_score(true, pred)
        metrics["mse" + "_" + prefix] = mean_squared_error(true, pred)
        metrics["mae" + "_" + prefix] = mean_absolute_error(true, pred)
        metrics["rmse" + "_" + prefix] = np.sqrt(metrics["mse" + "_" + prefix])
    
    elif log["task"] == "binary":
        true, pred = true[::20, ...], pred[::20, ...]
        true, pred = true.flatten(), pred.flatten()
        metrics["auc" + "_" + prefix] = roc_auc_score(true, pred)
        metrics["pr" + "_" + prefix] = average_precision_score(true, pred)
        metrics["f1" + "_" + prefix] = f1_score(true, pred >= 0.5)
    
    elif log["task"] == "multiclass":
        true, pred = true.flatten(), pred.flatten()
        metrics["acc" + "_" + prefix] = (true == pred).float().mean().item()
    return metrics

for path in paths:
    if not os.path.exists(path):
        continue
    
    configs = os.listdir(path)

    for conf in tqdm(configs):
        
        if not os.path.exists(path + conf + "/config.pkl"):
            continue

        log = joblib.load(path + conf + "/config.pkl")
        preds = joblib.load(path + conf + '/preds.pkl')
        
        for stage in ["train", "val", "test"]:
            all_pred, all_true = preds[stage]["preds"], preds[stage]["target"]
            
            for topk in [0, 5, 11]:
                pred, true = all_pred[:, topk, ...], all_true[:, topk, ...]
                log.update(calc_metrics(log, true, pred, str(topk + 1) + "_" + stage))
            
            log.update(calc_metrics(log, all_true, all_pred, "all" + "_" + stage))
            
            all_true, all_pred = all_true, all_pred
            if log["task"] == "reg":
                log["r2_curve_" + stage] = [r2_score(all_true[:, i, ...].flatten(), all_pred[:, i, ...].flatten()) for i in range(all_true.shape[1])]
            
        joblib.dump(log, path + conf + "/config.pkl")
