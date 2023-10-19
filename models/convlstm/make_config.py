from copy import deepcopy
import random
import pathlib
import joblib
import numpy as np
import torch

path = str(pathlib.Path(__file__).parent.absolute()) + "/" + "input/"
p = pathlib.Path(path)
p.mkdir(parents=True, exist_ok=True)

data_path = "../../data/data_configs.pkl"
configs = joblib.load(data_path)

for config in configs:
    c = deepcopy(config)
    c["model_name"] = "ConvLSTM"
    c["gpu"] = str(np.random.choice(["0"]))
    
    c["n_epochs"] = 30
    c["batch_size"] = 4
    
    c["opt_params"] = {'lr': 0.001, 'weight_decay': 1e-5}
    c["opt"] = "AdamW"
    
    if c["task"] == "reg":
        c["loss"] = "MSELoss"
    elif c["task"] == "binary":
        c["loss"] = "BCEWithLogitsLoss"
    elif c["task"] == "multiclass":
        c["loss"] = "CrossEntropyLoss"
    else:
        assert False, f"Wrong task: {c['task']}"
    
    seq_len = 16
    pred_len = 12
    c["seq_len"] = seq_len
    c["pred_len"] = pred_len
    
    val_data = torch.load(c["val_data"])
    T, H, W = val_data.shape
    
    C_out = 1
    if c["task"] == "multiclass":
        C_out = len(torch.unique(val_data))
    
    c["model_params"] = {
        "input_dim": 1,
        "hidden_dim": 24,
        "kernel_size": (3, 3),
        "num_layers": 3,
        "batch_first": True,
        "return_all_layers": False,
        "C_out": C_out,
        "pred_len": pred_len,
    }
    c["model_params"]["kernel_size"] = [c["model_params"]["kernel_size"]] * c["model_params"]["num_layers"]
    
    exp_name = str(c)
    random.seed(exp_name)
    c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
    joblib.dump(c, path + c["exp_id"] + ".pkl")
