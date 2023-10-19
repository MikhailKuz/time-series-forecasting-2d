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

for model_name in ["Linear", "NLinear", "DLinear"]:
    for config in configs:
        c = deepcopy(config)
        # if c["task"] != "binary":
        #     continue
        if c["task"] != "reg":
            continue
        # if c["task"] != "multiclass":
        #     continue
        
        c["model_name"] = model_name
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
        
        c["aux_feats"] = []
        
        c["model_params"] = {
            "input_shape": [seq_len, H, W, 1 + len(c["aux_feats"])],
            "target_shape": [pred_len, H, W, C_out],
            "kernel_size": 16,
            "emb_len": 16,
            "temperature": 1,
            "use_emb": False,
        }
        
        if c["model_name"] == "DLinear":
            c["model_params"]["dec_kernel_size"] = 3
        
        exp_name = str(c)
        random.seed(exp_name)
        c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
        joblib.dump(c, path + c["exp_id"] + ".pkl")


temps = [0.5, 1, 2]
emb_lens = [16, 32, 64]

for temp in temps:
    for emb_len in emb_lens:
        for model_name in ["Linear"]:
            for config in configs:
                c = deepcopy(config)
                # if c["task"] != "binary":
                #     continue
                if c["task"] != "reg":
                    continue
                # if c["task"] != "multiclass":
                #     continue
                
                c["model_name"] = model_name
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
                
                c["aux_feats"] = []
                
                c["model_params"] = {
                    "input_shape": [seq_len, H, W, 1 + len(c["aux_feats"])],
                    "target_shape": [pred_len, H, W, C_out],
                    "kernel_size": 16,
                    "emb_len": emb_len,
                    "temperature": temp,
                    "use_emb": True,
                }
                
                if c["model_name"] == "DLinear":
                    c["model_params"]["dec_kernel_size"] = 3
                
                exp_name = str(c)
                random.seed(exp_name)
                c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
                joblib.dump(c, path + c["exp_id"] + ".pkl")


for model_name in ["Linear"]:
    for config in configs:
        c = deepcopy(config)
        # if c["task"] != "binary":
        #     continue
        if c["task"] != "reg":
            continue
        # if c["task"] != "multiclass":
        #     continue
        
        c["model_name"] = model_name
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
        
        c["aux_feats"] = ["inter_force_power_2"]
        
        c["model_params"] = {
            "input_shape": [seq_len, H, W, 1 + len(c["aux_feats"])],
            "target_shape": [pred_len, H, W, C_out],
            "kernel_size": 16,
            "emb_len": 16,
            "temperature": 1,
            "use_emb": True,
        }
        
        if c["model_name"] == "DLinear":
            c["model_params"]["dec_kernel_size"] = 3
        
        exp_name = str(c)
        random.seed(exp_name)
        c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
        joblib.dump(c, path + c["exp_id"] + ".pkl")


for model_name in ["Linear", "NLinear", "DLinear"]:
    for config in configs:
        c = deepcopy(config)
        # if c["task"] != "binary":
        #     continue
        # if c["task"] != "reg":
        #     continue
        # if c["task"] != "multiclass":
        #     continue
        
        c["model_name"] = model_name
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
        
        c["aux_feats"] = ["inter_force_power_2"]
        
        c["model_params"] = {
            "input_shape": [seq_len, H, W, 1 + len(c["aux_feats"])],
            "target_shape": [pred_len, H, W, C_out],
            "kernel_size": 16,
            "emb_len": 16,
            "temperature": 1,
            "use_emb": True,
        }
        
        if c["model_name"] == "DLinear":
            c["model_params"]["dec_kernel_size"] = 3
        
        exp_name = str(c)
        random.seed(exp_name)
        c['exp_id'] = ''.join([str(random.randint(0, 9)) for i in range(20)])
        joblib.dump(c, path + c["exp_id"] + ".pkl")
