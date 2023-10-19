import os
import joblib
import pathlib
import torch

def preprocces_train(train, task):
    if task == 'reg':
        return train
    
    elif task == 'binary':
        mask = train < -2
        train[mask] = 1
        train[mask == False] = 0
    
    elif task == 'multiclass':
        mask1 = train < -2
        mask2 = train > 2
        mask3 = (mask1 == False) & (mask2 == False)
        
        train[mask1] = 2
        train[mask3] = 1
        train[mask2] = 0
    
    return train

input_dir = "./files/"
logs = []

for file in os.listdir("./files"):
    dt_name = file[:-4]
    orig_train = torch.load(input_dir + file)
    
    for task in ["reg", "binary", "multiclass"]:
        train = orig_train.clone()
        train = preprocces_train(train, task)
        N = len(train)
        
        percent_train_val = 0.7
        # percent_train_val = 0.1
        indx_test = int(N * percent_train_val)
        
        percent_train = 0.9
        # percent_train = 0.5
        indx_train = int(indx_test * percent_train)
        
        test = train[indx_test:]
        # test = train[-50:]
        val = train[indx_train:indx_test]
        train = train[:indx_train]
        
        paths = {}
        for dt, name in zip([train, val, test],
                            ['train', 'val', 'test']):
            path = "./splits/" + task + "/" + name
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            path += "/" + dt_name + '.pt'
            
            torch.save(dt.float(), path)
            paths[name + "_data"] = os.path.abspath(path)
        
        log = {
            "task": task,
            "data": dt_name,
            **paths
        }
        logs.append(log)

joblib.dump(logs, "./data_configs.pkl")
