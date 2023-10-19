import os
import pathlib
import argparse
from time import time
import sys
from box import Box
import torch
import numpy as np

sys.path.insert(0, '../')

from trainer import Trainer

import joblib


def get_time():
    torch.cuda.current_stream().synchronize()
    return time()

def model_forward(model, x, y):
    x = torch.cat([x, torch.zeros_like(y)], dim=1)
    # B, T, C, H, W
    layer_output_list, last_state_list = model(x.permute(0, 1, 4, 2, 3))
    pred = layer_output_list.permute(0, 1, 3, 4, 2)
    return pred

def main(args):
    path = str(pathlib.Path(__file__).parent.absolute()) + "/"
    log = joblib.load(path + "/input/" + args.config_name)
    log = Box(log)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = log.gpu
    
    cur_path = path + '/output/' + log.exp_id + "/"
    pathlib.Path(cur_path).mkdir(parents=True, exist_ok=True)
    log.output_path = cur_path
    
    log = Box(log)
    trainer = Trainer(log)
    trainer.model_forward = model_forward
    
    log["train_time"] = get_time()
    train_loss_epoch, val_loss_epoch = trainer.fit()
    log["train_time"] = get_time() - log["train_time"]
    
    log["train_loss_epoch"] = train_loss_epoch
    log["val_loss_epoch"] = val_loss_epoch
    
    log["min_val_loss_epoch"] = min(val_loss_epoch)
    log["min_train_loss_epoch"] = min(train_loss_epoch)
    
    log["epoch_min_val_loss_epoch"] = np.argmin(val_loss_epoch) + 1
    
    train_preds = trainer.predict_loader(trainer.train_loader)
    val_preds = trainer.predict_loader(trainer.val_loader)
    test_preds = trainer.predict_loader(trainer.test_loader)
    
    joblib.dump({"train": train_preds, "val": val_preds, "test": test_preds}, cur_path + 'preds.pkl')
    joblib.dump(log.to_dict(), cur_path + 'config.pkl')
    
    cur_path = path + '/input/' + log.exp_id + ".pkl"
    os.remove(cur_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params info you can get on")
    parser.add_argument('--config_name', nargs='?',
                    default='./',
                    help='')
    args = parser.parse_args()
    main(args)
