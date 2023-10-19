from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch
import os
from tqdm.auto import tqdm
import pathlib
import sys

try:
    from earthformer.model import *
except:
    pass

try:
    from convlstm.model import *
except:
    pass

try:
    from linear.model import *
except:
    pass

thismodule = sys.modules[__name__]

def force_power_2(x, t, h, w):
    power = 2
    
    r1 = torch.arange(1, t + 1).reshape(1, 1, t, 1, 1)
    r2 = torch.arange(h).reshape(1, 1, 1, h, 1) - h // 2
    r3 = torch.arange(w).reshape(1, 1, 1, 1, w) - w // 2
    
    r = r1 ** 2 + r2 ** 2 + r3 ** 2
    r = torch.sqrt(r)
    r = 1 / (r ** power)
    r = r.flip([2])
    
    r = torch.cat([torch.zeros_like(r), torch.zeros_like(r)[:, :, 0, :, :].unsqueeze(2), r], dim=2)
    x = torch.nn.functional.conv3d(x[None, None, ...], r, padding='same')
    x = x.squeeze()
    return x

ax_name_to_f = {
    "inter_force_power_2": force_power_2
}

class FlodDataset(Dataset):
    def __init__(self, data=None, **kwargs):
        self.data = data
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __len__(self):
        return self.data.shape[0] - self.pred_len - self.seq_len + 1

    def __getitem__(self, index):
        
        x_item = self.data[index:index + self.seq_len]
        y_item = self.data[index + self.seq_len:index + self.seq_len + self.pred_len]

        return x_item.float(), y_item[:, :, :, 0].unsqueeze(-1).float()

class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self._create_dir(config.output_path)
        
        dataset_args = {
            "seq_len": config.seq_len,
            "pred_len": config.pred_len,
        }
        
        train_data = torch.load(config.train_data)
        val_data = torch.load(config.val_data)
        test_data = torch.load(config.test_data)
        
        if config.task == "reg":
            self.mean, self.var = train_data.mean(dim=(0)), train_data.var(dim=(0))
            self.var[self.var == 0] = 1
            train_data = (train_data - self.mean) / self.var
            val_data = (val_data - self.mean) / self.var
            test_data = (test_data - self.mean) / self.var
            self.mean, self.var = self.mean.to(self.device), self.var.to(self.device)
        
        _train_data = [train_data]
        _val_data = [val_data]
        _test_data = [test_data]
        
        if "aux_feats" in self.config:
            for ax_f in self.config.aux_feats:
                f = ax_name_to_f[ax_f]
                _train_data.append(f(train_data, (self.config.seq_len - 1) // 2, self.config.model_params.kernel_size // 3, self.config.model_params.kernel_size // 4))
                _val_data.append(f(val_data, (self.config.seq_len - 1) // 2, self.config.model_params.kernel_size // 3, self.config.model_params.kernel_size // 4))
                _test_data.append(f(test_data, (self.config.seq_len - 1) // 2, self.config.model_params.kernel_size // 3, self.config.model_params.kernel_size // 4))
                
        train_data = torch.stack(_train_data, dim=-1)
        val_data = torch.stack(_val_data, dim=-1)
        test_data = torch.stack(_test_data, dim=-1)
        
        dataset_ett_train = FlodDataset(data=train_data, **dataset_args)
        dataset_ett_val = FlodDataset(data=val_data, **dataset_args)
        dataset_ett_test = FlodDataset(data=test_data, **dataset_args)
        
        dataloader_args = {
            "pin_memory": True,
            "batch_size": config.batch_size,
            "num_workers": 0,
        }
        self.train_loader = DataLoader(
            dataset_ett_train,
            shuffle=True,
            drop_last=True,
            **dataloader_args
        )
        self.val_loader = DataLoader(
            dataset_ett_val,
            shuffle=False,
            drop_last=False,
            **dataloader_args
        )
        self.test_loader = DataLoader(
            dataset_ett_test,
            shuffle=False,
            drop_last=False,
            **dataloader_args
        )
        
        self.model = getattr(thismodule, config.model_name)(**config.model_params)
        self.model = self.model.to(self.device)
    
    def _create_dir(self, path):
        path = os.path.abspath(path)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    def model_forward(self, model, x, y):
        assert False, "Need to implement"
    
    def fit(self):
        optim = getattr(torch.optim, self.config.opt)(self.model.parameters(), **self.config.opt_params)
        criterion = getattr(nn, self.config.loss)()
        
        train_loss_epoch = []
        val_loss_epoch = []
 
        for epoch in tqdm(range(self.config.n_epochs)):
            train_loss = []
            self.model.train()
            
            for batch in tqdm(self.train_loader):
                optim.zero_grad()
                batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model_forward(self.model, batch[0], batch[1])
                
                if self.config.task == "multiclass":
                    loss = criterion(pred.reshape(-1, pred.shape[-1]), batch[1].flatten().long())
                else:
                    loss = criterion(pred, batch[1])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                train_loss.append(loss.item())
            
            train_loss_epoch.append(np.mean(train_loss))
            self.model.train(False)
            val_loss = []
            
            with torch.autograd.no_grad():
                for batch in tqdm(self.val_loader):
                    batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
                    
                    pred = self.model_forward(self.model, batch[0], batch[1])
                    if self.config.task == "multiclass":
                        loss = criterion(pred.reshape(-1, pred.shape[-1]), batch[1].flatten().long())
                    else:
                        loss = criterion(pred, batch[1])
                    
                    val_loss.append(loss.item())

            val_loss_epoch.append(np.mean(val_loss))
            
            if np.argmin(val_loss_epoch) == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.config.output_path, "best_model.pth"))
            print("train loss:", np.round(train_loss_epoch[-1], 5), "\tval loss:", np.round(val_loss_epoch[-1], 5))
        
        self.model.load_state_dict(torch.load(os.path.join(self.config.output_path, "best_model.pth")))
        return train_loss_epoch, val_loss_epoch
        
    def predict_loader(self, loader):
        task = self.config.task
        test_preds = []
        target_preds = []

        self.model.train(False)
        with torch.autograd.no_grad():
            for batch in tqdm(loader):
                batch[0], batch[1] = batch[0].to(self.device), batch[1].to(self.device)
                
                pred = self.model_forward(self.model, batch[0], batch[1])
                batch[1] = batch[1][..., 0]
                
                if task == "reg":
                    pred = pred[..., 0]
                    pred = self.mean + pred * self.var
                    batch[1] = self.mean + batch[1] * self.var
                
                elif task == "binary":
                    pred = pred[..., 0]
                    pred = torch.sigmoid(pred).float()
                
                elif task == "multiclass":
                    pred = torch.argmax(pred, dim=-1)
                
                test_preds.append(pred)
                target_preds.append(batch[1])
        
        return {
            "target": torch.cat(target_preds, dim=0).detach().cpu(),
            "preds": torch.cat(test_preds, dim=0).detach().cpu()
        }
