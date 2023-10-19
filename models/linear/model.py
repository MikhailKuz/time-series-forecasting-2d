import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear(nn.Module):
    def __init__(self, **kwargs):
        super(Linear, self).__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        self.T_in, self.H, self.W, self.C_in = self.input_shape
        self.T_out, self.H, self.W, self.C_out = self.target_shape
        
        self.padding_H = self.H % self.kernel_size
        self.padding_W = self.W % self.kernel_size
        
        self.unfold = nn.Unfold(kernel_size=[self.kernel_size] * 2,
                                padding=(self.padding_H, self.padding_W),
                                stride=(self.kernel_size // 2, self.kernel_size // 2))
        self.fold = nn.Fold(output_size=(self.H, self.W),
                            kernel_size=[self.kernel_size] * 2,
                            padding=(self.padding_H, self.padding_W),
                            stride=(self.kernel_size // 2, self.kernel_size // 2))
        
        self.w = {}
        for _ in range(self.C_out * self.T_out):
            self.w.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })
        
        self.w = nn.ParameterDict(self.w)
        
        if self.use_emb:
            self.emb = nn.Embedding(self.emb_len, self.kernel_size ** 2)
            self.features_to_emb = nn.Linear(self.kernel_size ** 2, self.emb_len)
        
        
        
    def forward(self, x):
        B, T_in, H, W, C_in = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)
        x = self.unfold(x) # [B, ..., L]
        L = x.shape[-1]
        x = x.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]
        
        xx = []
        for i in range(self.C_out * self.T_out):
            w, bias = self.w[f"w_{i}"], self.w[f"bias_{i}"]
            xx.append((x * w).sum(dim=(2, 3)) + bias)  # [B, L, k, k]
        
        x = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
        
        if self.use_emb:
            x = F.softmax(self.features_to_emb(x.flatten(start_dim=-2)) / self.temperature, dim=-1) # [B, C_out * T_out, L, N_emb]
            x = x[..., None] * self.emb.weight[None, None, None, ...]
            x = x.sum(dim=-2)
            x = x.reshape(*(tuple(x.shape[:-1]) + (self.kernel_size, self.kernel_size,))) # [B, C_out * T_out, L, k, k]
        
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, L)
        x = self.fold(x) # [B, C_out * T_out, H, W]
        x = x.reshape(B, self.T_out, self.C_out, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x # [B, T, H, W, С_out]


class NLinear(nn.Module):
    def __init__(self, **kwargs):
        super(NLinear, self).__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        self.T_in, self.H, self.W, self.C_in = self.input_shape
        self.T_out, self.H, self.W, self.C_out = self.target_shape
        
        self.padding_H = self.H % self.kernel_size
        self.padding_W = self.W % self.kernel_size
        
        self.unfold = nn.Unfold(kernel_size=[self.kernel_size] * 2,
                                padding=(self.padding_H, self.padding_W),
                                stride=(self.kernel_size // 2, self.kernel_size // 2))
        self.fold = nn.Fold(output_size=(self.H, self.W),
                            kernel_size=[self.kernel_size] * 2,
                            padding=(self.padding_H, self.padding_W),
                            stride=(self.kernel_size // 2, self.kernel_size // 2))
        
        self.w = {}
        for _ in range(self.C_out * self.T_out):
            self.w.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })
        
        self.w = nn.ParameterDict(self.w)
        
        if self.use_emb:
            self.emb = nn.Embedding(self.emb_len, self.kernel_size ** 2)
            self.features_to_emb = nn.Linear(self.kernel_size ** 2, self.emb_len)
        
        
        
    def forward(self, x):
        B, T_in, H, W, C_in = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)
        x = self.unfold(x) # [B, ..., L]
        L = x.shape[-1]
        x = x.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]
        last = x[:, :, -1, 0, :, :]
        x = x - last[:, :, None, None, :, :]
        
        xx = []
        for i in range(self.C_out * self.T_out):
            w, bias = self.w[f"w_{i}"], self.w[f"bias_{i}"]
            xx.append((x * w).sum(dim=(2, 3)) + bias + last)  # [B, L, k, k]
        
        x = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
        
        if self.use_emb:
            x = F.softmax(self.features_to_emb(x.flatten(start_dim=-2)) / self.temperature, dim=-1) # [B, C_out * T_out, L, N_emb]
            x = x[..., None] * self.emb.weight[None, None, None, ...]
            x = x.sum(dim=-2)
            x = x.reshape(*(tuple(x.shape[:-1]) + (self.kernel_size, self.kernel_size,))) # [B, C_out * T_out, L, k, k]
        
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, L)
        x = self.fold(x) # [B, C_out * T_out, H, W]
        x = x.reshape(B, self.T_out, self.C_out, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x # [B, T, H, W, С_out]

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # [B, T_in, C]
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)) # [B, C, T_in]
        x = x.permute(0, 2, 1) # [B, T_in, C]
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, **kwargs):
        super(DLinear, self).__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        
        self.T_in, self.H, self.W, self.C_in = self.input_shape
        self.T_out, self.H, self.W, self.C_out = self.target_shape
        
        self.padding_H = self.H % self.kernel_size
        self.padding_W = self.W % self.kernel_size
        
        self.unfold = nn.Unfold(kernel_size=[self.kernel_size] * 2,
                                padding=(self.padding_H, self.padding_W),
                                stride=(self.kernel_size // 2, self.kernel_size // 2))
        self.fold = nn.Fold(output_size=(self.H, self.W),
                            kernel_size=[self.kernel_size] * 2,
                            padding=(self.padding_H, self.padding_W),
                            stride=(self.kernel_size // 2, self.kernel_size // 2))
        
        self.w_s = {}
        self.w_t = {}
        
        for _ in range(self.C_out * self.T_out):
            self.w_s.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })
        
        for _ in range(self.C_out * self.T_out):
            self.w_t.update({
                f"w_{_}": nn.Parameter(torch.randn(self.T_in, self.C_in, self.kernel_size, self.kernel_size),
                                    requires_grad=True),
                f"bias_{_}": nn.Parameter(torch.zeros((self.kernel_size, self.kernel_size)),
                                    requires_grad=True),
            })
        
        self.w_s = nn.ParameterDict(self.w_s)
        self.w_t = nn.ParameterDict(self.w_t)
        self.decompsition = series_decomp(self.dec_kernel_size)
        
        if self.use_emb:
            self.emb = nn.Embedding(self.emb_len, self.kernel_size ** 2)
            self.features_to_emb = nn.Linear(self.kernel_size ** 2, self.emb_len)
        
        
        
    def forward(self, x):
        B, T_in, H, W, C_in = x.shape
        x = x.permute(0, 1, 4, 2, 3) # [B, T_in, C_in, H, W]
        x = x.reshape(B, T_in, -1)
        
        decomp = self.decompsition(x)
        res = []
        for i, weights in enumerate([self.w_s, self.w_t]):
            x = decomp[i]
            x = x.reshape(B, T_in, C_in, H, W)
            x = x.reshape(B, -1, H, W)
            x = self.unfold(x) # [B, ..., L]
            L = x.shape[-1]
            x = x.reshape(B, T_in, C_in, self.kernel_size, self.kernel_size, L).permute(0, 5, 1, 2, 3, 4) # [B, L, T_in, C_in, k, k]
            
            xx = []
            for i in range(self.C_out * self.T_out):
                w, bias = weights[f"w_{i}"], weights[f"bias_{i}"]
                xx.append((x * w).sum(dim=(2, 3)) + bias)  # [B, L, k, k]
            
            x = torch.stack(xx, dim=1) # [B, C_out * T_out, L, k, k]
            res.append(x)
        
        x = torch.stack(res, dim=0).sum(dim=0)
        
        if self.use_emb:
            x = F.softmax(self.features_to_emb(x.flatten(start_dim=-2)) / self.temperature, dim=-1) # [B, C_out * T_out, L, N_emb]
            x = x[..., None] * self.emb.weight[None, None, None, ...]
            x = x.sum(dim=-2)
            x = x.reshape(*(tuple(x.shape[:-1]) + (self.kernel_size, self.kernel_size,))) # [B, C_out * T_out, L, k, k]
        
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, L)
        x = self.fold(x) # [B, C_out * T_out, H, W]
        x = x.reshape(B, self.T_out, self.C_out, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x # [B, T, H, W, С_out]
