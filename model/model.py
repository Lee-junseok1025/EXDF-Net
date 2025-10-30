
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from timm.layers import trunc_normal_

from typing import Dict, Iterable, Optional, Tuple


import numpy as np
from torch import Tensor
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
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


class TemporalEmbed(nn.Module):
    def __init__(self,embed_dim):
        super(TemporalEmbed,self).__init__()
        seasonal_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        self.seasonal_embed = nn.Embedding(seasonal_size,embed_dim)
        self.hour_embed = nn.Embedding(hour_size,embed_dim)
        self.week_embed = nn.Embedding(weekday_size,embed_dim)
        self.day_embed = nn.Embedding(day_size,embed_dim)
        self.month_embed = nn.Embedding(month_size,embed_dim)
    
    def forward(self,x):
        # minute = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        if x.dim() > 2:
            seasonal = self.seasonal_embed(x[:,:,4])
            hour = self.hour_embed(x[:,:,3])
            week = self.week_embed(x[:,:,2])
            day = self.day_embed(x[:,:,1])
            month = self.month_embed(x[:,:,0])
        else:
            seasonal = self.seasonal_embed(x[:,4])
            hour = self.hour_embed(x[:,3])
            week = self.week_embed(x[:,2])
            day = self.day_embed(x[:,1])
            month = self.month_embed(x[:,0])
        return  seasonal + hour + week + day + month

class SwishGLU(nn.Module):
    def __init__(self,embed_dim,expand):
        super(SwishGLU,self).__init__()
        dim_inner = embed_dim * expand
        self.fc1 = nn.Linear(embed_dim,dim_inner)
        self.fc2 = nn.Linear(embed_dim,dim_inner)
        self.fc3 = nn.Linear(dim_inner,embed_dim)

    def forward(self,x,tdf=None):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)

class Decoder(nn.Module):
    def __init__(
        self,
        in_features,
        embed_dim,
        seq_len,
        pred_len,
        individual
    ):
        super(Decoder,self).__init__()
        self.in_features = in_features
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.individual = individual
        self.channel = in_features

        kernel_size = 7 if pred_len < 95 else 25
        self.decompsition = series_decomp(kernel_size)

        if self.individual:
            self.seasonal_linear = nn.ModuleList()
            self.trend_linear = nn.ModuleList()
            
            for i in range(self.channel):
                self.seasonal_linear.append(nn.Linear(seq_len,pred_len))
                self.trend_linear.append(nn.Linear(seq_len,pred_len))
        else:
            self.seasonal_linear = nn.Linear(seq_len,pred_len)
            self.trend_linear = nn.Linear(seq_len,pred_len)
        # self.seasonal_linear.weight = nn.Parameter((1/seq_len)*torch.ones([pred_len,seq_len]))
        self.x_embed = TemporalEmbed(self.channel)
        self.y_embed = TemporalEmbed(self.channel)
        self.x_token_proj = nn.Linear(self.channel,self.channel)
        self.y_token_proj = nn.Linear(self.channel,self.channel) 
        self.proj = nn.Linear(pred_len,pred_len)
        
    def forward(
        self,
        dec_input: Tensor,
        x_token: Tensor,
        y_token: Tensor, 
    ):
        
    
        last_seq = dec_input[:,-1:,:]
        dec_input = dec_input - last_seq
        embed_x = self.x_token_proj(self.x_embed(x_token))
        embed_y = self.y_token_proj(self.y_embed(y_token)).permute(0,2,1)

        
        seasonal_init, trend_init = self.decompsition(dec_input)
        seasonal_init = seasonal_init.permute(0,2,1)
        trend_init = (trend_init + embed_x).permute(0,2,1)

        if self.individual:
            seasonal_out = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_out = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)

            for i in range(self.channel):
                seasonal_out[:,i,:] = self.seasonal_linear[i](seasonal_init[:,i,:])
                trend_out[:,i,:] = self.seasonal_linear[i](trend_init[:,i,:])
        else:
            seasonal_out = self.seasonal_linear(seasonal_init)
            trend_out = self.trend_linear(trend_init)
        
        output = seasonal_out + trend_out + embed_y
        output = self.proj(output).permute(0,2,1)
        return output + last_seq

class Model(nn.Module):
    def __init__(
        self,
        in_features,
        embed_dim,
        seq_len,
        label_len,
        pred_len,
        individual
    ):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.decoder = Decoder(
            in_features=in_features,
            embed_dim=embed_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual
        )
        
        # self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=.02)
    
    def forward(self, x, y, x_token, y_token)-> Dict[str, torch.Tensor]:
        # x_ = torch.cat([x,y[:, -self.pred_len:, :]], dim=1)
        y_token = y_token[:, -self.pred_len:, :]
        # enc_x = self.encoder(tdf=tdf,x_token=x_token)
        x = self.decoder(dec_input=x,x_token=x_token,y_token=y_token)
        return x
