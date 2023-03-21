#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Clabat
"""



import torch
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

class WindSampling(torch.utils.data.Dataset):
    def __init__(self, data_proj_numpy, mask, step_per_epoch, device):
        self.device = device
        torch_data = torch.from_numpy(data_proj_numpy).to(self.device)
        self.X = torch.clone(torch_data)
        self.X[mask,:] = torch.mean(self.X[mask == 0,:])
        self.y = torch.clone(torch_data)
        self.length = data_proj_numpy.shape[0]
        self.mask = torch.from_numpy(mask).to(self.device)
        self.step_per_epoch = step_per_epoch
    def __getitem__(self, index):
        return self.X, self.y, self.mask

    def __len__(self):
        # Returns length
        return self.step_per_epoch
    
class TorusDenoising(torch.utils.data.Dataset):
    def __init__(self, data_clean,data_noisy, step_per_epoch, device):
        self.device = device
        torch_data_clean = torch.from_numpy(data_clean).to(self.device)
        torch_data_noisy = torch.from_numpy(data_noisy).to(self.device)
        self.X = torch.clone(torch_data_noisy)
        self.y = torch.clone(torch_data_clean)
        self.step_per_epoch = step_per_epoch

    def __getitem__(self, index):
        return self.X, self.y

    def __len__(self):
        # Returns length
        return self.step_per_epoch

class WindPrediction(torch.utils.data.Dataset):
    def __init__(self, data_proj_numpy, time_window, device):
        self.device = device
        self.X = torch.from_numpy(data_proj_numpy).to(self.device)
        self.time_window = time_window

    def __getitem__(self, idx):
        x = self.X[idx:(idx+self.time_window),:,:]
        y = self.X[(idx+self.time_window):(idx+2*self.time_window),:,:]
        return x,y

    def __len__(self):
        return self.X.shape[0]-2*self.time_window

