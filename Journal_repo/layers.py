#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudio Battiloro
"""

import torch
import torch.nn as nn


# Graph Convolutional Neural Network Layer
class GNNLayer(nn.Module):
    
    def __init__(self, F_in, F_out, L, kappa,device, sigma):
        """
        Parameters
        ----------
        F_in: Numer of input signals
        F_out: Numer of outpu signals
        L: Shift Operator
        kappa: Filters order
        device: Device
        sigma: non-linearity
        """
        super(GNNLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.sigma = sigma
        self.L = L
        if self.L.type() == 'torch.cuda.DoubleTensor':
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device).double())
        else:
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device))
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        nn.init.xavier_uniform_(self.b.data, gain=gain)

    def forward(self, x):
        alpha_zero = torch.clone(self.L)
        data = torch.clone(x)
        alpha_k = torch.clone(alpha_zero)
        try:
            z_i = alpha_k @ torch.clone(data  @ self.W[0])
        except:
            alpha_k = alpha_k.to(data.device)
            z_i = alpha_k @ torch.clone(data  @ self.W[0])
        for k in range(1, self.K):
            alpha_k = alpha_k @ alpha_zero
            z_i += alpha_k  @  data  @ self.W[k]
        out = self.sigma(z_i)
        return out


# Graph Convolutional Neural Network Layer
class RGNNLayer(nn.Module):

    def __init__(self, F_in, F_out, L, kappa,device, sigma, time_window):
        """
        Parameters
        ----------
        F_in: Numer of input signals 
        F_out: Numer of outpu signals 
        L: Shift Operator
        kappa: Filters order 
        device: Device
        sigma: non-linearity 
        time_window: Prediction time window 
        """
        super(RGNNLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.sigma = sigma
        self.time_window = time_window
        self.L = L
        if self.L.type() == 'torch.cuda.DoubleTensor':
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.H = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device).double())
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device).double())
        else:
            self.W = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.H = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)).to(device))
            self.b = nn.Parameter(torch.empty(size=(1, 1)).to(device))
        self.reset_parameters()
        self.device = device

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W.data, gain=gain)
        nn.init.xavier_uniform_(self.H.data, gain=gain)
        nn.init.xavier_uniform_(self.b.data, gain=gain)

    def forward(self, x):
        # x is batch_sizeXhow_many_time_slotsXnumber_of_nodesXnumber_of_features
        alpha_zero = torch.clone(self.L)
        data = torch.clone(x).to(self.device).double()
        out = torch.zeros(data.shape)
        for data_point in range(data.shape[0]): # Batch Loop: inefficient, can be improved with PyTorch Geometric
            hidden_state = torch.zeros(data.shape[2:])
            for t in range(self.time_window): # Time Loop 
                alpha_k = torch.clone(alpha_zero)
                hidden_state = hidden_state.to(self.device).double()
                try:
                    z_i = alpha_k @ torch.clone(data[data_point,t,:,:]  @ self.W[0]) + alpha_k @ torch.clone(hidden_state  @ self.H[0])
                except:
                    alpha_k = alpha_k.to(data.device)
                    z_i = alpha_k @ torch.clone(data[data_point,t,:,:]  @ self.W[0]) + alpha_k @ torch.clone(hidden_state  @ self.H[0])
                for k in range(1, self.K):
                    alpha_k = alpha_k @ alpha_zero
                    z_i += alpha_k  @  data[data_point,t,:,:]  @ self.W[k] + alpha_k @ torch.clone(hidden_state  @ self.H[k])
                hidden_state = self.sigma(z_i)
                out[data_point,t,:,:] = hidden_state
        return out