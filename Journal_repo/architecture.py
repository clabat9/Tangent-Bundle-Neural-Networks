#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudio Battiloro
"""

#import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from layers import GNNLayer, RGNNLayer
import numpy as np

# Tangent Bundle Neural Network
class TNN(pl.LightningModule):

    def __init__(self, in_features, L, features,
                 lr, weight_decay, sigma, readout_sigma, kappa, n,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(TNN, self).__init__()
        self.lr = lr
        self.n = n
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.readout_sigma = readout_sigma
        in_features = [in_features] + [features[l]
                                       for l in range(len(features))]
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.readout_sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.sigma}
            tnn_layer = GNNLayer(**hparams).to(device)
            ops.extend([tnn_layer])
        self.tnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.tnn(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
            loss = self.loss_fn(y_hat_trim, y_trim)
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
            loss = self.loss_fn(y_hat_trim, x)
        self.mse_train = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
        loss = self.loss_fn(y_hat_trim, y_trim)
        self.mse_val = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}

    
# Manifold Neural Network (redundant in our experiments, just to keep things separated)
class MNN(pl.LightningModule):

    def __init__(self, in_features, L, features,
                 lr, weight_decay, sigma, readout_sigma, kappa, n,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(MNN, self).__init__()
        self.lr = lr
        self.n = n
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.readout_sigma = readout_sigma
        in_features = [in_features] + [features[l]
                                       for l in range(len(features))]
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.readout_sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "sigma": self.sigma}
            mnn_layer = GNNLayer(**hparams).to(device)
            ops.extend([mnn_layer])
        self.mnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.mnn(x)

    def training_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
            loss = self.loss_fn(y_hat_trim, y_trim)
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
            loss = self.loss_fn(y_hat_trim, x)
        self.mse_train = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y, mask = batch
            y_hat = self(x)
            y_trim = y[mask, :]
            y_hat_trim = y_hat[mask, :]
        except:
            x, y  = batch
            y_hat = self(x)
            y_trim = y
            y_hat_trim = y_hat
        loss = self.loss_fn(y_hat_trim, y_trim)
        self.mse_val = ((y_trim - y_hat_trim).square()).sum() / self.n
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}

# Recurrent Tangent Bundle Neural Network
class RTNN(pl.LightningModule):

    def __init__(self, in_features, time_window, L,
                 lr, weight_decay, sigma, kappa,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        time_window: Prediction time window
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(RTNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            rtnn_layer = RGNNLayer(**hparams).to(device)
            ops.extend([rtnn_layer])
        self.rtnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.rtnn(x)

    def training_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device).double()
        loss = self.loss_fn(xT_hat, xT)
        self.mse_train = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device)
        loss = self.loss_fn(xT_hat, xT)
        self.mse_val = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}


# Recurrent Manifold Neural Network (redundant in our experiments, just to keep things separated)
class RMNN(pl.LightningModule):

    def __init__(self, in_features, time_window, L,
                 lr, weight_decay, sigma, kappa,
                 loss_function, device):
        """
        Parameters
        ----------
        in_features : Input features
        time_window: Prediction time window
        L : List of Shift Operators (one per layer)
        features : List of hidden features
        lr: optimizer's learning rate
        weight_decay: Weight decay multiplier
        sigma : Non-linearity
        readout_sigma: Non-linearity of the last layer
        kappa : List of filters order
        n: Number of manifold points
        loss_function: Loss function
        device : device
        """
        super(RMNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.L = [l.to(device) for l in L]
        ops = []
        self.sigma = sigma
        self.N_layers = len(in_features)
        self.min_mse_train = 1e20
        self.min_mse_val = 1e20
        self.loss_fn = loss_function
        for l in range(self.N_layers-1):
            if l == self.N_layers-2:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            else:
                hparams = {"F_in": in_features[l],
                           "F_out": in_features[l+1],
                           "L": self.L[l],
                           "kappa": kappa[l],
                           "device": device,
                           "time_window": time_window,
                           "sigma": self.sigma}
            simplicial_attention_layer = RGNNLayer(**hparams).to(device)
            ops.extend([simplicial_attention_layer])
        self.rmnn = nn.Sequential(*ops)

    def forward(self, x):
        return self.rmnn(x)

    def training_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device).double()
        loss = self.loss_fn(xT_hat, xT)
        self.mse_train = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_train = min(self.mse_train, self.min_mse_train)
        self.log('train_mse', self.mse_train, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xt, xT = batch
        xT_hat = self(xt).to(self.device)
        loss = self.loss_fn(xT_hat, xT)
        self.mse_val = ((xT - xT_hat).square()).sum() / np.prod(xT.shape)
        self.min_mse_val = min(self.mse_val, self.min_mse_val)
        self.log('test_mse', self.mse_val, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log('test_loss', loss.item(), on_step=True,
                 on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outs):
        pass

    def validation_epoch_end(self, outs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'train_loss'}
