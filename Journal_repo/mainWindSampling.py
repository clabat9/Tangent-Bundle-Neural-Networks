#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudio Battiloro
"""
#import warnings
#warnings.filterwarnings("ignore") to suppress warnings
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from architecture import TNN,MNN
from data_util import WindSampling
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
import numpy as np
from utils import get_laplacians, project_data, topk
from tensorboard import program
import webbrowser
import numpy.ma as ma
import pickle as pkl

# Set Seeds
np.random.seed(0)
pl.seed_everything(0)
# Custom activation function: Identity activation
class linear_act(torch.nn.Module):
            def __init__(self):
                super(linear_act, self).__init__()
            def forward(self, x):
                return x
# Open Tensorboard            
open_tb = 0
# Select Architecture
tnn_or_mnn = sys.argv[1]

#%% Data Importing
with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/data/windfields/data2016.pkl', 'rb') as file:
    data_all = pkl.load(file)
# In the sampling and reconstrunction experiment we take a single day
data_all = data_all[0,:,:]
# Normalize the coordinates by the nominal earth radius to avoid numerical instability
R = 6356.8
data_all[:,:3] = data_all[:,:3]/R
# Scale the data to facilitate training
data_all[:,3:] = (data_all[:,3:])/(np.max(data_all[:,3:])-np.min(data_all[:,3:]))#-np.min(data_all[:,3:])
n_max = data_all.shape[0]
p = 3 # Ambient Space Dimension
d = 2 # Manifold Dimension

# MonteCarlo Simulation Parameters
outer_num_rel = 8
inner_num_rel = 8
num_avg_samples_coll = [100, 150, 200, 300, 400] # 1st Sampling: to reduce the initial dimensionality -> let us assume that the complete dataset is the complete manifold
avg_sample_pctg_coll = [.5, .7, .9] # 2nd Sampling: the actual mask

# Architecture Parameters
in_features = int((data_all.shape[1]-p)/d) if tnn_or_mnn == 'tnn' or tnn_or_mnn == 'ftnn' else data_all.shape[1]-p
features = [8,4,1] # The last number is the output features. The lenght is the number of layers
if tnn_or_mnn == "mnn" or tnn_or_mnn == "fmnn":
    features[-1] = features[-1]*d
dense = []
lr = 4e-4
if tnn_or_mnn == "fmnn" or tnn_or_mnn == "ftnn":
    readout_sigma =linear_act()# torch.nn.Tanh()
    sigma = linear_act()
else:
    readout_sigma =linear_act()# torch.nn.Tanh()
    sigma = torch.nn.Tanh()
kappa = [2]*len(features)
loss_function = torch.nn.MSELoss(reduction = 'sum')#reduction = 'mean'
weight_decay = 1e-3
max_epochs = 500
opt_step_per_epoch = 100 # Total optimization steps = step_per_epoch*max_epochs, division useful for loggin

# Logging Parameters
string = "Wind_Reconstruction" # Experiment Name
save_dir_ = '/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results' # Saving Directory
# Sheaf Laplacian Parameters
epsilon_pca = .8#.2#n**(-2/(true_d+1))# n^{-2/(d+1)}
gamma = .8
epsilon = .5
open_tb = 0 # Opens TensorBoard in the default browser
tracking_address = '/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string # TB Tracking Folder


for num_avg_samples in num_avg_samples_coll:
    print()                                                                  
    print("Testing with average number of points: "+str(num_avg_samples))
    print()
    p_samp = num_avg_samples/n_max   
    for avg_sample_pctg in avg_sample_pctg_coll:
        print()                                                                  
        print("Testing with masking propability: "+str(avg_sample_pctg))
        print()
        min_mse = np.zeros((outer_num_rel,inner_num_rel))
        # 1st Sampling
        for outer_rel in range(outer_num_rel):
            sampling_set = np.random.binomial(1, p_samp, n_max)>0
            data = data_all[sampling_set,-2:]
            coord = data_all[sampling_set,:3]
            n = coord.shape[0]
            print()
            print("Outer Realization number "+str(outer_rel)+": "+str(n) + " samples!")
            print()
            # Build the Sheaf Laplacian 
            if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                Delta_n_numpy, S,W,O_i_collection, d_hat, B_i_collection = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                data_proj = project_data(data, O_i_collection)
            else:
                Delta_n_numpy = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                data_proj = data
            
            # Laplacian Replicating for each layer
            Delta_n = len(features)*[torch.from_numpy(Delta_n_numpy)]
            # Net Prameters
            hparams ={'in_features': in_features,\
                      'L': Delta_n,\
                      'features': features,\
                      'lr': lr,\
                      'weight_decay': weight_decay,\
                      'sigma': sigma,\
                      'readout_sigma': readout_sigma,\
                      'kappa': kappa,\
                      'n': n,\
                      'loss_function': loss_function,\
                      'device': device}
            for inner_rel in range(inner_num_rel):    
                # 2nd Sampling 
                bern = np.random.binomial(1, avg_sample_pctg, n)
                if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                    mask = np.kron(np.ones((1,d)),np.expand_dims(bern,1)).flatten()>0
                else:
                    mask = bern > 0
                val_mask = mask==0  
                print()                                                                  
                print("Inner Realization number "+str(inner_rel)+": "+str(sum(mask)) + " masked points!")
                print()
                # Data and Net Instantiating
                data_torch = WindSampling(data_proj,mask,opt_step_per_epoch,device)
                data_torch_val = WindSampling(data_proj,val_mask,1,device)
                if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                    net = TNN(**hparams).to(device)
                else:
                    net = MNN(**hparams).to(device)
                train_loader = \
                    torch.utils.data.DataLoader(
                        data_torch, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)
                val_loader =\
                     torch.utils.data.DataLoader(
                        data_torch_val, batch_size=None, batch_sampler=None, shuffle=False,  num_workers=0)
                
                logger = pl.loggers.TensorBoardLogger(name=string, save_dir=save_dir_)
                early_stop_callback = EarlyStopping(monitor="test_mse", min_delta=1e-6, patience=5, verbose=False, mode="min")
                trainer = pl.Trainer(max_epochs=max_epochs,logger = logger, log_every_n_steps= 1,
                                     accelerator='gpu', devices=1, auto_select_gpus=False, callbacks=[early_stop_callback])
                trainer.fit(net, train_loader, val_loader)
                min_mse[outer_rel,inner_rel] = net.min_mse_val
        min_mse = min_mse[~np.isnan(min_mse).any(axis=1), :] # Removes eventual corrupted runs (divergent, outliers, etc...)
        to_delete = topk(min_mse,2)  # Removes the worst 2 (redundant in case of divergent but not NaN runs with results_aggregator.py)
        mask = np.logical_or(min_mse == to_delete[0], min_mse == to_delete[1])
        min_mse = ma.masked_array(min_mse, mask = mask)
        try:
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'rb') as file:
                mse_dic = pkl.load(file)
            print("Results file already exisisting... Updating!")
            try:
                tmp = mse_dic["avg_points"+str(num_avg_samples)]
                tmp["avg_mask"+str(avg_sample_pctg)] = {"avg_mse":min_mse.mean(),"std_mse": min_mse.std(), "complete_coll": min_mse}
                mse_dic["avg_points"+str(num_avg_samples)] = tmp
            except:
                mse_dic["avg_points"+str(num_avg_samples)] = {"avg_mask"+str(avg_sample_pctg):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'wb') as file:
                pkl.dump(mse_dic, file)   
            print("Updated!")
        except:
            print("Results file not found... Creating!")
            mse_dic = {"avg_points"+str(num_avg_samples):{"avg_mask"+str(avg_sample_pctg):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}}
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'wb') as file:
                pkl.dump(mse_dic, file)
print(mse_dic)
# Tensor Board Monitoring
if open_tb:
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    webbrowser.open_new(url) 
    input("Press Enter to Exit")

"""
print("Minimum TEST MSE: "+ str(net.min_mse_val))
print("Misc. Metrics:")
print(trainer.callback_metrics)
"""


    