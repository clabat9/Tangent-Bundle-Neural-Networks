#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudio Battiloro
"""
#import warnings
#warnings.filterwarnings("ignore") to suppress warnings
import sys
import pickle as pkl
import numpy.ma as ma
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from architecture import TNN,MNN
from data_util import TorusDenoising
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
import numpy as np
from utils import get_laplacians, project_data, topk
from tensorboard import program
import webbrowser

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

open_tb = 0

tnn_or_mnn = sys.argv[1]
#%% Synthetic Data Generation ()
res = 100 # The torus will be sampled on a regular grid of res^2 points
p = 3 
d = 2

# Torus sampling 
phi = np.linspace(0, 2*np.pi, res)
theta = np.linspace(0, 2*np.pi, res)
phi, theta = np.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()
r = .1
b = .3
x = np.expand_dims((r*np.cos(theta)+b)*np.cos(phi),1)
y = np.expand_dims((r*np.cos(theta)+b)*np.sin(phi),1)
z = np.expand_dims((r*np.sin(theta)),1)
coord_max = np.concatenate((x,y,z),1)

# Smooth Tangent vector field on Torus
X = np.expand_dims(-np.sin(theta),1)
Y = np.expand_dims(np.cos(theta),1)
Z = np.expand_dims(np.zeros(len(theta)),1)
data_all = np.concatenate((X,Y,Z),1)
n_max = data_all.shape[0]

# MonteCarlo Simulation Parameters
outer_num_rel = 8
inner_num_rel = 8
num_avg_samples_coll = [400]#[100, 200,150,300,450] # 1st Sampling: to reduce the initial dimensionality -> let us assume that the complete dataset is the complete manifold
noise_sds_coll = [7e-2, 1e-1, 3e-1] # 2nd Sampling: the actual mask

# Architecture Parameters
in_features = int(data_all.shape[1]/d) if tnn_or_mnn == 'tnn' or tnn_or_mnn == 'ftnn' else data_all.shape[1]
features = [8,4,1]
if tnn_or_mnn == "mnn" or tnn_or_mnn == "fmnn":
    features[-1] = features[-1]*p
dense = []
lr = 1e-3
if tnn_or_mnn == "fmnn" or tnn_or_mnn == "ftnn":
    readout_sigma = linear_act()
    sigma = linear_act()#torch.nn.ReLU()
else:
    readout_sigma = linear_act()
    sigma = torch.nn.Tanh()#torch.nn.ReLU()
kappa = [2]*len(features)
loss_function = torch.nn.MSELoss(reduction = 'sum')
weight_decay = 0.0
step_per_epoch = 100
max_epochs = 500

# Logging Parameters
string = "Torus_Denoising" # Experiment Name
save_dir_ = '/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results' # Saving Directory
# Sheaf Laplacian Parameters
epsilon_pca = .8#.2#n**(-2/(true_d+1))# n^{-2/(d+1)}
epsilon = .5
gamma = .8
open_tb = 0 # Opens TensorBoard in the default browser
tracking_address = '/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string # TB Tracking Folder

for num_avg_samples in num_avg_samples_coll:
    print()                                                                  
    print("Testing with average number of points: "+str(num_avg_samples))
    print()
    p_samp = num_avg_samples/n_max
    for noise_sd in noise_sds_coll:
        print()                                                                  
        print("Testing with noise variance: "+str(noise_sd))
        print()
        min_mse = np.zeros((outer_num_rel,inner_num_rel))   
        # 1st Sampling (to reduce the initial dimensionality and ensure random sampling -> let us assume that the complete dataset is the complete manifold)
        for outer_rel in range(outer_num_rel):
            sampling_set = np.random.binomial(1, p_samp, n_max)>0
            data = data_all[sampling_set,:]
            coord = coord_max[sampling_set,:]
            n = coord.shape[0]
            print()
            print("Outer Realization number "+str(outer_rel)+": "+str(n) + " samples!")
            print()
            for inner_rel in range(inner_num_rel): 
                print()                                                                  
                print("Inner Realization number "+str(inner_rel))
                print()
                # Adding Noise to Data
                data_noisy = data + np.random.normal(0.0, noise_sd, size=data.shape) #np.random.normal(0.0, sigma_noise, size=(1,))*
                # Build the Sheaf Laplacian
                if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                    Delta_n_numpy, S,W,O_i_collection, _, B_i_collection = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                    data_proj = project_data(data, O_i_collection) 
                    data_proj_noisy = project_data(data_noisy, O_i_collection) 
                else:
                    Delta_n_numpy = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                    data_proj = data
                    data_proj_noisy = data_noisy
                # Laplacian Replicating for each layer
                Delta_n = len(features)*[torch.from_numpy(Delta_n_numpy)]
                # Net Parameters
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
                # Data and Net Instantiating
                data_torch = TorusDenoising(data_proj,data_proj_noisy, step_per_epoch, device)
                if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                    net = TNN(**hparams).to(device)
                else:
                    net = MNN(**hparams).to(device)
                train_loader = \
                    torch.utils.data.DataLoader(
                        data_torch, batch_size=None, batch_sampler=None, shuffle=True, num_workers=0)
                logger = pl.loggers.TensorBoardLogger(name=string, save_dir='/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results')
                early_stop_callback = EarlyStopping(monitor="train_mse", min_delta=1e-6, patience=5, verbose=False, mode="min")
                trainer = pl.Trainer(max_epochs=max_epochs,logger = logger, log_every_n_steps= 1,
                                    accelerator='gpu', devices=1, auto_select_gpus=False, callbacks=[early_stop_callback])
                trainer.fit(net, train_loader)
                min_mse[outer_rel,inner_rel] = net.min_mse_train
        min_mse = min_mse[~np.isnan(min_mse).any(axis=1), :] # Removes eventual corrupted runs (divergent, outliers, etc...)
        to_delete = topk(min_mse,2) # Removes the worst 2 (redundant in case of divergent but not NaN runs with results_aggregator.py)
        mask = np.logical_or(min_mse == to_delete[0], min_mse == to_delete[1])
        min_mse = ma.masked_array(min_mse, mask = mask)
        try:
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'rb') as file:
                mse_dic = pkl.load(file)
            print("Results file already exisisting... Updating!")
            try:
                tmp = mse_dic["avg_points"+str(num_avg_samples)]
                tmp["noise_sd"+str(noise_sd)] = {"avg_mse":min_mse.mean(),"std_mse": min_mse.std(), "complete_coll": min_mse}
                mse_dic["avg_points"+str(num_avg_samples)] = tmp
            except:
                mse_dic["avg_points"+str(num_avg_samples)] = {"noise_sd"+str(noise_sd):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'wb') as file:
                pkl.dump(mse_dic, file)   
            print("Updated!")
        except:
            print("Results file not found... Creating!")
            mse_dic = {"avg_points"+str(num_avg_samples):{"noise_sd"+str(noise_sd):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}}
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'wb') as file:
                pkl.dump(mse_dic, file)
print(mse_dic)
# Tensor Board Monitoring
if open_tb:
    tracking_address = '/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    webbrowser.open_new(url) 
    input("Press Enter to Exit")



    