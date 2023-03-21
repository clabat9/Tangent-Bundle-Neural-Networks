#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Claudio Battiloro
"""
import warnings
#warnings.filterwarnings("ignore") to suppress warnings
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from architecture import RTNN, RMNN
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
import numpy as np
from utils import get_laplacians, project_data, topk
from data_util import WindPrediction
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
# Train
with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/Journal_repo/data/windfields/data2016.pkl', 'rb') as file:
    data_all = pkl.load(file)
#Test
with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/Journal_repo/data/windfieldsdata2017.pkl', 'rb') as file:
    data_all_test = pkl.load(file)

# Crop the data (the whole year will be slow)
how_many_days = 250 #250 ok
data_all = data_all[:how_many_days,:,:]
data_all_test = data_all_test[:how_many_days,:,:]

# Normalize the coordinates by the nominal earth radius to avoid numerical instability and 
R = 6356.8
data_all[:,:,:3] = data_all[:,:,:3]/R
data_all_test[:,:,:3] = data_all_test[:,:,:3]/R
# Scale the data for numerical stability
data_all[:,:,3:] = data_all[:,:,3:]/(np.max(data_all[:,:,3:])-np.min(data_all[:,:,3:])) #-np.min(data_all[:,:,3:]))
data_all_test[:,:,3:] = (data_all_test[:,:,3:]-np.min(data_all_test[:,:,3:]))/(np.max(data_all_test[:,:,3:])-np.min(data_all_test[:,:,3:]))
n_max = data_all.shape[1]
p = 3 # Ambient Space Dimension
d = 2 # Manifold Dimension

# MonteCarlo Simulation Parameters
outer_num_rel = 8
num_avg_samples_coll = [100, 200,300, 400] #  1st Sampling: to reduce the initial dimensionality -> let us assume that the complete dataset is the complete manifold
time_window_coll =  [20,50,80] # 2nd Sampling: the actual mask

# Architecture Parameters
in_features = int((data_all.shape[2]-p)/d) if tnn_or_mnn == 'tnn' or tnn_or_mnn == 'ftnn' else data_all.shape[2]-p # The last number is the output features. The lenght is the number of layers
n_layers = 3 
in_features = [in_features]*n_layers
dense = []
lr = 1e-3
if tnn_or_mnn == "fmnn" or tnn_or_mnn == "ftnn":
    sigma = linear_act()
else:
    sigma = torch.nn.Tanh()
kappa = [2]*n_layers 
num_epochs = 70
batch_size_ = 1
loss_function = torch.nn.MSELoss(reduction = 'sum')
weight_decay = 1e-3

# Logging Parameters
string = "Wind_Prediction" # Experiment Name
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
    # 1st Sampling (to reduce the initial dimensionality -> let us assume that the complete dataset is the complete manifold)
    p_samp = num_avg_samples/n_max
    for time_window in time_window_coll:
        print()                                                                  
        print("Testing with Time Window: "+str(time_window))
        print()
        min_mse = np.zeros((outer_num_rel,))
        # 1st Sampling
        for outer_rel in range(outer_num_rel):
            sampling_set = np.random.binomial(1, p_samp, n_max)>0
            data = data_all[:,sampling_set,-2:]
            data_test = data_all_test[:,sampling_set,-2:]
            coord = data_all[0,sampling_set,:3]
            n = coord.shape[0]
            if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                Delta_n_numpy, S,W,O_i_collection, d_hat, B_i_collection = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                data_proj = np.array([project_data(data[el,:,:], O_i_collection) for el in range(data.shape[0])])
                data_proj_test = np.array([project_data(data_test[el,:,:], O_i_collection) for el in range(data_test.shape[0])])
            if tnn_or_mnn == "mnn" or tnn_or_mnn == "fmnn":
                Delta_n_numpy = get_laplacians(coord,epsilon,epsilon_pca,gamma, tnn_or_mnn)
                data_proj = data
                data_proj_test = data_test
            if tnn_or_mnn == "rnn":
                Delta_n_numpy = np.eye(n)
                data_proj = data
                data_proj_test = data_test
            # Normalize Laplacians
            #[lambdas,_] = np.linalg.eigh(Delta_n_numpy)
            #Delta_n_numpy = Delta_n_numpy/np.max(np.real(lambdas))
            Delta_n = len(in_features)*[torch.from_numpy(Delta_n_numpy)]
            data_torch = WindPrediction(data_proj,time_window,device)
            data_torch_val = WindPrediction(data_proj_test,time_window,device)
            hparams ={'in_features': in_features,\
                    'L': Delta_n,\
                    'lr': lr,\
                    'weight_decay': weight_decay,\
                    'sigma': sigma,\
                    'kappa': kappa,\
                    'time_window': time_window,\
                    'loss_function': loss_function,\
                    'device': device}
            if tnn_or_mnn == "tnn" or tnn_or_mnn == "ftnn":
                net = RTNN(**hparams).to(device)
            else:
                net = RMNN(**hparams).to(device)
            train_loader = \
                torch.utils.data.DataLoader(
                    data_torch, batch_size=batch_size_, batch_sampler=None, shuffle=True, num_workers=0)
            val_loader =\
                torch.utils.data.DataLoader(
                    data_torch_val, batch_size=how_many_days-2*time_window, batch_sampler=None, shuffle=False,  num_workers=0)
            logger = pl.loggers.TensorBoardLogger(name=string, save_dir=save_dir_)
            early_stop_callback = EarlyStopping(monitor="test_mse", min_delta=1e-6, patience=5, verbose=False, mode="min")
            trainer = pl.Trainer(max_epochs=num_epochs,logger = logger, log_every_n_steps= 1,
                                accelerator='gpu', devices=1, auto_select_gpus=False, callbacks=[early_stop_callback])#,check_val_every_n_epoch=int(num_epochs/10)
            trainer.fit(net, train_loader,val_loader)
            min_mse[outer_rel] = net.min_mse_val
        min_mse = min_mse[~np.isnan(min_mse)] # Removes eventual corrupted runs (divergent, outliers, etc...)
        #min_mse = min_mse[min_mse < 1.5]
        to_delete = topk(min_mse,2)
        mask = np.logical_or(min_mse == to_delete[0], min_mse == to_delete[1])
        min_mse = ma.masked_array(min_mse, mask = mask)
        try:
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'rb') as file:
                mse_dic = pkl.load(file)
            print("Results file already exisisting... Updating!")
            try:
                tmp = mse_dic["avg_points"+str(num_avg_samples)]
                tmp["time_window"+str(time_window)] = {"avg_mse":min_mse.mean(),"std_mse": min_mse.std(), "complete_coll": min_mse}
                mse_dic["avg_points"+str(num_avg_samples)] = tmp
            except:
                mse_dic["avg_points"+str(num_avg_samples)] = {"time_window"+str(time_window):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}
            with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/results/'+string+'/res_'+tnn_or_mnn+'.pkl', 'wb') as file:
                pkl.dump(mse_dic, file)   
            print("Updated!")
        except:
            print("Results file not found... Creating!")
            mse_dic = {"avg_points"+str(num_avg_samples):{"time_window"+str(time_window):{"avg_mse":min_mse.mean(),"std_mse":min_mse.std(), "complete_coll": min_mse}}}
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



    