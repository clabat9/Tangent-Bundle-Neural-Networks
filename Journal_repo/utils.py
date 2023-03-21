#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Clabat
"""

import torch

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np
from scipy.linalg import expm
import sys

# %% Sheaf Laplacian Utils
def compute_neighbours(data,epsilon,epsilon_pca, option = 'mean_shift'):
    n = data.shape[0]
    X_i_collection = []
    neighbours_collection = np.zeros((n,n))
    distances_collection = []
    complete_distance_collection = []
    for point in range(n):
        x_i = data[point,:]
        x_i_dists = np.sum((x_i - data)**2,1)**.5
        neigh = (x_i_dists > 0.0) *\
                (x_i_dists < epsilon_pca**.5) 
        tmp_neigh = data[neigh,:]
        tmp_dist_trim_scaled_pcs = x_i_dists[neigh]/epsilon_pca**.5
        if option == 'point_shift':
            X_i_collection.append((tmp_neigh - x_i).T)
        if option == 'mean_shift':   
            X_i_collection.append((tmp_neigh - np.mean(tmp_neigh,0)).T)
        distances_collection.append(tmp_dist_trim_scaled_pcs)
        complete_distance_collection.append(x_i_dists/epsilon**.5)
        neighbours_collection[point,:] = neigh
    return X_i_collection,  distances_collection, neigh, complete_distance_collection


def truncated_gaussian_kernel(distances_collection):
    n = len(distances_collection)
    D_i_collection = []
    for point in range(n):
        dist = distances_collection[point]
        kernel_dist = np.sqrt(np.exp(-dist**2)) * (dist < 1.0) * (dist > 0.0) 
        D_i_collection.append(kernel_dist)
    return D_i_collection


def epanechnikov_kernel(distances_collection):
    n = len(distances_collection)
    D_i_collection = []
    for point in range(n):
        dist = distances_collection[point]
        kernel_dist = np.sqrt((1-dist**2)) * (dist < 1.0) * (dist > 0.0) 
        D_i_collection.append(kernel_dist)
    return D_i_collection
        
        
def compute_weighted_X_i(X_i_collection,distances_collection,option = 'epanechnikov'):
    n = len(X_i_collection)
    B_i_collection = []
    if option == 'epanechnikov': 
        D_i_collection = epanechnikov_kernel(distances_collection)
    if option == 'gaussian': 
        D_i_collection = truncated_gaussian_kernel(distances_collection) 
    for point in range(n):
        B_i = X_i_collection[point]@np.diag(D_i_collection[point])
        B_i_collection.append(B_i)
    return B_i_collection


def local_pca(B_i_collection, gamma):
    n = len(B_i_collection)
    U_i_collection = []
    dhat_i_collection = []
    for point in range(n):
        U_i,sigma_i,_ = np.linalg.svd(B_i_collection[point],full_matrices=False)
        U_i_collection.append(U_i)
        tmp_cumsum = np.sort(np.cumsum(sigma_i)/np.sum(sigma_i))
        d_hat_i = np.where(tmp_cumsum>gamma)[0][0]+1
        dhat_i_collection.append(d_hat_i)
    d_hat = int(np.median(dhat_i_collection))
    O_i_collection = []
    for point in range(n):
        O_i = U_i_collection[point][:,:d_hat]
        O_i_collection.append(O_i)
    return O_i_collection, d_hat


def build_S_W(O_i_collection, complete_distance_collection, option = 'gaussian'):
    n = len(O_i_collection)
    d_hat = O_i_collection[0].shape[1]
    S = np.zeros((n*d_hat,n*d_hat))
    if option == 'epanechnikov': 
        D_i_collection = epanechnikov_kernel(complete_distance_collection)
    if option == 'gaussian': 
        D_i_collection = truncated_gaussian_kernel(complete_distance_collection) 
    for point_i in range(n):
        for point_j in range(n):
            w_ij = D_i_collection[point_i][point_j]**2
            O_ij_tilde = O_i_collection[point_i].T@O_i_collection[point_j]
            U_i,_,Vt_i = np.linalg.svd(O_ij_tilde,full_matrices=False)
            O_ij = U_i@Vt_i
            S[point_i*d_hat:(point_i+1)*d_hat,point_j*d_hat:(point_j+1)*d_hat,]=w_ij*O_ij
    W = np.array(D_i_collection)       
    return S, (W+W.T)/2

def build_SheafLaplacian(S,W,d_hat,epsilon):
            D_cal_inv = np.diag(1/np.sum(W,1))
            W_1 = D_cal_inv@W@D_cal_inv
            D_cal_inv_block = np.kron(D_cal_inv, np.eye(d_hat))
            S_1 = D_cal_inv_block@S@D_cal_inv_block
            D_1_cal_inv = np.diag(1/np.sum(W_1,1))
            D_1_inv = np.kron(D_1_cal_inv, np.eye(d_hat))
            Delta_n = (1/epsilon)*(D_1_inv@S_1 - np.eye(S.shape[0]))
            Delta_n = expm(Delta_n)
            #Delta_n[Delta_n < 1e-10] = 0            
            return Delta_n
    
def get_laplacians(data, epsilon, epsilon_pca, gamma_svd,tnn_or_gnn):
    if tnn_or_gnn == "tnn":
        X_i_collection,  distances_collection, _, complete_distance_collection = compute_neighbours(data,epsilon,epsilon_pca)
        B_i_collection = compute_weighted_X_i(X_i_collection,distances_collection)
        O_i_collection, d_hat = local_pca(B_i_collection, gamma_svd)
        S,W =  build_S_W(O_i_collection, complete_distance_collection)
        Delta_n = (1/epsilon)*build_SheafLaplacian(S,W, d_hat, epsilon)
        return Delta_n, S,W,O_i_collection, d_hat, B_i_collection
    else:
        Delta_n = build_CloudLaplacian(data, heat_kernel_t = epsilon)
        return Delta_n

def project_data(data,O_i_collection):
    d_hat = O_i_collection[0].shape[1]
    data_proj = np.zeros((data.shape[0]*d_hat,1))
    for point in range(len(O_i_collection)):
        if data.shape[1] == d_hat:
            data_proj[point*d_hat:(point+1)*d_hat,:] = np.expand_dims(data[point,:],1)
        else:
            data_proj[point*d_hat:(point+1)*d_hat,:] = np.expand_dims(O_i_collection[point].T@data[point,:],1)
    return data_proj

def topk(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argsort(input, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind, axis=axis) 
    return val

# %% Cloud Laplacian Utils
from sklearn.metrics import pairwise_distances

#  From https://github.com/tegusi/RGCNN

def get_pairwise_euclidean_distance_matrix(tensor):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    tensor = torch.tensor(tensor)
    adj_matrix = torch.cdist(tensor,tensor)
    return adj_matrix

def get_pairwise_distance_matrix(tensor, t):
    """Compute pairwise distance of a tensor.
    Args:
        tensor: tensor (batch_size, num_points, num_dims)
        t: scalar
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    # t = 10.55 # Average distance of CIFAR10
    # t = 10.55**2 # Average distance square of CIFAR10
    if len(tensor.shape)== 2:
        tensor = np.expand_dims(tensor,0)
    tensor = torch.tensor(tensor)
    adj_matrix = torch.squeeze(torch.cdist(tensor,tensor))
    adj_matrix = torch.square(adj_matrix)
    adj_matrix = torch.div( adj_matrix, -4*t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements


    return adj_matrix

def build_CloudLaplacian(imgs, normalize_exp = True, heat_kernel_t = 10, clamp_value=None):

    adj_matrix = get_pairwise_distance_matrix(imgs, heat_kernel_t)
    # Remove large values
    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix > clamp_value, adj_matrix, zero_tensor)

    if normalize_exp:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0])
        D = torch.diag(1 / torch.sqrt(D))
        L = (torch.matmul(torch.matmul(D, adj_matrix), D) - eye).numpy()
        L = expm(-L)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = (D - adj_matrix).numpy()
    return L

def get_laplacian_from_adj(adj_matrix, normalize = False, heat_kernel_t = 10, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    adj_matrix = torch.exp(adj_matrix)
    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        # remove large values
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to('cuda') # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    L= L.fill_diagonal_(0)
    return L

def get_gau_adj_from_adj(X_unlab, adj_matrix, normalize = False, heat_kernel_t = 10, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    # adj_matrix = torch.div( adj_matrix, 4*heat_kernel_t)
    # adj_matrix = torch.div( adj_matrix, 0.4)
    adj_matrix = torch.div( adj_matrix, 0.2)

    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))

    adj_matrix = torch.exp(-adj_matrix)
    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))

    # e, V = np.linalg.eig(adj_matrix.cpu().detach().numpy())

    adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements
    # adj_matrix= torch.div(adj_matrix, torch.max(adj_matrix))
    zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
    # adj_matrix = torch.where(adj_matrix < 1e-5,  zero_tensor, adj_matrix)

    if clamp_value!=None:
        # remove large values
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)
    
    # Remove path through obstacles
    zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')

    for i in range(X_unlab.shape[0]):
        for j in range(i+1, X_unlab.shape[0]):
            x1 = X_unlab[i, 0]
            y1 = X_unlab[i, 1]
            x2 = X_unlab[j, 0]
            y2 = X_unlab[j, 1]
            kk = (y2 - y1) / (x2 - x1)
            if (5- x1) * kk + y1 <=10 and (5- x1) * kk + y1 >= 3 and x1 < 5 and x2 > 5:
                # print(adj_matrix[i, j])
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (5- x1) * kk + y1 <=10 and (5- x1) * kk + y1 >= 3 and x2 < 5 and x1 > 5:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (15- x1) * kk + y1 <= 7 and (15- x1) * kk + y1 >= 0 and x2 < 15 and x1 > 15:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            if (15- x1) * kk + y1 <= 7 and (15- x1) * kk + y1 >= 0 and x1 < 15 and x2 > 15:
                # print(adj_matrix[i, j])

                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
    return adj_matrix

def get_euclidean_laplacian_from_adj(adj_matrix, normalize = False, clamp_value=None):

    # Remove small values
    adj_matrix = torch.square(adj_matrix)

    # adj_matrix = torch.div( adj_matrix, -4*heat_kernel_t)
    # adj_matrix = torch.exp(adj_matrix)
    # adj_matrix = adj_matrix.fill_diagonal_(0) # Delete the diagonal elements

    if clamp_value!=None:
        zero_tensor = torch.zeros(adj_matrix.size()).to('cuda')
        adj_matrix = torch.where(adj_matrix < clamp_value, adj_matrix, zero_tensor)

    if normalize:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        eye = torch.eye(adj_matrix.size()[0]).to('cuda') # Juan Modified This
        D = torch.diag(1 / torch.sqrt(D))
        L = eye - torch.matmul(torch.matmul(D, adj_matrix), D)
    else:
        D = torch.sum(adj_matrix, axis=1)  # (batch_size,num_points)
        # eye = tf.ones_like(D)
        # eye = tf.matrix_diag(eye)
        # D = 1 / tf.sqrt(D)
        D = torch.diag(D)
        L = D - adj_matrix
    return L


def projsplx(tensor):
    hk1 = np.argsort(tensor)
    vals = tensor[hk1]
    n = len(vals)
    Flag = True
    i = n - 1
    while Flag:
        ti = (torch.sum(vals[i + 1:]) - 1) / (n - i)
        if ti >= vals[i]:
            Flag = False
            that = ti
        else:
            i = i - 1
        if i == 0:
            Flag = False
            that = (torch.sum(vals) - 1) / n
    vals = torch.nn.functional.relu(vals - that)
    vals = vals/torch.sum(vals).item()
    return vals[np.argsort(hk1)]
