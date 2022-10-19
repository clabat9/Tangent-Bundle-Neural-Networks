# Tangent-Bundle-Neural-Networks
This repo contains the code used for implementing the numerical results in the paper: 

**"Tangent Bundle Neural Networks: from Manifolds to Celullar Sheaves and Back"**

C. Battiloro, Z. Wang, H. Riess, A. Ribeiro, P. Di Lorenzo

![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fintuition%2Fwhat-the-heck-is-a-manifold-60b8750e9690&psig=AOvVaw07ply7qzTvlGbUeJRHouvA&ust=1666299087836000&source=images&cd=vfe&ved=0CA0QjRxqFwoTCIjhvf6V7foCFQAAAAAdAAAAABAD)

The data are generated via the Matlab script VecDiffMap.m, in which is possible to select the number of sampling points on the sphere, the number of sampling realizations and the number of noise realizations; the ouput files are saved in "~/VectorDiffusionMaps-master/data". We use the implementation of the Vector Diffusion Map algorithm from trgao10 (https://github.com/trgao10/VectorDiffusionMaps) that is based on the original implementation of the VDM paper [] author Hau-Tieng Wu; we plan to implement the VDM procedure in Python for the extension of this conference paper to have a better control e better computational performance.

The implementation of the architecture is performed using PyTorch via a tailored version of the "alegnn" repo (https://github.com/alelab-upenn/graph-neural-networks, thanks to Fernando Gama and Luana Ruiz). 

The code is commented and it is ready to run up to data and Laplacians directories to be specified on local machines. For any questions, comments or suggestions, please e-mail Claudio Battiloro at claudio.battiloro@uniroma1.it and/or  Zhiyang Wang at zhiyangw@seas.upenn.edu. 


## Script and Other files descriptions

1. __`VectorDiffusionMaps-master`__: This folder contains the implementation of the Vector Diffusion Maps algorithm. We invite to visit the VDM repo of trgao10 or Hu's homepage for further details.

2. __`SphereDenoisingDDTNN.py`__: 
	This python script contains the implementation of the DD-TNN architecture and the computation of the corresonding results.
  
3. __`SphereDenoisingMNN.py`__: 
	This python script contains the implementation of the MNN architecture from [1] and the computation of the corresonding results.
  
4. __`Experiments`__: 
	This folder contains the experiments logs (as explained in the scripts), e.g. results, seeds, best architectures,...
  
5. __`alegnnss`__: 
  This folder contains the "alegnn" library with some modifications.
