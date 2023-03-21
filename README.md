# Tangent Bundle Neural Networks
This repo contains the code used for implementing the numerical results in the papers: 

**1) "Tangent Bundle Convolutional Learning: from Manifolds to Celullar Sheaves and Back"** (extended journal paper)

**2) "Tangent Bundle Filters and Neural Networks: from Manifolds to Celullar Sheaves and Back"** (preliminary conference)



*C. Battiloro (1,2) , Z.Wang (1), H. Riess (3), P. Di Lorenzo (2), A. Ribeiro (1)*

(1) ESE Department, University of Pennsylvania, Philadelphia, USA 

(2) DIET Department, Sapienza University of Rome, Rome, Italy 

(3) , Duke University, Durham, USA

<p align="center">
	<img src="https://github.com/clabat9/Tangent-Bundle-Neural-Networks/blob/main/sphere_ex_cropped.jpg?raw=true" alt="drawing" width="400"/>
</p>

**1) "Tangent Bundle Convolutional Learning: from Manifolds to Celullar Sheaves and Back"** ("Journal Repo" folder)

## Abstract
 In this work we introduce a convolution operation over the tangent bundle of
Riemann manifolds in terms of exponentials of the Connection Laplacian
operator. We define tangent bundle filters and tangent bundle neural networks
(TNNs) based on this convolution operation, which are novel continuous
architectures operating on tangent bundle signals, i.e. vector fields over the
manifolds. Tangent bundle filters admit a spectral representation that
generalizes the ones of scalar manifold filters, graph filters and standard
convolutional filters in continuous time. We then introduce a discretization
procedure, both in the space and time domains, to make TNNs implementable,
showing that their discrete counterpart is a novel principled variant of the
very recently introduced sheaf neural networks. We formally prove that this
discretized architecture converges to the underlying continuous TNN. Finally,
we numerically evaluate the effectiveness of the proposed architecture on
various learning tasks, both on synthetic and real data.

## Summary
The implementation of the architecture is performed using PyTorch.

The code is commented and it is ready to run  (with data and Laplacians directories to be specified on local machines). For any questions, comments or suggestions, please e-mail Claudio Battiloro at claudio.battiloro@uniroma1.it and/or  Zhiyang Wang at zhiyangw@seas.upenn.edu. 


## Files description

1. __`data`__: The windfields data used for the experiments are available at https://drive.google.com/file/d/1VXAO7soE6HKG2Z7ReAD4TVDf1iARvF8c/view?usp=sharing

2. __`results`__: 
	This folders contains the pickles of the results showed in the paper.
  
3. __`mainTorusDenoising.py`__, __`mainWindSampling.py`__, __`mainWindPrediction.py`__: 
	These python scripts are the main scripts to reproduce the experiments of the paper, it is suffiecient to run them.
  
4. __`layers.py`__: 
	This python script contains the DD-TNNs and MNNs layers implementation.
	
5. __`layers.py`__: 
	This python script contains the DD-TNNs and MNNs architecture implementation.
  
6. __`data_utils.py`__: 
  	This python script contains Torch Dataset Classes used for the experiments.
	
7. __`utils.py`__: 
  	This python script contains various functions, among which the ones used to build the cloud and sheaf Laplacians.

**2) "Tangent Bundle Filters and Neural Networks: from Manifolds to Celullar Sheaves and Back"** ("Repo" folder)

## Abstract
In this work we introduce a convolution operation over the tangent bundles of Riemann manifolds starting from a (vector) heat diffusion process controlled by the Connection Laplacian operator. We exploit the convolution to define tangent bundle filters  and tangent bundle neural networks (TNNs), novel continuous architectures operating on tangent bundle-structured data. We then discretize TNNs both in space and time domains, showing that their discrete counterpart is a generalization of the recently introduced Sheaf Neural Networks. We formally prove that this discrete architecture converges to the underlying continuous TNN. Finally, we numerically evaluate the effectiveness of the proposed architecture on a denoising task of a tangent vector field of the unit 2-sphere.

## Summary
The data are generated via the Matlab script VecDiffMap.m, in which is possible to select the number of sampling points on the sphere, the number of sampling realizations and the number of noise realizations; the ouput files are saved in "~/VectorDiffusionMaps-master/data". We use the implementation of the Vector Diffusion Map algorithm from trgao10 (https://github.com/trgao10/VectorDiffusionMaps) that is based on the original implementation of the VDM paper [1] author Hau-Tieng Wu. 

The implementation of the architecture is performed using PyTorch via a tailored version of the "alegnn" repo (https://github.com/alelab-upenn/graph-neural-networks, thanks to Fernando Gama and Luana Ruiz). 

The code is commented and it is ready to run  (with data and Laplacians directories to be specified on local machines). For any questions, comments or suggestions, please e-mail Claudio Battiloro at claudio.battiloro@uniroma1.it and/or  Zhiyang Wang at zhiyangw@seas.upenn.edu. 

## Comments

This is a preliminary work whose main content is theoretical and methodological. We are planning to extend it and, for this reason, we expect to improve the code, design more complex experiments and validations, and implement the VDM procedure in Python and the TNN architecture in Pytorch Geometric to have a better control on the procedure and better computational performance. As a result, the actual code and pipeline could be for sure improved, but we believe it is simple and readable at this stage.

## Files description

1. __`VectorDiffusionMaps-master`__: This folder contains the implementation of the Vector Diffusion Maps algorithm. We invite to visit the VDM repo of trgao10 or Wu's homepage for further details.

2. __`SphereDenoisingDDTNN.py`__: 
	This python script contains the implementation of the DD-TNN architecture and the computation of the corresonding results.
  
3. __`SphereDenoisingMNN.py`__: 
	This python script contains the implementation of the MNN architecture from [2] and the computation of the corresonding results.
  
4. __`Experiments`__: 
	This folder contains the experiments logs (as explained in the scripts), e.g. results, seeds, best architectures,...
  
5. __`alegnnss`__: 
  This folder contains the "alegnn" library with some modifications.
  
## References

[1] "Vector Diffusion Maps and the Connection Laplacian", Amit Singer and Hau-tieng Wu, https://arxiv.org/abs/1102.0075

[2] "Convolutional neural networks on manifolds: From graphs and back" Zhiyang Wang, Luana Ruiz, and Alejandro Ribeiro, https://zhiyangw.com/Papers/convolution-asilomar2022.pdf
