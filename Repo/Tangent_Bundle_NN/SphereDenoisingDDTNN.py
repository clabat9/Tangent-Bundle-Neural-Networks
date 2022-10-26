# 2022/10/20~
# Claudio Battiloro, clabat@seas.upenn.edu/claudio.battiloro@uniroma1.it
# Zhiyang Wang, zhiyangw@seas.upenn.edu
# Hans Riess

# Thanks to:
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
# for implementing the "alegnn" library.

# This is the code used for implementing the numerical results in the paper:
# "Tangent Bundle Neural Networks: from Manifolds to Celullar Sheaves and Back"
# C. Battiloro, Z. Wang, H. Riess, A. Ribeiro, P. Di Lorenzo
# In particular, this code implements the denoising task described in the paper 
# over the vector field (-y,x,0) tangent to the unitary sphere S2 using DD-TNN
# Obs: This code could be used also for a denoising+reconstruction task, indeed
# a random sampler is implemented. For the denoising-only task, it is sufficient,
# tos et the sample percentage to 1.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       parameters of each realization on a directory named 'savedModels'.
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       in pickle format. These variables are saved in a trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. 

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:


import os
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy
import torch;    torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#\\\ Own libraries:
import alegnnss.utils.graphML as gml
import   alegnnss.utils.dataTools as dt
import alegnnss.modules.architectures as archit
import alegnnss.modules.model as model
import alegnnss.modules.training as training
import alegnnss.modules.evaluation as evaluation
from alegnnss.modules.loss import MSE_semisup
#\\\ Separate functions:
from alegnnss.utils.miscTools import writeVarValues
from alegnnss.utils.miscTools import saveSeed


#\\\ Activation function: just a scaled Tanh
class alphaTanh(nn.Module):
            def __init__(self, alpha = 2):
                super(alphaTanh, self).__init__()
                self.alpha = alpha
            def forward(self, x):
                return self.alpha*torch.tanh(x)
            
# Start measuring time
startRunTime = datetime.datetime.now()


#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################


thisFilename = 'tangentbundlenn' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters and results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)
useGPU = True# If true, and GPU is available, use it.

########
# DATA #
########

nDataSplits = 5 # Number of data realizations
nNoiseSplits = 5 # Number of noise realizations --> Total Num. of experiments = nDataSplits*nNoiseSplits
dhat = 2 # Estimated underlying manifold dimension
sigma_noise = 1e-1 #5e-2 # Noise variance
sample_percentage = 1 # The sampling mask is generated with P(retain_sample)=sample_percentage


############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
l2pen = 1e-5
learningRate = 0.0005 # In all options
beta1 = 0.8 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.9 # ADAM option only

#\\\ Loss function choice
lossFunction = MSE_semisup
reg_smooth = None

#\\\ Overall training options
nEpochs = 5000 #Number of epochs
batchSize = 1 # Batch size (Semisupervised task, so it is set to 1 such that nEpochs + number of training steps)
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = int(nEpochs/3) # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'nDataSplits': nDataSplits,
                'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'beta2': beta2,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

for nPoints in [200,800]: # The Available Data are for 200, 800 and 1200. Of course other
                               # can be generated from the Matlab script
    
    #################
    # ARCHITECTURES #
    #################
    # Tangent Bundle Neural Networks are implemented via Sheaf Neural Netwroks; 
    # Sheaf Neural Networks can be implemented via a Selection-GNN with the GSO set
    # as the Sheaf Laplacian and vector data stacked on the columns of the input matrix
    # (as explained in the paper).
    # We exploit a two layered DD-TNN and Discretized Space/Time Filters (denoted as DD-TF). 
    # The main difference is that the DD-TNN includes a non-linearity function.
    
    # In this section, we determine the (hyper)parameters of models that we are
    # going to train. This only sets the parameters. The architectures need to be
    # created later below. Do not forget to add the name of the architecture
    # to modelList.
    
    # If the model dictionary is called 'model' + name, then it can be
    # picked up immediately later on, and there's no need to recode anything after
    # the section 'Setup' (except for setting the number of nodes in the 'N' 
    # variable after it has been coded).
    
    # The name of the keys in the model dictionary have to be the same
    # as the names of the variables in the architecture call, because they will
    # be called by unpacking the dictionary.
    
    modelList = []
    
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    #\\\ DD-TNN via Selection GNN \\\
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    #\\\ Basic parameters for both DD-TNN and DD-TF
    modelSelGNN = {} # Model parameters for the Selection GNN (SelGNN)
    modelSelGNN['name'] = 'TNN'
    modelSelGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Chosen architecture
    modelSelGNN['archit'] = archit.SelectionGNN
    # Layers Parameters
    modelSelGNN['dimNodeSignals'] = [1,1] # Features per layer
    modelSelGNN['nFilterTaps'] = [5] # Number of filter taps per layer
    modelSelGNN['bias'] = False # Decide whether to include a bias term
        
    # Pooling (Set To No Pooling)
    modelSelGNN['poolingFunction'] = gml.NoPool # Summarizing function
    modelSelGNN['nSelectedNodes'] = [dhat*nPoints] 
    modelSelGNN['poolingSize'] = [0] # poolingSize-hop neighborhood that is affected by the summary
        
    # Full MLP readout layer (not needed in our semisupervised setting)
    modelSelGNN['dimLayersMLP'] = [] # Dimension of the fully connected
        # layers after the GCN layers, we are doing a binary classification problem.
        
    # Cellular Sheaf structure
    modelSelGNN['GSO'] = None # Sheaf Shift Operator, to be determined later on, based on data
    modelSelGNN['order'] = None # Not used because there is no pooling
    modelSelGNN['lossFunction'] = lossFunction # Loss Function
       
    #\\\ TRAINER
    modelSelGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    modelSelGNN['evaluator'] = evaluation.evaluate
    
    #\\\\\\\\\\\\\\\\\\\\\\\
    #\\\ MODEL 1: DD-TNN \\\
    #\\\\\\\\\\\\\\\\\\\\\\\
    modelDDTNN = deepcopy(modelSelGNN)
    modelDDTNN['name'] = 'DDTNN' # Name of the architecture
    # Nonlinearity
    modelDDTNN['nonlinearity'] = nn.PReLU
    # Save Values:
    writeVarValues(varsFile, modelDDTNN)
    modelList += [modelDDTNN['name']]
    
    #\\\\\\\\\\\\\\\\\\\\\\
    #\\\ MODEL 2: DD-TF \\\
    #\\\\\\\\\\\\\\\\\\\\\\
    #modelDDTF = deepcopy(modelSelGNN)
    #modelDDTF['name'] = 'DDTF' # Name of the architecture
    # Nonlinearity
    #modelDDTF['nonlinearity'] = gml.NoActivation
    # Save Values:
    #writeVarValues(varsFile, modelDDTF)
    #modelList += [modelDDTF['name']]
    
    
    ###########
    # LOGGING #
    ###########
    
    # Options:
    doPrint = True # Decide whether to print stuff while running
    doLogging = False # Log into tensorboard
    doSaveVars = True # Save (pickle) useful variables
    doFigs = False # Plot some figures (this only works if doSaveVars is True)
    # Parameters:
    printInterval = 16000 # After how many training steps, print the partial results
    #   0 means to never print partial results while training
    xAxisMultiplierTrain = 1 # How many training steps in between those shown in
        # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
    xAxisMultiplierValid = 1 # How many validation steps in between those shown,
        # same as above.
    figSize = 5 # Overall size of the figure that contains the plot
    lineWidth = 2 # Width of the plot lines
    markerShape = 'o' # Shape of the markers
    markerSize = 3 # Size of the markers
    
    #\\\ Save values:
    writeVarValues(varsFile,
                   {'doPrint': doPrint,
                    'doLogging': doLogging,
                    'doSaveVars': doSaveVars,
                    'doFigs': doFigs,
                    'saveDir': saveDir,
                    'printInterval': printInterval,
                    'figSize': figSize,
                    'lineWidth': lineWidth,
                    'markerShape': markerShape,
                    'markerSize': markerSize})
    
    
    #%%##################################################################
    #                                                                   #
    #                    SETUP                                          #
    #                                                                   #
    #####################################################################
    
    #\\\ Determine processing unit:
    if useGPU and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    #\\\ Notify of processing units
    if doPrint:
        print("Selected devices:")
        for thisModel in modelList:
            modelDict = eval('model' + thisModel)
            print("\t%s: %s" % (thisModel, modelDict['device']))
    
    #\\\ Logging options
    if doLogging:
        # If logging is on, load the tensorboard visualizer and initialize it
        from alegnnss.utils.visualTools import Visualizer
        logsTB = os.path.join(saveDir, 'logsTB')
        logger = Visualizer(logsTB, name='visualResults')
    
    #\\\ Save variables during evaluation.
    # We will save all the evaluations obtained for each of the trained models.
    # It basically is a dictionary, containing a list. The key of the
    # dictionary determines the model, then the first list index determines
    # which split realization. Then, this will be converted to numpy to compute
    # mean and standard deviation (across the split dimension).
    costBest = {} # Cost for the best model (Evaluation cost: RMSE)
    costLast = {} # Cost for the last model
    costBestDiff = {}
    InitCost ={}
    
    for thisModel in modelList: # Create an element for each realization,
        costBest[thisModel] = np.zeros((nDataSplits,nNoiseSplits))
        costLast[thisModel] = np.zeros((nDataSplits,nNoiseSplits))
        costBestDiff[thisModel] = np.zeros((nDataSplits,nNoiseSplits))
        InitCost[thisModel] = np.zeros((nDataSplits,nNoiseSplits))
    
    

    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    lossValid = {}
    costValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = np.zeros((nDataSplits,nNoiseSplits,nEpochs))
        costTrain[thisModel] = np.zeros((nDataSplits,nNoiseSplits,nEpochs))
        lossValid[thisModel] = np.zeros((nDataSplits,nNoiseSplits,np.ceil(nEpochs/validationInterval).astype(int)))
        costValid[thisModel] = np.zeros((nDataSplits,nNoiseSplits,np.ceil(nEpochs/validationInterval).astype(int)))
    
    
    ####################
    # TRAINING OPTIONS #
    ####################
    # Training phase. It has a lot of options that are input through a
    # dictionary of arguments.
    # The value of these options was decided above with the rest of the parameters.
    # This just creates a dictionary necessary to pass to the train function.
    
    trainingOptions = {}
    if doLogging:
        trainingOptions['logger'] = logger
    if doSaveVars:
        trainingOptions['saveDir'] = saveDir
    if doPrint:
        trainingOptions['printInterval'] = printInterval
    if doLearningRateDecay:
        trainingOptions['learningRateDecayRate'] = learningRateDecayRate
        trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
    trainingOptions['validationInterval'] = validationInterval
    
    # And in case each model has specific training options, then we create a 
    # separate dictionary per model.
    trainingOptsPerModel= {}
    
    #%%##################################################################
    #                                                                   #
    #                    DATA LOADING AND SAMPLING                      #
    #                                                                   #
    #####################################################################
    # Data and Laplcians have been precedently computed via the MatLab VDM implementation.
    for split in range(nDataSplits):
        for rel in range(nNoiseSplits):
            # Loads the Sheaf Laplacian (Delta_n), the Laplacian, their Exp and the specific data realization
            #SLaplacian = pd.read_csv('/home/claudio/Dropbox/VectorDiffusionMaps-master/data/data_samples_'+str(nPoints)+'_realization_'\
                                  #+str(split)+'/SLaplacian.csv',header = None).to_numpy()
            SLaplacian = pd.read_csv('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Repo/VectorDiffusionMaps-master/data/data_samples_'+str(nPoints)+'_realization_'\
                                  +str(split+1)+'/expSLaplacian.csv',header = None).to_numpy()
            [lambdas,_] = np.linalg.eigh(SLaplacian)
            SLaplacian = SLaplacian/np.max(np.real(lambdas))
            data_np = pd.read_csv('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Repo/VectorDiffusionMaps-master/data/data_samples_'+str(nPoints)+'_realization_'\
                                  +str(split+1)+'/projData.csv',header = None).to_numpy()
            sampling_mask = np.kron(np.ones((1,dhat)),np.expand_dims(np.random.binomial\
                                                                     (1, sample_percentage, size=nPoints),1)).flatten() # Sampling Mask
            # Defines the training set as the masked signals and adds noise
            train_indices =(np.arange(1, nPoints*dhat+1)*sampling_mask-1).astype(int)
            test_indices = np.arange(0,nPoints*dhat)[train_indices == -1]#np.delete(np.arange(0, nPoints),train_indices)
            train_np = pd.read_csv('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Repo/VectorDiffusionMaps-master/data/data_samples_'+str(nPoints)+'_realization_'\
                                  +str(split+1)+'/projData_sd_'+str(sigma_noise)+'_nrel_'+str(rel+1)+'.csv',header = None).to_numpy()
            train_indices_torch = torch.from_numpy(np.expand_dims(train_indices[train_indices != -1],1).T).to('cuda:0' if (useGPU and torch.cuda.is_available()) \
                                             else 'cpu')
            test_indices_torch = torch.from_numpy(np.expand_dims(test_indices,1).T).to('cuda:0' if (useGPU and torch.cuda.is_available()) \
                                             else 'cpu')
            train_np[test_indices] = 0#np.mean(train_np[train_indices_torch]) 
    
            # Data Object Instatiating 
            # Clearly we have one data vector, train data are noisy, validation/test is
            # computed evaluating the denoising error
            data = dt._dataForSemisupervised()
            data.dataType = torch.float64
            data.nTrain = 1
            data.nValid = 1
            data.nTest = 1
            data.test_indices = test_indices
            data.test_indices_torch = test_indices_torch
            data.evaluate_only_test = 0 # If data are also sampled, this attribute allowes to evaluate
                                        # the model only on unsampled points
            data.samples = {}
            data.samples['train'] = {}
            data.samples['train']['signals'] = torch.from_numpy(train_np.T)
            data.samples['train']['targets'] = torch.from_numpy(train_np.T)
            data.samples['valid'] = {}
            data.samples['valid']['signals'] = torch.from_numpy(train_np.T)
            data.samples['valid']['targets'] = torch.from_numpy(data_np.T)
            data.samples['test'] = {}
            data.samples['test']['signals'] = torch.from_numpy(train_np.T)
            data.samples['test']['targets'] = torch.from_numpy(data_np.T)
            data.nPoints = nPoints
            data.expandDims()
        
        
            #%%##################################################################
            #                                                                   #
            #                    MODELS INITIALIZATION                          #
            #                                                                   #
            #####################################################################
        
            # This is the dictionary where we store the models (in a model.Model
            # class, that is then passed to training).
            modelsTNN = {}
        
            # If a new model is to be created, it should be called for here.
            
            if doPrint:
                print("Model initialization...", flush = True)
                
            for thisModel in modelList:
                
                # Get the corresponding parameter dictionary
                modelDict = deepcopy(eval('model' + thisModel))
                modelDict['GSO'] = SLaplacian
                # and training options
                trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
                
                # Now, this dictionary has all the hyperparameters that we need to pass
                # to the architecture function, but it also has other keys that belong
                # to the more general model (like 'name' or 'device'), so we need to
                # extract them and save them in seperate variables for future use.
                thisName = modelDict.pop('name')
                callArchit = modelDict.pop('archit')
                thisDevice = modelDict.pop('device')
                thisTrainer = modelDict.pop('trainer')
                thisEvaluator = modelDict.pop('evaluator')
                thisLossFunction = modelDict.pop('lossFunction')(train_indices_torch, reg_smooth,\
                                                                 torch.Tensor(SLaplacian).to(thisDevice))
           
                # If more than one data realization is going to be carried out,
                # we are going to store all of those models separately, so that any of
                # them can be brought back and studied in detail.
                if nDataSplits > 1:
                    thisName += '_num_sampl_'+str(nPoints)+'_data_real_'+str(split)+'_noise_real_'+str(rel)
                    
                if doPrint:
                    print("\tInitializing %s..." % thisName,
                          end = ' ',flush = True)
                    
                ##############
                # PARAMETERS #
                ##############
        
                #\\\ Optimizer options
                #   (If different from the default ones, change here.)
                thisOptimAlg = optimAlg
                thisLearningRate = learningRate
                thisBeta1 = beta1
                thisBeta2 = beta2
                
                ################
                # ARCHITECTURE #
                ################
        
                thisArchit = callArchit(**modelDict)
                thisArchit.to(thisDevice)
                train_indices_torch.to(thisDevice)
                
                #############
                # OPTIMIZER #
                #############
        
                if thisOptimAlg == 'ADAM':
                    thisOptim = optim.Adam(thisArchit.parameters(),
                                           lr = learningRate,
                                           betas = (beta1, beta2), weight_decay=l2pen)
                elif thisOptimAlg == 'SGD':
                    thisOptim = optim.SGD(thisArchit.parameters(),
                                          lr = learningRate)
                elif thisOptimAlg == 'RMSprop':
                    thisOptim = optim.RMSprop(thisArchit.parameters(),
                                              lr = learningRate, alpha = beta1)
        
              
        
                #########
                # MODEL #
                #########
                # Create the model
                modelCreated = model.Model(thisArchit,
                                           thisLossFunction,
                                           thisOptim,
                                           thisTrainer,
                                           thisEvaluator,
                                           thisDevice,
                                           thisName,
                                           saveDir)
        
                # Store it
                modelsTNN[thisName] = modelCreated
        
                # Write the main hyperparameters
                writeVarValues(varsFile,
                               {'name': thisName,
                                'thisOptimizationAlgorithm': thisOptimAlg,
                                'thisTrainer': thisTrainer,
                                'thisEvaluator': thisEvaluator,
                                'thisLearningRate': thisLearningRate,
                                'thisBeta1': thisBeta1,
                                'thisBeta2': thisBeta2})
        
                if doPrint:
                    print("OK")
                    
            if doPrint:
                print("Model initialization... COMPLETE")
        
            #%%##################################################################
            #                                                                   #
            #                    TRAINING                                       #
            #                                                                   #
            #####################################################################
            
            print("")
            
            # We train each model separately
            
            for thisModel in modelsTNN.keys():
                
                if doPrint:
                    print("Training model %s..." % thisModel)
                
                # Remember that modelsTNN.keys() has the split numbering as well as the
                # name, while modelList has only the name. So we need to map the
                # specific model for this specific split with the actual model name,
                # since there are several variables that are indexed by the model name
                # (for instance, the training options, or the dictionaries saving the
                # loss values)
                for m in modelList:
                    if m in thisModel:
                        modelName = m
                
                # Identify the specific split number at training time
                if nDataSplits > 1:
                    trainingOptsPerModel[modelName]['graphNo'] = split
                
                # Train the model
                thisTrainVars = modelsTNN[thisModel].train(data,
                                                           nEpochs,
                                                           batchSize,
                                                           **trainingOptsPerModel[modelName])
        
                # Find which model to save the results (when having multiple
                # realizations)
                lossTrain[modelName][split,rel,:] = thisTrainVars['lossTrain']
                costTrain[modelName][split,rel,:] = thisTrainVars['costTrain']
                lossValid[modelName][split,rel,:] = thisTrainVars['lossValid']
                costValid[modelName][split,rel,:] = thisTrainVars['costValid']
                # Store Best RMSE
                costBest[modelName][split,rel]= min(costValid[modelName][split,rel,:])
                            
            # And we also need to save 'nBatches' but is the same for all models, so
            if doFigs:
                nBatches = thisTrainVars['nBatches']
    
    #Computes and Saves Best Performances as a DataFrame
    meancostBest = {}
    
    for thisModel in modelList:
        mask = np.logical_or(costBest[thisModel] == costBest[thisModel].max(keepdims = 1), costBest[thisModel] == costBest[thisModel].min(keepdims = 1))
        tmp = ma.masked_array(costBest[thisModel], mask = mask)
        meancostBest[thisModel] = {"Mean":np.mean(tmp),"Std":np.std(tmp)}
    pd.DataFrame(meancostBest).to_csv(saveDir+'/best_summary_'+str(nPoints)+'.csv')
    
    #%%##################################################################
    #                                                                   #
    #                    PLOT                                           #
    #                                                                   #
    #####################################################################
    
    # Finally, we might want to plot several quantities of interest
    
    if doFigs and doSaveVars:
    
        ###################
        # DATA PROCESSING #
        ###################
        
        #\\\ FIGURES DIRECTORY:
        saveDirFigs = os.path.join(saveDir,'figs')
        # If it doesn't exist, create it.
        if not os.path.exists(saveDirFigs):
            os.makedirs(saveDirFigs)
    
        #\\\ COMPUTE STATISTICS:
        # The first thing to do is to transform those into a matrix with all the
        # realizations, so create the variables to save that.
        meanLossTrain = {}
        meanCostTrain = {}
        meanLossValid = {}
        meanCostValid = {}
        stdDevLossTrain = {}
        stdDevCostTrain = {}
        stdDevLossValid = {}
        stdDevCostValid = {}
        # Initialize the variables
        for thisModel in modelList:
            # Transform into np.array
            lossTrain[thisModel] = np.array(lossTrain[thisModel])
            costTrain[thisModel] = np.array(costTrain[thisModel])
            lossValid[thisModel] = np.array(lossValid[thisModel])
            costValid[thisModel] = np.array(costValid[thisModel])
            # Each of one of these variables should be of shape
            # nDataSplits x nNoiseSplits x nEpochs
            # And compute the statistics
            meanLossTrain[thisModel] = np.mean(np.mean(lossTrain[thisModel], axis = 1), axis = 0)
            meanCostTrain[thisModel] = np.mean(np.mean(costTrain[thisModel], axis = 1), axis = 0)
            meanLossValid[thisModel] = np.mean(np.mean(lossValid[thisModel], axis = 1), axis = 0)
            meanCostValid[thisModel] = np.mean(np.mean(costValid[thisModel], axis = 1), axis = 0)
            stdDevLossTrain[thisModel] = np.std(np.mean(lossTrain[thisModel], axis = 1), axis = 0)
            stdDevCostTrain[thisModel] = np.std(np.mean(costTrain[thisModel], axis = 1), axis = 0)
            stdDevLossValid[thisModel] = np.std(np.mean(lossValid[thisModel], axis = 1), axis = 0)
            stdDevCostValid[thisModel] = np.std(np.mean(costValid[thisModel], axis = 1), axis = 0)
    
        ####################
        # SAVE FIGURE DATA #
        ####################
    
        # And finally, we can plot. But before, let's save the variables mean and
        # stdDev so, if we don't like the plot, we can re-open them, and re-plot
        # them, a piacere.
        #   Pickle, first:
        varsPickle = {}
        varsPickle['nEpochs'] = nEpochs
        varsPickle['nBatches'] = nBatches
        varsPickle['meanLossTrain'] = meanLossTrain
        varsPickle['stdDevLossTrain'] = stdDevLossTrain
        varsPickle['meanCostTrain'] = meanCostTrain
        varsPickle['stdDevCostTrain'] = stdDevCostTrain
        varsPickle['meanLossValid'] = meanLossValid
        varsPickle['stdDevLossValid'] = stdDevLossValid
        varsPickle['meanCostValid'] = meanCostValid
        varsPickle['stdDevCostValid'] = stdDevCostValid
        with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
            pickle.dump(varsPickle, figVarsFile)
    
        ########
        # PLOT #
        ########
    
        # Compute the x-axis
        xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
        xValid = np.arange(0, nEpochs * nBatches, \
                              validationInterval*xAxisMultiplierValid)
        xTest = [pow(10, r) for r in np.linspace(-3, 0 , num = 5)]
    
    
        # If we do not want to plot all the elements (to avoid overcrowded plots)
        # we need to recompute the x axis and take those elements corresponding
        # to the training steps we want to plot
        if xAxisMultiplierTrain > 1:
            # Actual selected samples
            selectSamplesTrain = xTrain
            # Go and fetch tem
            for thisModel in modelList:
                meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                        [selectSamplesTrain]
                stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                            [selectSamplesTrain]
                meanCostTrain[thisModel] = meanCostTrain[thisModel]\
                                                        [selectSamplesTrain]
                stdDevCostTrain[thisModel] = stdDevCostTrain[thisModel]\
                                                            [selectSamplesTrain]
        # And same for the validation, if necessary.
        if xAxisMultiplierValid > 1:
            selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                           xAxisMultiplierValid)
            for thisModel in modelList:
                meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                        [selectSamplesValid]
                stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                            [selectSamplesValid]
                meanCostValid[thisModel] = meanCostValid[thisModel]\
                                                        [selectSamplesValid]
                stdDevCostValid[thisModel] = stdDevCostValid[thisModel]\
                                                            [selectSamplesValid]
        
        #\\\ LOSS (Training and validation) for EACH MODEL
        for key in meanLossTrain.keys():
            lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
            plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                         color = '#01256E', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                         color = '#95001A', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.ylabel(r'Loss')
            plt.xlabel(r'Training steps')
            plt.legend([r'Training', r'Validation'])
            plt.title(r'%s' % key)
            lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                            bbox_inches = 'tight')
            
        
        plot_lag = 1
        # LOSS (training) for ALL MODELS
        allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
        for key in meanLossTrain.keys():
            plt.errorbar(xTrain[plot_lag:], meanLossTrain[key][plot_lag:], yerr = stdDevLossTrain[key][plot_lag:],
                         linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend(list(meanLossTrain.keys()))
        allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                        bbox_inches = 'tight')
        
        #\\\ RMSE (Training and validation) for EACH MODEL
        for key in meanCostTrain.keys():
            costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
            plt.errorbar(xTrain[plot_lag:], meanCostTrain[key][plot_lag:], yerr = stdDevCostTrain[key][plot_lag:],
                         color = '#01256E', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.errorbar(xValid[plot_lag:], meanCostValid[key][plot_lag:], yerr = stdDevCostValid[key][plot_lag:],
                         color = '#95001A', linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
            plt.ylabel(r'RMSE')
            plt.xlabel(r'Training steps')
            plt.legend([r'Training', r'Validation'])
            plt.title(r'%s' % key)
            costFig.savefig(os.path.join(saveDirFigs,'cost%s.pdf' % key),
                            bbox_inches = 'tight')
            
        # RMSE (validation) for ALL MODELS
        allCostValidFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        for key in meanCostValid.keys():
            plt.errorbar(xValid[plot_lag:], meanCostValid[key][plot_lag:], yerr = stdDevCostValid[key][plot_lag:],
                         linewidth = lineWidth,
                         marker = markerShape, markersize = markerSize)
        plt.ylabel(r'RMSE')
        plt.xlabel(r'Training steps')
        plt.legend(list(meanCostValid.keys()))
        allCostValidFig.savefig(os.path.join(saveDirFigs,'allCostValid.pdf'),
                        bbox_inches = 'tight')

    

    

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))
    
