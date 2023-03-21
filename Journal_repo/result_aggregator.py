import pickle as pkl
import sys
import numpy as np
import numpy.ma as ma

string = sys.argv[1]

with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/Journal_repo/results/'+string+'/res_tnn.pkl', 'rb') as file:
    mse_dic_tnn = pkl.load(file)
with open('/home/claudio/Desktop/Tangent-Bundle-Neural-Networks-main/Code_for_Journal/TNNs/Journal_repo/results/'+string+'/res_mnn.pkl', 'rb') as file:
    mse_dic_mnn = pkl.load(file) 

try:
    assert mse_dic_tnn.keys() == mse_dic_mnn.keys()
    print("All Sample Sizes Coincide!")
    keys = mse_dic_tnn.keys()
except: 
    print("Not all Sample Sizes Coincide! Using the intersection...")
    keys = set(mse_dic_tnn.keys()).intersection(set(mse_dic_mnn.keys()))

thresh = float(sys.argv[2])
verbose = int(sys.argv[3])
keys_mask = {}
for sample_size in keys:
    try:
        assert mse_dic_tnn[sample_size].keys() == mse_dic_mnn[sample_size].keys()
        print("All Masks/Noise Variances Coincide for Sample Size "+sample_size+"!")
        keys_mask[sample_size] = mse_dic_tnn[sample_size].keys()
    except: 
        print("Not all Masks/Noise Variances Coincide for Sample Size "+sample_size+"!" +"Using the intersection...")
        keys_mask[sample_size] = set(mse_dic_tnn[sample_size].keys()).intersection(set(mse_dic_mnn[sample_size].keys()))
    for mask_size in keys_mask[sample_size]:
        # Delete Runs over Threshold (Divergent or Badly Trained)
        if np.sum(mse_dic_tnn[sample_size][mask_size]["complete_coll"]>thresh) > 0:
            if verbose:
                print("Architecture: TNN")
                print("Sample Size:")
                print(sample_size)
                print("Mask Size/Noise Variance")
                print(mask_size)
                print("Before:")
                print(mse_dic_tnn[sample_size][mask_size])
            tmp = mse_dic_tnn[sample_size][mask_size]["complete_coll"]
            mask = tmp > thresh
            min_mse = ma.masked_array(tmp, mask = mask)
            mse_dic_tnn[sample_size][mask_size]["complete_coll"] = min_mse
            mse_dic_tnn[sample_size][mask_size]["avg_mse"] = min_mse.mean()
            mse_dic_tnn[sample_size][mask_size]["std_mse"] = min_mse.std()
            if verbose:
                print("After:")
                print(mse_dic_tnn[sample_size][mask_size])

        if np.sum(mse_dic_mnn[sample_size][mask_size]["complete_coll"]>thresh) > 0:
            if verbose:
                print("Architecture: MNN")
                print("Sample Size:")
                print(sample_size)
                print("Mask Size/Noise Variance:")
                print(mask_size)
                print("Before:")
                print(mse_dic_mnn[sample_size][mask_size])
            tmp = mse_dic_mnn[sample_size][mask_size]["complete_coll"]
            mask = tmp > thresh
            min_mse = ma.masked_array(tmp, mask = mask)
            mse_dic_mnn[sample_size][mask_size]["complete_coll"] = min_mse
            mse_dic_mnn[sample_size][mask_size]["avg_mse"] = min_mse.mean()
            mse_dic_mnn[sample_size][mask_size]["std_mse"] = min_mse.std()
            if verbose:
                print("After:")
                print(mse_dic_mnn[sample_size][mask_size])    


aggregated_results = {}
for sample_size in keys:
    for mask_size in keys_mask[sample_size]:
            print("Sample Size:")
            print(sample_size)
            print("Mask Size/Noise Variance:")
            print(mask_size)
            print("Who's Better?")
            print("TNN" if mse_dic_tnn[sample_size][mask_size]['avg_mse']<mse_dic_mnn[sample_size][mask_size]['avg_mse']\
                        else "MNN")
            tmp = {'avg_mse_tnn': mse_dic_tnn[sample_size][mask_size]['avg_mse'], 'std_mse_tnn':mse_dic_tnn[sample_size][mask_size]['std_mse'],\
                            'avg_mse_mnn': mse_dic_mnn[sample_size][mask_size]['avg_mse'], 'std_mse_mnn': mse_dic_mnn[sample_size][mask_size]['std_mse'] }  
            if sample_size in aggregated_results.keys():
                aggregated_results[sample_size][mask_size] = tmp
            else:
                aggregated_results[sample_size] = {mask_size: tmp}
            #print(mse_dic_tnn[sample_size][mask_size])
            #print(mse_dic_mnn[sample_size][mask_size])
            print(tmp)
print(aggregated_results.keys())