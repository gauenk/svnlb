"""
Compare the Python API with the C++ Results

"""

# -- python imports --
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict

# -- this package --
import pyvnlb

# -- local imports --
from data_loader import load_dataset


#
#  ---------------- Compare Results ----------------
#

def run_comparison(vnlb_dataset):

    print(f"Running Comparison on {vnlb_dataset}")

    # -- run method using different image IOs when computing Optical Flow  --
    res_vnlb,res_pyvnlb_cv2 = run_method(vnlb_dataset,"cv2")
    res_vnlb,res_pyvnlb_cpp = run_method(vnlb_dataset,"cpp") # exactly matches C++ code
    res_vnlb = {'cv2':res_vnlb,'cpp':res_vnlb} # both the same
    res_pyvnlb = {'cv2':res_pyvnlb_cv2,'cpp':res_pyvnlb_cpp}

    # -- compare results --
    results = defaultdict(dict)
    for imageIO in ['cv2','cpp']:
        for field in ["noisyForFlow","noisyForVnlb","fflow","bflow","basic","denoised"]:
            cppField = res_vnlb[imageIO][field]
            pyField = res_pyvnlb[imageIO][field]
            totalError = np.sum(np.abs(cppField - pyField)/(np.abs(cppField)+1e-12))
            rkey = f"Total Error ({imageIO})"
            results[field][rkey] = totalError
    results = pd.DataFrame(results)
    print(results.to_markdown())

#
# -- Comparison Code --
#

def run_method(vnlb_dataset,ioForFlow):

    #
    #  ---------------- Setup Parameters ----------------
    #

    pyvnlb.check_omp_num_threads()
    flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
                   "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
                   "verbose":False,"testing":False,'bw':False}

    #
    #  ---------------- Read Data (Image & VNLB-C++ Results) ----------------
    #

    res_vnlb,paths,fmts = load_dataset(vnlb_dataset)
    clean,noisy,std = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std
    noisyForFlow = pyvnlb.readVideoForFlow(noisy.shape,fmts.noisy)
    noisyForVnlb = pyvnlb.readVideoForVnlb(noisy.shape,fmts.noisy)
    if ioForFlow == "cv2":
        flowImages = pyvnlb.rgb2bw(noisy)
        vnlbImages = noisy
    else:
        flowImages = noisyForFlow
        vnlbImages = noisyForVnlb

    #
    #  ---------------- TV-L1 Optical Flow ----------------
    #

    fflow,bflow = pyvnlb.runPyFlow(flowImages,std,flow_params)

    #
    #  ---------------- Video Non-Local Bayes ----------------
    #

    tensors={'fflow':fflow,'bflow':bflow}
    vnlb_params={'verbose':False,'testing':True}
    res_pyvnlb = pyvnlb.runPyVnlb(noisy,std,tensors,vnlb_params)

    #
    #  ---------------- Add Noisy Images to Show IO Changes ----------------
    #

    res_vnlb['noisyForFlow'] = noisyForFlow
    res_pyvnlb['noisyForFlow'] = flowImages
    res_vnlb['noisyForVnlb'] = noisyForVnlb
    res_pyvnlb['noisyForVnlb'] = vnlbImages

    return res_vnlb,res_pyvnlb

if __name__ == "__main__":

    # -- dataset example 1 --
    vnlb_dataset = "davis_64x64"
    run_comparison(vnlb_dataset)

    # -- dataset example 2 --
    # vnlb_dataset = "davis"
    # run_comparison(vnlb_dataset)

    # -- dataset example 3 --
    # vnlb_dataset = "gmobil"
    # run_comparison(vnlb_dataset)

