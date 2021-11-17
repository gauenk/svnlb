"""
Compare the Python API with the C++ Results

"""

# -- python imports --
import os
import numpy as np
import pandas as pd
from einops import rearrange
from easydict import EasyDict as edict
from collections import defaultdict

# -- this package --
import vnlb.pylib as pyvnlb

# -- local imports --
from data_loader import load_dataset




#
#  ---------------- Compare Results ----------------
#

def run_comparison():

    # -- run method using different image IOs when computing Optical Flow  --
    res_vnlb,res_pyvnlb_cv2 = run_method("cv2")
    res_vnlb,res_pyvnlb_iio = run_method("iio") # exactly matches C++ code
    res_vnlb = {'cv2':res_vnlb,'iio':res_vnlb} # both the same
    res_pyvnlb = {'cv2':res_pyvnlb_cv2,'iio':res_pyvnlb_iio}    
    
    # -- compare results --
    results = defaultdict(dict)
    for imageIO in ['cv2','iio']:
        for field in ["denoised",'fflow','bflow',"basic"]:
            cppField = res_vnlb[imageIO][field]
            pyField = res_pyvnlb[imageIO][field]
            totalError = np.sum(np.abs(cppField - pyField))
            rkey = f"Total Error ({imageIO})"
            results[field][rkey] = totalError
    results = pd.DataFrame(results)
    print(results)

#
# -- Comparison Code --
#

def run_method_tmp(ioForFlow):
    zeros = np.zeros(10)
    a = {"denoised":zeros,'fflow':zeros,'bflow':zeros,"basic":zeros}
    b = {"denoised":zeros,'fflow':zeros,'bflow':zeros,"basic":zeros}
    return a,b

def run_method(ioForFlow):

    #
    #  ---------------- Setup Parameters ----------------
    #
    
    omp_nthreads = int(os.getenv('OMP_NUM_THREADS'))
    assert omp_nthreads == 4,"run `export OMP_NUM_THREADS=4`"
    flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
                   "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
                   "verbose":False,"testing":False,'bw':False}
    
    #
    #  ---------------- Read Data (Image & VNLB-C++ Results) ----------------
    #
    
    res_vnlb,paths,fmts = load_dataset("davis_pariasm_vnlb")
    clean,noisy,std = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std
    noisyForFlow = pyvnlb.readVideoForFlow(noisy.shape,fmts.noisy)
    if ioForFlow == "cv2": noisyForFlow = noisy
    
    #
    #  ---------------- TV-L1 Optical Flow ----------------
    #
    
    fflow,bflow = pyvnlb.runPyFlow(noisyForFlow,std,flow_params)
    
    
    #
    #  ---------------- Video Non-Local Bayes ----------------
    #
    
    vnlb_params = {'fflow':fflow,'bflow':bflow,'testing':True}
    res_pyvnlb = pyvnlb.runPyVnlb(noisy,std,vnlb_params)
    
    return res_vnlb,res_pyvnlb

run_comparison() # exec file


"""
# ----------------------------------
# 
#     Compare Results with C++
# 
# ----------------------------------

# -- load c++ results --

# -- compate with cpp --
from pathlib import Path
print(paths['noisy'])
print(Path(paths['noisy'][0]).parents[0])
video_paths = fmts.noisy#Path(paths['noisy'][0]).parents[0] / "%03d.tif"
noisyForVnlb = pyvnlb.readVideoForVnlb(noisy.shape,video_paths,{'verbose':False})
print("Delta: ",np.sum(np.abs(noisy - noisyForVnlb)))
noisy_bw = pyvnlb.rgb2bw(noisy)
noisyForFlow = pyvnlb.readVideoForFlow(noisy_bw.shape,video_paths,{'verbose':False})
print("Delta: ",np.sum(np.abs(noisy_bw - noisyForFlow)))

# -- exec python --
pyargs = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
          "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
          "verbose":False,"testing":False,'bw':False}
fflow,bflow = pyvnlb.runPyFlow(noisyForFlow,std,pyargs)
pyargs = {'fflow':fflow,'bflow':bflow,'testing':True}
# pyargs = {'testing':True}
# pyargs['fflow'] = np.ascontiguousarray(res_vnlb.fflow)
# pyargs['bflow'] = np.ascontiguousarray(res_vnlb.bflow)
res_pyvnlb = pyvnlb.runPyVnlb(noisy,std,pyargs)
# res_pyvnlb = {}

# -- prepare dict --
res_pyvnlb['fflow'] = fflow
res_pyvnlb['bflow'] = bflow
pyvnlb.expand_flows(res_pyvnlb,axis=0) # nflows must match nframes

#
# -- compare outputs --
#

# fields = ['fflow','bflow']
fields = ['denoised','fflow','bflow',"basic"]

maxes = {}
for field in fields:
    pymax = np.abs(res_pyvnlb[field]).max()
    cppmax = np.abs(res_vnlb[field]).max()
    fmax = max(pymax,cppmax)
    maxes[field] = fmax if fmax < 255. else 255.
print(maxes)

for field in fields:
    print("\n\n\n\n")
    print(f"Results for {field}")
    cppField = res_vnlb[field]
    pyField = res_pyvnlb[field]
    psnrs = np.mean(pyvnlb.compute_psnrs(cppField,pyField,maxes[field]))
    rel = np.mean(np.abs(cppField - pyField)/(np.abs(cppField)+1e-10))
    print(f"[{field}] PSNR: %2.2f | RelError: %2.1e" % (psnrs,rel))
    if field in ['fflow','bflow']:
        cppField = rearrange(cppField,'c t h w -> t c h w')
        pyField = rearrange(pyField,'c t h w -> t c h w')
        save_images(f"cpp_{field}_0.png",cppField[:,[0]],imax=1.)
        save_images(f"py_{field}_0.png",pyField[:,[0]],imax=1.)
        save_images(f"cpp_{field}_1.png",cppField[:,[1]],imax=1.)
        save_images(f"py_{field}_1.png",pyField[:,[1]],imax=1.)
    else:
        psnrs = np.mean(pyvnlb.compute_psnrs(cppField,clean))
        print(f"Denoising PSNR [CPP,{field}]: %2.3f" % psnrs)
        psnrs = np.mean(pyvnlb.compute_psnrs(pyField,clean))
        print(f"Denoising PSNR [Py,{field}]: %2.3f" % psnrs)
        save_images(f"cpp_{field}.png",cppField)
        save_images(f"py_{field}.png",pyField)
"""
