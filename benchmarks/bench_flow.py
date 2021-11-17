
"""
Comparing the pyvnlb (this repo) 
v.s. the vnlb (the original repo)
TVL1Flow output

"""
import os,cv2
import pickle
import subprocess
from PIL import Image

import torch
import torchvision.utils as tvUtils

import numpy as np
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

import vnlb.pylib as pyvnlb
from vnlb.benchmarks.utils import *
from vnlb.benchmarks.io import read_result
from vnlb.benchmarks.create_noisy_burst import get_noisy_burst,get_vnlb_burst

# VNLB_ROOT = Path("/home/gauenk/Documents/packages/vnlb")
# PYVNLB_ROOT = Path("/home/gauenk/Documents/packages/pyvnlb")

def th_save_image(burst,fn):
    burst = torch.FloatTensor(burst)
    # burst = rearrange(burst,'c t h w -> t c h w')
    tvUtils.save_image(burst,fn)

def unpack_vnlb(vnlb_path,fstart,nframes):
    # -- read flow,basic,denoised --
    results = edict()
    results.fflow = read_result(vnlb_path,"tvl1_%03d_f.flo",fstart,nframes)
    results.bflow = read_result(vnlb_path,"tvl1_%03d_b.flo",fstart,nframes)

    return results

def exec_vnlb(vnlb_path,npaths,std,fstart,nframes):

    # -- read info [if exsists] --
    results = unpack_vnlb(vnlb_path,fstart,nframes)
    print(len(results.values()))
    any_none = False
    for elem in results.values():
        if elem is None: any_none = True
    if any_none: 
        print("Error: please run the vnlb separately.")
        exit()

    # -- format --
    results['fflow'] = rearrange(results['fflow'],'t h w two -> t two h w')
    results['bflow'] = rearrange(results['bflow'],'t h w two -> t two h w')

    return results

def exec_pyflow(pyvnlb_path,noisy,std):

    # -- read info [if exsists] --
    rerun = True
    if pyvnlb_path.exists() and not(rerun):
        results = pickle.load(open(str(pyvnlb_path),'rb'))
        if 'flow' in results: del results['flow']
        return results

    # -- exec --
    pyargs = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
              "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
              "verbose":True,"testing":True,"bw":True}
    fflow,bflow= pyvnlb.runPyFlowFB(noisy,std,pyargs)
    results = {'fflow':fflow,'bflow':bflow}
    if 'flow' in results.keys(): del results['flow']
    pyvnlb.expand_flows(results)

    # -- save to file --
    pickle.dump(results,open(str(pyvnlb_path),'wb'))

    return results

def run_comparison():

    # -- parms --
    std,fstart,nframes = 20,0,5
    ipath = Path("../vnlb/data/davis_baseball_64x64/")
    opath = Path(f"../vnlb/output/davis_baseball_64x64_{std}/")
    vnlb_path = opath / "./vnlb/"
    pyvnlb_path = opath / f"./pyvnlb_vnlb.pkl"
    pyflow_path = opath / f"./pyvnlb_flow.pkl"

    # -- get images --
    clean,noisy,npaths = get_vnlb_burst(ipath,vnlb_path,fstart,nframes)

    # -- to black and white --
    # bwnoisy = []
    # for t in range(noisy.shape[0]):
    #     # frame = rearrange(noisy[t],'h w c -> c h w')
    #     frame = noisy[t]
    #     # frame = .299 * frame[...,0] + .587 * frame[...,1] + .114 * frame[...,2]
    #     # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    #     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) 
    #     bwnoisy.append(np.array(frame)[:,:,None])
    # bwnoisy = np.stack(bwnoisy)
    # noisy  = bwnoisy


    noisy = rearrange(noisy,'t h w c -> t c h w')
    clean = rearrange(clean,'t h w c -> t c h w')

    # import torch
    # import torchvision.utils as tvUtils
    # tvUtils.save_image(torch.FloatTensor(noisy)/255.,"burst_bw_outer.png")
    

    # th_save_image(noisy/255.,"noisy.png")
    # noisy = np.flip(noisy,axis=0).copy()

    # -- print info --
    nmin,nmax,nmean = noisy.min(),noisy.max(),noisy.mean()
    print("[noisy]:  %2.2e, %2.2e, %2.2e" % (nmin,nmax,nmean))

    # -- exec vnlb --
    vnlb_res = exec_vnlb(vnlb_path,npaths,std,fstart,nframes)

    # -- exec pyvnlb --    
    pyvnlb_res = exec_pyflow(pyflow_path,noisy,std)

    # -- compare vnlb & pyvnlb --    
    fields = list(pyvnlb_res.keys())
    print(fields)
    for field in fields:
        cppField = vnlb_res[field]
        pyField  = pyvnlb_res[field]
        # cppField = np.around(vnlb_res[field],7)
        # pyField  = np.around(pyvnlb_res[field],7)
        print(cppField.shape,pyField.shape)
        for i in range(2):
            pblock = np.stack([cppField[:,i,32,32],pyField[:,i,32,32]],axis=-1)
            print(field,pblock)
            # print(field,cppField[:,i,38,38],pyField[:,i,38,38])
        delta = np.abs(cppField - pyField)
        dmean = np.sum(delta)
        psnrs = compute_psnrs(cppField,pyField)
        psnr = np.mean(psnrs)
        rel = relative_error(pyField,cppField)
        print(field)
        print(psnrs)
        print("[%s]: %2.3f | %2.2f | %2.2e" % (field,dmean,psnr,rel))
        save_field(field,cppField,pyField)
        print("delta.shape: ",delta.shape)
        print(delta.max())
        th_save_image(200000.*delta,"delta_bench_flow.png")
        if field in ["denoised","basic"]:
            th_save_image(clean/255.,"clean.png")
            cpp_psnr = compute_psnrs(cppField,clean)
            py_psnr = compute_psnrs(pyField,clean)
            print("[PSNRS]: Cpp: %2.2f Python: %2.2f" % (cpp_psnr,py_psnr))

if __name__ == "__main__":
    run_comparison()
