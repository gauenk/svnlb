
"""
Comparing the pyvnlb (this repo) 
v.s. the vnlb (the original repo)
TVL1Flow output

"""
import os
import pickle
import subprocess
import numpy as np
import torch
import torchvision.utils as tvUtils
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

import vnlb.pylib as pyvnlb
from vnlb.benchmarks.utils import *
from vnlb.benchmarks.io import read_result
from vnlb.benchmarks.create_noisy_burst import get_noisy_burst,get_vnlb_burst
from vnlb.benchmarks.bench_flow import exec_pyflow

def unpack_vnlb(vnlb_path,fstart,nframes):
    # -- read flow,basic,denoised --
    results = edict()
    results.fflow = read_result(vnlb_path,"tvl1_%03d_f.flo",fstart,nframes)
    results.bflow = read_result(vnlb_path,"tvl1_%03d_b.flo",fstart,nframes)
    results.basic = read_result(vnlb_path,"bsic_%03d.tif",fstart,nframes)
    results.denoised = read_result(vnlb_path,"deno_%03d.tif",fstart,nframes)

    # -- reshaping --
    for key,val in results.items():
        if key == "std": continue
        results[key] = rearrange(val,'t h w c -> t c h w')
    # results.denoised = rearrange(results.denoised,'t h w c -> c t h w')
    # results.basic = rearrange(results.basic,'t h w c -> c t h w')

    return results

def exec_vnlb(vnlb_path,npaths,std,fstart,nframes):

    # -- read info [if exsists] --
    results = unpack_vnlb(vnlb_path,fstart,nframes)
    any_none = False
    for elem in results.values():
        if elem is None: any_none = True
    if any_none: 
        print("Error: please run the vnlb separately.")
        exit()

    # -- format --
    results['fflow'] = rearrange(results['fflow'],'t h w two -> two t h w')
    results['bflow'] = rearrange(results['bflow'],'t h w two -> two t h w')

    return results
    
def exec_pyvnlb(pyvnlb_path,pyflow_path,noisy,std,flows=None):

    # -- read info [if exsists] --
    rerun = True
    if pyvnlb_path.exists() and not(rerun):
        results = pickle.load(open(str(pyvnlb_path),'rb'))
        if 'final' in results: del results['final']
        return results

    # -- exec flow --
    if flows is None:
        # pyargs = {'testing':True,'fflow':fflow,'bflow':bflow}
        # fresults = exec_pyflow(pyflow_path,noisy,std)
        pyargs = {'testing':True}
        fresults = pyvnlb.runPyTvL1Flow(noisy,std,pyargs)
        fflow,bflow = fresults['fflow'],fresults['bflow']
        _fflow = rearrange(fflow,'two t h w -> t two h w')
        _bflow = rearrange(bflow,'two t h w -> t two h w')
        # print(fflow.shape)
        # print(noisy.shape)
        # print(fflow[:,0].max(),fflow[:,0].min())
        # print(fflow[:,1].max(),fflow[:,1].min())
        # print(bflow[:,0].max(),bflow[:,0].min())
        # print(bflow[:,1].max(),bflow[:,1].min())
    else:
        fflow = flows['fflow']
        bflow = flows['bflow']
        _fflow = rearrange(fflow,'two t h w -> t two h w')
        _bflow = rearrange(bflow,'two t h w -> t two h w')
        _fflow = np.ascontiguousarray(_fflow.copy())
        _bflow = np.ascontiguousarray(_bflow.copy())
    
    # -- exec denoiser --
    pyargs = {'testing':True,'fflow':_fflow,'bflow':_bflow}
    results = pyvnlb.runPyVnlb(noisy,std,pyargs)
    del results['final']

    # -- save to file --
    results['fflow'] = fflow
    results['bflow'] = bflow
    pickle.dump(results,open(str(pyvnlb_path),'wb'))

    return results

def run_comparison():

    # -- parms --
    std,fstart,nframes = 20,0,5
    ipath = Path("/home/gauenk/Documents/packages/vnlb/")
    # ipath = ipath / Path("output/davis_baseball_64x64_20/vnlb/")
    ipath = ipath / Path("data/davis_baseball/")
    # ipath = Path("../vnlb/data/davis_baseball_64x64/")
    opath = Path(f"../vnlb/output/davis_baseball_64x64_20/")
    vnlb_path = opath / "./vnlb/"
    pyvnlb_path = opath / f"./pyvnlb_vnlb.pkl"
    pyflow_path = opath / f"./pyvnlb_flow.pkl"

    # -- get images --
    clean,noisy,npaths = get_vnlb_burst(ipath,vnlb_path,fstart,nframes)
    noisy = rearrange(noisy,'t h w c -> t c h w')
    clean = rearrange(clean,'t h w c -> t c h w')
    
    # -- exec vnlb --
    vnlb_res = exec_vnlb(vnlb_path,npaths,std,fstart,nframes)

    # -- exec pyvnlb --    
    # flows = None
    flows = {'fflow':vnlb_res['fflow'],'bflow':vnlb_res['bflow']}
    # flows = {'fflow':np.zeros_like(vnlb_res['fflow']),
    #          'bflow':np.zeros_like(vnlb_res['bflow'])}
    pyvnlb_res = exec_pyvnlb(pyvnlb_path,pyflow_path,noisy,std,flows)

    # -- compare vnlb & pyvnlb --    
    fields = list(pyvnlb_res.keys())
    print(fields)
    for field in fields:
        cppField = vnlb_res[field]
        pyField  = pyvnlb_res[field]
        print(cppField.shape,pyField.shape)
        delta = np.mean(np.abs(cppField - pyField))
        psnr = compute_psnrs(cppField,pyField)
        rel = relative_error(pyField,cppField)
        print("[%s]: %2.3f | %2.2f | %2.2e" % (field,delta,psnr,rel))
        # save_field(field,cppField,pyField)
        # if field in ["denoised","basic"]:
        #     th_save_image(clean/255.,"clean.png")
        #     cpp_psnr = compute_psnrs(cppField,clean)
        #     py_psnr = compute_psnrs(pyField,clean)
        #     print("[PSNRS]: Cpp: %2.2f Python: %2.2f" % (cpp_psnr,py_psnr))


if __name__ == "__main__":
    run_comparison()
