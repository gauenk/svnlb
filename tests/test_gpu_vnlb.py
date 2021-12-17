
# -- python --
import torch,time
import cv2,tqdm,copy
import numpy as np
import unittest
import vnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- package helper imports --
from vnlb.testing.data_loader import load_dataset
from vnlb.testing.file_io import save_images
from vnlb.utils import groups2patches,patches2groups,patches_at_indices

# -- python impl --
from vnlb.gpu import initMask,processNLBayes,runPythonVnlb
from vnlb.utils import idx2coords,compute_psnrs

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestProcNlb(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,vnlb_dataset):

        #  -- Read Data (Image & VNLB-C++ Results) --
        res_vnlb,paths,fmts = load_dataset(vnlb_dataset)
        clean,noisy,std = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std
        fflow,bflow = res_vnlb.fflow,res_vnlb.bflow

        #  -- TV-L1 Optical Flow --
        flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,
                       "nscales":100,"fscale":1,"zfactor":0.5,"nwarps":5,
                       "epsilon":0.01,"verbose":False,"testing":False,'bw':True}
        fflow,bflow = vnlb.swig.runPyFlow(noisy,std,flow_params)

        # -- pack data --
        data = edict()
        data.clean = clean
        data.noisy = noisy
        data.fflow = fflow
        data.bflow = bflow
        data.t,data.c,data.h,data.w = data.noisy.shape

        return data,std

    def do_load_rand_data(self,t,c,h,w):

        # -- create data --
        data = edict()
        sigma = 20.
        data.clean = np.random.rand(t,c,h,w)*255.
        data.noisy = np.random.normal(data.clean,sigma)
        data.fflow = (np.random.rand(t,2,h,w)-0.5)*5.
        data.bflow = (np.random.rand(t,2,h,w)-0.5)*5.

        # -- format + shape --
        for key,val in data.items():
            data[key] = data[key].astype(np.float32)
        data.t,data.c,data.h,data.w = data.noisy.shape

        return data,sigma

    def init_py_gpu(self):

        # -- unpack --
        tensors,sigma = self.do_load_rand_data(3,3,16,16)
        clean = tensors.clean.copy()
        noisy = tensors.noisy.copy()
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=None)
        tensors = {}

        # -- process --
        basic = np.zeros_like(noisy)
        py_params = copy.deepcopy(params)
        py_params['nstreams'] = [4,4]
        processNLBayes(noisy,basic,sigma,0,tensors,py_params)

    #
    # -- [Exec] Comp Agg --
    #

    def do_run_proc_nlb(self,tensors,sigma,in_params,save=True):


        # -- init --
        hwSlice = slice(0,32)
        # hwSlice = slice(0,64)
        # hwSlice = slice(0,256)
        # hwSlice = slice(200,200+256)
        # noisy = tensors.noisy[:3,:,:16,:16].copy()
        # noisy = tensors.noisy[:3,:,-16:,-16:].copy()
        clean = tensors.clean[:,:,hwSlice,hwSlice].copy()
        noisy = tensors.noisy[:,:,hwSlice,hwSlice].copy()
        # noisy = tensors.noisy[:3,:,16:32,16:32].copy()
        # noisy = tensors.noisy[:3,:,:36,:36].copy()
        # noisy = tensors.noisy[:3,:,:96,:96].copy()
        # noisy = tensors.noisy[:3,:,:64,:64].copy()
        # noisy = tensors.noisy
        # clean = tensors.clean.copy()
        # noisy = tensors.noisy.copy()
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- testing info --
        print("nosiy.shape: ",noisy.shape)
        print("clean.shape: ",clean.shape)

        # -- parse parameters --
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
        # tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        tensors = {}

        # -- python exec --
        basic = np.zeros_like(noisy)
        py_params = copy.deepcopy(params)
        py_params['nstreams'] = [4,4]
        print("start python.")
        start = time.perf_counter()
        py_results = runPythonVnlb(noisy,sigma,tensors,py_params)
        end = time.perf_counter() - start
        print("[py] exec time: ",end)

        # -- unpack --
        py_denoised = py_results['denoised'].cpu().numpy()
        py_basic = py_results['basic'].cpu().numpy()
        # py_ngroups = py_results['ngroups']

        # -- python --
        py_min = py_basic.min().item()
        py_mean = py_basic.mean().item()
        py_max = py_basic.max().item()
        py_psnr = compute_psnrs(py_basic,clean)
        print("[py.basic] psnrs: ",py_psnr)
        py_psnr = compute_psnrs(py_denoised,clean)
        print("[py.denoised] psnrs: ",py_psnr)

        # -- cpp exec --
        cpp_params = copy.deepcopy(params)
        print("start cpp.")
        start = time.perf_counter()
        cpp_results = vnlb.swig.runPyVnlb(noisy,sigma,tensors,cpp_params)
        end = time.perf_counter() - start
        print("[cpp] exec time: ",end)

        # -- unpack --
        cpp_denoised = cpp_results['denoised']
        cpp_basic = cpp_results['basic']
        # cpp_ngroups = cpp_results['ngroups']

        # -- cpp --
        cpp_min = cpp_basic.min().item()
        cpp_mean = cpp_basic.mean().item()
        cpp_max = cpp_basic.max().item()
        cpp_psnr = compute_psnrs(cpp_basic,clean)
        print("[cpp.basic] psnrs: ",cpp_psnr)
        cpp_psnr = compute_psnrs(cpp_denoised,clean)
        print("[cpp.denoised] psnrs: ",cpp_psnr)

        # -- delta --
        delta = np.abs(cpp_basic - py_basic)
        delta = np.sum(delta).item()
        print("[basic] delta: %2.3f\n" % delta)

        # -- delta --
        delta = np.abs(cpp_basic - py_basic)
        delta = np.sum(delta).item()
        print("[denoised] delta: %2.3f\n" % delta)

        # -- save samples --
        print("save: ",save)
        if save:
            save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
            save_images(cpp_denoised,SAVE_DIR / "./cpp_denoised.png",imax=255.)
            save_images(py_denoised,SAVE_DIR / "./py_denoised.png",imax=255.)
            save_images(cpp_basic,SAVE_DIR / "./cpp_basic.png",imax=255.)
            save_images(py_basic,SAVE_DIR / "./py_basic.png",imax=255.)

            # -- delta images --
            delta_basic = np.sum(np.abs(cpp_basic - py_basic),axis=1)[:,None]
            dmax = delta_basic.max()
            print(dmax)
            save_images(delta_basic,SAVE_DIR / "./delta_basic.png",imax=dmax)

            delta_denoised = np.abs(cpp_denoised - py_denoised)
            dmax = delta_denoised.max()
            save_images(delta_denoised,SAVE_DIR / "./delta_denoised.png",imax=dmax)


        # -- debug center --
        # print(py_basic.shape)
        # delta = np.abs(cpp_basic-py_basic)
        # args = np.where(delta > 1.)
        # print(args)

        # -- compare --
        # np.testing.assert_allclose(cpp_denoised,py_denoised)
        # np.testing.assert_allclose(cpp_basic,py_basic,rtol=1.5e-3)


    #
    # -- Call the Tests --
    #

    def test_proc_nlb(self):

        # -- init save path --
        seed = 234
        torch.manual_seed(seed)
        np.random.seed(seed)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        self.init_py_gpu()

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        # vnlb_dataset = "davis"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        # self.do_run_proc_nlb(tensors,sigma,pyargs)


        # -- random data & modified patch size --
        pyargs = {}
        # tensors,sigma = self.do_load_rand_data(5,3,854,480)
        # tensors,sigma = self.do_load_rand_data(10,3,96,96)
        # tensors,sigma = self.do_load_rand_data(10,3,128,128)
        # tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_proc_nlb(tensors,sigma,pyargs)

