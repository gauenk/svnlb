
# -- python --
import cv2,tqdm,copy
import numpy as np
import unittest
import pyvnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- package helper imports --
from pyvnlb.pylib.tests.data_loader import load_dataset
from pyvnlb.pylib.tests.file_io import save_images
from pyvnlb import groups2patches,patches2groups,patches_at_indices

# -- python impl --
from pyvnlb.pylib.py_impl import initMask,processNLBayes,idx2coords

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
        fflow,bflow = pyvnlb.runPyFlow(noisy,std,flow_params)

        # -- pack data --
        data = edict()
        data.noisy = noisy
        data.fflow = fflow
        data.bflow = bflow
        data.t,data.c,data.h,data.w = data.noisy.shape

        return data,std

    def do_load_rand_data(self,t,c,h,w):

        # -- create data --
        data = edict()
        data.noisy = np.random.rand(t,c,h,w)*255.
        data.fflow = (np.random.rand(t,2,h,w)-0.5)*5.
        data.bflow = (np.random.rand(t,2,h,w)-0.5)*5.
        sigma = 20.

        # -- format + shape --
        for key,val in data.items():
            data[key] = data[key].astype(np.float32)
        data.t,data.c,data.h,data.w = data.noisy.shape

        return data,sigma

    #
    # -- [Exec] Comp Agg --
    #

    def do_init_mask(self,tensors,sigma,in_params,save=True):

        # -- init --
        # noisy = tensors.noisy[:3,:,:36,:36].copy()
        # tensors.noisy = noisy
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
        # tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}

        # -- exec pix --
        tchecks,nchecks = 2,0
        checks = np.random.permutation(h*w*c*(t-1))[:1000]
        for pidx in checks:

            # -- check boarder --
            pidx = pidx.item()
            step = 0
            ti,ci,wi,hi = idx2coords(pidx,w,h,c)
            valid_w = (wi + params.sizePatch[step]) < w
            valid_h = (hi + params.sizePatch[step]) < h
            if not(valid_w and valid_h): continue

            # -- cpp exec --
            cpp_results = pyvnlb.init_mask(noisy.shape,params)
            cpp_mask = cpp_results.mask
            cpp_ngroups = cpp_results.ngroups

            # -- py exec --
            py_results = initMask(noisy.shape,params)
            py_mask = py_results.mask
            py_ngroups = py_results.ngroups

            # -- compare --
            assert np.abs(cpp_ngroups - py_ngroups) < 1e-8
            np.testing.assert_allclose(cpp_mask,py_mask)

            if save:
                save_images(cpp_mask[:,None],SAVE_DIR/"cpp_mask.png",imax=1.)
                save_images(py_mask[:,None],SAVE_DIR/"py_mask.png",imax=1.)


    def do_run_proc_nlb(self,tensors,sigma,in_params,save=True):

        # -- init --
        # noisy = tensors.noisy[:3,:,:36,:36].copy()
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
        tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        # tensors = {}

        # -- cpp exec --
        cpp_params = copy.deepcopy(params)
        cpp_results = pyvnlb.processNLBayes(noisy,sigma,0,tensors,cpp_params)

        # -- unpack --
        cpp_denoised = cpp_results['denoised']
        cpp_basic = cpp_results['basic']
        cpp_ngroups = cpp_results['ngroups']

        # -- python exec --
        py_params = copy.deepcopy(params)
        py_results = processNLBayes(noisy,sigma,0,tensors,py_params)

        # -- unpack --
        py_denoised = py_results['denoised']
        py_basic = py_results['basic']
        py_ngroups = py_results['ngroups']

        # -- simple checks --
        assert abs(cpp_ngroups - py_ngroups) == 0

        # -- debug center --
        # print("-"*10 + " debug " + "-"*10)
        # s1 = np.stack([cpp_denoised[:,:,48:49,16:17],
        #                py_denoised[:,:,48:49,16:17]],-1)
        # s2 = np.stack([cpp_basic[:,:,48:49,16:17],
        #                py_basic[:,:,48:49,16:17]],-1)
        # print(s1)
        # print(s2)
        # print("-"*30)

        # -- save samples --
        if save:
            print(SAVE_DIR)
            save_images(cpp_denoised,SAVE_DIR / "./cpp_denoised.png",imax=255.)
            save_images(py_denoised,SAVE_DIR / "./py_denoised.png",imax=255.)
            save_images(cpp_basic,SAVE_DIR / "./cpp_basic.png",imax=255.)
            save_images(py_basic,SAVE_DIR / "./py_basic.png",imax=255.)

        # -- debug center --
        print(py_basic.shape)
        delta = np.abs(cpp_basic-py_basic)
        args = np.where(delta > 1.)
        print(args)

        # -- compare --
        np.testing.assert_allclose(cpp_denoised,py_denoised)
        np.testing.assert_allclose(cpp_basic,py_basic,rtol=5e-4)


    #
    # -- Call the Tests --
    #

    def test_init_mask(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        # self.do_init_mask(tensors,sigma,pyargs)

    def test_proc_nlb(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_proc_nlb(tensors,sigma,pyargs)

        # # -- modify patch size --
        # pyargs = {}
        # tensors,sigma = self.do_load_rand_data(5,3,32,32)
        # self.do_run_proc_nlb(tensors,sigma,pyargs)

