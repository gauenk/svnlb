
# -- python --
import cv2,tqdm
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
from vnlb.cpu import computeAggregation,idx2coords,initMask

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")


#
#
# -- Primary Testing Class --
#
#

class TestCompAgg(unittest.TestCase):

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

    def do_run_comp_agg(self,tensors,sigma,in_params,save=True):

        # -- init --
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
        # tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}

        tchecks,nchecks = 10,0
        checks = np.random.permutation(h*w*c*(t-1))[:1000]
        for pidx in checks:

            # -- check boarder --
            pidx = pidx.item()
            step = 0
            ti,ci,wi,hi = idx2coords(pidx,w,h,c)
            valid_w = (wi + params.sizePatch[step]) < w
            valid_h = (hi + params.sizePatch[step]) < h
            if not(valid_w and valid_h): continue
            # print(pidx,ti,ci,wi,hi)

            # -- estimate similar patches --
            sim_data = vnlb.swig.simPatchSearch(noisy,sigma,pidx,tensors,params)
            indices = sim_data['indices']
            sim_patches = sim_data["patchesNoisy"]
            sim_groupNoisy = sim_data["groupNoisy"]
            sim_groupBasic = sim_data["groupBasic"]
            nSimP = sim_data['npatches']
            nSimP_og = sim_data['ngroups']

            # -- bayes denoiser --
            groupNoisy = sim_groupNoisy
            groupNoisy = sim_groupNoisy

            # -- remove zero-filled color channels --
            groupNoisy[...,1] = groupNoisy[...,0].copy()
            groupNoisy[...,2] = groupNoisy[...,0].copy()
            groupNoisy.ravel()[0] = 100.
            # bayes_results = vnlb.swig.computeBayesEstimate(sim_groupNoisy.copy(),
            #                                             sim_groupBasic.copy(),0.,
            #                                             nSimP,shape,params,step)
            # groupNoisy = bayes_results['groupNoisy']

            # -- init mask --
            rmask = initMask(noisy.shape,params,0)
            mask = rmask.mask
            ngroups = rmask.ngroups
            # indices[...] = 64*64*3+32

            # -- cpp exec --
            cpp_deno = np.zeros_like(noisy)
            cpp_group = groupNoisy
            cpp_weights = np.zeros((t,h,w),dtype=np.float32)
            cpp_mask = mask.copy()
            results = vnlb.swig.computeAggregation(cpp_deno,cpp_group,
                                                   indices,cpp_weights,
                                                   cpp_mask,nSimP)

            # -- unpack --
            cpp_deno = results['deno']
            cpp_mask = results['mask']
            cpp_weights = results['weights']
            cpp_nmasked = results['nmasked']
            psX,psT = results['psX'],results['psT']

            # -- python exec --
            py_deno = np.zeros_like(noisy)
            py_group = groupNoisy
            py_weights = np.zeros((t,h,w),dtype=np.float32)
            py_mask = mask.copy()
            py_results = computeAggregation(py_deno,py_group,indices,
                                            py_weights,py_mask,nSimP,params)

            # -- unpack --
            py_deno = py_results['deno']
            py_mask = py_results['mask']
            py_weights = py_results['weights']
            py_nmasked = py_results['nmasked']
            py_psX,py_psT = py_results['psX'],py_results['psT']

            # -- save samples --
            if save:
                save_images(cpp_mask[:,None],SAVE_DIR / "./cpp_mask.png",imax=1.)
                save_images(py_mask[:,None],SAVE_DIR / "./py_mask.png",imax=1.)

            # -- simple checks --
            assert abs(psX - py_psX) == 0
            assert abs(psT - py_psT) == 0
            assert abs(cpp_nmasked - py_nmasked) == 0

            # -- compare --
            np.testing.assert_array_equal(cpp_mask,py_mask)
            np.testing.assert_array_equal(cpp_weights,py_weights)
            np.testing.assert_array_equal(cpp_deno,py_deno)

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break


    #
    # -- Call the Tests --
    #

    def test_comp_agg(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_comp_agg(tensors,sigma,pyargs)

        # -- modify patch size --
        pyargs = {}
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_comp_agg(tensors,sigma,pyargs)


