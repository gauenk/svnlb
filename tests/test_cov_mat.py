
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
from vnlb.cpu import computeCovMat
from vnlb.utils import idx2coords

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestCovMat(unittest.TestCase):

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
    # -- [Exec] Cov Checks --
    #

    def do_run_cov_mat(self,tensors,sigma,in_params,save=True):

        # -- init --
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape
        step = 0

        # -- parse parameters --
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)

        # -- estimate similar patches --
        tchecks,nchecks = 10,0
        checks = np.random.permutation(h*w*c*(t-1))[:1000]
        for pidx in checks:

            # -- check border --
            pidx = pidx.item()
            ti,ci,wi,hi = idx2coords(pidx,w,h,c)
            valid_w = (wi + params.sizePatch[step]) < w
            valid_h = (hi + params.sizePatch[step]) < h
            if not(valid_w and valid_h): continue

            # -- unpack --
            rank = params.rank[0]
            params.use_imread = [False,False]
            sim_data = vnlb.swig.simPatchSearch(noisy,sigma,pidx,tensors,params)
            patches = sim_data["patchesNoisy"]
            groupNoisy = sim_data["groupNoisy"]
            nSimP = sim_data['npatches']
            nSimP_og = sim_data['ngroups']

            # -- cpp exec --
            cpp_results = vnlb.swig.computeCovMat(groupNoisy[:,:,0],rank)

            # -- unpack --
            cpp_cov = cpp_results['covMat']
            cpp_evals = cpp_results['covEigVals']
            cpp_evecs = cpp_results['covEigVecs']

            # -- python exec --
            py_results = computeCovMat(groupNoisy[:,:,0],rank)

            # -- unpack --
            py_cov = py_results['covMat']
            py_evals = py_results['covEigVals']
            py_evecs = py_results['covEigVecs']


            # -- compare --
            np.testing.assert_allclose(cpp_cov,py_cov,rtol=5e-5)

            # -- compare eig vecs & allow for coeff swapping --
            delta_pos = np.abs(py_evecs + cpp_evecs)
            delta_neg = np.abs(py_evecs - cpp_evecs)
            delta = np.minimum(delta_pos,delta_neg).mean()
            assert delta < 1.5e-7

            # -- relative error to max value;  --
            # -- eigVals num. stability related to e_max/e_min ratio --
            emax = cpp_evals.max()
            delta = np.abs(py_evals-cpp_evals)/emax
            delta = delta.sum()
            assert delta < 5e-5

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break


    #
    # -- Call the Tests --
    #

    def test_cov_mat(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_cov_mat(tensors,sigma,pyargs)

        # # -- modify patch size --
        # pyargs = {}
        # tensors,sigma = self.do_load_rand_data(5,3,32,32)
        # self.do_run_proc_nlb(tensors,sigma,pyargs)

