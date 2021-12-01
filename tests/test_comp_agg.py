
# -- python --
import cv2,tqdm
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
from pyvnlb.pylib.py_impl import computeAggregation,idx2coords

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

def check_if_reordered(data_a,data_b):
    delta = np.zeros((len(data_a),len(data_b)))
    for a_i in range(len(data_a)):
        for b_i in range(len(data_b)):
            delta[a_i,b_i] = np.sum(np.abs(data_a[a_i]-data_b[b_i]))
    row_ind,col_ind = optimize.linear_sum_assignment(delta)
    perc_nz = (delta[row_ind, col_ind] > 0.).astype(np.float32).mean()*100
    return perc_nz

def print_value_order(group_og,gt_patches_og,c,psX,psT,nSimP):

    # -- create patches --
    order = []
    size = psX * psX * psT * c
    start,pidx = 5000,0
    gidx = 30
    group_og_f = group_og.ravel()[:size*nSimP]
    patch_cmp = gt_patches_og.ravel()[:size*nSimP]

    # -- message --
    print("Num Eq: ",len(np.where(np.abs(patch_cmp - group_og_f)<1e-10)[0]))
    print(np.where(np.abs(patch_cmp - group_og_f)<1e-10)[0])

    print("Num Neq: ",len(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0]))
    print(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0])

    # -- the two are zero at _exactly_ the same indices --
    # pzeros = np.where(np.abs(patch_cmp)<1e-10)[0]
    # print(pzeros,len(pzeros))
    # gzeros = np.where(np.abs(group_og_f)<1e-10)[0]
    # print(gzeros,len(gzeros))
    # print(np.sum(np.abs(gzeros-pzeros)))

    return

def print_neq_values(group_og,gt_patches_og):
    order = []
    skip,pidx = 0,0
    gidx = 20
    for gidx in range(0,103):
        group_og_f = group_og[0,...,gidx].ravel()
        patch_cmp = gt_patches_og[0,...,gidx].ravel()
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            if (i+skip*pidx) in idx: idx = i
            elif len(idx) > 0: idx = idx[0]
            else: idx = -1
            if idx != i:
                print(gidx,i,np.abs(patch_cmp[i] - group_og_f[i]))

def print_neq_values_fix_pix(group_og,gt_patches_og):
    order = []
    skip,pidx,gidx = 0,0,20
    shape = group_og.shape
    for gidx in range(1,2):
        group_og_f = group_og.reshape(shape[0],-1,shape[-1]).ravel()#[:,1,:].ravel()
        patch_cmp = gt_patches_og.reshape(shape[0],-1,shape[-1])[:,gidx,:].ravel()
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            print(gidx,i,idx)

def check_pairwise_diff(vals,tol=1e-4):
    # all the "same" value;
    # the order change is
    # due to small (< tol) differences
    nvals = vals.shape[0]
    # delta = np.zeros(nvals,nvals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[0]):
            # delta[i,j] = np.abs(vals[i] - vals[j])
            assert np.abs(vals[i] - vals[j]) < tol

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

    def do_run_comp_agg(self,tensors,sigma,in_params,save=True):

        # -- init --
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
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
            sim_data = pyvnlb.simPatchSearch(noisy,sigma,pidx,tensors,params)
            indices = sim_data['indices']
            sim_patches = sim_data["patchesNoisy"]
            sim_groupNoisy = sim_data["groupNoisy"]
            nSimP = sim_data['npatches']
            nSimP_og = sim_data['ngroups']

            # -- cpp exec --
            cpp_deno = np.zeros_like(noisy)
            cpp_group = sim_groupNoisy
            cpp_weights = np.zeros((t,h,w),dtype=np.float32)
            cpp_mask = np.zeros((t,h,w),dtype=np.int8)
            results = pyvnlb.computeAggregation(cpp_deno,cpp_group,
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
            py_group = sim_groupNoisy
            py_weights = np.zeros((t,h,w),dtype=np.float32)
            py_mask = np.zeros((t,h,w),dtype=np.int8)
            py_results = computeAggregation(py_deno,py_group,indices,
                                            py_weights,py_mask,nSimP,params)

            # -- unpack --
            py_deno = py_results['deno']
            py_mask = py_results['mask']
            py_weights = py_results['weights']
            py_nmasked = py_results['nmasked']
            py_psX,py_psT = py_results['psX'],py_results['psT']

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


            # -- save samples --
            if save:
                save_images(cpp_deno,SAVE_DIR / "./cpp_deno.png",imax=255.)


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


