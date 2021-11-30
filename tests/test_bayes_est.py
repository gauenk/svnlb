
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
from pyvnlb.pylib.py_impl import runBayesEstimate,idx2coords

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

class TestBayesEstimate(unittest.TestCase):

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
    # -- [Exec] Bayes Estimate --
    #

    def do_run_bayes_est(self,tensors,sigma,in_params,save=True):

        # -- init --
        noisy = tensors.noisy
        shape = noisy.shape
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
        # tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}

        tchecks,nchecks = 3,0
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
            cpp_data = pyvnlb.simPatchSearch(noisy,sigma,pidx,tensors,params)
            cpp_group = cpp_data["groupNoisy"]
            cpp_group_og = cpp_data["groupNoisy_og"]
            nSimP = cpp_data['npatches']
            nSimP_og = cpp_data['npatches_og']

            # -- cpp exec --
            results = pyvnlb.computeBayesEstimate(cpp_group_og,cpp_group_og,
                                                  0.,nSimP_og,shape,params)

            # -- unpack --
            cpp_groupNoisy = results['groupNoisy']
            cpp_groupBasic = results['groupBasic']
            cpp_group = results['group']
            cpp_center = results['center']
            cpp_covMat = results['covMat']
            cpp_covEigVecs = results['covEigVecs']
            cpp_covEigVals = results['covEigVals']
            cpp_rank_var = results['rank_var']
            psX = results['psX']
            psT = results['psT']

            # -- python exec --
            py_results = runBayesEstimate(cpp_group_og,cpp_group_og,
                                          0.,nSimP_og,shape,params)

            # -- unpack --
            py_groupNoisy = py_results['groupNoisy']
            py_groupBasic = py_results['groupBasic']
            py_group = py_results['group']
            py_center = py_results['center']
            py_covMat = py_results['covMat']
            py_covEigVecs = py_results['covEigVecs']
            py_covEigVals = py_results['covEigVals']
            py_rank_var = py_results['rank_var']
            py_psX = py_results['psX']
            py_psT = py_results['psT']

            # -- simple checks --
            assert abs(py_psX - psX) == 0
            assert abs(py_psT - psT) == 0

            # -- tests --
            np.testing.assert_array_equal(py_groupNoisy,cpp_groupNoisy)
            np.testing.assert_array_equal(py_groupBasic,cpp_groupBasic)
            np.testing.assert_array_equal(py_group,cpp_group)
            np.testing.assert_array_equal(py_center,cpp_center)
            np.testing.assert_array_equal(py_covMat,cpp_covMat)
            np.testing.assert_array_equal(py_covEigVecs,cpp_covEigVecs)
            np.testing.assert_array_equal(py_covEigVals,cpp_covEigVals)
            np.testing.assert_array_equal(py_rank_var,cpp_rank_var)

            # -- neq messages --
            # neq_idx = np.where(cpp_indices != py_indices)
            # print(neq_idx[0])
            # perc_nz = check_if_reordered(py_indices[neq_idx],cpp_indices[neq_idx])
            # perc_ic = len(neq_idx[0]) / (1.*len(py_indices)) * 100.
            # print(perc_nz,perc_ic)

            # -- expore --
            # print(cpp_indices)
            # print(py_indices)
            # print(py_indices[neq_idx[0]])
            # print(cpp_indices[neq_idx[0]])
            # print(py_indices[neq_idx])
            # print(cpp_indices[neq_idx])
            # examples = np.stack([py_indices[:3],cpp_indices[:3]],axis=-1).T
            # print(examples)

            # -- allow for swapping of "close" values --
            # try:
            #     np.testing.assert_array_equal(py_indices,cpp_indices)
            # except:
            #     print("SWAPPED!")
            #     neq_idx = np.where(cpp_indices != py_indices)
            #     check_pairwise_diff(py_vals[neq_idx])

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break

            # -- save samples --
            if save:

                patches = groups2patches(cpp_group,1,psX,psT,nSimP)
                save_images(patches,SAVE_DIR / "./patches.png",imax=255.)

                patches = groups2patches(cpp_groupNoisy,c,psX,psT,nSimP)
                print(patches.shape,c,psX,psT,nSimP)
                save_images(patches,SAVE_DIR / "./groupNoisy.png",imax=255.)

                patches_ave = patches.mean(axis=0)
                print(patches.shape,c,psX,psT,nSimP)
                save_images(patches,SAVE_DIR / "./groupNoisy.png",imax=255.)

                patches = groups2patches(cpp_groupBasic,c,psX,psT,nSimP)
                print(patches.shape,c,psX,psT,nSimP)
                save_images(patches,SAVE_DIR / "./groupBasic.png",imax=255.)



    #
    # -- Call the Tests --
    #

    def test_run_bayes_estimate(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_bayes_est(tensors,sigma,pyargs)

        # -- modify patch size --
        pyargs = {}
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        # self.do_run_bayes_est(tensors,sigma,pyargs)


