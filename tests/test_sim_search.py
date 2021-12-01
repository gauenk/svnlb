
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
from pyvnlb.pylib.py_impl import runSimSearch,idx2coords

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

class TestSimSearch(unittest.TestCase):

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

        return data,std

    def do_load_rand_data(self,t,c,h,w):

        # -- create data --
        data = edict()
        data.noisy = np.random.rand(t,c,h,w)*255.
        data.fflow = (np.random.rand(t,2,h,w)-0.5)*5.
        data.bflow = (np.random.rand(t,2,h,w)-0.5)*5.
        sigma = 20.

        for key,val in data.items():
            data[key] = data[key].astype(np.float32)

        return data,sigma


    #
    # -- [Exec] Sim Search --
    #

    def do_run_sim_search(self,tensors,sigma,in_params,save=True):

        # -- init --
        noisy = tensors.noisy
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
        params.use_imread = True
        tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
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

            # -- cpp exec --
            cpp_data = pyvnlb.simPatchSearch(noisy,sigma,pidx,tensors,params)

            # -- unpack --
            cpp_patches = cpp_data["patchesNoisy"]
            cpp_group = cpp_data["groupNoisy"]
            cpp_indices = cpp_data['indices']
            cpp_psX,cpp_psT = cpp_data['psX'],cpp_data['psT']
            cpp_nSimP = cpp_data['npatches']
            cpp_ngroups = cpp_data['ngroups']
            cpp_nParts = cpp_data['nparts_omp']

            # -- python exec --
            py_data = runSimSearch(noisy,sigma,pidx,tensors,params)

            # -- unpack --
            py_patches = py_data.patches
            py_vals = py_data.values
            py_indices = py_data.indices
            nSimP = len(py_indices)
            nflat = py_data.nflat

            # -- compare --
            np.testing.assert_allclose(py_patches,cpp_patches,rtol=1e-5)

            # -- allow for swapping of "close" values --
            try:
                np.testing.assert_array_equal(py_indices,cpp_indices)
            except:
                neq_idx = np.where(cpp_indices != py_indices)
                check_pairwise_diff(py_vals[neq_idx])

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break

    #
    # -- [Exec] Patches2Groups and Groups2Patches --
    #

    def do_run_patches_xfer_groups(self,vnlb_dataset,in_params,save=True):

        # -- data --
        tensors,sigma = self.do_load_data(vnlb_dataset)
        noisy = tensors.noisy
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)
        params.use_imread = True
        tchecks,nchecks = 10,0
        checks = np.random.permutation(h*w*c*(t-1))[:100]
        flows = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}

        for pidx in checks:

            # -- check boarder --
            pidx = pidx.item()
            step = 0
            ti,ci,wi,hi = idx2coords(pidx,w,h,c)
            valid_w = (wi + params.sizePatch[step]) < w
            valid_h = (hi + params.sizePatch[step]) < h
            if not(valid_w and valid_h): continue

            # -- cpp exec --
            cpp_data = pyvnlb.simPatchSearch(noisy,sigma,pidx,
                                             tensors=flows,
                                             params=params)
            # -- unpack --
            patches = cpp_data["patchesNoisy"]
            groups = cpp_data["groupNoisy"]
            indices = cpp_data['indices']
            psX,psT = cpp_data['psX'],cpp_data['psT']
            nSimP = cpp_data['npatches']
            ngroups = cpp_data['ngroups']
            nParts = cpp_data['nparts_omp']

            # -- ground truth patches --
            gt_patches = patches_at_indices(noisy,indices,psX,psT)
            gt_groups = patches2groups(gt_patches,c,psX,psT,ngroups,nParts)
            gt_patches_r2 = groups2patches(gt_groups,c,psX,psT,nSimP)

            # -- compare --
            np.testing.assert_array_equal(gt_patches,patches)
            np.testing.assert_array_equal(gt_groups,groups)
            np.testing.assert_array_equal(gt_patches_r2,patches)

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break

        # -- save [the pretty] results --
        if save:
            save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
            save_images(patches,SAVE_DIR / f"./patches_pyvnlb.png",imax=255.)
            save_images(gt_patches,SAVE_DIR / f"./patches_gt.png",imax=255.)


    #
    # -- Call the Tests --
    #

    def test_patches_xfer_groups(self):

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        self.do_run_patches_xfer_groups(vnlb_dataset,pyargs)

        # -- modify patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        self.do_run_patches_xfer_groups(vnlb_dataset,pyargs)

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- modify patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- random data & modified patch size --
        pyargs = {}
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- random data & modified patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_sim_search(tensors,sigma,pyargs)

