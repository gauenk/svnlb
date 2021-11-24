
import cv2
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

SAVE_DIR = Path("./output/tests/")


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

class TestSimSearch(unittest.TestCase):

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


    def do_run_sim_search_patch_indexing(self,vnlb_dataset,in_params,save=True):

        # -- data --
        tensors,sigma = self.do_load_data(vnlb_dataset)
        noisy = tensors.noisy
        t,c,h,w = noisy.shape

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=in_params)

        for pidx in range(900,1000):

            # -- cpp exec --
            cpp_data = pyvnlb.simPatchSearch(noisy,sigma,pidx,
                                             tensors={'fflow':tensors['fflow'],
                                                      'bflow':tensors['bflow']},
                                             params=params)
            # -- unpack --
            group = cpp_data["groupNoisy"]
            group_og = cpp_data["groupNoisy_og"]
            indices = cpp_data['indices']
            psX,psT = cpp_data['psX'],cpp_data['psT']
            nSimP = cpp_data['npatches']
            nSimOG = cpp_data['npatches_og']
            nParts = cpp_data['nparts_omp']

            # -- ground truth patches --
            gt_patches = patches_at_indices(noisy,indices,psX,psT)
            gt_patches_og = patches2groups(gt_patches,c,psX,psT,nSimP,nSimOG,nParts)
            gt_patches_rs = groups2patches(gt_patches_og,c,psX,psT,nSimP)

            # -- [temp] --
            # print_value_order(group_og,gt_patches_og,c,psX,psT,nSimP)
            # print_value_order(group,gt_patches_rs,c,psX,psT,nSimP)
            # print_neq_values(group_og,gt_patches_og)
            # print_neq_values_fix_pix(group_og,gt_patches_og)

            # -- compare --
            np.testing.assert_array_equal(gt_patches,group)
            np.testing.assert_array_equal(gt_patches_og,group_og)
            np.testing.assert_array_equal(gt_patches_rs,group)

        # -- save [the pretty] results --
        if save:
            save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
            save_images(group,SAVE_DIR / f"./patches_pyvnlb.png",imax=255.)
            save_images(gt_patches,SAVE_DIR / f"./patches_gt.png",imax=255.)


    def test_sim_search(self):

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        self.do_run_sim_search_patch_indexing(vnlb_dataset,pyargs)

        # -- modify patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        self.do_run_sim_search_patch_indexing(vnlb_dataset,pyargs)

        # # -- modify number of elems --
        # pyargs = {'k':3}
        # self.do_run_sim_search(vnlb_dataset,pyargs)

        # # -- modify patch size & number of elems --
        # pyargs = {'ps_x':11,'k':3}
        # self.do_run_sim_search(vnlb_dataset,pyargs)
