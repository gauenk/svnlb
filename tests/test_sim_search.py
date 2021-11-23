
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


SAVE_DIR = Path("./output/tests/")

def index2indices(index,shape):
    t,c,h,w = shape

    tidx = index // (c*h*w)
    t_mod = index % (c*h*w)

    cidx = t_mod // (h*w)
    c_mod = t_mod % (h*w)

    hidx = c_mod // (h)
    h_mod = c_mod % (h)

    widx = h_mod# // w
    # c * wh + index + ht * whc + hy * w + hx
    indices = [tidx,cidx,hidx,widx]
    return indices

def patch_at_index(noisy,index,psX,psT):
    indices = index2indices(index,noisy.shape)
    tslice = slice(indices[0],indices[0]+psT)
    cslice = slice(indices[1],indices[1]+psX)
    hslice = slice(indices[2],indices[2]+psX)
    wslice = slice(indices[3],indices[3]+psX)
    return noisy[tslice,cslice,hslice,wslice]

def patches_at_indices(noisy,indices,psX,psT):
    patches = []
    for index in indices:
        patches.append(patch_at_index(noisy,index,psX,psT))
    patches = np.stack(patches)
    return patches

def reorder_patches_to_img(group,c,psX,psT,nSimP):

    # -- setup --
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = nSimP * psX * psX * psT * c
    group_f = group.ravel()[:numNz]

    # -- [og -> img] --
    group = group_f.reshape(c,psT,-1)
    group = ncat(group,axis=1)
    group = group.reshape(c*psT,psX**2,nSimP).transpose(2,0,1)
    group = ncat(group,axis=0)

    # -- final reshape --
    group = group.reshape(nSimP,psT,c,psX,psX)

    return group


def reorder_patches_to_og(group,c,psX,psT,nSimP,nSimOG,nParts):

    # -- setup --
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = nSimP * psX * psX * psT * c
    group = group.ravel()[:numNz]

    # -- [img -> og] --
    group = group.reshape(nSimP,psX*psX,c*psT).transpose(1,2,0)
    group = ncat(group,axis=0)
    group = group.reshape(psT,c,nSimP*psX*psX)
    group = ncat(group,axis=1)

    # -- fill with zeros --
    group_f = group.ravel()[:numNz]
    group = np.zeros(size*nSimOG)
    group[:size*nSimP] = group_f[...]
    group = group.reshape(nParts,psT,c,psX,psX,nSimOG)

    return group

def print_value_order(group_og,gt_patches_og,c,psX,psT,nSimP):
    order = []
    size = psX * psX * psT * c
    start,pidx = 5000,0
    gidx = 30
    group_og_f = group_og.ravel()[:size*nSimP]
    patch_cmp = gt_patches_og.ravel()[:size*nSimP]
    # group_og_f = group_og[0,...,gidx].ravel()
    # patch_cmp = gt_patches_og[0,...,gidx].ravel()
    # print(np.stack([group_og_f,patch_cmp],axis=-1))
    # print(group_og_f)
    # print(patch_cmp)
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
    for i in range(patch_cmp.shape[0]):
        idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
        print(i,idx)
        # idx = idx[0]
        if (i+skip*pidx) in idx: idx = i
        elif len(idx) > 0: idx = idx[0]
        else: idx = -1
        if idx != i:
            print(gidx,i,np.abs(patch_cmp[i] - group_og_f[i]))
        #     print(i,idx)
        #     break
        order.append(idx)
    print(order)

def print_neq_values(group_og,gt_patches_og):
    order = []
    skip,pidx = 0,0
    gidx = 20
    for gidx in range(0,103):
        group_og_f = group_og[0,...,gidx].ravel()
        patch_cmp = gt_patches_og[0,...,gidx].ravel()
        # print(len(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0]))
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            # print(i,idx)
            # idx = idx[0]
            if (i+skip*pidx) in idx: idx = i
            elif len(idx) > 0: idx = idx[0]
            else: idx = -1
            if idx != i:
                print(gidx,i,np.abs(patch_cmp[i] - group_og_f[i]))
            #     print(i,idx)
            #     break
        #     order.append(idx)
        # print(order)

def print_neq_values_fix_pix(group_og,gt_patches_og):
    order = []
    skip,pidx = 0,0
    gidx = 20
    shape = group_og.shape
    for gidx in range(1,2):
        group_og_f = group_og.reshape(shape[0],-1,shape[-1]).ravel()#[:,1,:].ravel()
        patch_cmp = gt_patches_og.reshape(shape[0],-1,shape[-1])[:,gidx,:].ravel()
        # print(len(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0]))
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            print(gidx,i,idx)
        #     # print(i,idx)
        #     # idx = idx[0]
        #     if (i+skip*pidx) in idx: idx = i
        #     elif len(idx) > 0: idx = idx[0]
        #     else: idx = -1
        #     if idx != i:
        #         print(gidx,i,np.abs(patch_cmp[i] - group_og_f[i]))
        #     #     break
        # #     order.append(idx)
        # # print(order)


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
        # pidx = h*(w//2)+w//2
        pidx = 968

        # -- parse parameters --
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,in_params)

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
        gt_patches_og = reorder_patches_to_og(gt_patches,c,psX,psT,nSimP,nSimOG,nParts)
        gt_patches_rs = reorder_patches_to_img(gt_patches_og,c,psX,psT,nSimP)
        # gt_patches_rs = reorder_patches_to_img(group_og,c,psX,psT,nSimP)

        # -- [temp] --
        # print_value_order(group_og,gt_patches_og,c,psX,psT,nSimP)
        # print_value_order(group,gt_patches_rs,c,psX,psT,nSimP)
        # print_neq_values(group_og,gt_patches_og)
        # print_neq_values_fix_pix(group_og,gt_patches_og)


        # -- compare --
        np.testing.assert_array_equal(gt_patches,group)
        np.testing.assert_array_equal(gt_patches_og,group_og)

        # -- save [the pretty] results --
        if save:
            save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
            save_images(group,SAVE_DIR / f"./patches_pyvnlb.png",imax=255.)
            save_images(gt_patches,SAVE_DIR / f"./patches_gt.png",imax=255.)

        # # -- show top indices --
        # print(indices[:10])


    def test_sim_search(self):

        # -- init save path --
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        self.do_run_sim_search_patch_indexing(vnlb_dataset,pyargs)

        # # -- modify patch size --
        # pyargs = {'ps_x':11}
        # self.do_run_sim_search(vnlb_dataset,pyargs)


        # # -- modify number of elems --
        # pyargs = {'k':3}
        # self.do_run_sim_search(vnlb_dataset,pyargs)

        # # -- modify patch size & number of elems --
        # pyargs = {'ps_x':11,'k':3}
        # self.do_run_sim_search(vnlb_dataset,pyargs)
