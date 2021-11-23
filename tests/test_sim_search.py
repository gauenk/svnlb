
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

def index_to_indices(index,shape):
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
    indices = index_to_indices(index,noisy.shape)
    tslice = slice(indices[0],indices[0]+psT)
    cslice = slice(indices[1],indices[1]+psX)
    hslice = slice(indices[2],indices[2]+psX)
    wslice = slice(indices[3],indices[3]+psX)
    return noisy[tslice,cslice,hslice,wslice]

def patch_at_indices(noisy,indices,psX,psT):
    patches = []
    for index in indices:
        patches.append(patch_at_index(noisy,index,psX,psT))
    patches = np.stack(patches)
    return patches


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


    def do_run_sim_search_patch_indexing(self,vnlb_dataset,in_params):

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
        indices = cpp_data['indices']
        psX,psT = cpp_data['psX'],cpp_data['psT']

        # -- ground truth patches --
        gt_patches = patch_at_indices(noisy,indices,psX,psT)

        # -- compare --
        np.testing.assert_array_equal(gt_patches,group)

        # -- save results --
        save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
        save_images(group[0],SAVE_DIR / f"./patches_pyvnlb.png",imax=255.)
        save_images(gt_patches,SAVE_DIR / f"./patches_gt.png",imax=255.)

        # -- show top indices --
        print(indices[:10])


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
