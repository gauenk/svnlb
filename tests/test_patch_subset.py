
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
from vnlb.testing.file_io import save_images,save_image
from vnlb.utils import groups2patches,patches2groups,patches_at_indices

# -- python impl --
from vnlb.gpu import initMask,processNLBayes,runPythonVnlb,patch_est_plot
from vnlb.gpu import exec_patch_subset,runSimSearch
from vnlb.utils import idx2coords,compute_psnrs

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

#
# -- Primary Testing Class --
#

class TestPatchSubset(unittest.TestCase):

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

    def do_run_patch_subset(self,tensors,sigma,in_params,save=True):


        # -- init --
        print("tensors.noisy.shape: ",tensors.noisy.shape)
        # hwSlice = slice(0,32)
        # hwSlice = slice(0,64)
        # hwSlice = slice(0,256)
        # hwSlice = slice(300,300+256)

        hwSliceX = slice(0,0+64)
        hwSliceY = slice(0,0+64)

        # hwSliceX = slice(200,200+256)
        # hwSliceY = slice(600,600+256)

        # hwSliceX = slice(264,264+128)
        # hwSliceY = slice(664,664+128)

        # noisy = tensors.noisy[:3,:,:16,:16].copy()
        # noisy = tensors.noisy[:3,:,-16:,-16:].copy()
        clean = tensors.clean[:,:,hwSliceX,hwSliceY].copy()
        noisy = tensors.noisy[:,:,hwSliceX,hwSliceY].copy()

        # noisy = tensors.noisy[:3,:,16:32,16:32].copy()
        # noisy = tensors.noisy[:3,:,:36,:36].copy()
        # noisy = tensors.noisy[:3,:,:96,:96].copy()
        # noisy = tensors.noisy[:3,:,:64,:64].copy()
        # noisy = tensors.noisy
        # clean = tensors.clean.copy()
        # noisy = tensors.noisy.copy()
        shape = noisy.shape
        t,c,h,w = noisy.shape
        save_images(noisy,SAVE_DIR / "./noisy.png",imax=255.)
        for ti in range(t):
            save_dir = Path("./data/dcrop/")
            if not(save_dir.exists()): save_dir.mkdir()
            fn = save_dir / ("%05d.jpg" % ti)
            print(fn)
            save_image(clean[ti],fn,imax=255.)

        # -- testing info --
        print("nosiy.shape: ",noisy.shape)
        print("clean.shape: ",clean.shape)

        # -- parse parameters --
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
        # tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        tensors = {}

        # -- create patches --
        gpu_params = copy.deepcopy(params)
        gpu_params.nstreams = 4
        gpu_params.rand_mask = False
        start = time.perf_counter()
        py_data = runSimSearch(noisy,sigma,tensors,gpu_params,0)
        patches = py_data.patches
        patches = rearrange(patches[0],'b n pt c ph pw -> b n (pt c ph pw)')
        patches = patches[:128]
        print("patches: ",patches.shape)

        # -- patch est plot --
        print("creating error bars.")
        # patch_est_plot(noisy,clean,sigma,tensors,params)

        # -- python exec --
        basic = np.zeros_like(noisy)
        py_params = copy.deepcopy(params)
        py_params['nstreams'] = [1,1]
        # py_params['nSimilarPatches'] = [100,60]
        print("start python.")
        start = time.perf_counter()
        stats,orders,bias = exec_patch_subset(patches,sigma)
        end = time.perf_counter() - start
        print("[py] exec time: ",end)

        print(stats)


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
        sigma = 50.
        shape = list(tensors.clean.shape)
        tensors.noisy = tensors.clean + sigma * np.random.normal(size=shape)
        # self.do_run_proc_nlb(tensors,sigma,pyargs)


        # -- random data & modified patch size --
        pyargs = {}
        # tensors,sigma = self.do_load_rand_data(5,3,854,480)
        # tensors,sigma = self.do_load_rand_data(10,3,96,96)
        # tensors,sigma = self.do_load_rand_data(10,3,128,128)
        # tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_patch_subset(tensors,sigma,pyargs)

