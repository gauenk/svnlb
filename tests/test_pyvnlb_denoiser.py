import cv2,copy
import pandas as pd
import numpy as np
import unittest
import pyvnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from collections import defaultdict
from easydict import EasyDict as edict

# -- package helper imports --
from pyvnlb.pylib.tests.data_loader import load_dataset
from pyvnlb.pylib.tests.file_io import save_images
from pyvnlb import groups2patches,patches2groups,patches_at_indices
from pyvnlb import check_omp_num_threads

# -- python impl --
from pyvnlb.pylib.py_impl import runPythonVnlb,processNLBayes
SAVE_DIR = Path("./output/tests/")


class TestPythonVnlbDenoiser(unittest.TestCase):

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
        data._clean = clean

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
    # -- Define C++ & Python calls --
    #

    def do_run_cpp(self,tensors,sigma,params):
        noisy = tensors.noisy
        # flows = {}
        flows = {'fflow':tensors.fflow,'bflow':tensors.bflow}
        results = pyvnlb.runPyVnlb(noisy,sigma,flows,params)
        return results

    def do_run_python(self,tensors,sigma,params):
        noisy = tensors.noisy
        # flows = {}
        flows = {'fflow':tensors.fflow,'bflow':tensors.bflow}
        params = copy.deepcopy(params)
        results = runPythonVnlb(noisy,sigma,flows,params)
        return results

    #
    # -- Run Comparison --
    #

    def do_run_comparison(self,tensors,sigma,pyargs):

        # -- parse parameters --
        # noisy = tensors.noisy[:3,:,:36,:36].copy()
        noisy = tensors.noisy
        tensors.noisy = noisy
        params = pyvnlb.setVnlbParams(noisy.shape,sigma,params=pyargs)

        # -- exec both types --
        cpp_results = self.do_run_cpp(tensors,sigma,copy.deepcopy(params))
        py_results = self.do_run_python(tensors,sigma,params)

        # -- save results --
        results = defaultdict(dict)
        fields = ["basic","denoised"]
        for field in fields:
            cppField = cpp_results[field]
            pyField = py_results[field]
            save_images(SAVE_DIR / f"cpp_{field}.png",cppField,imax=255.)
            save_images(SAVE_DIR / f"py_{field}.png",pyField,imax=255.)
            delta = np.abs(cppField - pyField)
            if delta.max() > 0:
                delta /= delta.max()
            save_images(SAVE_DIR / f"delta_{field}.png",delta,imax=1.)

        # -- compare results --
        clean = tensors._clean
        results = defaultdict(dict)
        fields = ["basic","denoised"]
        for field in fields:
            cppField = cpp_results[field]
            pyField = py_results[field]
            msg = f"[{field}] check failed."
            np.testing.assert_allclose(cppField,pyField,rtol=5e-3,err_msg=msg)

            cpp_psnrs = pyvnlb.compute_psnrs(clean,cppField)
            py_psnrs = pyvnlb.compute_psnrs(clean,pyField)
            for ti in range(len(cpp_psnrs)):
                field_t = field + "_" + str(ti)
                results['py'][field_t] = py_psnrs[ti]
                results['cpp'][field_t] = cpp_psnrs[ti]
            # print(np.stack([cppField,pyField],axis=-1))
            # totalError = np.abs(cppField - pyField)/(np.abs(cppField)+1e-12)
            # totalError = np.mean(totalError)
            # totalError = np.around(totalError,9)
            # tgt = 0.
            # delta = np.abs(totalError-tgt)/(tgt+1e-12)
            # assert delta < 1e-5
        results = pd.DataFrame(results)
        print(results.to_markdown())


    def test_python_denoiser(self):

        # -- init save path --
        np.random.seed(234)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- check omp threads --
        check_omp_num_threads(nthreads=1)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        self.do_run_comparison(tensors,sigma,pyargs)
