import cv2
import numpy as np
import unittest
import vnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from collections import defaultdict
import pytest

# -- add data/file io --
from scripts.compare_cpp import run_method

@pytest.mark.order(1)
class TestVnlbDenoiser(unittest.TestCase):

    def do_run_get_target_results(self,vnlb_dataset):
        non_zeros = {'cv2':{},'cpp':{}}
        if vnlb_dataset == "davis_64x64":
            non_zeros['cv2']['noisyForFlow'] = 0.000505755
            non_zeros['cv2']["fflow"] = 504.308
            non_zeros['cv2']["bflow"] = 21.643
        return non_zeros

    def do_run_comparison(self,vnlb_dataset):

        # -- load dataset --
        non_zeros = self.do_run_get_target_results(vnlb_dataset)
        res_vnlb,res_pyvnlb_cv2 = run_method(vnlb_dataset,"cv2")
        res_vnlb,res_pyvnlb_cpp = run_method(vnlb_dataset,"cpp")

        # -- compare --
        res_vnlb = {'cv2':res_vnlb,'cpp':res_vnlb} # both the same
        res_pyvnlb = {'cv2':res_pyvnlb_cv2,'cpp':res_pyvnlb_cpp}

        # -- compare results --
        results = defaultdict(dict)
        fields = ["noisyForFlow","noisyForVnlb","fflow","bflow","basic","denoised"]
        for imageIO in ['cv2','cpp']:
            imageIO_nz = non_zeros[imageIO]
            for field in fields:
                cppField = res_vnlb[imageIO][field]
                pyField = res_pyvnlb[imageIO][field]
                totalError = np.abs(cppField - pyField)/(np.abs(cppField)+1e-12)
                totalError = np.sum(totalError)
                totalError = np.around(totalError,9)
                tgt = 0.
                if field in imageIO_nz:
                    tgt = imageIO_nz[field]
                delta = np.abs(totalError-tgt)/(tgt+1e-12)
                assert delta < 1e-5

    def test_denoiser(self):
        vnlb_dataset = "davis_64x64"
        self.do_run_comparison(vnlb_dataset)

