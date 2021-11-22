
import cv2
import numpy as np
import unittest
import pyvnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path

class TestSetVnlbParams(unittest.TestCase):

    def do_exec_denoiser(self,noisy,std,pyargs):
        # -- TV-L1 Optical Flow --
        fflow,bflow = pyvnlb.runPyFlow(noisy,std)
        pyargs['fflow'] = fflow
        pyargs['bflow'] = bflow

        # -- Video Non-Local Bayes --
        pyargs['verbose'] = True
        pyargs['testing'] = True
        results = pyvnlb.runPyVnlb(noisy,std,pyargs)
        basic = results['basic']
        denoised = results['denoised']

        return basic,denoised

    def do_vnlb_param_pair(self,t,c,h,w):

        # -- create video  --
        noisy = np.uint16(255.*np.random.rand(t,c,h,w))
        std = 30.

        # -- exec default params --
        pyargs = {'default':True}
        results = self.do_exec_denoiser(noisy,std,pyargs)
        cppParser_basic,cppParser_denoised = results

        # -- exec default params --
        pyargs = pyvnlb.setVnlbParams(noisy.shape,std)
        results = self.do_exec_denoiser(noisy,std,pyargs)
        pyParser_basic,pyParser_denoised = results

        # -- compare --
        np.testing.assert_array_equal(pyParser_basic,cppParser_basic)
        np.testing.assert_array_equal(pyParser_denoised,cppParser_denoised)

    def test_vnlb_params(self):
        self.do_vnlb_param_pair(3,3,32,32)
        # self.do_vnlb_param_pair(10,3,256,256)
