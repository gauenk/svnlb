
import cv2
import numpy as np
import unittest
import vnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path


class TestVideoIO(unittest.TestCase):


    def do_create_temp_video(self,t,c,h,w):

        # -- create video  --
        data = np.uint16(255.*np.random.rand(t,h,w,c))

        # -- create temp file --
        dir_path = tempfile.mkdtemp()
        path = str(Path(dir_path) / "%03d.tif")

        # -- fill file with video --
        try:
            for tidx in range(t):
                path_t = path % tidx
                cv2.imwrite(path_t,data[tidx])
        except:
            shutil.rmtree(dir_path)

        return dir_path,path,data

    def do_read_video_for_flow(self,t,c,h,w):
        dir_path,path,data = self.do_create_temp_video(t,c,h,w)
        try:
            data = rearrange(data,'t h w c -> t c h w')
            data = np.flip(data,axis=1)
            data = vnlb.utils.rgb2bw(data)
            read_data = vnlb.swig.readVideoForFlow((t,c,h,w),path)
        finally:
            shutil.rmtree(dir_path)
        delta = np.abs(read_data - data)
        conda = np.all(delta < 1.1)
        condb = np.mean(delta/255.) < 1e-2
        assert (conda and condb)

    def do_read_video_for_vnlb(self,t,c,h,w):
        dir_path,path,data = self.do_create_temp_video(t,c,h,w)
        try:
            data = rearrange(data,'t h w c -> t c h w')
            data = np.flip(data,axis=1)
            read_data = vnlb.swig.readVideoForVnlb((t,c,h,w),path)
        finally:
            shutil.rmtree(dir_path)
        np.testing.assert_array_equal(read_data,data)

    def test_read_video_for_flow(self):
        self.do_read_video_for_flow(10,3,64,64)
        self.do_read_video_for_flow(3,3,256,256)

    def test_read_video_for_vnlb(self):
        self.do_read_video_for_vnlb(10,3,64,64)
        self.do_read_video_for_vnlb(3,3,256,256)
