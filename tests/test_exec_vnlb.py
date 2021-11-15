"""
Test execution of the VNLB library

"""

import vnlb.pylib as pyvnlb
import numpy as np

def test_exec_vnlb():

    t,h,w,c = 3,64,64,3
    noisy = np.random.rand(t,h,w,c)
    print(dir(pyvnlb))
    pyvnlb.runPyVnlb(noisy,1.0)
