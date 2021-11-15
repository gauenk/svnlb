"""
Test execution of the VNLB library

"""

import vnlb.pylib as pyvnlb
import numpy as np

def test_exec_vnlb():

    c,t,h,w = 3,5,32,32
    noisy = np.random.rand(c,t,h,w)
    print(dir(pyvnlb))
    pyvnlb.runPyVnlb(noisy,1.0)
