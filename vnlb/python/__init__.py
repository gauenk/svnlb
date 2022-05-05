
# -- import inferace files --
# from .vnlb.interface import runPyVnlb,runPyVnlbTimed,setVnlbParams
# from .vnlb.interface import simPatchSearch,computeBayesEstimate,computeAggregation
# from .vnlb.interface import processNLBayes,init_mask
# from .vnlb.interface import computeCovMat
# from .vnlb.sim_utils import groups2patches,patches2groups,patches_at_indices
# from .flow.interface import runPyTvL1Flow,runPyFlow,runPyFlowFB
# from .video_io.interface import readVideoForVnlb,readVideoForFlow
# from .flow.flow_utils import flow2img,flow2burst
# from .utils import compute_psnrs,expand_flows,rgb2bw,check_omp_num_threads
# from .utils import check_omp_num_threads
# import vnlb.pylib.tests as tests# import *

# -- swig vars --
_swig_enabled = False
__version__ = "%d.%d.%d" % (0,0,0)
from .utils import check_omp_num_threads
import svnlb.swig as swig
# import svnlb.gpu as gpu
import svnlb.cpu as cpu
import svnlb.utils as utils
from .loader import *


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       API Default Settings
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


import numpy as np
from easydict import EasyDict as edict

def compute_flow(noisy,sigma):
    assert isinstance(noisy,np.ndarray)
    flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
                   "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
                   "verbose":False,"testing":False,'bw':False}
    flowImages = utils.rgb2bw(noisy)
    fflow,bflow = swig.runPyFlow(flowImages,sigma,flow_params)
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow
    return flows


