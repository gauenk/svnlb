
# -- python deps --
import numpy as np
from easydict import EasyDict as edict

# -- package imports --
import svnlb

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .proc_nlb import processNLBayes


# -- project imports --
from svnlb.utils import groups2patches,patches2groups

def runPythonVnlb(noisy,sigma,flows,params=None,clean=None):
    """

    A Python implementation of the C++ code.

    """

    # -- create params --
    if params is None:
        params = vnlb.swig.setVnlbParams(noisy.shape,sigma,None)
        # print(list(params.keys()))
        # print(params.aggreBoost)
        # params.aggreBoost = [False,False]
        # params.psX = [3,3]
        # params.nSimilarPatches = [100,60]
    # print(params.sizePatch)
    # print(params.nSimilarPatches)
    # print(params.aggreBoost)

    # -- step 1 --
    step_results = processNLBayes(noisy,sigma,0,flows,params,clean)
    step1_results = step_results
    basic = step1_results.basic.copy()

    # -- step 2 --
    tensors = edict(flows)
    tensors.basic = step_results.basic.copy()
    step_results = processNLBayes(noisy,sigma,1,tensors,params)

    # -- format --
    results = edict()
    results.basic = basic
    results.denoised = step_results.denoised

    return results

