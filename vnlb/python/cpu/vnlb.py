
# -- python deps --
import numpy as np
from easydict import EasyDict as edict

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .proc_nlb import processNLBayes


# -- project imports --
from vnlb.utils import groups2patches,patches2groups

def runPythonVnlb(noisy,sigma,flows,params):
    """

    A Python implementation of the C++ code.

    """

    # -- step 1 --
    step_results = processNLBayes(noisy,sigma,0,flows,params)
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

