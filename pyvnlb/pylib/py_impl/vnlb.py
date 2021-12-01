
# -- python deps --
import numpy as np
from easydict import EasyDict as edict

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation


# -- project imports --
from pyvnlb import groups2patches,patches2groups

def runPythonVnlb(noisy,sigma,flows,params):
    """

    A Python implementation of the C++ code.

    """

    # -- init params --
    t,c,h,w = noisy.shape
    basic = np.zeros_like(noisy)
    mask = np.zeros((t,h,w),dtype=np.int8)
    weights = np.zeros((t,h,w),dtype=np.float32)

    # -- step 1 --
    step_results_1 = exec_step(noisy,basic,mask,weights,sigma,flows,params,0)

    # -- step 2 --
    step_results_2 = exec_step(noisy,basic,mask,weights,sigma,flows,params,1)

    # -- format --
    results = edict()
    results.basic = step_results_1.basic
    results.denoised = step_results_2.denoised

    return results

def exec_step(noisy,basic,mask,weights,sigma,flows,params,step):

    # init_mask(mask)
    npixels = 3#noisy.size
    for pidx in range(npixels):
        results = exec_step_at_pixel(noisy,basic,mask,weights,sigma,
                                     pidx,flows,params,step)
        denoised = results['deno']

    # -- pack results --
    results = edict()
    results.denoised = denoised
    results.basic = denoised

    return results


def exec_step_at_pixel(noisy,basic,mask,weights,sigma,pidx,flows,params,step):

    # -- sim search --
    params.use_imread = False
    sim_results = runSimSearch(noisy,sigma,pidx,flows,params,step)
    patches = sim_results.patches
    indices = sim_results.indices
    nSimP = sim_results.nSimP
    nsearch = sim_results.nsearch

    # -- to groups --
    ps,ps_t = sim_results.ps,sim_results.ps_t
    t,c,h,w = noisy.shape
    groups = patches2groups(patches,c,ps,ps_t,nsearch,1)

    # -- bayes estimate --
    shape = noisy.shape
    rank_var = 0.
    bayes_results = runBayesEstimate(groups,groups,rank_var,
                                     nSimP,shape,params,step)
    groupNoisy = bayes_results['groupNoisy']

    # -- aggregate results --
    agg_results = computeAggregation(basic,groupNoisy,indices,weights,mask,
                                     nSimP,params,step=step)
    return agg_results
