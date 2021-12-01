
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
import numpy as np

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
    step_results = exec_step(noisy,basic,mask,weights,sigma,flows,params,0)

    # -- step 2 --
    step_results = exec_step(noisy,basic,mask,weights,sigma,flows,params,1)

    # -- format --
    results = edict()
    results.deno = step_results['deno']

    return results

def exec_step(noisy,basic,mask,weights,sigma,flows,params,step):

    # init_mask(mask)
    npixels = noisy.size
    for pidx in range(npixels):
        exec_step_at_pixel(noisy,basic,mask,weights,sigma,pidx,flows,params,step)


def exec_step_at_pixel(noisy,basic,mask,weights,sigma,pidx,flows,params,step):

    # -- sim search --
    sim_results = runSimSearch(noisy,sigma,pidx,flows,params,step)
    groupNoisy = sim_results['groupNoisy_og']
    groupBasic = sim_results['groupBasic_og']
    nSimP = sim_results['npatches']
    indices = sim_results['indices']

    # -- bayes estimate --
    shape = noisy.shape
    rank_var = params.rank_var
    bayes_results = runBayesEstimate(groupNoisy,groupBasic,rank_var,
                                     nSimP,shape,params,step)
    groupNoisy = bayes_results['groupNoisy']

    # -- aggregate results --
    agg_results = computeAggregation(basic,groupNoisy,indices,weights,mask,
                                     nSimP,params,step=step)
    return agg
