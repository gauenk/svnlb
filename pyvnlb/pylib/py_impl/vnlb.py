
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation


def runPythonVnlb(noisy,sigma,pidx,tensors,params,0):
    """

    A Python implementation of the C++ code.

    """

    # -- init params --
    mask = np.zeros(noisy.shape,dtype=np.int8)
    basic = np.zeros_like(noisy)

    # -- step 1 --
    step_results = exec_step(noisy,basic,mask,sigma,tensors,params,0)

    # -- step 2 --
    step_results = exec_step(noisy,basic,mask,sigma,tensors,params,1)

    # -- format --
    results = edict()
    results.deno = step_results['deno']deno

    return results

def exec_step(noisy,basic,mask,sigma,tensors,params,step):

    # init_mask(mask)
    npixesl = noisy.size
    for pidx in range(npixels):
        exec_step_at_pixel(noisy,basic,mask,sigma,pidx,tensors,params,step)


def exec_step_at_pixel(noisy,basic,mask,sigma,pidx,tensors,params,step):

    sim_results = runSimSearch(noisy,sigma,pidx,tensors,params,step)
    bayes_results = runBayesEstimate(groupNoisy,groupBasic,rank_var,
                                     nSimP,shape,params,step)
    agg_results = computeAggregation(deno,group,indices,weights,mask,
                                     nSimP,params,step=step)
    return agg
