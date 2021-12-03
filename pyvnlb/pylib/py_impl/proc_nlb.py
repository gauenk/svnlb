
# -- python deps --
import copy
import numpy as np
from easydict import EasyDict as edict
import pyvnlb

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .init_mask import initMask
from .utils import idx2coords,coords2idx,patches2groups,groups2patches
from .utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp

# -- project imports --
from pyvnlb import groups2patches,patches2groups

def optional(pydict,key,default):
    if pydict is None: return default
    if key in pydict: return pydict[key]
    else: return default

def processNLBayes(noisy,sigma,step,tensors,params):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- init outputs --
    t,c,h,w = noisy.shape
    zero_basic = np.zeros_like(noisy)
    denoised = np.zeros_like(noisy)
    basic = optional(tensors,'basic',zero_basic)
    weights = np.zeros((t,h,w),dtype=np.float32)
    flows = tensors

    # -- run the step --
    step_results = exec_step(noisy,basic,weights,sigma,flows,params,step)

    # -- format outputs --
    results = edict()
    results.denoised = step_results.denoised
    results.basic = step_results.basic
    results.ngroups = step_results.ngroups

    return results

def exec_step(noisy,basic,weights,sigma,flows,params,step):


    # -- init mask --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    deno = basic if step == 0 else np.zeros_like(noisy)
    minfo = initMask(noisy.shape,params,step)
    mask,n_groups = minfo['mask'],minfo['ngroups']

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)

    # -- init looping vars --
    npixels = noisy.size
    g_remain = n_groups
    g_counter = 0

    # -- run npixels --
    for pidx in range(npixels):

        # -- coords --
        ti,ci,hi,wi = idx2coords(pidx,c,h,w)
        pidx3 = coords2idx(ti,hi,wi,1,h,w)
        if not(mask[ti,hi,wi]): continue
        g_counter += 1
        if g_counter > 2: break
        # print("pidx: %d" % pidx)

        # -- sim search --
        sim_results = estimateSimPatches(noisy,basic,sigma,pidx,flows,params,step)
        groupNoisy,groupBasic,indices = sim_results
        nSimP = len(indices)

        # -- bayes estimate --
        groupNoisy,rank_var = computeBayesEstimate(groupNoisy,groupBasic,nSimP,
                                                   shape,params,step)
        rank_var = 0.

        # -- debug zone. --
        # from pyvnlb.pylib.tests import save_images
        # print(groups.shape)
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1])[:100]
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1],1)
        # patches_rgb = yuv2rgb_cpp(patches_yuv)
        # print(patches_rgb)
        # save_images(patches_rgb,f"output/patches_{pidx}.png",imax=255.)

        # -- aggregate results --
        deno,weights,mask,nmasked = computeAgg(deno,groupNoisy,indices,weights,
                                               mask,nSimP,params,step)
        g_remain -= nmasked

    # -- reduce using weighted ave --
    weightedAggregation(deno,noisy_yuv,weights)
    # deno = numpy_div0(deno,weights[:,None],0.)
    # print(weights[0,0,0])
    # print(deno[0,0,0])

    # -- re-colorize --
    deno = yuv2rgb_cpp(deno)
    # basic = yuv2rgb_cpp(basic)

    # -- pack results --
    results = edict()
    results.denoised = deno if step == 1 else np.zeros_like(deno)
    results.basic = deno
    results.ngroups = g_counter

    return results


def estimateSimPatches(noisy,basic,sigma,pidx,flows,params,step):

    # -- unpack --
    t,c,h,w = noisy.shape
    psX = params.sizePatch[step]
    psT = params.sizePatchTime[step]

    # -- cpp exec --
    # sim_results = pyvnlb.simPatchSearch(noisy,sigma,pidx,flows,params)
    # sim_results = edict(sim_results)

    # groups = sim_results.groupNoisy
    # indices = sim_results.indices
    # nsearch = sim_results.nsearch

    # -- sim search --
    params.use_imread = [False,False]
    tensors = edict({k:v for k,v in flows.items()})
    tensors.basic = basic
    sim_results = runSimSearch(noisy,sigma,pidx,tensors,params,step)

    patchesNoisy = sim_results.patchesNoisy
    patchesBasic = sim_results.patchesBasic
    indices = sim_results.indices
    nSimP = sim_results.nSimP
    ngroups = sim_results.ngroups
    groupsNoisy = patches2groups(patchesNoisy,c,psX,psT,ngroups,1)
    groupsBasic = patches2groups(patchesBasic,c,psX,psT,ngroups,1)

    # -- debug zone --
    # print(groups.shape)
    # print("nsearch: %d\n" % nsearch)
    # print("gsize: %d\n" % gsize)
    # print(indices.dtype)

    # from pyvnlb.pylib.tests import save_images
    # print("patches.shape: ",patches.shape)
    # patches_rgb = yuv2rgb_cpp(patches)
    # save_images(patches_rgb,"output/patches.png",imax=255.)

    return groupsNoisy,groupsBasic,indices

def computeBayesEstimate(groupNoisy,groupBasic,nSimP,shape,params,step):

    # -- sim search --
    rank_var = 0.
    bayes_results = pyvnlb.computeBayesEstimate(groupNoisy.copy(),
                                                groupBasic.copy(),0.,
                                                nSimP,shape,params,step)

    # bayes_results = runBayesEstimate(groupNoisy.copy(),groupBasic.copy(),
    #                                  rank_var,nSimP,shape,params,step)

    groups = bayes_results['groupNoisy']
    rank_var = bayes_results['rank_var']

    return groups,rank_var

def computeAgg(deno,groupNoisy,indices,weights,mask,nSimP,params,step):

    # -- cpp version --
    print(groupNoisy.shape)
    print(deno.shape)
    results = pyvnlb.computeAggregation(deno,groupNoisy,
                                        indices,weights,
                                        mask,nSimP,params)
    deno = results['deno']
    mask = results['mask']
    weights = results['weights']
    nmasked = results['nmasked']

    # -- python version --
    # agg_results = computeAggregation(deno,groupNoisy,indices,weights,mask,
    #                                  nSimP,params,step=step)
    # deno = agg_results['deno']
    # weights = agg_results['weights']
    # mask = agg_results['mask']
    # nmasked = agg_results['nmasked']

    return deno,weights,mask,nmasked

def weightedAggregation(deno,noisy,weights):
    gtz = np.where(weights > 0)
    eqz = np.where(weights == 0)
    for c in range(deno.shape[1]):
        deno[gtz[0],c,gtz[1],gtz[2]] /= weights[gtz]
        # deno[gtz[0],c,gtz[1],gtz[2]] = 0#weights[gtz]
        deno[eqz[0],c,eqz[1],eqz[2]] = noisy[eqz[0],c,eqz[1],eqz[2]]


