
# -- python deps --
import copy
import numpy as np
from easydict import EasyDict as edict
import vnlb

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .init_mask import initMask
from .flat_areas import runFlatAreas
from vnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches
from vnlb.utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp

# -- project imports --
from vnlb.utils import groups2patches,patches2groups

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
    """

    Primary sub-routine of VNLB

    """


    # -- init denoised image --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    chnls = c
    # deno = basic if step == 0 else np.zeros_like(noisy)
    deno = np.zeros_like(noisy)

    # -- init mask --
    minfo = initMask(noisy.shape,params,step)
    # minfo = vnlb.init_mask(noisy.shape,params,step)
    mask,n_groups = minfo['mask'],minfo['ngroups']

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)

    # -- init looping vars --
    npixels = t*h*w
    g_remain = n_groups
    g_counter = 0

    # -- run npixels --
    for pidx in range(npixels):

        # -- pix index to coords --
        # pidx = t*wh + y*width + x;
        ti = pidx // (w*h)
        hi = (pidx - ti*w*h) // w
        wi = pidx - ti*w*h - hi*w

        # pidx3 = t*whc + c*wh + y*width + x

        # ti,ci,hi,wi = idx2coords(pidx,c,h,w)
        # pidx3 = coords2idx(ti,hi,wi,1,h,w)
        # t1,c1,h1,w1 = idx2coords(pidx3,1,h,w)
        pidx3 = ti*w*h*c + hi*w + wi

        # -- skip masked --
        if not(mask[ti,hi,wi] == 1): continue
        # print("mask: ",mask[0,0,0],mask[0,0,1],mask[0,0,2],mask[0,0,3])

        # -- inc counter --
        # if g_counter > 2: break
        # print("group_counter: %d" % g_counter)
        g_counter += 1
        # print("(t,h,w,-): %d,%d,%d,%d" %(ti,hi,wi,mask[1,0,24]))
        # print("ij,ij3: %d,%d\n" % (pidx,pidx3))

        # -- sim search --
        sim_results = estimateSimPatches(noisy,basic,sigma,pidx3,flows,params,step)
        groupNoisy,groupBasic,indices = sim_results
        nSimP = len(indices)

        # -- optional flat patch --
        flatPatch = False
        if params.flatAreas[step]:
            # flatPatch = runFlatAreas(groupNoisy,groupBasic,nSimP,chnls)
            psX,psT = params.sizePatch[step],params.sizePatchTime[step]
            gamma = params.gamma[step]
            flatPatch = runFlatAreas(groupNoisy,psX,psT,nSimP,chnls,gamma,sigma)

        # -- bayes estimate --
        rank_var = 0.
        groupNoisy,rank_var = computeBayesEstimate(groupNoisy,groupBasic,
                                                   nSimP,shape,params,
                                                   step,flatPatch)
        # print(groupNoisy.ravel()[0])

        # -- debug zone. --
        # from vnlb.pylib.tests import save_images
        # print(groups.shape)
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1])[:100]
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1],1)
        # patches_rgb = yuv2rgb_cpp(patches_yuv)
        # print(patches_rgb)
        # save_images(patches_rgb,f"output/patches_{pidx}.png",imax=255.)

        # -- aggregate results --
        deno,weights,mask,nmasked = computeAgg(deno,groupNoisy,indices,weights,
                                               mask,nSimP,params,step)
        # print("deno.ravel()[0]: ",deno.ravel()[0])
        # print("deno.ravel()[1]: ",deno.ravel()[1])
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
    # sim_results = vnlb.simPatchSearch(noisy,sigma,pidx,flows,params,step)
    # sim_results = edict(sim_results)
    # groupsNoisy = sim_results.groupNoisy
    # groupsBasic = sim_results.groupBasic
    # indices = sim_results.indices

    # sim_groupsNoisy = sim_results.groupNoisy
    # sim_groupsBasic = sim_results.groupBasic
    # sim_indices = sim_results.indices

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

    # -- check -- # 563
    # delta = np.sum(np.sort(indices) - np.sort(sim_indices))
    # if delta >1e-3:
    #     print(pidx,step)
    #     print(np.stack([np.sort(indices),np.sort(sim_indices)],-1))
    # assert delta < 1e-3

    # py_order = np.argsort(indices)
    # sim_order = np.argsort(sim_indices)


    # py_patches = patchesNoisy[py_order]
    # sim_patches = groups2patches(sim_groupsNoisy,3,7,2,nSimP)[sim_order]
    # delta = np.abs(py_patches-sim_patches)
    # if np.any(delta>1e-3):
    #     print(np.unique(np.where(delta>1e-3)[0]))
    #     print(np.stack([py_patches[0],sim_patches[0]],-1))
    #     assert False


    # from vnlb.pylib.tests import save_images
    # print("patches.shape: ",patches.shape)
    # patches_rgb = yuv2rgb_cpp(patches)
    # save_images(patches_rgb,"output/patches.png",imax=255.)

    return groupsNoisy,groupsBasic,indices

def computeBayesEstimate(groupNoisy,groupBasic,nSimP,shape,params,step,flatPatch):

    # -- prepare --
    rank_var = 0.

    # -- exec --
    # bayes_results = vnlb.computeBayesEstimate(groupNoisy.copy(),
    #                                             groupBasic.copy(),0.,
    #                                             nSimP,shape,params,step)
    bayes_results = runBayesEstimate(groupNoisy.copy(),groupBasic.copy(),
                                     rank_var,nSimP,shape,params,step,flatPatch)

    # -- format --
    groups = bayes_results['groupNoisy']
    rank_var = bayes_results['rank_var']


    return groups,rank_var

def computeAgg(deno,groupNoisy,indices,weights,mask,nSimP,params,step):

    # -- cpp version --
    # params.isFirstStep[step] = step == 0
    # results = vnlb.computeAggregation(deno,groupNoisy,
    #                                     indices,weights,
    #                                     mask,nSimP,params)
    # deno = results['deno']
    # mask = results['mask']
    # weights = results['weights']
    # nmasked = results['nmasked']

    # -- python version --
    agg_results = computeAggregation(deno,groupNoisy,indices,weights,mask,
                                     nSimP,params,step)
    deno = agg_results['deno']
    weights = agg_results['weights']
    mask = agg_results['mask']
    nmasked = agg_results['nmasked']

    return deno,weights,mask,nmasked

def weightedAggregation(deno,noisy,weights):
    gtz = np.where(weights > 0)
    eqz = np.where(weights == 0)
    for c in range(deno.shape[1]):
        deno[gtz[0],c,gtz[1],gtz[2]] /= weights[gtz]
        # deno[gtz[0],c,gtz[1],gtz[2]] = 0#weights[gtz]
        deno[eqz[0],c,eqz[1],eqz[2]] = noisy[eqz[0],c,eqz[1],eqz[2]]


