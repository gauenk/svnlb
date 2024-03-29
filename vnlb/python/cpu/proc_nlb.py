
# -- python deps --
import copy
import numpy as np
from easydict import EasyDict as edict
import svnlb

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .init_mask import initMask
from .flat_areas import runFlatAreas
from svnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches
from svnlb.utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp
from svnlb.testing import save_images

# -- project imports --
from svnlb.utils import groups2patches,patches2groups

def optional(pydict,key,default):
    if pydict is None: return default
    if key in pydict: return pydict[key]
    else: return default

def processNLBayes(noisy,sigma,step,tensors,params,clean=None):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- init outputs --
    t,c,h,w = noisy.shape
    zero_basic = np.zeros_like(noisy)
    denoised = np.zeros_like(noisy)
    basic = optional(tensors,'basic',zero_basic)
    weights = np.zeros((t,h,w),dtype=np.float32)

    # -- extract flows --
    flows = {}
    if 'fflow' in tensors and 'bflow' in tensors:
        flows = edict({'fflow':tensors['fflow'],'bflow':tensors['bflow']})

    # -- run the step --
    step_results = exec_step(noisy,basic,weights,sigma,flows,params,step,clean)

    # -- format outputs --
    results = edict()
    results.denoised = step_results.denoised
    results.basic = step_results.basic
    results.ngroups = step_results.ngroups

    return results

def exec_step(noisy,basic,weights,sigma,flows,params,step,clean=None):
    """

    Primary sub-routine of VNLB

    """


    # -- shapes --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    nframes,chnls,height,width = t,c,h,w
    ps,ps_t = params['sizePatch'][step],params['sizePatchTime'][step]

    # -- init denoised --
    deno = basic if step == 0 else np.zeros_like(noisy)
    # deno = np.zeros_like(noisy)

    # -- init mask --
    minfo = initMask(noisy.shape,params,step)
    # minfo = vnlb.init_mask(noisy.shape,params,step)
    mask,n_groups = minfo['mask'],minfo['ngroups']
    access = np.zeros_like(mask)

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
        pidx4 = ti*w*h*c + hi*w + wi

        # -- skip if invalid --
        valid_t = (ti + ps_t - 1) < nframes
        valid_h = (hi + ps - 1) < height
        valid_w = (wi + ps - 1) < width
        valid = valid_t and valid_h and valid_w
        if not(valid): continue
        access[ti,hi,wi] = 1

        # -- skip masked --
        if (mask[ti,hi,wi] == 0): continue
        # print("mask: ",mask[0,0,0],mask[0,0,1],mask[0,0,2],mask[0,0,3])
        # print("pidx3: ",pidx,pidx3,ti,hi,wi)


        # -- inc counter --
        # if g_counter > 2: break
        # print("group_counter: %d" % g_counter)
        g_counter += 1
        # print("(t,h,w,-): %d,%d,%d,%d" %(ti,hi,wi,mask[1,0,24]))
        # print("ij,ij3: %d,%d\n" % (pidx,pidx3))

        # -- sim search --
        sim_results = estimateSimPatches(noisy,basic,sigma,pidx4,flows,
                                         params,step,clean)
        groupNoisy,groupBasic,groupClean,indices = sim_results
        nSimP = len(indices)

        # -- optional flat patch --
        flatPatch = False
        if params.flatAreas[step]:
            psX,psT = params.sizePatch[step],params.sizePatchTime[step]
            gamma = params.gamma[step]
            flatPatch = runFlatAreas(groupNoisy,psX,psT,nSimP,chnls,gamma,sigma)

        # -- bayes estimate --
        rank_var = 0.
        groupNoisy,rank_var = computeBayesEstimate(groupNoisy,groupBasic,
                                                   nSimP,shape,params,
                                                   step,flatPatch,groupClean)

        # -- debug zone. --
        # from vnlb.pylib.tests import save_images
        # print(groups.shape)
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1])[:100]
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1],1)
        # patches_rgb = yuv2rgb_cpp(patches_yuv)
        # print(patches_rgb)
        # save_images(patches_rgb,f"output/patches_{pidx}.png",imax=255.)

        # -- stats --
        # n_neg = np.sum(indices == -1)
        # n_zero = np.sum(indices == 0)
        # n_elems = indices.size * 1.
        # perc_neg = n_neg / n_elems * 100
        # perc_zero = n_zero / n_elems * 100
        # print("Perc Neg: %2.1f" % perc_neg)
        # print("Perc Zero: %2.1f" % perc_zero)

        # -- aggregate results --
        deno,weights,mask,nmasked = computeAgg(deno,groupNoisy,indices,weights,
                                               mask,nSimP,params,step)
        # print("deno.ravel()[0]: ",deno.ravel()[0])
        # print("deno.ravel()[1]: ",deno.ravel()[1])
        g_remain -= nmasked

    # -- save --
    wmax = weights.max().item()
    save_images(weights[:,None],"output/weights.png",imax=wmax)

    # amax = access.max().item()
    # print("amax")
    # save_images(access[:,None],"output/access.png",imax=amax)

    # -- reduce using weighted ave --
    wimg = noisy if params.use_imread[step] else noisy_yuv
    weightedAggregation(deno,wimg,weights)
    # deno = numpy_div0(deno,weights[:,None],0.)
    # print(weights[0,0,0])
    # print(deno[0,0,0])

    # -- re-colorize --
    if not(params.use_imread[step]):
        deno[...] = yuv2rgb_cpp(deno)
    # basic = yuv2rgb_cpp(basic)

    # -- pack results --
    results = edict()
    # results.denoised = deno if step == 1 else np.zeros_like(deno)
    results.denoised = deno
    results.basic = basic
    results.ngroups = g_counter

    return results


def estimateSimPatches(noisy,basic,sigma,pidx,flows,params,step,clean=None):

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
    # params.use_imread = [False,False]
    tensors = edict({k:v for k,v in flows.items()})
    tensors.basic = basic
    sim_results = runSimSearch(noisy,sigma,pidx,tensors,params,step,clean)

    patchesNoisy = sim_results.patchesNoisy
    patchesBasic = sim_results.patchesBasic
    patchesClean = sim_results.patchesClean
    indices = sim_results.indices
    nSimP = sim_results.nSimP
    ngroups = sim_results.ngroups
    groupsNoisy = patches2groups(patchesNoisy,c,psX,psT,ngroups,1)
    groupsBasic = patches2groups(patchesBasic,c,psX,psT,ngroups,1)
    groupsClean = None
    if not(patchesClean is None):
        groupsClean = patches2groups(patchesClean,c,psX,psT,ngroups,1)

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

    return groupsNoisy,groupsBasic,groupsClean,indices

def computeBayesEstimate(groupNoisy,groupBasic,nSimP,shape,params,
                         step,flatPatch,groupClean):

    # -- prepare --
    rank_var = 0.

    # -- exec --
    # bayes_results = vnlb.computeBayesEstimate(groupNoisy.copy(),
    #                                             groupBasic.copy(),0.,
    #                                             nSimP,shape,params,step)
    bayes_results = runBayesEstimate(groupNoisy.copy(),groupBasic.copy(),
                                     rank_var,nSimP,shape,params,step,
                                     flatPatch,groupClean)

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


