
# -- python deps --
import copy,torch
import torch as th
from easydict import EasyDict as edict

# -- package --
import vnlb

# -- local imports --
from .init_mask import initMask
from .flat_areas import runFlatAreas

# -- wrapped functions --
from .wrapped import weightedAggregation,computeAgg
from .wrapped import computeBayesEstimate,estimateSimPatches

from vnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches
# from vnlb.utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp

# -- project imports --
from vnlb.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp

# -- project imports --
from vnlb.utils import groups2patches,patches2groups,optional

def processNLBayes(noisy,sigma,step,tensors,params,gpuid=0):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- place on cuda --
    if not(th.is_tensor(noisy)):
        device = gpuid
        noisy = th.FloatTensor(noisy).to(device)
        flows = edict({k:th.FloatTensor(v).to(device) for k,v in tensors.items()})

    # -- init outputs --
    t,c,h,w = noisy.shape
    zero_basic = th.zeros_like(noisy)
    denoised = th.zeros_like(noisy)
    basic = optional(tensors,'basic',zero_basic)
    weights = th.zeros((t,h,w),dtype=th.float32)
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
    # deno = basic if step == 0 else th.zeros_like(noisy)
    deno = th.zeros_like(noisy)

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
    results.denoised = deno if step == 1 else th.zeros_like(deno)
    results.basic = deno
    results.ngroups = g_counter

    return results

