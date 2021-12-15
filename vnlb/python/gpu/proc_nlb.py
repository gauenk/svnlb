
# -- python deps --
import copy,torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package --
import vnlb

# -- local imports --
from .init_mask import initMask
from .flat_areas import runFlatAreas
from .sim_search import sim_search_batch
from .bayes_est import bayes_estimate_batch
from .comp_agg import compute_agg_batch


# -- wrapped functions --
from .wrapped import weightedAggregation,computeAgg
from .wrapped import computeBayesEstimate,estimateSimPatches

from vnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches
# from vnlb.utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp

# -- project imports --
from vnlb.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp

# -- project imports --
from vnlb.utils import groups2patches,patches2groups,optional
from vnlb.testing import save_images

# -- streams
from vnlb.gpu.sim_search.streams import init_streams,wait_streams,get_hw_batches
from vnlb.gpu.sim_search.streams import view_batch,vprint


def processNLBayes(noisy,basic,sigma,step,flows,params,gpuid=0):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- place on cuda --
    device = gpuid
    if not(th.is_tensor(noisy)):
        noisy = th.FloatTensor(noisy).to(device)
        zero_basic = th.zeros_like(noisy)
        basic = optional(basic,'basic',zero_basic)
        basic = basic.to(device)

    # -- init outputs --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    deno = th.zeros_like(noisy)
    nstreams = optional(params,'nstreams',1)
    flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})

    # -- to device flow --
    flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})
    zflow = torch.zeros((t,2,h,w)).to(device)
    fflow = optional(flows,'fflow',zflow)
    bflow = optional(flows,'bflow',zflow)

    # -- unpack --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = params['nSimilarPatches'][step]
    w_s = params['sizeSearchWindow'][step]
    nWt_f = params['sizeSearchTimeFwd'][step]
    nWt_b = params['sizeSearchTimeBwd'][step]
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    check_steps(step1,step)
    sigma2 = params['sigma'][step]**2
    beta = params['beta'][step]
    sigmaBasic2 = params['sigmaBasic'][step]**2
    sigmab2 = beta * sigmaBasic2 if step==1 else sigma**2
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    flat_areas = params['flatAreas'][step]
    gamma = params['gamma'][step]
    step_s = params['procStep'][step]
    t,c,h,w = shape
    group_chnls = 1 if couple_ch else c
    step_s = 1
    print("step_s: ",step_s)

    # -- create mask --
    mask = initMask(noisy.shape,params,step)['mask']
    mask = torch.ByteTensor(mask).to(device)

    # -- run the step --
    exec_step(noisy,basic,deno,mask,fflow,bflow,sigma2,sigmab2,rank,ps,
              ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
              thresh,flat_areas,gamma,step,nstreams)

    # -- format outputs --
    results = edict()
    results.basic = basic
    results.denoised = deno
    results.ngroups = npatches

    return results

def mask2inds(mask,h_bsize,w_bsize):
    index = torch.nonzero(mask-1)
    bsize = h_bsize * w_bsize
    return index[:bsize]

def exec_step(noisy,basic,deno,mask,fflow,bflow,sigma2,sigmab2,rank,ps,ps_t,npatches,
              step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,thresh,flat_areas,gamma,
              step,nstreams):

    """
    ** Our "simsearch" is not the same as "vnlb" **

    1. the concurrency of using multiple cuda-streams creates io issues
       for using the mask
    2. if with no concurrency, the details of using an "argwhere" each batch
       seems strange
    3. it is unclear if we will want this functionality for future uses
       of this code chunk
    """

    # -- unpack info --
    use_imread = False
    device = noisy.device
    shape = noisy.shape
    nframes,chnls,height,width = noisy.shape

    # -- init tensors --
    deno = basic if step == 0 else deno
    # weights = th.zeros((nframes,height,width),dtype=th.float32)

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,nframes-1)
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    bsize = 8
    h_batches,w_batches = get_hw_batches(height,width,bsize*step_s)

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    ns,np,t,c = nstreams,npatches,nframes,chnls
    tf32 = torch.float32
    patchesNoisy = torch.zeros(ns,np,t,ps_t,c,ps,ps,bsize,bsize).type(tf32).to(device)
    patchesBasic = torch.zeros(ns,np,t,ps_t,c,ps,ps,bsize,bsize).type(tf32).to(device)
    indices = torch.zeros(ns,np,t,bsize,bsize).type(torch.int32).to(device)
    vals = torch.zeros(ns,np,t,bsize,bsize).type(tf32).to(device)
    weights = torch.zeros(nframes,height,width).type(tf32).to(device)
    access = torch.zeros(nframes,height,width).type(tf32).to(device)

    # -- print statements --
    # mins = noisy.min(0).values.min(1).values.min(1).values
    # maxs = noisy.max(0).values.max(1).values.max(1).values
    # print("noisy: ",mins,maxs)
    # mins = noisy_yuv.min(0).values.min(1).values.min(1).values
    # maxs = noisy_yuv.max(0).values.max(1).values.max(1).values
    # print("noisy_yuv: ",mins,maxs)

    # -- exec search --
    for h_start in h_batches:
        h_start = h_start.item()

        for w_start in w_batches:
            w_start = w_start.item()

            # -- batch info --
            binfo = [h_start,w_start,bsize]
            print("h_start,w_start: %d,%d" %(h_start,w_start))

            # -- get indies from mask --
            access = mask2inds(mask,bsize,bsize)

            # -- assign to stream --
            cs = curr_stream
            torch.cuda.set_stream(streams[cs])
            cs_ptr = streams[cs].cuda_stream

            # -- select data for stream --
            patchesNoisy_s = patchesNoisy[cs]
            patchesBasic_s = patchesBasic[cs]
            vals_s = vals[cs]
            inds_s = indices[cs]
            # weights_s = weights[cs]

            # -- sim_search_block --
            sim_search_batch(noisy_yuv,basic_yuv,patchesNoisy_s,patchesBasic_s,
                             access,vals_s,inds_s,fflow,bflow,step_s,h_start,w_start,
                             bsize,ps,ps_t,w_s,nWt_f,nWt_b,step==0,cs,cs_ptr)
            # nzeros = torch.sum(patchesNoisy_s==0).item()
            # nelems = patchesNoisy_s.numel()*1.
            # print("[patchesNoisy] percent zero: ",nzeros/nelems*100)

            # -- get inds info --
            # nzero = torch.sum(inds_s==0).item()
            # size = inds_s.numel()
            # print("[inds_s] perc zero: %2.3f" % (nzero / size * 100))

            # -- optional flat patch --
            flat_patch = False
            # if flat_areas:
            #     flat_patch = runFlatAreas(patchesNoisy_s,ps,ps_t,gamma,sigma2)

            # -- bayes filter --
            rank_var = 0.
            res = bayes_estimate_batch(patchesNoisy_s,
                                       patchesBasic_s,sigma2,
                                       sigmab2,rank,group_chnls,thresh,
                                       step==1,flat_patch,cs,cs_ptr)
            # rank_var,patchesNoisy_s = res
            rank_var = res[0]
            torch.cuda.synchronize()

            # -- save patches noisy --
            # patches = patchesNoisy_s
            # print("patches.shape: ",patches.shape)
            # shape_str = 'n t pt c ph pw h w -> n (t h w) pt c ph pw'
            # patches = rearrange(patches,shape_str).cpu().numpy()
            # print("saving.")
            # sidx = 0
            # for b in range(patches.shape[0]-1,0,-1):
            #     save_images(patches[b,0],f"patches_{b}.png")
            #     if sidx > 10: break
            #     sidx += 1

            # -- inds stats --
            n_invalid = torch.sum(inds_s == -1).item()
            nelems = inds_s.numel() * 1.
            perc_invalid = n_invalid / nelems * 100.
            print("Perc Invalid: %2.1f" % (perc_invalid))

            # -- aggregate results --
            # print("[pre agg] deno: ",deno.min(),deno.max())
            compute_agg_batch(deno,patchesNoisy_s,inds_s,weights,ps,ps_t,cs_ptr)
            # print("[post agg] deno: ",deno.min(),deno.max())

            # -- change stream --
            if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

    # -- wait for all streams --
    torch.cuda.synchronize()

    # -- reweight --
    # print("noisy: ",noisy.min(),noisy.max(),noisy.shape)
    # print("noisy_yuv: ",noisy_yuv.min(),noisy_yuv.max(),noisy_yuv.shape)
    # print("all: ",torch.all(weights>0))
    # print("weights: ",weights.min(),weights.max(),weights.shape)
    # print("deno: ",deno.min(),deno.max(),deno.shape)
    wmax = weights.max().item()
    save_images("weights.png",weights.cpu().numpy()[:,None],imax=wmax)
    wmax = weights.max().item()
    weights = repeat(weights,'t h w -> t c h w',c=chnls)
    print(weights[0,0,:3,:3])
    print(weights[0,0,8:10,8:10])
    print("wmax: ",wmax)

    index = torch.nonzero(weights,as_tuple=True)
    # for ci in range(chnls):
    #     deno[:,ci][index] /= weights[index]
    # print("[post] deno: ",deno.min(),deno.max())
    deno[index] /= weights[index]
    # print("[post] deno: ",deno.min(),deno.max())

    # -- yuv 2 rgb --
    if not(use_imread):
        yuv2rgb_cpp(deno)
    print("[post-2] deno: ",deno.min(),deno.max())



def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1
