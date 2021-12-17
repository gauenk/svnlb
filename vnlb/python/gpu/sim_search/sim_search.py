
# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package imports --
from vnlb.utils import get_patch_shapes_from_params,optional,groups2patches,check_flows,check_and_expand_flows
from vnlb.utils.gpu_utils import apply_color_xform_cpp
from vnlb.testing import save_images

# -- local imports --
from ..init_mask import initMask,mask2inds,update_mask
from .streams import init_streams,wait_streams,get_hw_batches,view_batch,vprint
from .subave_impl import compute_subset_ave
from .l2norm_impl import compute_l2norm_cuda
from .fill_patches import fill_patches,fill_patches_img

#
# -- exec across concurrent streams --
#

def runSimSearch(noisy,sigma,tensors,params,step=0,gpuid=0):

    # -- move to device --
    noisy = th.FloatTensor(noisy).to(gpuid)
    device = noisy.device

    # -- extract info for explicit call --
    t,c,h,w = noisy.shape
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = params['nSimilarPatches'][step]
    nwindow_xy = params['sizeSearchWindow'][step]
    nfwd = params['sizeSearchTimeFwd'][step]
    nbwd = params['sizeSearchTimeBwd'][step]
    nwindow_t = nfwd + nbwd + 1
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    use_imread = params['use_imread'][step] # use rgb for patches or yuv?
    step_s = params['procStep'][step]
    basic = optional(tensors,'basic',th.zeros_like(noisy))
    assert ps_t == 2,"patchsize for time must be 2."
    nstreams = optional(params,'nstreams',1)

    # -- format flows for c++ (t-1 -> t) --
    if check_flows(tensors):
        check_and_expand_flows(tensors,t)

    # -- extract tensors --
    zflow = th.zeros((t,2,h,w),dtype=th.float32).to(gpuid)
    fflow = optional(tensors,'fflow',zflow.clone()).to(gpuid)
    bflow = optional(tensors,'bflow',zflow.clone()).to(gpuid)

    # -- color transform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)

    # -- create mask --
    nframes,chnls,height,width = noisy.shape
    mask = torch.zeros(nframes,height,width).type(torch.int8).to(device)
    # mask = torch.ByteTensor(mask).to(device)

    # -- find the best patches using c++ logic --
    srch_img = noisy_yuv if step1 else basic_yuv
    results = exec_sim_search(srch_img,basic_yuv,fflow,bflow,mask,
                              sigma,ps,ps_t,npatches,step_s,nwindow_xy,
                              nfwd,nbwd,couple_ch,step1,nstreams)
    patchesNoisy,patchesBasic,dists,indices = results
    dists = rearrange(dists,'nb b p -> (nb b) p')
    indices = rearrange(indices,'nb b p -> (nb b) p')

    # -- group the values and indices --
    img_noisy = noisy if use_imread else noisy_yuv
    img_basic = basic if use_imread else basic_yuv
    # patchesNoisy = fill_patches_img(img_noisy,indices,ps,ps_t)
    # patchesBasic = fill_patches_img(img_basic,indices,ps,ps_t)

    # -- groups from patches --
    # patchesNoisy = groups2patches(groupNoisy)
    # patchesBasic = groups2patches(groupBasic)
    # groupNoisy = groups2patches(patchesNoisy.cpu().numpy())
    # groupBasic = groups2patches(patchesBasic.cpu().numpy())
    groupNoisy = None
    groupBasic = None

    # -- extract some params info --
    i_params = edict({k:v[step] if isinstance(v,list) else v for k,v in params.items()})
    pinfo = get_patch_shapes_from_params(i_params,c)
    patch_num,patch_dim,patch_chnls = pinfo
    nsearch = nwindow_xy * nwindow_xy * (nfwd + nbwd + 1)

    # -- pack results --
    results = edict()
    results.patches = patchesNoisy
    results.patchesNoisy = patchesNoisy
    results.patchesBasic = patchesBasic
    results.groupNoisy = groupNoisy
    results.groupBasic = groupBasic
    results.indices = indices
    results.nSimP = len(indices)
    results.nflat = results.nSimP * ps * ps * ps_t * c
    results.values = dists
    results.nsearch = nsearch
    results.ngroups = patch_num
    results.npatches = len(patchesNoisy)
    results.ps = ps
    results.ps_t = ps_t
    results.psX = ps
    results.psT = ps_t
    results.access = None

    return results

def exec_sim_search(noisy,basic,fflow,bflow,mask,sigma,ps,ps_t,npatches,
                    step_s,w_s,nWt_f,nWt_b,couple_ch,step1,nstreams):
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
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,t-1)
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    bsize = 64
    nelems = torch.sum(mask==0).item()
    nbatches = divUp(nelems,bsize)

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    ns,np = nstreams,npatches
    patchesNoisy = torch.zeros(nbatches,bsize,npatches,ps_t,c,ps,ps).to(device)
    patchesBasic = torch.zeros(nbatches,bsize,npatches,ps_t,c,ps,ps).to(device)
    vals = torch.zeros(nbatches,bsize,npatches).type(torch.float32).to(device)
    inds = -torch.ones(nbatches,bsize,npatches).type(torch.int32).to(device)
    # vals = torch.zeros(npatches,t,h,w).type(torch.float32).to(device)
    # inds = torch.zeros(npatches,t,h,w).type(torch.int32).to(device)

    # -- exec search --
    for batch in range(nbatches):

        # -- assign to stream --
        cs = curr_stream
        torch.cuda.set_stream(streams[cs])
        cs_ptr = streams[cs].cuda_stream

        # -- grab access --
        access = mask2inds(mask,bsize)
        if access.shape[0] == 0: break

        # -- grab data for current stream --
        vals_s = vals[batch]
        inds_s = inds[batch]
        patchesNoisy_s = patchesNoisy[batch]
        patchesBasic_s = patchesBasic[batch]

        # -- sim search block --
        sim_search_batch(noisy,basic,patchesNoisy_s,patchesBasic_s,
                         access,vals_s,inds_s,fflow,bflow,
                         step_s,bsize,ps,ps_t,w_s,nWt_f,nWt_b,step1,cs,cs_ptr)

        # -- update mask naccess --
        update_mask(mask,access)

        # -- change stream --
        if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

    # -- wait for all streams --
    torch.cuda.synchronize()

    return patchesNoisy,patchesBasic,vals,inds

def sim_search_batch(noisy,basic,patchesNoisy,patchesBasic,access,
                     vals,inds,fflow,bflow,step_s,bsize,ps,ps_t,w_s,
                     nWt_f,nWt_b,step1,cs,cs_ptr):


    # -- compute difference --
    srch_img = noisy if step1 else basic
    l2_dists,l2_inds = compute_l2norm_cuda(srch_img,fflow,bflow,access,step_s,
                                           ps,ps_t,w_s,nWt_f,nWt_b,step1,cs_ptr)
    # -- get inds info --
    # nzero = torch.sum(l2_inds==0).item()
    # size = l2_inds.numel()
    # print("[sim_search: l2_inds] perc zero: %2.3f" % (nzero / size * 100))

    # nzero = torch.sum(l2_inds==-1).item()
    # size = l2_inds.numel()
    # print("[sim_search: l2_inds] perc invalid: %2.3f" % (nzero / size * 100))

    # -- compute topk --
    get_topk(l2_dists,l2_inds,vals,inds)

    # -- get inds info --
    # nzero = torch.sum(inds==0).item()
    # size = inds.numel()
    # print("[sim_search: inds] perc zero: %2.3f" % (nzero / size * 100))

    # nzero = torch.sum(inds==-1).item()
    # size = inds.numel()
    # print("[sim_search: inds] perc invalid: %2.3f" % (nzero / size * 100))

    # -- fill noisy patches --
    fill_patches(patchesNoisy,noisy,inds,cs_ptr)

    # -- fill basic patches --
    if not(step1): fill_patches(patchesBasic,basic,inds,cs_ptr)

def get_topk(l2_vals,l2_inds,vals,inds):

    # -- take mins --
    order = torch.argsort(l2_vals,dim=1,descending=False)

    # -- get top k --
    b,_ = l2_vals.shape
    _,k = vals.shape
    vals[:b,:] = torch.gather(l2_vals,1,order[:,:k])
    inds[:b,:] = torch.gather(l2_inds,1,order[:,:k])

# ------------------------------
#
#      Swap Tensor Dims
#
# ------------------------------

# -- swap dim --
def swap_2d_dim(tensor,dim):
    tensor = tensor.clone()
    tmp = tensor[0].clone()
    tensor[0] = tensor[1].clone()
    tensor[1] = tmp
    return tensor

def divUp(a,b): return (a-1)//b+1


