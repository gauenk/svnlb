
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
    mask = torch.zeros_like(noisy).type(torch.byte)
    # mask = torch.ByteTensor(mask).to(device)

    # -- find the best patches using c++ logic --
    srch_img = noisy_yuv if step1 else basic_yuv
    results = exec_sim_search(srch_img,basic_yuv,fflow,bflow,mask,
                              sigma,ps,ps_t,npatches,step_s,nwindow_xy,
                              nfwd,nbwd,couple_ch,step1,nstreams)
    patchesNoisy,patchesBasic,dists,indices = results

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

def mask2inds(mask,bsize):
    index = torch.nonzero(mask-1)
    return index[:bsize]

def update_mask(mask,inds):
    mask[inds] = 1

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
    bsize = 16
    h_batches,w_batches = get_hw_batches(h,w,bsize*step_s)

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    ns,np = nstreams,npatches
    patchesNoisy = torch.zeros(npatches,t,ps_t,c,ps,ps,h,w).to(device)
    patchesBasic = torch.zeros(npatches,t,ps_t,c,ps,ps,h,w).to(device)
    vals = torch.zeros(np,t,h,w).type(torch.float32).to(device)
    inds = -torch.ones(np,t,h,w).type(torch.int32).to(device)
    # vals = torch.zeros(npatches,t,h,w).type(torch.float32).to(device)
    # inds = torch.zeros(npatches,t,h,w).type(torch.int32).to(device)

    # -- exec search --
    for h_start in h_batches:
        h_start = h_start.item()

        for w_start in w_batches:
            w_start = w_start.item()

            # -- assign to stream --
            cs = curr_stream
            torch.cuda.set_stream(streams[cs])
            cs_ptr = streams[cs].cuda_stream

            # -- grab access --
            access = mask2inds(mask,bsize*bsize)

            # -- grab data for current stream --
            vals_s = view_batch(vals,h_start,w_start,bsize)
            inds_s = view_batch(inds,h_start,w_start,bsize)
            patchesNoisy_s = view_batch(patchesNoisy,h_start,w_start,bsize)
            patchesBasic_s = view_batch(patchesBasic,h_start,w_start,bsize)

            # -- sim search block --
            sim_search_batch(noisy,basic,patchesNoisy_s,patchesBasic_s,
                             access,vals_s,inds_s,fflow,bflow,h_start,w_start,
                             step_s,bsize,ps,ps_t,w_s,nWt_f,nWt_b,step1,cs,cs_ptr)

            # -- update mask naccess --
            update_mask(mask,access)

            # -- change stream --
            if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

    # -- wait for all streams --
    torch.cuda.synchronize()

    return patchesNoisy,patchesBasic,vals,inds


def sim_search_batch(noisy,basic,patchesNoisy,patchesBasic,access,vals,inds,
                     fflow,bflow,step_s,h_start,w_start,bsize,ps,ps_t,w_s,
                     nWt_f,nWt_b,step1,cs,cs_ptr):


    # -- compute difference --
    srch_img = noisy if step1 else basic
    l2_dists,l2_inds = compute_l2norm_cuda(srch_img,fflow,bflow,step_s,
                                           access,h_start,w_start,bsize,ps,ps_t,
                                           w_s,nWt_f,nWt_b,step1,cs_ptr)
    # -- get inds info --
    nzero = torch.sum(l2_inds==0).item()
    size = l2_inds.numel()
    print("[sim_search: l2_inds] perc zero: %2.3f" % (nzero / size * 100))

    # -- compute topk --
    get_topk(l2_dists,l2_inds,vals,inds)

    # -- [toy] top k --
    # print(inds.shape)
    # t,c,h,w = noisy.shape
    # bh,bw = patchesNoisy.shape[-2:]
    # inds_toy = (torch.arange(bh*bw*t)*c)%(h*w*c)#*(t-1)*c)
    # inds_toy = repeat(inds_toy,'(bt bh bw) -> n bt bh bw',n=100,bt=t,bh=bh)
    # inds[...] = inds_toy

    # -- get inds info --
    nzero = torch.sum(inds==0).item()
    size = inds.numel()
    print("[sim_search: inds] perc zero: %2.3f" % (nzero / size * 100))

    # -- fill noisy patches --
    fill_patches(patchesNoisy,noisy,inds,cs_ptr)

    # -- fill basic patches --
    if not(step1): fill_patches(patchesBasic,basic,inds,cs_ptr)

def get_topk(dists,inds,distsView,indsView):

    # -- reshape for selection --
    dists = rearrange(dists,'wt wh ww t h w -> (wt wh ww) t h w')
    inds = rearrange(inds,'wt wh ww t h w -> (wt wh ww) t h w')

    # -- take mins --
    order = torch.argsort(dists,dim=0,descending=False)

    # -- get top k --
    print(order.shape)
    print(distsView.shape,indsView.shape)
    k = distsView.shape[0]
    distsView[...] = torch.gather(dists,0,order[:k])
    indsView[...] = torch.gather(inds,0,order[:k])

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


