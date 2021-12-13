
# -- python imports --
import torch
import torch as th
from einops import rearrange
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
    basic = optional(tensors,'basic',th.zeros_like(noisy))
    assert ps_t == 2,"patchsize for time must be 2."

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
    print("cuda.sim: ",noisy_yuv[0,0,0,0])

    # -- find the best patches using c++ logic --
    srch_img = noisy_yuv if step1 else basic_yuv
    patches,dists,indices,access = exec_sim_search(srch_img,fflow,bflow,sigma,
                                                   ps,ps_t,npatches,nwindow_xy,nfwd,nbwd,
                                                   couple_ch,step1)

    # -- group the values and indices --
    img_noisy = noisy if use_imread else noisy_yuv
    img_basic = basic if use_imread else basic_yuv
    patchesNoisy = fill_patches_img(img_noisy,indices,ps)
    patchesBasic = fill_patches_img(img_basic,indices,ps)

    # patchesNoisy = exec_select_cpp_patches(img_noisy,indices,ps,ps_t)
    # patchesBasic = exec_select_cpp_patches(img_basic,indices,ps,ps_t)
    # groupNoisy = exec_select_cpp_groups(img_noisy,indices,ps,ps_t)
    # groupBasic = exec_select_cpp_groups(img_basic,indices,ps,ps_t)

    # -- groups from patches --
    # patchesNoisy = groups2patches(groupNoisy)
    # patchesBasic = groups2patches(groupBasic)
    groupNoisy = groups2patches(patchesNoisy.cpu().numpy())
    groupBasic = groups2patches(patchesBasic.cpu().numpy())

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
    results.access = access

    return results

def exec_sim_search(noisy,fflow,bflow,sigma,ps,ps_t,
                    npatches,nwindow_xy,nWt_f,nWt_b,couple_ch,step1):

    # -- unpack info --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- search region aliases --
    w_s = nwindow_xy
    w_t = min(nWt_f + nWt_b + 1,t-1)
    # w_t = nWt_f + nWt_b + 1
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    bsize = 32
    h_batches,w_batches = get_hw_batches(h,w,bsize)

    # -- synch before start --
    curr_stream = 0
    nstreams = 1
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    patches = torch.zeros(npatches,t,ps,ps,h,w).to(device)
    indices = torch.zeros(npatches,t,h,w).type(torch.int32).to(device)
    access = torch.zeros(3,w_t,w_s,w_s,t,h,w).type(torch.int32).to(device)
    vals = torch.zeros(npatches,t,h,w).type(torch.float32).to(device)

    """
    ** Our "simsearch" is not the same as "vnlb" **

    1. the concurrency of using multiple cuda-streams creates io issues
       for using the mask
    2. if with no concurrency, the details of using an "argwhere" each batch
       seems strange
    3. it is unclear if we will want this functionality for future uses
       of this code chunk
    """

    # -- exec search --
    for h_start in h_batches:
        h_start = h_start.item()

        for w_start in w_batches:
            w_start = w_start.item()

            # -- assign to stream --
            cs = curr_stream
            torch.cuda.set_stream(streams[cs])

            # -- grab data of batch --
            # bufs.noisyView[cs] = view_batch(noisy,h_start,w_start,bsize)
            bufs.patchesView[cs] = view_batch(patches,h_start,w_start,bsize)
            bufs.distsView[cs] = view_batch(vals,h_start,w_start,bsize)
            bufs.indsView[cs] = view_batch(indices,h_start,w_start,bsize)
            bufs.accessView[cs] = view_batch(access,h_start,w_start,bsize)
            fflowView = view_batch(fflow,h_start,w_start,bsize)
            bflowView = view_batch(bflow,h_start,w_start,bsize)

            # -- compute difference --
            # compute_l2norm_cuda(noisy,h_start,w_start,h_batch,w_batch,
            #                     ps,ps_t,w_s,w_t,k=1)
            h_batch,w_batch = bufs.patchesView[cs].shape[-2:]
            dists,inds,_access = compute_l2norm_cuda(noisy,fflow,bflow,
                                                     h_start,w_start,
                                                     h_batch,w_batch,ps,ps_t,w_s,w_t,
                                                     nWt_f,nWt_b)

            # -- sanity check --
            bufs.accessView[cs][...] = _access
            args = torch.where(dists<100000.)
            geqz = torch.all(inds[args]>=0).item()
            print("All non-zero if not-infty: ",geqz)


            # -- compute topk --
            get_topk(dists,inds,npatches,
                     bufs.distsView[cs],bufs.indsView[cs])

            # -- fill patches --
            fill_patches(bufs.patchesView[cs],noisy,bufs.indsView[cs])

            # -- change stream --
            if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

        # -- wait for all streams --
        wait_streams([streams[curr_stream]],streams)

    return patches,vals,indices,access

def get_topk(dists,inds,k,distsView,indsView):

    # -- reshape for selection --
    dists = rearrange(dists,'wt wh ww t h w -> (wt wh ww) t h w')
    inds = rearrange(inds,'wt wh ww t h w -> (wt wh ww) t h w')

    # -- take mins --
    order = torch.argsort(dists,dim=0,descending=False)

    # -- get top k --
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


