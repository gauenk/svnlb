
import torch
from einops import rearrange,repeat

# -- package --
from svnlb.gpu.patch_utils import yuv2rgb_patches

# -- local --
from .ksd_test import ksd_test,hotelling_t_test
from .grad_weights import create_grad_weights
from .rand_subset import compute_chi_random_subset

# -- hids --
import hids

def exec_patch_subset(patches,sigma,ref_patches=None,clean=None,ngroups=20,
                      gsize=5,nkeep=100,inds=None,
                      hypo_method="hotelling",split_method="perm"):

    return hids_fxn(patches,ref_patches,sigma,gsize,nkeep,clean,inds)
    # return exec_chi_rand_subset(patches,ref_patches,sigma,gsize,nkeep,clean,inds)
    # return exec_patch_influence(patches,ref_patches,sigma,gsize,nkeep,clean,inds)
    # return exec_patch_subset_hypo(patches,ngroups=ngroups,gsize=gsize,
    #                               method=hypo_method,split_method=split_method)


# --------------------------------------------
#
#   Use influence function to find subset
#
# --------------------------------------------

def exec_patch_subset_filter(patches,inds,sigma,nkeep,cs_ptr,**kwargs):

    # -- shaping --
    device = patches.device
    b,n,pt,c,ph,pw = patches.shape
    patches_rs = rearrange(patches,'b n pt c ph pw -> b n (pt c ph pw)')

    # -- edge case --
    if n <= nkeep: return patches,inds

    # -- yuv => rgb --
    # patches_rs = patches_rs.clone()
    patches_rs = yuv2rgb_patches(patches.clone())

    # -- keep only valid? --
    ivalid = torch.where(torch.all(inds!=-1,1))[0]
    print("ivalid.shape: ",ivalid.shape)
    if len(ivalid) < 128: return None,inds,None,None

    # -- exec search --
    rpatches,cpatches = None,None
    valid_patches = patches_rs[ivalid]/255.
    print("sigma: ",sigma)
    h_vals,h_inds = hids.subset_search(valid_patches,sigma/255.,nkeep,
                                       "beam",**kwargs)
    # h_inds = repeat(torch.arange(nkeep),'n -> b n',b=b).to(device)

    # h_inds,rpatches,cpatches = hids.subset_search(patches_rs,sigma,nkeep,
    #                                               "coreset",**kwargs)
    # rpatches = rearrange(rpatches,'b n (t c h w) -> b n t c h w',t=2,c=3,h=7)
    # if not(cpatches is None):
    #     cpatches = rearrange(cpatches,'b n (t c h w) -> b n t c h w',t=2,c=3,h=7)
    print("menu: ",torch.sort(inds[0]).values)

    # -- gather --
    patches_keep = hids.gather_data(patches_rs,h_inds)
    inds_keep = hids.gather_data(inds[:,:,None],h_inds)[:,:,0]
    # delta = torch.abs(inds[:,:2] - inds_keep[:,:2]).sum().item()
    # assert delta < 1e-8

    return patches_keep,inds_keep,cpatches,rpatches

def hids_fxn(patches,ref_patches,sigma,gsize,nkeep,clean,inds):

    # -- subset inds --
    h_vals,h_inds = hids.subset_search(patches,sigma,nkeep,"beam")
    # h_vals,h_inds = hids.subset_search(patches,sigma,nkeep,"coreset")

    # patches = torch.gather(patches,inds)

    return final_patches,orders,final_weights,bias


def prepare_patches(patches,color):
    pass

def exec_chi_rand_subset(patches,ref_patches,sigma,gsize,nkeep,clean=None,inds=None):

    # -- shape --
    device = patches.device
    bsize,nsamples,dim = patches

    # -- reshape --
    outputs = compute_chi_random_subset(patches,sigma,gsize,nkeep)
    final_patches,orders,final_weights,bias = outputs


    return final_patches,orders,final_weights,bias

def exec_patch_influence(patches,ref_patches,sigma,gsize=30,nkeep=100,
                         clean=None,inds=None):

    # -- shape --
    device = patches.device
    color = 3
    bsize,nsamples,dim = patches.shape

    # -- split datasets --
    if ref_patches is None:
        ref_patches = patches[:,:10]#gsize]
    # candidate_patches = patches[:,gsize:]

    # -- reshape patches --
    # print("patches.shape: ",patches.shape)
    ref_patches = rearrange(ref_patches,'(b c) n p -> b n c p',c=color)
    patches = rearrange(patches,'(b c) n p -> b n (c p)',c=color)
    candidate_patches = rearrange(patches,'b n (c p) -> b n c p',c=color)
    # print(ref_patches.shape,candidate_patches.shape)

    # -- single v.s. muli color --
    # ref_patches = ref_patches[:,:,0,:]
    # candidate_patches = candidate_patches[:,:,0,:]

    # ref_patches = rearrange(ref_patches,'b n c p -> b n (c p)')
    # candidate_patches = rearrange(candidate_patches,'b n c p -> b n (c p)')

    print("ref_patches.shape: ",ref_patches.shape)
    ref_patches = rearrange(ref_patches,'b n c p -> (b c) n p')
    candidate_patches = rearrange(candidate_patches,'b n c p -> (b c) n p')

    # -- exec weight selection --
    # print("[impl.py] patches.shape: ",patches.shape)
    # print("candidate_patches.shape: ",candidate_patches.shape)
    orders,weights,bias = create_grad_weights(ref_patches,candidate_patches,
                                              sigma,nkeep,clean)
    # print("weights.shape: ",weights.shape)

    # -- create return weights --
    # weights = rearrange(weights,'(b c) p -> b c p',c=3)
    # for ci in range(color):
    #     weights[:,ci] = torch.gather(weights[:,ci],1,orders)
    # weights = rearrange(weights,'b c p -> (b c) p',c=3)
    selected_weights = weights


    # selected_weights = torch.gather(weights,1,orders)
    # print("selected_weights.shape: ",selected_weights.shape)
    # wrefs = torch.zeros((bsize,gsize),device=device)
    # final_weights = torch.cat([wrefs,selected_weights],1)
    final_weights = selected_weights
    # final_weights = torch.exp(-final_weights)
    print("final_weights.shape: ",final_weights.shape)
    # final_weights /= torch.sum(final_weights,1,keepdim=True)

    # -- create return patches --
    selected_patches = candidate_patches
    # aug_orders = repeat(orders,'b n -> b n p',p=dim)
    # spatches = selected_patches
    # spatches = rearrange(spatches,'(b c) n p -> b c n p',c=color)
    # for ci in range(color):
    #     spatches[:,ci] = torch.gather(spatches[:,ci],1,aug_orders)
    # spatches = rearrange(spatches,'b c n p -> (b c) n p')
    # selected_patches = spatches
    # print("selected_patches.shape: ",selected_patches.shape)

    # final_patches = torch.cat([ref_patches,selected_patches],1)
    final_patches = selected_patches
    # print("final_patches.shape: ",final_patches.shape)


    # patches = final_weights[:,:,None] * final_patches#patches
    patches = final_patches#patches

    # -- fix shape back --
    # patches = rearrange(patches,'b n (c p) -> (b c) n p',c=color)

    # print("weights.shape: ",weights.shape)
    # print("patches.shape: ",patches.shape)
    final_patches = patches

    # -- select inds --
    # if not(inds is None):
    #     # print("inds.shape: ",inds.shape)
    #     # print("orders.shape: ",orders.shape)
    #     # nsave = orders.shape[1]
    #     inds[...] = torch.gather(inds,1,orders)

    # -- testing --
    # print("-"*15)
    # print("testing.")
    # print("-"*15)
    # delta = selected_patches[0,None,:] - selected_patches[0,:,None]
    # print(delta.shape)
    # delta = torch.sum(delta**2,-1)
    # print(delta.shape)
    # print(torch.where(delta < 1e-8))
    # print(final_weights[0,:])
    # final_weights[...] = 1.

    return final_patches,orders,final_weights,bias


# --------------------------------------------
#
#   Use random subsets of a hypothesis test
#
# --------------------------------------------


def exec_patch_subset_hypo(patches,ngroups=20,gsize=10,
                                  method="hotelling",split_method="perm"):



    # -- shape --
    device = patches.device
    bsize,nsamples,dim = patches.shape

    # -- split datasets --
    ref_patches = patches[:,:gsize]
    candidate_patches = patches[:,gsize:]

    # -- select groups --
    group_inds = create_group_inds(nsamples-gsize,ngroups,gsize,split_method)
    group_inds = group_inds.to(device)
    print("c: ",candidate_patches.shape)
    print("g: ",group_inds.shape)

    # -- create groups --
    grouped_patches = torch.zeros(bsize,ngroups,gsize,dim).to(device)
    for g in range(ngroups):
        gpatches = candidate_patches[:,group_inds[g],:]
        grouped_patches[:,g,:,:] = gpatches

    # -- run hypothesis tests --
    print("p: ",ref_patches.shape)
    print("g: ",grouped_patches.shape)
    stats = run_hypothesis_test(ref_patches,grouped_patches,method)

    # -- ave stat of each group --
    counts = torch.zeros(bsize,nsamples,device=device)
    weights = torch.zeros(bsize,nsamples,device=device)
    for g in range(ngroups):
        weights[:,group_inds[g]] += stats[:,[g]]
        counts[:,group_inds[g]] += 1
    nzi = torch.where(counts > 0)
    weights[nzi] /= counts[nzi]
    zi = torch.where(counts == 0)
    weights[zi] = float("inf")

    # -- sort according to ave stat --
    orders = torch.argsort(weights,dim=1)
    print(orders)
    assert torch.all(weights>0) is True

    # -- prints --
    print("s: ",stats.shape)
    print("p: ",patches.shape)
    print("o: ",orders.shape)
    print("g: ",group_inds.shape)
    print("w: ",weights.shape)

    return patches[orders],orders,0.

def run_hypothesis_test(r_patches,g_patches,method="ksd"):
    if method == "ksd":
        return ksd_test(r_patches,g_patches)
    elif method == "hotelling":
        return hotelling_t_test(r_patches,g_patches)
    else:
        raise ValueError(f"Uknown hypothesis test [{method}]")

def create_group_inds(nsamples,ngroups,gsize,method="rint"):

    # -- permutation method [slower] --
    if method == "perm":
        inds = torch.zeros(ngroups,gsize)
        for g in range(ngroups):
            inds[g] = torch.randperm(nsamples)[:gsize]
    # -- rand int method [fast] --
    elif method == "rint":
        inds = torch.randint(0,nsamples,(ngroups,gsize))
    else:
        raise ValueError(f"Uknown method for simulating groups: [{method}]")

    return inds.long()


