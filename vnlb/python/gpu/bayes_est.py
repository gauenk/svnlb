
import torch
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

# from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups

def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1

def runBayesEstimate(patchesNoisy,patchesBasic,rank_var,nSimP,shape,params,
                     step=0,flatPatch=False):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- init outputs --
    t,c,h,w = noisy.shape
    zero_basic = th.zeros_like(noisy)
    deno = th.zeros_like(noisy)
    basic = optional(tensors,'basic',zero_basic)
    nstreams = optional(params,'nstreams',1)
    flows = tensors
    deno = th.zeros_like(noisy)
    fflow = flows['fflow']
    bflow = flows['bflow']

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
    t,chnls,h,w = shape
    group_chnls = 1 if couple_ch else c

    # -- exec python version --
    results = exec_bayes_estimate(patchesNoisy,patchesBasic,sigma2,sigmab2,rank,
                                  nSimP,chnls,group_chnls,thresh,step==1,flatPatch)

    rank_var = results['rank_var']
    return rank_var

def centering(patches,center=None):
    if center is None:
        center = patches.mean(dim=2,keepdim=True)
    patches[...] -= center
    return center

def centering_patches(patchesNoisy,patchesBasic,step2,flat_patch):
    # -- center basic --
    centerBasic = None
    if step2:
        centerBasic = centering(patchesBasic)

    # -- center noisy --
    centerNoisy = None
    if step2 and flat_patch:
        centerNoisy = centerBasic
    centerNoisy = centering(patchesNoisy,centerNoisy)
    return centerNoisy,centerBasic

def compute_cov_mat(patches,rank):
    # return compute_cov_mat_v1(patches,rank)
    return compute_cov_mat_v2(patches,rank)

def compute_cov_mat_v1(patches,rank):

    # -- cov mat --
    device = patches.device
    bsize,chnls,num,pdim = patches.shape
    patches = rearrange(patches,'b c n p -> (b c) n p')
    covMat = torch.matmul(patches.transpose(2,1),patches)
    covMat /= num

    # -- eigs --
    kwargs = {"compute_v":1,"range":'I',"lower":0,"vl":-1,"vu":0,
              "il":pdim-rank+1,"iu":pdim,"abstol":0,"overwrite_a":1}
    eigVals,eigVecs = [],[]
    covMat_np = covMat.cpu().numpy()
    for n in range(covMat.shape[0]):

        # -- eigen stuff --
        _covMat = covMat_np[n]
        results = scipy_linalg.lapack.ssyevx(_covMat,**kwargs)

        # -- format eigs --
        _eigVals,_eigVecs = results[0],results[1]
        _eigVals = _eigVals.astype(np.float32)
        _eigVecs = _eigVecs.astype(np.float32)

        # -- to pytorc --
        _eigVals = torch.FloatTensor(_eigVals).to(device)
        _eigVecs = torch.FloatTensor(_eigVecs).to(device)

        # -- pytorch eig --
        # _eigVals,_eigVecs = torch.linalg.eig(covMat[n])

        # -- append --
        eigVals.append(_eigVals)
        eigVecs.append(_eigVecs)

    # -- stack --
    eigVals = torch.stack(eigVals)
    eigVecs = torch.stack(eigVecs)

    # -- reshape --
    eigVals = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
    eigVecs = rearrange(eigVecs,'(b c) p p2 -> b c p p2',b=bsize)

    # -- rank --
    # eigVals = torch.real(eigVals)#[...,:rank])
    # eigVecs = torch.real(eigVecs[...,:rank])
    # eigVecs = eigVecs[...,:rank]
    # eigVals[...,rank:] = 0

    # shape = list(eigVecs.shape)
    # shape[-1] = shape[-2]
    # _eigVecs = torch.zeros(shape).to(device)
    # _eigVecs[...,:rank] = eigVecs


    return covMat,eigVals,eigVecs

def compute_cov_mat_v2(patches,rank):

    # -- cov mat --
    bsize,chnls,num,pdim = patches.shape

    patches = rearrange(patches,'b c n p -> (b c) n p')
    covMat = torch.matmul(patches.transpose(2,1),patches)
    covMat /= num

    # -- eigen stuff --
    eigVals,eigVecs = torch.linalg.eig(covMat)

    # -- eig --
    eigVals = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
    eigVecs = rearrange(eigVecs,'(b c) p p2 -> b c p p2',b=bsize)

    # -- rank --
    eigVals = torch.real(eigVals)#[...,:rank])
    eigVecs = torch.real(eigVecs[...,:rank])
    eigVals[...,rank:] = 0

    return covMat,eigVals,eigVecs

def denoise_eigvals(eigVals,sigmab2,mod_sel,rank):
    if mod_sel == "clipped":
        th_sigmab2 = torch.FloatTensor([sigmab2]).reshape(1,1,1)
        emin = torch.min(eigVals[:,:,:rank],th_sigmab2.to(eigVals.device))
        eigVals[:,:,:rank] -= emin
    else:
        raise ValueError(f"Uknown eigen-stuff modifier: [{mod_sel}]")
    return eigVals

def bayes_filter_coeff(eigVals,sigma2,thresh):
    bayes_filter_coeff_v2(eigVals,sigma2,thresh)

def bayes_filter_coeff_v2(eigVals,sigma2,thresh):
    for n in range(eigVals.shape[0]):
        for c in range(eigVals.shape[1]):
            geq = torch.where(eigVals[n,c] > (thresh*sigma2))
            leq = torch.where(eigVals[n,c] <= (thresh*sigma2))
            eigVals[n,c][geq] = 1. / (1. + sigma2 / eigVals[n,c][geq])
            eigVals[n,c][leq] = 0.

def bayes_filter_coeff_v1(eigVals,sigma2,thresh):
    geq = torch.where(eigVals > (thresh*sigma2))
    leq = torch.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.

def filter_patches(patches,eigVals,eigVecs,rank):
    # filter_patches_v2(patches,eigVals,eigVecs,rank)
    filter_patches_v1(patches,eigVals,eigVecs,rank)

def filter_patches_v2(patches,eigVals,eigVecs,rank):
    # patches.shape = (b c n p)
    for b in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            _patches = patches[b,c]
            _eigVals = eigVals[b,c]
            _eigVecs = eigVecs[b,c]

            Z = _patches @ _eigVecs

            # R = _eigVecs @ torch.diag(_eigVals[:rank])
            R = _eigVecs * _eigVals[:rank][None,:]


            hX = Z @ R.transpose(1,0)
            # hX = hX.transpose(1,0)

            patches[b,c,...] = hX

def filter_patches_v1(patches,eigVals,eigVecs,rank):

    # reshape
    bsize = patches.shape[0]
    print(patches.shape,eigVals.shape,eigVecs.shape)
    patches_rs = rearrange(patches,'b c n p -> (b c) n p')
    eigVals = rearrange(eigVals,'b c p -> (b c) p')
    eigVecs = rearrange(eigVecs,'b c p r -> (b c) p r')

    # ---------------------------------
    #      hX' = X' * U * (W * U')
    # ---------------------------------

    # Z = X'*U; (n x p) x (p x r) = (n x r)
    Z = torch.matmul(patches_rs,eigVecs)

    # R = U*W; p x r
    R = eigVecs * eigVals[:,None,:rank]
    # print("R")
    # print(R[0,:2,:2])
    # print("Ratio")
    # print(R[0,:2,:2]/eigVecs[0,:2,:2])
    # print(eigVals[0,:2])

    # hX' = Z'*R' = (X'*U)'*(U*W)'; (n x r) x (r x p) = (n x p)
    tmp = torch.matmul(Z,R.transpose(2,1))
    tmp = rearrange(tmp,'(b c) n p -> b c n p',b=bsize)
    patches[...] = tmp

def bayes_estimate_batch(in_patchesNoisy,patchesBasic,sigma2,
                         sigmab2,rank,group_chnls,thresh,
                         step2,flat_patch,cs,cs_ptr,mod_sel="clipped"):

    # -- shaping --
    patchesNoisy = in_patchesNoisy
    shape = list(patchesNoisy.shape)
    # shape[1],shape[-1] = 1,nSimP
    # print("patchesNoisy.shape: ",patchesNoisy.shape)
    num,t_bsize,ps_t,chnls,ps,ps,h_bsize,w_bsize = patchesNoisy.shape
    bsize = h_bsize*w_bsize*t_bsize
    pdim = ps*ps*ps_t

    # -- ravel out that pdim --
    shape_str = "n tb pt c ph pw hb wb -> (tb hb wb) c n (pt ph pw)"
    patchesNoisy = rearrange(patchesNoisy,shape_str)
    if step2: patchesBasic = rearrange(patchesBasic,shape_str)
    # shape = (b c n p)

    # -- group noisy --
    centerNoisy,centerBasic = centering_patches(patchesNoisy,patchesBasic,
                                                step2,flat_patch)
    # -- denoising! --
    rank_var = 0.

    # -- compute eig stuff --
    patchesInput = patchesNoisy if not(step2) else patchesBasic
    covMat,eigVals,eigVecs = compute_cov_mat(patchesInput,rank)
    eigVals = denoise_eigvals(eigVals,sigmab2,mod_sel,rank)
    rank_var = torch.mean(torch.sum(eigVals,dim=2),dim=1)
    bayes_filter_coeff(eigVals,sigma2,thresh)
    filter_patches(patchesNoisy,eigVals,eigVecs,rank)

    # -- add back center --
    patchesNoisy[...] += centerNoisy

    # -- reshape --
    shape_str = '(bt bh bw) c n (pt px py) -> n bt pt c px py bh bw'
    kwargs = {"pt":ps_t,"px":ps,"bh":h_bsize,"bw":w_bsize}
    patchesNoisy = rearrange(patchesNoisy,shape_str,**kwargs)
    if step2:
        patchesBasic = rearrange(patchesBasic,shape_str,**kwargs)
    in_patchesNoisy[...] = patchesNoisy

    # -- pack results --
    results = {}
    # results['patchesNoisy'] = patchesNoisy
    # results['patchesBasic'] = patchesBasic
    # results['patches'] = patchesNoisy
    # results['center'] = centerNoisy
    # results['covMat'] = covMat
    # results['covEigVecs'] = eigVecs
    # results['covEigVals'] = eigVals
    # results['rank_var'] = rank_var
    # return results

    return rank_var,patchesNoisy
