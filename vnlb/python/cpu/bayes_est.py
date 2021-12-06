
import scipy
import numpy as np
from einops import rearrange,repeat
import vnlb

from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups

def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1

def runBayesEstimate(groupNoisy,groupBasic,rank_var,nSimP,shape,params,
                     step=0,flatPatch=False):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- extract info for explicit call --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = params['nSimilarPatches'][step]
    nwindow_xy = params['sizeSearchWindow'][step]
    nfwd = params['sizeSearchTimeFwd'][step]
    nbwd = params['sizeSearchTimeBwd'][step]
    nwindow_t = nfwd + nbwd + 1
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    check_steps(step1,step)
    sigma = params['sigma'][step]
    sigmab2 = params['beta'][step] * params['sigmaBasic'][step]**2 if step==1 else sigma**2
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    t,c,h,w = shape
    group_chnls = 1 if couple_ch else c

    # -- exec python version --
    results = exec_bayes_estimate(groupNoisy,groupBasic,sigma,sigmab2,rank,nSimP,
                                  c,group_chnls,thresh,step==1,flatPatch)

    # -- format results --
    results['psX'] = ps
    results['psT'] = ps_t

    return results

def index_groups(group,nSimP,pdim,c):
    igroup = group.ravel()[nSimP*pdim * c:nSimP*pdim * (c+1)]
    return igroup.reshape(pdim,nSimP)

def comp_center(group,nSimP,pdim,c):
    group_f = group.ravel()[:nSimP*pdim*c]
    return np.mean(group_f.reshape(pdim*c,nSimP),axis=-1).reshape(c,pdim)

def centering(group,nSimP,pdim,c,centers=None):
    if centers is None:
        centers = comp_center(group,nSimP,pdim,c)
    nelems = nSimP*pdim*c
    groupv = group.ravel()[:nelems].reshape((c,pdim,nSimP))
    groupv -= centers[...,None]
    # for ci in range(c):
    #     index = slice(nSimP*pdim*ci,nSimP*pdim*(ci+1))
    #     # rcenter = repeat(centers[ci],'p -> (s p)',s=nSimP)
    #     group.ravel()[index].view(pdim,nSimP) -= centers[ci,None]
    return centers

def add_back_center(group,centers,nSimP,pdim,c):
    # for ci in range(c):
    #     index = slice(nSimP*pdim * ci,nSimP*pdim * (ci+1))
    #     # rcenter = repeat(centers[ci],'p -> (s p)',s=nSimP)
    groupv = group.ravel()[:nSimP*pdim*c].reshape((c,pdim,nSimP))
    groupv += centers[...,None]

def centering_v1(groupNoisy,groupBasic):

    # -- xfter to patches --
    # groupNoisy = groups2patches(groupNoisy)
    # groupBasic = groups2patches(groupBasic)

    # -- center noisy patches --
    # groupNoisy = rearrange(groupNoisy,'p c psT ps1 ps2 n -> c (p psT ps1 ps2) n')
    groupNoisy = rearrange(groupNoisy,'n psT c ps1 ps2 -> c (psT ps1 ps2) n')
    centerNoisy = groupNoisy.mean(axis=-1)
    groupNoisy -= centerNoisy[:,:,None]

    # -- if step 2, center basic patches --
    centerBasic = None
    if step2:
        # groupBasic = rearrange(groupBasic,'p c psT ps1 ps2 n -> c (p psT ps1 ps2) n')
        groupBasic = rearrange(groupBasic,'n psT c ps1 ps2 -> c (psT ps1 ps2) n')
        centerBasic = groupBasic.mean(axis=-1)
        groupBasic -= centerBasic[:,:,None]
    return groupNoisy,groupBasic,centerNoisy,centerBasic

def exec_bayes_estimate(groupNoisy,groupBasic,sigma,sigmab2,
                        rank,nSimP,channels,group_chnls,thresh,
                        step2,flatPatch,mod_sel="clipped"):

    # -- shaping --
    shape = list(groupNoisy.shape)
    shape[1] = 1
    shape[-1] = nSimP
    p,c,psT,psX,psX,n = groupNoisy.shape
    pdim = psX*psX*psT*p

    # -- group noisy --
    centerBasic = None
    if step2:
        centerBasic = centering(groupBasic,nSimP,pdim,c)
    # centers = centering_v1(groupNoisy,groupBasic)
    # groupNoisy,groupBasic,centerNoisy,centerBasic = centers

    # -- group basic --
    centerNoisy = None
    if step2 and flatPatch:
        centerNoisy = centerBasic
    centerNoisy = centering(groupNoisy,nSimP,pdim,c,centerNoisy)

    # -- denoising! --
    rank_var = 0.
    for chnl in range(group_chnls):

        # -- compute denoiser params --
        # group_c = groupNoisy[chnl] if not(step2) else groupBasic[chnl]
        groupInput = groupNoisy if not(step2) else groupBasic
        group_c = index_groups(groupInput,nSimP,pdim,chnl)

        covMat,eigVals,eigVecs = compute_eig_stuff(group_c,shape,rank)
        eigVals = denoise_eigvals(eigVals,sigmab2,mod_sel,rank)
        rank_var += np.sum(eigVals[:rank].astype(np.float32))
        eigVals = bayes_filter_coeff(eigVals,sigma,thresh)

        # -- run the denoiser --
        # groupNoisy_c = groupNoisy[chnl]
        groupNoisy_c = index_groups(groupNoisy,nSimP,pdim,chnl)

        group_c,mat_group,eigVecs = update_group(groupNoisy_c,eigVals,eigVecs,rank)

        groupNoisy.ravel()[nSimP*pdim * chnl:nSimP*pdim * (chnl+1)] = group_c.ravel()

    # -- add back center --
    add_back_center(groupNoisy,centerNoisy,nSimP,pdim,c)
    # print(groupNoisy.shape)
    # groupNoisy += centerNoisy[:,:,None]

    # -- rearrange --
    # shape_str = 'c (p pst ps1 ps2) n -> p c pst ps1 ps2 n'
    # kwargs = {'p':p,'pst':psT,'ps1':psX,'ps2':psX}
    # gnoisy = rearrange(groupNoisy,shape_str,**kwargs)
    gnoisy = groupNoisy
    gbasic = groupBasic
    # if step2:
    #     gbasic = rearrange(groupBasic,shape_str,**kwargs)

    # -- pack results --
    results = {}
    results['groupNoisy'] = gnoisy
    results['groupBasic'] = gbasic
    results['group'] = gnoisy
    results['center'] = centerNoisy
    results['covMat'] = covMat
    results['covEigVecs'] = eigVecs
    results['covEigVals'] = eigVals
    results['rank_var'] = rank_var
    return results

def compute_eig_stuff(group,shape,rank):

    # -- exec --
    group = group.reshape(shape)
    # results = vnlb.swig.computeCovMat(group,rank)
    results = computeCovMat(group,rank)

    # -- unpack --
    covMat = results.covMat
    eigVals = results.covEigVals
    eigVecs = results.covEigVecs

    return covMat,eigVals,eigVecs

def denoise_eigvals(eigVals,sigmab2,mod_sel,rank):
    if mod_sel == "clipped":
        emin = np.minimum(eigVals[:rank],sigmab2)
        eigVals[:rank] -= emin
    else:
        raise ValueError(f"Uknown eigen-stuff modifier: [{mod_sel}]")
    return eigVals

def bayes_filter_coeff(eigVals,sigma,thresh):
    sigma2 = sigma**2
    geq = np.where(eigVals > (thresh*sigma2))
    leq = np.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.
    return eigVals

def update_group(groupInput,eigVals,eigVecs,rank):

    #  hX' = X' * U * (W * U')
    eigVecs = eigVecs[:,:rank]
    eigVals = eigVals[:rank]

    # Z = X'*U
    Z = groupInput.transpose(1,0) @ eigVecs

    # R = U*W
    R = eigVecs @ np.diag(eigVals)

    # hX' = Z'*R' = (X'*U)'*(U*W)'
    group = Z @ R.transpose(1,0)

    # -- back to standard format --
    group = group.transpose(1,0)

    return group,Z,R

