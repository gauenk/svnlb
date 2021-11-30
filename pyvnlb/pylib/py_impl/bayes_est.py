
import scipy
import numpy as np
from einops import rearrange

def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1

def runBayesEstimate(groupNoisy,groupBasic,rank_var,nSimP,shape,params,step=0):

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
    sigmab = 20.#params['sigmaBasic'][step]
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    t,c,h,w = shape
    group_chnls = 1 if couple_ch else c

    # -- exec python version --
    groupInput = groupNoisy if step1 else groupBasic
    results = exec_bayes_estimate(groupInput,sigma,sigmab,rank,nSimP,
                                  c,group_chnls,thresh)

    # -- format results --
    # results['groupNoisy'] = groupNoisy
    # results['groupBasic'] = groupBasic
    # group_key = "groupNoisy" if step1 else "groupBasic"
    # results[group_key] = results['
    results['psX'] = ps
    results['psT'] = ps_t

    return results

def exec_bayes_estimate(groupInput,sigma,sigmab,rank,nSimP,
                        channels,group_chnls,thresh,mod_sel="clipped"):

    # -- shaping --
    p,pst,c,ps1,ps2,n = groupInput.shape
    pdim = ps1*ps2*pst*p
    groupInput = rearrange(groupInput,'p pst c ps1 ps2 n -> c (p pst ps1 ps2) n')

    # -- main logic --
    # TODO: index within the loop using group_chnls
    assert groupInput.shape[0] == group_chnls
    rank_var = 0.
    group = np.zeros_like(groupInput)
    center = np.zeros((group_chnls,pdim),dtype=np.float32)
    for chnl in range(group_chnls):
        group_c,center_c = center_data(groupInput[chnl])
        covMat = compute_cov_matrix(group_c,nSimP)
        eigVals,eigVecs = compute_eig_stuff(covMat,rank)
        eigVals = denoise_eigvals(eigVals,sigmab,mod_sel,rank)
        rank_var += np.sum(eigVals[:rank])
        eigVals = bayes_filter_coeff(eigVals,sigma,thresh)
        group_c,mat_group,eigVecs = update_group(group_c,eigVals,eigVecs,rank)
        group_c += center_c
        group[chnl] = group_c
        center[chnl] = center_c[:,0]
        break

    # -- rearrange --
    shape_str = 'c (p pst ps1 ps2) n -> p pst c ps1 ps2 n'
    kwargs = {'p':1,'pst':2,'ps1':ps1,'ps2':ps2}
    gnoisy = rearrange(group,shape_str,**kwargs)
    gbasic = rearrange(group,shape_str,**kwargs)

    # -- pack results --
    results = {}
    results['groupNoisy'] = gnoisy
    results['groupBasic'] = gbasic
    results['group'] = group_c
    results['center'] = center
    results['covMat'] = covMat
    results['covEigVecs'] = eigVecs
    results['covEigVals'] = eigVals
    results['rank_var'] = rank_var
    return results

def center_data(groupInput):
    center = groupInput.mean(axis=1)[:,None]
    centered = groupInput - center
    return centered,center

def compute_cov_matrix(groups,nSimP):
    dim,npatches = groups.shape
    covs = np.matmul(groups,groups.transpose(1,0))/nSimP
    return covs

def compute_eig_stuff(covMat,rank):
    n,n = covMat.shape
    results = np.linalg.eigh(covMat)
    results = scipy.linalg.lapack.ssyevx(covMat,1,'I',1,
                                         0,1,n-rank+1,n,
                                         overwrite_a=0)
    eigVals,eigVecs = results[0],results[1]
    return eigVals,eigVecs

def denoise_eigvals(eigVals,sigmab,mod_sel,rank):
    if mod_sel == "clipped":
        emin = np.minimum(eigVals[:rank],sigmab**2)
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

