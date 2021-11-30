

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
    rank =  params['rank'][step]

    # -- exec python version --
    groupInput = groupNoisy if step1 else groupBasic
    t,c,h,w = shape
    results = exec_bayes_estimate(groupInput,sigma,rank,nSimP,c)

    # -- format results --
    results['groupNoisy'] = groupNoisy
    results['groupBasic'] = groupBasic
    # group_key = "groupNoisy" if step1 else "groupBasic"
    # results[group_key] = results['
    results['psX'] = ps
    results['psT'] = ps_t

    return results

def exec_bayes_estimate(groupInput,sigma,rank,nSimP,channels):


    # -- main logic --
    center = center_data(groupInput)
    covMat = compute_cov_matrix(groupInput)
    eigVals,eigVecs = compute_eig_stuff(covMat,rank)
    eigVals,eigVecs,rank_var = modify_eig_stuff(eigVals,eigVecs,rank)
    group = update_group(groupInput,eigVals,eigVecs)

    # -- pack results --
    results = {}
    results['group'] = group
    results['center'] = center
    results['covMat'] = covMat
    results['covEigVecs'] = eigVecs
    results['covEigVals'] = eigVals
    results['rank_var'] = rank_var
    return results

def center_data(groupInput):
    center = None
    return center

def compute_cov_matrix(groupInput):
    covMat = None
    return covMat

def compute_eig_stuff(covMat,rank):
    eigVals,eigVecs = None,None
    return eigVals,eigVecs

def modify_eig_stuff(eigVals,eigVecs,rank):
    return eigVals,eigVecs,0.

def update_group(groupInput,eigVals,eigVecs):
    return groupInput

