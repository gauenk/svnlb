
# -- python imports --
import numpy
from einops import rearrange
from easydict import EasyDict as edict

# -- vnlb imports --
import pyvnlb

# -- local imports --
from ..utils import optional,optional_swig_ptr,assign_swig_args
from .parser import parse_args,parse_params
from .sim_parser import sim_parser,reorder_sim_group
from .bayes_parser import parse_bayes_params
from .agg_parser import parse_agg_params

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --     Exec VNLB Denoiser    --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyVnlb(noisy,sigma,tensors=None,params=None):
    res = runVnlb_np(noisy,sigma,tensors,params)
    return res

def runVnlb_np(noisy,sigma,tensors=None,params=None):

    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,swig_args,tensors,swig_tensors = parse_args(noisy,sigma,tensors,params)

    # -- exec using numpy --
    pyvnlb.runVnlb(swig_args[0],swig_args[1],swig_tensors)

    # -- format & create results --
    res = {}
    res['denoised'] = tensors.denoised# t c h w
    res['basic'] = tensors.basic
    res['fflow'] = tensors.fflow #t c h w
    res['bflow'] = tensors.bflow

    return res

def runPyVnlbTimed(noisy,sigma,tensors=None,params=None):

    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,swig_args,tensors,swig_tensors = parse_args(noisy,sigma,params)

    # -- exec using numpy --
    pyvnlb.runVnlbTimed(swig_args[0],swig_args[1],swig_tensors)

    # -- format & create results --
    res = {}
    res['denoised'] = tensors.denoised# t c h w
    res['basic'] = tensors.basic
    res['fflow'] = tensors.fflow #t c h w
    res['bflow'] = tensors.bflow

    return res


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --  VNLB Interior Functions  --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def setVnlbParams(shape,sigma,tensors=None,params=None):
    # -- create python-params for parser --
    py_params,swig_params = parse_params(shape,sigma,params)
    return py_params

def simPatchSearch(noisy,sigma,pidx,tensors=None,params=None):

    # -- create python-params for parser --
    # noisy = noisy.copy(order="C")
    py_params,swig_params = parse_params(noisy.shape,sigma,params)
    py_params = edict({k:v[0] for k,v in py_params.items()})
    nParts = 1
    tensors,swig_tensors = sim_parser(noisy,sigma,nParts,tensors,py_params)

    # -- search everything if a negative pixel index is input --
    if pidx < 0: all_pix = True
    else: all_pix = False

    # -- exec search --
    simParams = pyvnlb.PySimSearchParams()
    simParams.nParts = nParts
    simParams.nSimP = 0
    simParams.pidx = pidx
    simParams.all_pix = all_pix
    pyvnlb.runSimSearch(swig_params[0], swig_tensors, simParams)

    # -- fix-up groups --
    psX = swig_params[0].sizePatch
    psT = swig_params[0].sizePatchTime
    t,c,h,w = noisy.shape
    nSimP = simParams.nSimP
    gNoisy_og = tensors.groupNoisy
    gBasic_og = tensors.groupBasic
    gNoisy = reorder_sim_group(tensors.groupNoisy,psX,psT,c,nSimP)
    gBasic = reorder_sim_group(tensors.groupBasic,psX,psT,c,nSimP)
    indices = rearrange(tensors.indices[:,:nSimP],'nparts nsimp -> (nparts nsimp)')

    # -- pack results --
    results = {}
    results['groupNoisy'] = gNoisy
    results['groupBasic'] = gBasic
    results['groupNoisy_og'] = gNoisy_og
    results['groupBasic_og'] = gBasic_og
    results['indices'] = indices
    results['npatches_og'] = gNoisy_og.shape[-1]
    results['npatches'] = simParams.nSimP
    results['psX'] = psX
    results['psT'] = psT
    results['nparts_omp'] = nParts

    return results

def computeBayesEstimate(groupNoisy,groupBasic,rank_var,nSimP,shape,params=None):

    # -- create python-params for parser --
    empty = numpy.zeros(shape,dtype=numpy.float32)
    params,swig_params,_,_ = parse_args(empty,0.,None,params)
    params = edict({k:v[0] for k,v in params.items()})

    # -- exec search --
    bayesParams,swig_bayesParams = parse_bayes_params(groupNoisy,groupBasic,nSimP,
                                                      rank_var,shape,params)
    pyvnlb.runBayesEstimate(swig_params[0], swig_bayesParams)

    # -- pack results --
    results = {}
    results['groupNoisy'] = bayesParams.groupNoisy
    results['groupBasic'] = bayesParams.groupBasic
    results['group'] = bayesParams.mat_group
    results['center'] = bayesParams.mat_center
    results['covMat'] = bayesParams.mat_covMat
    results['covEigVecs'] = bayesParams.mat_covEigVecs.T
    results['covEigVals'] = bayesParams.mat_covEigVals
    results['rank_var'] = swig_bayesParams.rank_var
    results['psX'] = params.sizePatch
    results['psT'] = params.sizePatchTime


    return results

def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None):

    # -- create python-params for parser --
    params,swig_params,_,_ = parse_args(deno,0.,None,params)
    params = edict({k:v[0] for k,v in params.items()})

    # -- exec search --
    aggParams,swig_aggParams = parse_agg_params(deno,group,indices,weights,
                                                mask,nSimP,params)
    pyvnlb.runAggregation(swig_params[0], swig_aggParams)

    # -- pack results --
    results = {}
    results['deno'] = aggParams.imDeno
    results['mask'] = aggParams.mask
    results['weights'] = aggParams.weights
    results['nmasked'] = aggParams.nmasked
    results['psX'] = params.sizePatch
    results['psT'] = params.sizePatchTime

    return results

def modifyEigVals(noisy,sigma,tensors=None,params=None):
    pass


