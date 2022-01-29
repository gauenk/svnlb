
# -- python imports --
import numpy
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- vnlb imports --
import svnlb
from svnlb.utils import groups2patches

# -- local imports --
from svnlb.utils import optional,optional_swig_ptr,assign_swig_args
from .parser import parse_args,parse_params
from .sim_parser import sim_parser
from .mask_parser import mask_parser
# from .sim_utils import groups2patches
from .bayes_parser import parse_bayes_params
from .agg_parser import parse_agg_params
from .covmat_parser import covmat_parser
from .flat_areas_parser import flat_areas_parser

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
    svnlb.runVnlb(swig_args[0],swig_args[1],swig_tensors)

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
    svnlb.runVnlbTimed(swig_args[0],swig_args[1],swig_tensors)

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

def simPatchSearch(noisy,sigma,pidx,tensors=None,params=None,step=0):

    # -- create python-params for parser --
    # noisy = noisy.copy(order="C")
    py_params,swig_params = parse_params(noisy.shape,sigma,params)
    py_params = edict({k:v[step] for k,v in py_params.items()})
    nParts = 1
    tensors,swig_tensors = sim_parser(noisy,sigma,nParts,tensors,py_params)

    # -- search everything if a negative pixel index is input --
    if pidx < 0: all_pix = True
    else: all_pix = False

    # -- exec search --
    simParams = svnlb.PySimSearchParams()
    simParams.nParts = nParts
    simParams.nSimP = 0
    simParams.pidx = pidx
    simParams.all_pix = all_pix
    svnlb.runSimSearch(swig_params[step], swig_tensors, simParams)

    # -- fix-up groups --
    psX = swig_params[step].sizePatch
    psT = swig_params[step].sizePatchTime
    t,c,h,w = noisy.shape
    nSimP = simParams.nSimP
    groupNoisy = tensors.groupNoisy
    groupBasic = tensors.groupBasic
    patchesNoisy = groups2patches(tensors.groupNoisy,c,psX,psT,nSimP)
    patchesBasic = groups2patches(tensors.groupBasic,c,psX,psT,nSimP)
    indices = rearrange(tensors.indices[:,:nSimP],'nparts nsimp -> (nparts nsimp)')

    # -- pack results --
    results = {}
    results['patchesNoisy'] = patchesNoisy
    results['patchesBasic'] = patchesBasic
    results['groupNoisy'] = groupNoisy
    results['groupBasic'] = groupBasic
    results['ngroups'] = groupNoisy.shape[-1]
    results['indices'] = indices
    results['npatches'] = simParams.nSimP
    results['psX'] = psX
    results['psT'] = psT
    results['nparts_omp'] = nParts

    return results

def computeBayesEstimate(groupNoisy,groupBasic,rank_var,nSimP,shape,params=None,step=0):

    # -- create python-params for parser --
    empty = numpy.zeros(shape,dtype=numpy.float32)
    params,swig_params,_,_ = parse_args(empty,0.,None,params)
    params = edict({k:v[step] for k,v in params.items()})
    assert nSimP > 3,"refactor this issue out."

    # -- exec search --
    bayesParams,swig_bayesParams = parse_bayes_params(groupNoisy,groupBasic,nSimP,
                                                      rank_var,shape,params)
    svnlb.runBayesEstimate(swig_params[step], swig_bayesParams)

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

def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None,step=0):

    # -- create python-params for parser --
    params,swig_params,_,_ = parse_args(deno,0.,None,params)
    params = edict({k:v[step] for k,v in params.items()})

    # -- exec search --
    aggParams,swig_aggParams = parse_agg_params(deno,group,indices,weights,
                                                mask,nSimP,params)
    nmasked = 0
    nmasked = svnlb.runAggregation(swig_params[step], swig_aggParams, nmasked)

    # -- pack results --
    results = {}
    results['deno'] = aggParams.imDeno
    results['mask'] = aggParams.mask
    results['weights'] = aggParams.weights
    results['nmasked'] = nmasked
    results['psX'] = params.sizePatch
    results['psT'] = params.sizePatchTime

    return results

def processNLBayes(noisy,sigma,step,tensors=None,params=None):

    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,swig_args,tensors,swig_tensors = parse_args(noisy,sigma,tensors,params)

    # -- compute border --
    border0 = 2*(args['sizeSearchWindow'][0]//2) + args['sizePatch'][0]-1
    border1 = 2*(args['sizeSearchWindow'][1]//2) + args['sizePatch'][1]-1
    border = max(border0,border1)

    # -- exec using numpy --
    ngroups = 0
    ngroups = svnlb.processNLBayesCpp(swig_args[step],swig_tensors,
                                       ngroups,border)

    # -- format & create results --
    res = {}
    res['denoised'] = tensors.denoised# t c h w
    res['basic'] = tensors.basic
    res['ngroups'] = ngroups

    return res

def computeCovMat(groups,rank):

    # -- unpack shapes --
    # groups.shape = (p c pst ps1 ps2 na)
    groups = groups.copy() # usually not contiguous b/c previously indexed
    shape = groups.shape
    nParts,chnls,nSimP = shape[0],shape[1],shape[-1]
    pdim = groups.size // (nParts*nSimP)

    # -- parser --
    cinfo = covmat_parser(groups,pdim,nSimP,chnls,rank)
    params,covMat,covEigVals,covEigVecs = cinfo

    # -- exec --
    svnlb.computeCovMatCpp(params)

    # -- format output --
    results = edict()
    results.covMat = covMat
    results.covEigVals = covEigVals
    results.covEigVecs = covEigVecs.T

    return results


def init_mask(shape,vnlb_params,step=0,info=None):

    # -- parse inputs --
    t,c,h,w = shape
    mask = np.zeros((t,h,w),dtype=np.int8)
    vnlb_params = {k:v[step] for k,v in vnlb_params.items()}
    params = mask_parser(mask,vnlb_params,info)

    # -- exec using numpy --
    ngroups = 0
    ngroups = svnlb.init_mask_cpp(params,ngroups)

    # -- format & create results --
    results = edict()
    results.mask = mask
    results.ngroups = ngroups

    return results


def runFlatAreas(groupNoisy,groupBasic,nSimP,c,params):

    # -- parse inputs --
    flatAreas = params.flatAreas
    flat_params = flat_areas_parser(groupNoisy,groupBasic,flatAreas,nSimP,c)

    # -- exec --
    svnlb.runFlatAreasCpp(flat_params,params)

    # --> [no output] --> [in-place fxn]

