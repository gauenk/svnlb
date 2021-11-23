
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
    swig_params[0].verbose = True
    pyvnlb.runSimSearch(swig_params[0], swig_tensors, simParams)

    # -- fix-up groups --
    psX = swig_params[0].sizePatch
    psT = swig_params[0].sizePatchTime
    t,c,h,w = noisy.shape
    nSimP = simParams.nSimP
    gNoisy = reorder_sim_group(tensors.groupNoisy,psX,psT,c,nSimP)
    gBasic = reorder_sim_group(tensors.groupBasic,psX,psT,c,nSimP)
    indices = rearrange(tensors.indices[:,:nSimP],'nparts nsimp -> (nparts nsimp)')

    # -- pack results --
    results = {}
    results['groupNoisy'] = gNoisy
    results['groupBasic'] = gBasic
    results['indices'] = indices
    results['npatches'] = simParams.nSimP
    results['psX'] = psX
    results['psT'] = psT

    return results

def computeBayesEstimate(noisy,sigma,tensors=None,params=None):
    pass

def modifyEigVals(noisy,sigma,tensors=None,params=None):
    pass


"""
estimateSimilarPatches
	# vector<float> groupNoisy(            patch_num * patch_dim * patch_chnls);
	# vector<float> groupBasic(step1 ? 0 : patch_num * patch_dim * patch_chnls);

float* noisy
float* basic
float* fflow
float* bflow
float* gNoisy
float* gBasic
uint32* indices
uint32 pidx


	bool step1 = params.isFirstStep;
	int sWx   = params.sizeSearchWindow;
	int sWy   = params.sizeSearchWindow;
	const int sWt_f = params.sizeSearchTimeFwd;
	const int sWt_b = params.sizeSearchTimeBwd;
	const int sPx   = params.sizePatch;
	const int sPt   = params.sizePatchTime;
        params.tau
        params.nSimilarPatches

	const bool step1 = params.isFirstStep;
	const unsigned sWx = params.sizeSearchWindow;
	const unsigned sWt = params.sizeSearchTimeFwd +
	                     params.sizeSearchTimeBwd + 1;// VIDEO
	const unsigned sPx = params.sizePatch;
	const unsigned sPt = params.sizePatchTime;
	const VideoSize sz = imNoisy.sz;

	vector<unsigned> indices(patch_num);



// add in a list of pixels
nlbParams params


estimateSimilarPatches(
	Video<float> const& imNoisy,
	Video<float> const& imBasic,
	Video<float> const& fflow,
	Video<float> const& bflow,
	std::vector<float> &groupNoisy,//output
	std::vector<float> &groupBasic, //output
	std::vector<unsigned> &indices, //output
	const unsigned pidx,
	const nlbParams &params,
	Video<float> const &imClean)


	# matWorkspace mat;
	# mat.group     .resize(patch_num * patch_dim);
	# mat.covMat    .resize(patch_dim * patch_dim);
	# mat.center.resize(patch_dim * patch_chnls);
computeBayesEstimate
	std::vector<float> &groupNoisy,
	std::vector<float> &groupBasic,
	matWorkspace &mat, //output
	nlbParams const& params,
	const unsigned nSimP,
	const unsigned channels,
	const bool flatPatch)



modifyEigVals
void modifyEigVals(matWorkspace & mat,
		   float sigmab2, int rank, 
		   int pdim, int nSimP, VAR_MODE mode){


"""
