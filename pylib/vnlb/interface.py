
# -- python imports --
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from ..utils import optional
from .parser import parse_args

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --     Exec VNLB Denoiser    --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyVnlb(noisy,sigma,pyargs=None):
    res = runVnlb_np(noisy,sigma,pyargs)
    return res

def runVnlb_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlb(sargs)

    # -- format & create results --
    res = {}
    res['final'] = args.final# t c h w 
    res['basic'] = args.basic
    res['fflow'] = args.fflow #t c h w
    res['bflow'] = args.bflow

    # -- alias some vars --
    res['denoised'] = res['final']

    return res

def runPyVnlbTimed(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlbTimed(sargs)

    # -- format & create results --
    res = {}
    res['final'] = args.final# t c h w 
    res['basic'] = args.basic
    res['fflow'] = args.fflow #t c h w
    res['bflow'] = args.bflow

    # -- alias some vars --
    res['denoised'] = res['final']

    return res


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --  VNLB Interior Functions  --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def cppParseVnlbParams(noisy,sigma,pyargs=None):
    pass

def simPatchSearch(noisy,sigma,pyargs=None):
    pass

def computeBayesEstimate(noisy,sigma,pyargs=None):
    pass

def modifyEigVals(noisy,sigma,pyargs=None):
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
