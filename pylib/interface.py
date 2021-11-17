
# -- python imports --
import torch
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from .utils import optional
from .vnlb_param_parser import parse_args as parse_vnlb_args
from .flow_param_parser import parse_args as parse_flow_args
from .tests_param_parser import parse_args as parse_videoio_args

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --      Exec VNLB Denoiser --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyVnlb(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runVnlb_np(noisy,sigma,pyargs)
    return res

def runVnlb_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_vnlb_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlb(sargs)

    # -- format & create results --
    res = {}
    # res['final'] = rearrange(args.final,'t c h w -> c t h w')
    # res['basic'] = rearrange(args.basic,'t c h w -> c t h w')
    res['final'] = args.final# t c h w 
    res['basic'] = args.basic

    # -- alias some vars --
    res['denoised'] = res['final']

    return res

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --      Exec TVL1 Flow       --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyFlow(noisy,sigma,pyargs=None):
    return runPyFlowFB(noisy,sigma,pyargs)

def runPyFlowFB(noisy,sigma,pyargs=None):
    if pyargs is None: pyargs = {}

    # -- exec --
    pyargs['direction'] = 0
    resFwd = runPyTvL1Flow(noisy,sigma,pyargs)
    pyargs['direction'] = 1
    resBwd = runPyTvL1Flow(noisy,sigma,pyargs)

    # -- format --
    fflow,bflow = resFwd['fflow'],resBwd['bflow']
    fflow = numpy.ascontiguousarray(fflow.copy())
    bflow = numpy.ascontiguousarray(bflow.copy())

    return fflow,bflow

def runPyTvL1Flow(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runPyTvL1Flow_np(noisy,sigma,pyargs)
    return res

def runPyTvL1Flow_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_flow_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runTV1Flow(sargs)

    # -- format & create results --
    res = {}
    res['fflow'] = args.fflow #t c h w
    res['bflow'] = args.bflow

    # -- alias some vars --
    direction = optional(pyargs,'direction',0)
    if direction == 0: res['flow'] = res['fflow']
    else: res['flow'] = res['bflow']

    return res



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --     Test IO Precision     --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def loadVideoForVnlb(burst,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = burst.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    use_bw = optional(pyargs,'bw',False)
    assert use_bw == False,"This test shouldn't convert color to bw."

    # -- parse args --
    args,sargs = parse_videoio_args(burst,video_paths,pyargs)

    # -- exec function --
    vnlb.testLoadVideo(sargs)

    # -- result --
    delta = args.delta

    return delta

def loadVideoForFlow(burst,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = burst.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    if pyargs is None: pyargs = {}
    pyargs['bw'] = True

    # -- parse args --
    args,sargs = parse_videoio_args(burst,video_paths,pyargs)

    # -- exec function --
    vnlb.testIIORead(sargs)

    # -- result --
    delta = args.delta

    return delta

