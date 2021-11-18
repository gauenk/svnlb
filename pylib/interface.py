
# -- python imports --
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from .utils import optional
from .vnlb_param_parser import parse_args as parse_vnlb_args
from .flow_param_parser import parse_args as parse_flow_args
from .videoio_param_parser import parse_args as parse_videoio_args

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --      Exec VNLB Denoiser --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyVnlb(noisy,sigma,pyargs=None):
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
    res['final'] = args.final# t c h w 
    res['basic'] = args.basic
    res['fflow'] = args.fflow #t c h w
    res['bflow'] = args.bflow

    # -- alias some vars --
    res['denoised'] = res['final']

    return res

def runVnlbTimed(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_vnlb_args(noisy,sigma,pyargs)

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
# --      Exec TVL1 Flow       --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyFlow(noisy,sigma,pyargs=None):
    return runPyFlowFB(noisy,sigma,pyargs)

def runPyFlowFB(noisy,sigma,pyargs=None):

    # -- extract info --
    t,c,h,w = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_flow_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    sargs.direction = 0
    fflow = args.fflow
    vnlb.runTV1Flow(sargs)
    sargs.direction = 1
    vnlb.runTV1Flow(sargs)
    bflow = args.bflow
    
    return fflow,bflow

def runPyTvL1Flow(noisy,sigma,pyargs=None):
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

def readVideoForVnlb(shape,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w = shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    use_bw = optional(pyargs,'bw',False)
    assert use_bw == False,"This test shouldn't convert color to bw."

    # -- parse args --
    args,sargs = parse_videoio_args(shape,video_paths,pyargs)

    # -- exec function --
    vnlb.readVideoForVnlb(sargs)

    return args.read_video


def readVideoForFlow(shape,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w = shape
    if c != 1:
        shape = list(shape)
        shape[1] = 1
    # assert c == 1,"bw input shapes please."

    # -- parse args --
    args,sargs = parse_videoio_args(shape,video_paths,pyargs)

    # -- exec function --
    vnlb.readVideoForFlow(sargs)

    return args.read_video

