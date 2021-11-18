
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
