
# -- python imports --
import numpy
from einops import rearrange

# -- pyvnlb imports --
import pyvnlb

# -- local imports --
from ..utils import optional
from .parser import parse_args

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
    args,sargs = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    sargs.direction = 0
    fflow = args.fflow
    pyvnlb.runTV1Flow(sargs)
    sargs.direction = 1
    pyvnlb.runTV1Flow(sargs)
    bflow = args.bflow
    
    return fflow,bflow

def runPyTvL1Flow(noisy,sigma,pyargs=None):
    res = runPyTvL1Flow_np(noisy,sigma,pyargs)
    return res

def runPyTvL1Flow_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,sargs = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    pyvnlb.runTV1Flow(sargs)

    # -- format & create results --
    res = {}
    res['fflow'] = args.fflow #t c h w
    res['bflow'] = args.bflow

    # -- alias some vars --
    direction = optional(pyargs,'direction',0)
    if direction == 0: res['flow'] = res['fflow']
    else: res['flow'] = res['bflow']

    return res



