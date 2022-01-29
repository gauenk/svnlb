
# -- python imports --
import numpy
from einops import rearrange

# -- swig-vnlb imports --
import svnlb

# -- local imports --
from svnlb.utils import optional
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
    args,swig_args,tensors,swig_tensors = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    swig_args.direction = 0
    svnlb.runTV1Flow(swig_args,swig_tensors)
    fflow = tensors.fflow
    swig_args.direction = 1
    svnlb.runTV1Flow(swig_args,swig_tensors)
    bflow = tensors.bflow

    return fflow,bflow

def runPyTvL1Flow(noisy,sigma,pyargs=None):
    res = runPyTvL1Flow_np(noisy,sigma,pyargs)
    return res

def runPyTvL1Flow_np(noisy,sigma,pyargs=None):

    # -- extract info --
    t,c,h,w  = noisy.shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    args,swig_args,tensors,swig_tensors = parse_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    svnlb.runTV1Flow(swig_args,swig_tensors)

    # -- format & create results --
    res = {}
    res['fflow'] = tensors.fflow #t c h w
    res['bflow'] = tensors.bflow

    # -- alias some vars --
    direction = optional(pyargs,'direction',0)
    if direction == 0: res['flow'] = res['fflow']
    else: res['flow'] = res['bflow']

    return res



