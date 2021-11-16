
# -- python imports --
import torch
import numpy
from einops import rearrange

# -- vnlb imports --
import vnlb

# -- local imports --
from .vnlb_param_parser import parse_args as parse_vnlb_args
from .tvl1_param_parser import parse_args as parse_tvl1_args

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
    c,t,h,w  = noisy.shape
    args,sargs = parse_vnlb_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runVnlb(sargs)

    # -- format & create results --
    res = {}
    res['final'] = rearrange(args.final,'t c h w -> c t h w')

    # -- alias some vars --
    res['denoised'] = res['final']

    return res

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# --      Exec TVL1 Flow       --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def runPyFlowFB(noisy,sigma,pyargs=None):
    if pyargs is None: pyargs = {}
    pyargs['direction'] = 0
    resFwd = runPyTvL1Flow(noisy,sigma,pyargs)
    pyargs['direction'] = 1
    # resBwd = runPyTvL1Flow(noisy,sigma,pyargs)
    resBwd = resFwd
    res = {'fflow':resFwd['fflow'],'bflow':resBwd['bflow']}
    return res

def runPyFlow(noisy,sigma,pyargs=None):
    return runPyTvL1Flow(noisy,sigma,pyargs)

def runPyTvL1Flow(noisy,sigma,pyargs=None):
    if torch.is_tensor(noisy):
        noisy = noisy.cpu().numpy()
    res = runPyTvL1Flow_np(noisy,sigma,pyargs)
    return res

def runPyTvL1Flow_np(noisy,sigma,pyargs=None):
    
    # -- extract info --
    c,t,h,w  = noisy.shape
    args,sargs = parse_tvl1_args(noisy,sigma,pyargs)

    # -- exec using numpy --
    vnlb.runTV1Flow(sargs)

    # -- format & create results --
    res = {}
    res['fflow'] = rearrange(args.fflow,'t c h w -> c t h w')
    res['bflow'] = rearrange(args.bflow,'t c h w -> c t h w')

    # -- alias some vars --
    res['flow'] = res['fflow']

    return res

