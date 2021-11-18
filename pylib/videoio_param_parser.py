import numpy as np
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import vnlb

from .utils import optional,optional_swig_ptr


def verify_video_paths(video_paths,fstart,fstep,nframes):
    for t in range(fstart,nframes,fstep):
        path = str(video_paths) % t
        assert Path(path).exists(),f"path {path} must exist for test."

def create_swig_args(args):
    sargs = vnlb.ReadVideoParams()
    for key,val in args.items():
        sval = optional_swig_ptr(val)
        setattr(sargs,key,sval)
    return sargs

def parse_args(shape,video_paths,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',False)
    t,c,h,w  = shape

    # -- video checks --
    nframes = t
    fstart = optional(pyargs,'fstart',0)
    fstep = optional(pyargs,'fstep',1)
    verify_video_paths(video_paths,fstart,fstep,nframes)

    # -- params --
    args = edict()

    # -- set required numeric values --
    args.c = c
    args.t = t
    args.h = h
    args.w = w
    args.video_paths = bytes(str(video_paths),'utf-8')
    args.first_frame = fstart
    args.last_frame = list(range(fstart,nframes,fstep))[-1]
    args.frame_step = fstep
    args.read_video = np.zeros(shape,dtype=np.float32)
    args.verbose = verbose

    # -- copy to swig --
    sargs = create_swig_args(args)

    return args, sargs
