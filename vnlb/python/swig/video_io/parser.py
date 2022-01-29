
import numpy as np
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import svnlb

from svnlb.utils import optional,optional_swig_ptr,assign_swig_args


def verify_video_paths(video_paths,fstart,fstep,nframes):
    for t in range(fstart,nframes,fstep):
        path = str(video_paths) % t
        assert Path(path).exists(),f"path {path} must exist for test."

def parse_args(shape,video_paths,pyargs):

    # -- extract info --
    verbose = optional(pyargs,'verbose',False)
    t,c,h,w  = shape

    # -- video checks --
    nframes = t
    fstart = optional(pyargs,'fstart',0)
    fstep = optional(pyargs,'fstep',1)
    verify_video_paths(video_paths,fstart,fstep,nframes)

    # -- set required numeric values --
    args = edict()
    args.video_paths = bytes(str(video_paths),'utf-8')
    args.first_frame = fstart
    args.last_frame = list(range(fstart,nframes,fstep))[-1]
    args.frame_step = fstep
    args.verbose = verbose

    # -- set tensors --
    tensors = edict()
    tensors.c = c
    tensors.t = t
    tensors.h = h
    tensors.w = w
    tensors.noisy = np.zeros(shape,dtype=np.float32)

    # -- copy to swig --
    sargs = svnlb.ReadVideoParams()
    assign_swig_args(args,sargs)
    targs = svnlb.VnlbTensors()
    assign_swig_args(tensors,targs)

    return args, sargs, tensors, targs
