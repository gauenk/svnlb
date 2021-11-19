
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
# -- Read Images with C++ Code --
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def readVideoForVnlb(shape,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w = shape
    assert c in [1,3,4],"must have the color channel be 1, 3, or 4"
    use_bw = optional(pyargs,'bw',False)
    assert use_bw == False,"This test shouldn't convert color to bw."

    # -- parse args --
    args,sargs = parse_args(shape,video_paths,pyargs)

    # -- exec function --
    pyvnlb.readVideoForVnlbCpp(sargs)

    return args.read_video


def readVideoForFlow(shape,video_paths,pyargs=None):
    
    # -- extract info --
    t,c,h,w = shape
    if c != 1:
        shape = list(shape)
        shape[1] = 1
    # assert c == 1,"bw input shapes please."

    # -- parse args --
    args,sargs = parse_args(shape,video_paths,pyargs)

    # -- exec function --
    pyvnlb.readVideoForFlowCpp(sargs)

    return args.read_video

