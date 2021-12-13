
"""
Compute the average of a subset of the burst
using (i) the original burst, (ii) indices,
and (iii) weights indicating the burst's subsets

"""


# -- python --
import sys
import torch
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda
from vnlb.utils import divUp

def compute_subset_ave(burst,indices,ps):

    # -- shapes --
    device = burst.device
    c,t,h,w = burst.shape
    indT,psX,psX,h_batch,w_batch = indices.shape
    assert t >= indT

    # -- reshape indices for interpretation --
    indices = rearrange(indices,'st sh sw hb wh -> (st sh sw) hb wh')
    nsearch,h_batch,w_batch = indices.shape

    # -- init ave --
    ave = torch.zeros(c,ps,ps,nsearch,h_batch,w_batch).to(device)

    # -- run launcher --
    compute_subset_ave_launcher(ave,burst,indices)

    # -- format from kernel --
    # none!

    return ave


def compute_subset_ave_launcher(ave,burst,indices,mask):

    # -- shapes --
    c,t,h,w = burst.shape
    nsearch,h_batch,w_batch = indices.shape
    c,ps,ps,nsearch,h_batch,w_batch = ave.shape

    # -- numbify the torch tensors --
    ave_nba = cuda.as_cuda_array(ave)
    burst_nba = cuda.as_cuda_array(burst)
    indices_nba = cuda.as_cuda_array(indices)

    # -- tiling --
    hTile = 4
    wTile = 4
    sTile = 4

    # -- launch params --
    hBlocks = divUp(h_batch,hTile)
    wBlocks = divUp(w_batch,wTile)
    sBlocks = divUp(nsearch,sTile)
    blocks = (hBlocks,wBlocks,sBlocks)
    threads = (c,ps,ps)

    # -- launch kernel --
    compute_subset_ave_kernel[blocks,threads](ave_nba,burst_nba,
                                              indices_nba,
                                              hTile,wTile,sTile)

@cuda.jit
def compute_subset_ave_kernel(ave,burst,indices,hTile,wTile,sTile):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        return val

    # -- shapes --
    f,t,h,w = burst.shape
    f,ps,ps,s,h_batch,w_batch = ave.shape
    psHalf = ps//2

    # -- access with blocks and threads --
    hStart = hTile*cuda.blockIdx.x
    wStart = wTile*cuda.blockIdx.y
    sStart = sTile*cuda.blockIdx.z
    fi = cuda.threadIdx.x
    pi = cuda.threadIdx.y
    pj = cuda.threadIdx.z

    # -- compute dists --
    for hiter in range(hTile):
        hi = hStart + hiter
        if hi >= h: continue
        for witer in range(wTile):
            wi = wStart + witer
            if wi >= w: continue
            for siter in range(sTile):
                si = sStart + siter
                if si >= s: continue

                d_val,a_val,z_val = 0,0,0
                for ti in range(t):

                    ind = indices[ti,si,hi,wi]
                    blkH = indices[0,ti,si,hi,wi]
                    blkW = indices[1,ti,si,hi,wi]
                    top,left = blkH-psHalf,blkW-psHalf

                    bH = bounds(top+pi,h-1)
                    bW = bounds(left+pj,w-1)
                    b_val = burst[fi][ti][bH][bW]
                    a_val += w_val*b_val
                    z_val += w_val

                ave[fi][pi][pj][si][hi][wi] = a_val/z_val

