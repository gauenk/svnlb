

# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda

# -- local imports --
from .streams import init_streams,wait_streams,get_hw_batches,view_batch,vprint

def fill_patches_img(noisy,indices,ps,ps_t,patches=None):

    # -- init --
    device = noisy.device
    t,c,h,w = noisy.shape
    k = indices.shape[0]
    if patches is None:
        patches = torch.zeros(k,t,ps_t,c,ps,ps,h,w).to(device)

    # -- iter over batches --
    bsize = 32
    hbatches,wbatches = get_hw_batches(h,w,bsize)

    # -- synch before start --
    curr_stream = 0
    nstreams = 1
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- exec search --
    for h_start in hbatches:
        h_start = h_start.item()

        for w_start in wbatches:
            w_start = w_start.item()

            # -- assign to stream --
            cs = curr_stream
            torch.cuda.set_stream(streams[cs])

            # -- views --
            bufs.noisyView[cs] = view_batch(noisy,h_start,w_start,bsize)
            bufs.patchesView[cs] = view_batch(patches,h_start,w_start,bsize)
            bufs.indsView[cs] = view_batch(indices,h_start,w_start,bsize)

            # -- fill_patches --
            fill_patches(bufs.patchesView[cs],bufs.noisyView[cs],bufs.indsView[cs],cs)

            # -- change stream --
            if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

        # -- wait for all streams --
        wait_streams([streams[curr_stream]],streams)

    return patches

def fill_patches(patches,noisy,inds,cs):

    # -- create output --
    t,c,h,w = noisy.shape
    k,t,h_batch,w_batch = inds.shape
    k,t,ps_t,c,ps,ps,h_batch,w_batch = patches.shape

    # -- run launcher --
    fill_patches_launcher(patches,noisy,inds,cs)


def fill_patches_launcher(patches,noisy,inds,cs):

    # -- create output --
    t,c,h,w = noisy.shape
    k,t,h_batch,w_batch = inds.shape
    k,t,ps_t,c,ps,ps,h_batch,w_batch = patches.shape

    # -- to numba --
    patches_nba = cuda.as_cuda_array(patches)
    noisy_nba = cuda.as_cuda_array(noisy)
    inds_nba = cuda.as_cuda_array(inds)
    cs_nba = cuda.external_stream(cs)

    # -- thread and blocks --
    blocks = (h_batch,w_batch)
    threads = k

    # -- launch --
    fill_patches_kernel[blocks,threads,cs_nba](patches_nba,noisy_nba,inds_nba)

@cuda.jit
def fill_patches_kernel(patches,noisy,inds):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        return val

    def idx2coords(idx,color,height,width):

        # -- get shapes --
        whc = width*height*color
        wh = width*height

        # -- compute coords --
        t = (idx      ) // whc
        c = (idx % whc) // wh
        y = (idx % wh ) // width
        x = idx % width

        return t,c,y,x

    # -- shapes --
    nframes,color,height,width = noisy.shape
    # w_t,w_s,w_s,t,h_batch,w_batch = inds.shape
    k,bt,ps_t,color,ps,ps,h_batch,w_batch = patches.shape
    t = nframes
    whc = width*height*color
    wh = width*height

    # -- access with blocks and threads --
    ni = cuda.threadIdx.x
    bH,bW = cuda.blockIdx.x,cuda.blockIdx.y

    # -- compute dists --
    for bT in range(nframes):
        ind = inds[ni,bT,bH,bW]
        nT,_,nH,nW = idx2coords(ind,color,height,width)
        # nT = (ind      ) // whc
        # nH = (ind % wh ) // width
        # nW = ind % width

        for pt in range(ps_t):
            for ci in range(color):
                for pi in range(ps):
                    for pj in range(ps):
                        val = noisy[nT+pt,ci,nH+pi,nW+pj]
                        patches[ni,bT,pt,ci,pi,pj,bH,bW] = val
