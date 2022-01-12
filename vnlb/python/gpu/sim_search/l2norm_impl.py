"""

Compute the differences for each block
using the centroids

"""

# -- python --
import sys
import torch
import torchvision
import numpy as np
from einops import rearrange,repeat

# -- numba --
from numba import jit,njit,prange,cuda


def divUp(a,b): return (a-1)//b+1

def compute_l2norm_cuda(noisy,fflow,bflow,access,step_s,ps,ps_t,w_s,
                        nWt_f,nWt_b,step1,offset,cs,k=1):
    # todo: remove "step_s"

    # -- create output --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- init dists --
    # (w_s = windowSpace), (w_t = windowTime)
    bsize,three = access.shape
    w_t = min(nWt_f + nWt_b + 1,t-1)
    # print(bsize,w_t,w_s,w_s)
    dists = torch.ones(bsize,w_t,w_s,w_s).type(torch.float32).to(device)
    dists *= float("inf")
    indices = torch.zeros(bsize,w_t,w_s,w_s).type(torch.int32).to(device)
    bufs = torch.zeros(bsize,3,w_t,w_s,w_s).type(torch.int32).to(device)

    # -- run launcher --
    dists *= torch.inf
    indices[...] = -1
    # print("cuda_l2: ",noisy[0,0,0,0])
    # print("[l2norm_cuda] ps,ps_t: ",ps,ps_t)
    compute_l2norm_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                            ps,ps_t,nWt_f,nWt_b,step1,offset,cs)
    # -- reshape --
    dists = rearrange(dists,'b wT wH wW -> b (wT wH wW)')
    indices = rearrange(indices,'b wT wH wW -> b (wT wH wW)')

    return dists,indices



def create_frame_range(nframes,nWt_f,nWt_b,ps_t,device):
    tranges,n_tranges,min_tranges = [],[],[]
    for t_c in range(nframes-ps_t+1):

        # -- limits --
        shift_t = min(0,t_c - nWt_b) + max(0,t_c + nWt_f - nframes + ps_t)
        t_start = max(t_c - nWt_b - shift_t,0)
        t_end = min(nframes - ps_t, t_c + nWt_f - shift_t)+1

        # -- final range --
        trange = [t_c]
        trange_s = np.arange(t_c+1,t_end)
        trange_e = np.arange(t_start,t_c)[::-1]
        for t_i in range(trange_s.shape[0]):
            trange.append(trange_s[t_i])
        for t_i in range(trange_e.shape[0]):
            trange.append(trange_e[t_i])

        # -- aug vars --
        n_tranges.append(len(trange))
        min_tranges.append(np.min(trange))

        # -- add padding --
        for pad in range(nframes-len(trange)):
            trange.append(-1)

        # -- to tensor --
        trange = torch.IntTensor(trange).to(device)
        tranges.append(trange)

    tranges = torch.stack(tranges).to(device)
    n_tranges = torch.IntTensor(n_tranges).to(device)
    min_tranges = torch.IntTensor(min_tranges).to(device)

    return tranges,n_tranges,min_tranges

def compute_l2norm_launcher(dists,indices,fflow,bflow,access,bufs,noisy,
                            ps,ps_t,nWt_f,nWt_b,step1,offset,cs):

    # -- shapes --
    nframes,c,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = indices.shape
    tranges,n_tranges,min_tranges = create_frame_range(nframes,nWt_f,nWt_b,
                                                       ps_t,noisy.device)

    # -- numbify the torch tensors --
    dists_nba = cuda.as_cuda_array(dists)
    indices_nba = cuda.as_cuda_array(indices)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    access_nba = cuda.as_cuda_array(access)
    bufs_nba = cuda.as_cuda_array(bufs)
    noisy_nba = cuda.as_cuda_array(noisy)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)
    cs_nba = cuda.external_stream(cs)

    # -- batches per block --
    batches_per_block = 4
    bpb = batches_per_block

    # -- launch params --
    w_thread = min(w_s,32)
    nthread_loops = divUp(w_s,32)
    threads = (w_thread,w_thread)
    blocks = divUp(bsize,batches_per_block)

    # -- launch kernel --
    compute_l2norm_kernel[blocks,threads,cs_nba](dists_nba,indices_nba,
                                                 fflow_nba,bflow_nba,
                                                 access_nba,bufs_nba,
                                                 noisy_nba,tranges_nba,
                                                 n_tranges_nba,min_tranges_nba,
                                                 bpb,ps,ps_t,nWt_f,nWt_b,
                                                 nthread_loops,step1,offset)


@cuda.jit(max_registers=64)
def compute_l2norm_kernel(dists,inds,fflow,bflow,access,bufs,noisy,tranges,
                          n_tranges,min_tranges,bpb,ps,ps_t,nWt_f,nWt_b,
                          nthread_loops,step1,offset):

    # -- local function --
    def bounds(val,lim):
        if val < 0: val = -val
        if val > lim: val = 2*lim - val
        return val

    def valid_frame_bounds(ti,nframes):
        leq = ti < nframes
        geq = ti >= 0
        return (leq and geq)

    def valid_top_left(n_top,n_left,h,w,ps):
        valid_top = (n_top + ps) < h
        valid_top = valid_top and (n_top >= 0)

        valid_left = (n_left + ps) < w
        valid_left = valid_left and (n_left >= 0)

        valid = valid_top and valid_left

        return valid

    # -- shapes --
    nframes,color,h,w = noisy.shape
    bsize,w_t,w_s,w_s = dists.shape
    bsize,w_t,w_s,w_s = inds.shape
    chnls = 1 if step1 else color
    height,width = h,w
    Z = ps*ps*ps_t*chnls
    nWxy = w_s

    # -- cuda threads --
    cu_tidX = cuda.threadIdx.x
    cu_tidY = cuda.threadIdx.y
    blkDimX = cuda.blockDim.x
    blkDimY = cuda.blockDim.y
    # tidX = cuda.threadIdx.x
    # tidY = cuda.threadIdx.y

    # # -- pixel we are sim-searching for --
    # top,left = h_start+hi,w_start+wi

    # -- create a range --
    w_t = nWt_f + nWt_b + 1

    # ---------------------------
    #
    #      search frames
    #
    # ---------------------------

    # -- access with blocks and threads --
    block_start = cuda.blockIdx.x*bpb

    # -- we want enough work per thread, so we process multiple per block --
    for _bidx in range(bpb):

        # ---------------------------
        #    extract anchor pixel
        # ---------------------------

        bidx = block_start + _bidx
        if bidx >= access.shape[0]: continue
        ti = access[bidx,0]
        hi = access[bidx,1]
        wi = access[bidx,2]
        top,left = hi,wi

        # ---------------------------
        #     valid (anchor pixel)
        # ---------------------------

        valid_t = (ti+ps_t-1) < nframes
        valid_t = valid_t and (ti >= 0)

        valid_top = (top+ps-1) < height
        valid_top = valid_top and (top >= 0)

        valid_left = (left+ps-1) < width
        valid_left = valid_left and (left >= 0)

        valid_anchor = valid_t and valid_top and valid_left

        # ---------------------------------------
        #     searching loop for (ti,top,left)
        # ---------------------------------------

        trange = tranges[ti]
        n_trange = n_tranges[ti]
        min_trange = min_tranges[ti]

        # -- we loop over search space if needed --
        for x_tile in range(nthread_loops):
            tidX = cu_tidX + blkDimX*x_tile
            if tidX >= w_s: continue

            for y_tile in range(nthread_loops):
                tidY = cu_tidY + blkDimY*y_tile
                if tidY >= w_s: continue

                for tidZ in range(n_trange):

                    # -------------------
                    #    search frame
                    # -------------------
                    n_ti = trange[tidZ]
                    dt = trange[tidZ] - min_trange

                    # ------------------------
                    #      init direction
                    # ------------------------

                    direction = max(-1,min(1,n_ti - ti))
                    if direction != 0:
                        cw0 = bufs[bidx,0,dt-direction,tidX,tidY]
                        ch0 = bufs[bidx,1,dt-direction,tidX,tidY]
                        ct0 = bufs[bidx,2,dt-direction,tidX,tidY]

                        flow = fflow if direction > 0 else bflow

                        cw_f = cw0 + flow[ct0,0,ch0,cw0]
                        ch_f = ch0 + flow[ct0,1,ch0,cw0]

                        cw = max(0,min(w-1,round(cw_f)))
                        ch = max(0,min(h-1,round(ch_f)))
                        ct = n_ti
                    else:
                        cw = left
                        ch = top
                        ct = ti

                    # ----------------
                    #     update
                    # ----------------

                    bufs[bidx,0,dt,tidX,tidY] = cw#cw_vals[ti-direction]
                    bufs[bidx,1,dt,tidX,tidY] = ch#ch_vals[t_idx-direction]
                    bufs[bidx,2,dt,tidX,tidY] = ct#ct_vals[t_idx-direction]

                    # --------------------
                    #      init dists
                    # --------------------
                    dist = 0

                    # --------------------------------
                    #   search patch's top,left
                    # --------------------------------

                    # -- target pixel we are searching --
                    if (n_ti) < 0: dist = np.inf
                    if (n_ti) >= (nframes-ps_t+1): dist = np.inf

                    # -----------------
                    #    spatial dir
                    # -----------------

                    # ch,cw = top,left
                    shift_w = min(0,cw - (nWxy-1)//2) \
                        + max(0,cw + (nWxy-1)//2 - w  + ps)
                    shift_h = min(0,ch - (nWxy-1)//2) \
                        + max(0,ch + (nWxy-1)//2 - h  + ps)

                    # -- spatial endpoints --
                    sh_start = max(0,ch - (nWxy-1)//2 - shift_h)
                    sh_end = min(h-ps,ch + (nWxy-1)//2 - shift_h)+1

                    sw_start = max(0,cw - (nWxy-1)//2 - shift_w)
                    sw_end = min(w-ps,cw + (nWxy-1)//2 - shift_w)+1

                    n_top = sh_start + tidX
                    n_left = sw_start + tidY

                    # ---------------------------
                    #      valid (search "n")
                    # ---------------------------

                    valid_t = (n_ti+ps_t-1) < nframes
                    valid_t = valid_t and (n_ti >= 0)

                    valid_top = n_top < sh_end
                    valid_top = valid_top and (n_top >= 0)

                    valid_left = n_left < sw_end
                    valid_left = valid_left and (n_left >= 0)

                    valid = valid_t and valid_top and valid_left
                    valid = valid and valid_anchor
                    if not(valid): dist = np.inf

                    # ---------------------------------
                    #
                    #  compute delta over patch vol.
                    #
                    # ---------------------------------

                    # -- compute difference over patch volume --
                    for pt in range(ps_t):
                        for pi in range(ps):
                            for pj in range(ps):

                                # -- inside entire image --
                                vH = top+pi#bounds(top+pi,h-1)
                                vW = left+pj#bounds(left+pj,w-1)
                                vT = ti + pt

                                nH = n_top+pi#bounds(n_top+pi,h-1)
                                nW = n_left+pj#bounds(n_left+pj,w-1)
                                nT = n_ti + pt

                                # -- all channels --
                                for ci in range(chnls):

                                    # -- get data --
                                    v_pix = noisy[vT][ci][vH][vW]/255.
                                    n_pix = noisy[nT][ci][nH][nW]/255.

                                    # -- compute dist --
                                    if dist < np.infty:
                                        dist += (v_pix - n_pix)**2

                    # -- dists --
                    dist = dist-offset if dist < np.infty else dist
                    dist = dist if dist > 0 else 0.
                    dists[bidx,tidZ,tidX,tidY] = dist/Z

                    # -- inds --
                    ind = n_ti * height * width * color
                    ind += n_top * width
                    ind += n_left
                    inds[bidx,tidZ,tidX,tidY] = ind if dist < np.infty else -1

                    # -- access pattern --
                    # access[0,dt,tidX,tidY,ti,hi,wi] = n_ti
                    # access[1,dt,tidX,tidY,ti,hi,wi] = n_top
                    # access[2,dt,tidX,tidY,ti,hi,wi] = n_left

