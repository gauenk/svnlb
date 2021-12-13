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


def compute_l2norm_cuda(noisy,fflow,bflow,h_start,w_start,h_batch,w_batch,ps,ps_t,
                        w_s,w_t,nWt_f,nWt_b,k=1):

    # -- create output --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- init dists --
    # (w_s = windowSpace), (w_t = windowTime)
    dists = torch.ones(w_t,w_s,w_s,t,h_batch,w_batch).type(torch.float32).to(device)
    indices = torch.zeros(w_t,w_s,w_s,t,h_batch,w_batch).type(torch.int32).to(device)
    access = torch.zeros(3,w_t,w_s,w_s,t,h_batch,w_batch).type(torch.int32).to(device)
    bufs = torch.zeros(3,w_t,w_s,w_s,h_batch,w_batch).type(torch.int32).to(device)

    # -- run launcher --
    dists *= torch.inf
    # print("cuda_l2: ",noisy[0,0,0,0])
    compute_l2norm_launcher(dists,indices,fflow,bflow,bufs,noisy,access,
                            h_start,w_start,ps,ps_t,nWt_f,nWt_b,True)

    return dists,indices,access


def create_frame_range(nframes,nWt_f,nWt_b,ps_t,device):
    tranges,n_tranges,min_tranges = [],[],[]
    for t_c in range(nframes-1):

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

def compute_l2norm_launcher(dists,indices,fflow,bflow,
                            bufs,noisy,access,h_start,w_start,
                            ps,ps_t,nWt_f,nWt_b,step1):

    # -- shapes --
    nframes,c,h,w = noisy.shape
    w_t,w_s,w_s,t,h_batch,w_batch = dists.shape
    w_t,w_s,w_s,t,h_batch,w_batch = indices.shape
    tranges,n_tranges,min_tranges = create_frame_range(nframes,nWt_f,nWt_b,
                                                       ps_t,noisy.device)

    # -- numbify the torch tensors --
    dists_nba = cuda.as_cuda_array(dists)
    indices_nba = cuda.as_cuda_array(indices)
    fflow_nba = cuda.as_cuda_array(fflow)
    bflow_nba = cuda.as_cuda_array(bflow)
    bufs_nba = cuda.as_cuda_array(bufs)
    access_nba = cuda.as_cuda_array(access)
    noisy_nba = cuda.as_cuda_array(noisy)
    tranges_nba = cuda.as_cuda_array(tranges)
    n_tranges_nba = cuda.as_cuda_array(n_tranges)
    min_tranges_nba = cuda.as_cuda_array(min_tranges)

    # -- launch params --
    threads = (w_s,w_s)
    blocks = (h_batch,w_batch)
    # print(threads,blocks)
    # print(tranges)
    # print(n_tranges)
    # print(min_tranges)

    # -- launch kernel --
    compute_l2norm_kernel[blocks,threads](dists_nba,indices_nba,
                                          fflow_nba,bflow_nba,bufs_nba,access_nba,
                                          noisy_nba,tranges_nba,n_tranges_nba,
                                          min_tranges_nba,
                                          h_start,w_start,
                                          ps,ps_t,nWt_f,nWt_b,step1)


@cuda.jit(max_registers=64)
def compute_l2norm_kernel(dists,inds,fflow,bflow,bufs,access,noisy,tranges,n_tranges,
                          min_tranges,h_start,w_start,ps,ps_t,nWt_f,nWt_b,step1):

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
    w_t,w_s,w_s,t,h_batch,w_batch = dists.shape
    w_t,w_s,w_s,t,h_batch,w_batch = inds.shape
    chnls = 1 if step1 else color
    Z = ps*ps*ps_t*chnls
    nWxy = w_s

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y

    # -- cuda threads --
    tidX = cuda.threadIdx.x
    tidY = cuda.threadIdx.y

    # -- thread idx --
    si = cuda.threadIdx.x - w_s//2
    sj = cuda.threadIdx.y - w_s//2

    # -- pixel we are sim-searching for --
    top,left = h_start+hi,w_start+wi

    # -- create a range --
    w_t = nWt_f + nWt_b + 1

    # -- we want enough work per thread, so we keep the "t" loop --
    for ti in range(nframes-1):

        trange = tranges[ti]

        # print(np.where(trange==-1))
        # n_trange = np.min(np.where(trange==-1)[0])
        # print("n_trange: ",n_trange)
        n_trange = n_tranges[ti]
        min_trange = min_tranges[ti]
        for tidZ in range(n_trange):

            # -------------------
            #    search frame
            # -------------------
            st = trange[tidZ]
            dt = trange[tidZ] - min_trange

            # ------------------------
            #      init direction
            # ------------------------

            direction = max(-1,min(1,st - ti))
            # print("(t_i,t_idx,dir): (%d,%d,%d)" % (t_i,t_idx,direction))
            if direction != 0:
                cw0 = bufs[0,dt-direction,tidX,tidY,hi,wi]#cw_vals[ti-direction]
                ch0 = bufs[1,dt-direction,tidX,tidY,hi,wi]#ch_vals[t_idx-direction]
                ct0 = bufs[2,dt-direction,tidX,tidY,hi,wi]#ct_vals[t_idx-direction]

                flow = fflow if direction > 0 else bflow

                # print(cw0,ch0,ct0)
                cw_f = cw0 + flow[ct0,0,ch0,cw0]
                ch_f = ch0 + flow[ct0,1,ch0,cw0]
                # print("(cw0,ch0,ct0): (%d,%d,%d)" % (cw0,ch0,ct0))
                # print("(cw_f,ch_f,dir): (%2.3f,%2.3f)" % (cw_f,ch_f))

                cw = max(0,min(w-1,round(cw_f)))
                ch = max(0,min(h-1,round(ch_f)))
                ct = st
            else:
                cw = left
                ch = top
                ct = ti

            # ----------------
            #     update
            # ----------------

            bufs[0,dt,tidX,tidY,hi,wi] = cw#cw_vals[ti-direction]
            bufs[1,dt,tidX,tidY,hi,wi] = ch#ch_vals[t_idx-direction]
            bufs[2,dt,tidX,tidY,hi,wi] = ct#ct_vals[t_idx-direction]


            # --------------------
            #      init dists
            # --------------------
            dist = 0

            # --------------------------------
            #   search patch's top,left
            # --------------------------------

            # -- frames offset --
            # st = tidZ - nWt_b
            # st = t_start + tidZ
            # st = trange[tidZ]
            # st = trange[tidZ] - min_trange
            # if (ti + st) < 0: dist = np.inf
            # if (ti + st) >= nframes: dist = np.inf

            # -- target pixel we are searching --
            # n_top,n_left = top+si,left+sj
            n_ti = st
            if (n_ti) < 0: dist = np.inf
            if (n_ti) >= nframes: dist = np.inf

            # -----------------
            #    spatial dir
            # -----------------

            # ch,cw = top,left
            shift_w = min(0,cw - (nWxy-1)//2) + max(0,cw + (nWxy-1)//2 - w  + ps)
            shift_h = min(0,ch - (nWxy-1)//2) + max(0,ch + (nWxy-1)//2 - h  + ps)

            # -- spatial endpoints --
            sh_start = max(0,ch - (nWxy-1)//2 - shift_h)
            sh_end = min(h-ps,ch + (nWxy-1)//2 - shift_h)+1

            sw_start = max(0,cw - (nWxy-1)//2 - shift_w)
            sw_end = min(w-ps,cw + (nWxy-1)//2 - shift_w)+1

            n_top = sh_start + tidX
            n_left = sw_start + tidY

            # i_ind = ti * h * w * color
            # i_ind += top * w
            # i_ind += left

            # n_ind = n_ti * h * w * color
            # n_ind += n_top * w
            # n_ind += n_left

            # -------------------
            #      valid
            # -------------------

            valid_t = (n_ti+ps_t-1) < nframes
            valid_t = valid_t and (n_ti >= 0)

            valid_top = (n_top + ps - 1) < h
            valid_top = valid_top and (n_top >= 0)

            valid_left = (n_left + ps - 1) < w
            valid_left = valid_left and (n_left >= 0)

            valid = valid_t and valid_top and valid_left
            if not(valid): dist = np.inf
            # if not(valid): continue

            # 712 = 11 + 8*64
            # 14596 = 4 + 36*64 + 64*64*3
            p_top = n_top == 11
            p_left = n_left == 8
            p_t = n_ti == 0
            # if p_top and p_left and p_t:# and i_ind == 14596:
            #     print("ptop: ", n_ind)

            p_top = top == 36
            p_left = left == 4
            p_t = ti == 1
            # if p_top and p_left and p_t:# and i_ind == 14596:
            #     print("ptop: ", i_ind,n_ind,shift_h,shift_w)


            # if n_ind == 712 and i_ind == 14596:
            #     print("[cuda] t: ",ti,n_ti,ps_t,nframes)
            #     print("[cuda] h: ",n_top,sh_start,sh_end,shift_h,n_ind,i_ind)
            #     print("[cuda] w: ",n_left,sw_start,sw_end,shift_w,n_ind,i_ind)
            #     # for iii in range(len(trange)):
            #     #     print("trange: %d" % (trange[iii]))

            #     num = 1 if valid else 0
            #     num_t = 1 if valid_t else 0
            #     num_top = 1 if valid_top else 0
            #     num_left = 1 if valid_left else 0
            #     print("[misc] valid,valid_t,valid_top,valid_left :",
            #           num,num_t,num_top,num_left)

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
            dists[dt,tidX,tidY,ti,hi,wi] = dist/Z

            # -- inds --
            ind = n_ti * h * w * color
            ind += n_top * w
            ind += n_left
            inds[dt,tidX,tidY,ti,hi,wi] = ind

            # -- access pattern --
            access[0,dt,tidX,tidY,ti,hi,wi] = n_ti
            access[1,dt,tidX,tidY,ti,hi,wi] = n_top
            access[2,dt,tidX,tidY,ti,hi,wi] = n_left

