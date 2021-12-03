

# -- python deps --
import numpy as np
from easydict import EasyDict as edict

# -- numba --
from numba import njit,prange

# -- parser for cpp --
from pyvnlb.pylib.vnlb.mask_parser import mask_parser

def initMask(shape,vnlb_params,step=0,info=None):

    # -- parse inputs --
    t,c,h,w = shape
    mask = np.zeros((t,h,w),dtype=np.int8)
    vnlb_params = {k:v[step] for k,v in vnlb_params.items()}
    mask_params = mask_parser(mask,vnlb_params,info)
    params = comp_params(mask_params,t,h,w)

    # -- exec --
    ngroups = fill_mask_launcher(mask,params)

    # -- format results --
    results = edict()
    results.mask = mask
    results.ngroups = ngroups

    return results

def comp_params(mask_params,t,h,w):

    # -- init --
    params = edict()
    sPx = mask_params.ps
    sPt = mask_params.ps_t
    sWx = mask_params.sWx
    sWt = mask_params.sWt

    # -- borders --
    params.border_w0 = mask_params.origin_w > 0
    params.border_h0 = mask_params.origin_h > 0
    params.border_t0 = mask_params.origin_t > 0
    params.border_w1 = mask_params.ending_w < w
    params.border_h1 = mask_params.ending_h < h
    params.border_t1 = mask_params.ending_t < t

    # -- origins --
    border_s = sPx-1 + sWx//2
    border_t = sPt-1 + sWt//2
    params.ori_w = border_s if params.border_w0 else 0
    params.ori_h = border_s if params.border_h0 else 0
    params.ori_t = border_t if params.border_t0 else 0
    params.end_w = (w - border_s) if params.border_w1 else (w-sPx+1)
    params.end_h = (h - border_s) if params.border_h1 else (h-sPx+1)
    params.end_t = (t - border_t) if params.border_t1 else (t-sPt+1)

    # -- copy over misc. --
    params.sPx = mask_params.ps
    params.sPt = mask_params.ps_t
    params.sWx = mask_params.sWx
    params.sWt = mask_params.sWt
    params.step_t = mask_params.step_t
    params.step_h = mask_params.step_h
    params.step_w = mask_params.step_w

    return params


def fill_mask_launcher(mask,params):
    # -- unpack --
    step_t = params.step_t
    step_h = params.step_h
    step_w = params.step_w
    border_t0 = params.border_t0
    border_t1 = params.border_t1
    border_h0 = params.border_h0
    border_h1 = params.border_h1
    border_w0 = params.border_w0
    border_w1 = params.border_w1
    ori_t = params.ori_t
    ori_h = params.ori_h
    ori_w = params.ori_w
    end_t = params.end_t
    end_h = params.end_h
    end_w = params.end_w
    ngroups = 0
    ngroups = fill_mask(mask,ngroups,step_t,step_h,step_w,
                        border_t0,border_t1,border_h0,
                        border_h1,border_w0,border_w1,
                        ori_t,ori_h,ori_w,end_t,end_h,end_w)
    return ngroups

@njit
def fill_mask(mask,ngroups,step_t,step_h,step_w,
              border_t0,border_t1,border_h0,
              border_h1,border_w0,border_w1,
              ori_t,ori_h,ori_w,end_t,end_h,end_w):

    # -- init --
    t_size = end_t - ori_t
    h_size = end_h - ori_h
    w_size = end_w - ori_w

    # -- fill it up! --
    ngroups = 0
    for dt in prange(t_size):
        for dh in prange(h_size):
            for dw in prange(w_size):

                # -- unpack --
                ti = ori_t + dt
                hi = ori_h + dh
                wi = ori_w + dw

                # -- bools --
                take_t_step = dt % step_t == 0
                last_t = ti == (end_t-1) and not(border_t1)
                if take_t_step or last_t:

                    phase_h = 0 if last_t else ti//step_t
                    take_h_step = dh % step_h == phase_h % step_h
                    first_h = not(border_h0) and hi == ori_h
                    last_h = not(border_h1) and hi == (end_h-1)

                    if (take_h_step or first_h or last_h):

                        phase_w = 0 if last_h else phase_h + hi//step_h
                        take_w_step = dw % step_w == phase_w % step_w
                        first_w = not(border_w0) and wi == 0
                        last_w = not(border_w1) and wi == (end_w-1)

                        if (take_w_step or first_w or last_w):
                            mask[ti,hi,wi] = True
                            ngroups+=1

    return ngroups
