
# -- python deps --
import scipy
import numpy as np
from einops import rearrange

# -- numba --
from numba import njit

# -- package --
from vnlb.utils import groups2patches

def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None,step=0):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- extract info for explicit call --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    onlyFrame = params['onlyFrame'][step]
    aggreBoost =  params['aggreBoost'][step]

    # -- convert groups to patches  --
    t,c,h,w = deno.shape
    nSimP = len(indices)
    patches = groups2patches(group,c,ps,ps_t,nSimP)

    # -- exec search --
    deno_clone = deno.copy()
    nmasked = exec_aggregation(deno,patches,indices,weights,mask,
                               ps,ps_t,onlyFrame,aggreBoost)

    # -- pack results --
    results = {}
    results['deno'] = deno
    results['weights'] = weights
    results['mask'] = mask
    results['nmasked'] = nmasked
    results['psX'] = ps
    results['psT'] = ps_t

    return results


@njit
def exec_aggregation(deno,patches,indices,weights,mask,
                     ps,ps_t,onlyFrame,aggreBoost):

    # -- def functions --
    def idx2coords(idx,width,height,color):

        # -- get shapes --
        whc = width*height*color
        wh = width*height

        # -- compute coords --
        t = (idx      ) // whc
        c = (idx % whc) // wh
        y = (idx % wh ) // width
        x = idx % width

        return t,c,y,x

    def pixRmColor(ind,c,w,h):
        whc = w*h*c
        wh = w*h
        ind1 = (ind // whc) * wh + ind % wh;
        return ind1

    # -- init --
    nmasked = 0
    t,c,h,w = deno.shape
    nSimP = len(indices)

    # -- update [deno,weights,mask] --
    for n in range(indices.shape[0]):

        # -- get the sim locaion --
        ind = indices[n]
        ind1 = pixRmColor(ind,c,h,w)
        t0,c0,h0,w0 = idx2coords(ind,w,h,c)
        t1,c1,h1,w1 = idx2coords(ind1,w,h,1)

        # -- handle "only frame" case --
        if onlyFrame >= 0 and onlyFrame != t0:
            continue

        # -- set using patch info --
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    for ci in range(c):
                        ij = ind + ci*w*h
                        gval = patches[n,pt,ci,pi,pj]
                        deno[t0+pt,ci,h0+pi,w0+pj] += gval
                    weights[t1+pt,h1+pi,w1+pj] += 1

        # -- apply paste trick --
        if (mask[t1,h1,w1] == 1): nmasked += 1
        mask[t1,h1,w1] = False

        if (aggreBoost):
            if ( (h1 > 2*ps) and (mask[t1,h1-1,w1]==1) ): nmasked += 1
            if ( (h1 < (h - 2*ps)) and (mask[t1,h1+1,w1]==1) ): nmasked += 1
            if ( (w1 > 2*ps) and (mask[t1,h1,w1-1]==1) ): nmasked += 1
            if ( (w1 < (w - 2*ps)) and (mask[t1,h1,w1+1]==1) ): nmasked += 1

            if (h1 > 2*ps):  mask[t1,h1-1,w1] = False
            if (h1 < (h - 2*ps)): mask[t1,h1+1,w1] = False
            if (w1 > 2*ps):  mask[t1,h1,w1-1] = False
            if (w1 < (w - 2*ps)): mask[t1,h1,w1+1] = False

    return nmasked

