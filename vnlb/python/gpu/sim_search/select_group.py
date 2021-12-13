

#
# -- select patches of noisy regions --
#

def exec_select_cpp_groups(noisy,indices,ps,ps_t):
    t,c,h,w = noisy.shape
    npatches = indices.shape[0]
    groups = th.zeros((1,c,ps_t,ps,ps,npatches),dtype=th.float32)
    groups_f = groups.ravel()
    numba_select_cpp_groups(groups_f,noisy,indices,ps,ps_t)
    return groups

# @njit
def numba_select_cpp_groups(groups,noisy,indices,ps,ps_t):

    # -- init shapes --
    t,c,h,w = noisy.shape
    nframes,color,height,width = t,c,h,w

    # def idx2coords(idx):

    #     # -- get shapes --
    #     whc = width*height*color
    #     wh = width*height

    #     # -- compute coords --
    #     t = (idx      ) // whc
    #     c = (idx % whc) // wh
    #     y = (idx % wh ) // width
    #     x = idx % width

    #     return t,c,y,x

    # -- exec copy --
    k = 0
    for ci in range(c):
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    for n in range(indices.shape[0]):
                        ind = indices[n]
                        # ti,_,hi,wi = idx2coords(ind)
                        # groups[k] = noisy[ti+pt,ci,hi+pi,wi+pj]
                        groups[k] = noisy.ravel()[ci * w*h + ind + pt*w*h*c + pi*w + pj]
                        k+=1

