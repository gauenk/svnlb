import numpy as np


def index2indices(index,shape):
    t,c,h,w = shape

    tidx = index // (c*h*w)
    t_mod = index % (c*h*w)

    cidx = t_mod // (h*w)
    c_mod = t_mod % (h*w)

    hidx = c_mod // (h)
    h_mod = c_mod % (h)

    widx = h_mod# // w
    # c * wh + index + ht * whc + hy * w + hx
    indices = [tidx,cidx,hidx,widx]
    return indices

def patch_at_index(noisy,index,psX,psT):
    indices = index2indices(index,noisy.shape)
    tslice = slice(indices[0],indices[0]+psT)
    cslice = slice(indices[1],indices[1]+psX)
    hslice = slice(indices[2],indices[2]+psX)
    wslice = slice(indices[3],indices[3]+psX)
    return noisy[tslice,cslice,hslice,wslice]

def patches_at_indices(noisy,indices,psX,psT):
    patches = []
    for index in indices:
        patches.append(patch_at_index(noisy,index,psX,psT))
    patches = np.stack(patches)
    return patches

def groups2patches(group,c,psX,psT,npatches):

    # -- setup --
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = npatches * psX * psX * psT * c
    group_f = group.ravel()[:numNz]

    # -- [og -> img] --
    group = group_f.reshape(c,psT,-1)
    group = ncat(group,axis=1)
    group = group.reshape(c*psT,psX**2,npatches).transpose(2,0,1)
    group = ncat(group,axis=0)

    # -- final reshape --
    group = group.reshape(npatches,psT,c,psX,psX)

    return group


def patches2groups(patches,c,psX,psT,nsearch,nParts):

    # -- setup --
    npatches = patches.shape[0]
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = npatches * psX * psX * psT * c
    group = patches.ravel()[:numNz]

    # -- [img -> og] --
    group = group.reshape(npatches,psX*psX,c*psT).transpose(1,2,0)
    group = ncat(group,axis=0)
    group = group.reshape(psT,c,npatches*psX*psX)
    group = ncat(group,axis=1)

    # -- fill with zeros --
    group_f = group.ravel()[:numNz]
    group = np.zeros(size*nsearch)
    group[:size*npatches] = group_f[...]
    group = group.reshape(nParts,psT,c,psX,psX,nsearch)

    return group
