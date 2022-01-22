
"""
Exec a batch of hypothesis testing compared to a reference group

ref = N x P
groups = B x M x P

"""


# -- python imports --
import torch
import torch as th
import numpy as np
from einops import rearrange


def multidim_gaussian_influence(sample):
    val = 0.

    return val

def ksd_test(ref,groups):

    # -- shapes --
    N,P = ref.shape
    B,M,P = groups.shape
    return hotelling_t_test(ref,groups)

def hotelling_t_test(ref,groups):

    # -- means --
    print("ref.shape: ",ref.shape)
    G,N,P = ref.shape
    device = ref.device
    ref = ref[:,None]
    r_means = torch.mean(ref,dim=2,keepdim=True)
    g_means = torch.mean(groups,dim=2,keepdim=True)
    diff_means = r_means - g_means
    print("r_means.shape: ",r_means.shape)
    print("g_means.shape: ",g_means.shape)

    # -- cov mats --
    ref_cov = th_cov(ref,r_means)
    group_cov = th_cov(groups,g_means)
    print("ref_cov.shape: ",ref_cov.shape)
    print("group_cov.shape: ",group_cov.shape)

    # -- compute deltas --
    cov = ref_cov + group_cov
    diff_means = diff_means[:,:,0]
    print(ref_cov.shape,group_cov.shape)
    # icov = th.linalg.solve(cov, diff_means)
    print("cov.shape: ",cov.shape)
    print("diff_means.shape: ",diff_means.shape)
    # icov = th.linalg.solve(cov.cpu(), diff_means.cpu()).to(device)
    # stats = th.sum(icov * diff_means,2)
    stats = th.sum(diff_means*diff_means,2)/(N*(N-1))
    print("stats.shape: ",stats.shape)

    return stats


def th_cov(patches,mean):

    # -- cov mats --
    B,G,N,P = patches.shape

    # -- zero mean --
    patches -= mean

    # -- cov mat --
    patches = rearrange(patches,'b g n p -> (b g) n p')
    patchesT = rearrange(patches,'bg n p -> bg p n')
    print(patches.shape)
    cov = patchesT @ patches / (N - 1)
    print("[1] cov.shape: ",cov.shape)
    cov = rearrange(cov,'(b g) p1 p2 -> b g p1 p2',b=B)
    print("[2] cov.shape: ",cov.shape)

    return cov
