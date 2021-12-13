
# -- python imports --
import copy
from collections import OrderedDict
from easydict import EasyDict as edict

# -- project imports --
from .mesh import create_meshgrid,apply_mesh_filters

def create_experiment_grid():

    # -- create patchsize grid --
    ps_1 = [7,]
    ps_2 = [7,]

    # -- create noise level grid --
    sigma = [25.]

    # -- create frame number grid --
    nframes = [5,] # [31]

    # -- dataset name --
    dataset = ["davis_64x64"]

    # -- frame size --
    frame_size = ['64_64']

    # -- random seed --
    random_seed = [123,]

    # -- aggreboost --
    aggreBoost_1 = [True]
    aggreBoost_2 = [True]

    # -- nSimilarSearch --
    nSimSearch_1 = [100]
    nSimSearch_2 = [60]

    # -- nSimilarSearch --
    use_clean = [True,False]

    # -- create a list of arrays to mesh --
    lists = [ps_1,ps_2,sigma,nframes,dataset,
             frame_size,random_seed,aggreBoost_1,
             aggreBoost_2,nSimSearch_1,
             nSimSearch_2,use_clean]
    order = ['ps_1','ps_2','sigma','nframes',
             'dataset','frame_size','random_seed',
             'aggreBoost_1','aggreBoost_2',
             'nSimSearch_1','nSimSearch_2','use_clean']
    named_params = edict({o:l for o,l in zip(order,lists)})

    # -- create mesh --
    mesh = create_meshgrid(lists)

    # -- name each element --
    named_mesh = []
    for elem in mesh:
        named_elem = edict(OrderedDict(dict(zip(order,elem))))
        named_mesh.append(named_elem)

    return named_mesh,order

def format_exp_config(exp):

    # -- create a copy --
    params = edict()

    # -- imread --
    params.use_imread = [True,True]

    # -- set patchsize --
    params.patchsize = [exp.ps_1,exp.ps_2]

    # -- aggregation boost --
    params.aggreBoost = [exp.aggreBoost_1,exp.aggreBoost_2]

    # -- num of similar patches from search --
    params.nSimilarPatches = [exp.nSimSearch_1,exp.nSimSearch_2]

    # -- num of similar patches from search --
    params.use_clean = exp.use_clean

    # -- setup the rank of the inv cov --
    rank_1 = min(exp.nSimSearch_1,39)
    rank_2 = min(exp.nSimSearch_2,39)
    params.rank = [rank_1,rank_2]

    # -- random seed --
    params.random_seed = exp.random_seed

    # -- set frame size
    params.frame_size = exp.frame_size

    # -- fix dataset --
    params.dataset = exp.dataset

    # -- number of frames --
    params.nframes = int(exp.nframes)

    # -- set noise params --
    params.sigma = [exp.sigma,exp.sigma]

    # -- setup fields to delete before passing --
    del_fields = ['nframes','use_clean','random_seed','dataset','frame_size']

    return params,del_fields
