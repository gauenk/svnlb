
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict
from collections.abc import Iterable

import svnlb

# from .ptr_utils import py2swig
from svnlb.utils.image_utils import est_sigma
from svnlb.utils.utils import optional,optional_pair,optional_swig_ptr,ndarray_ctg_dtype
from svnlb.utils.utils import check_flows,check_none,assign_swig_args,check_and_expand_flows

#
# --Vnlb Parameters --
#

def set_function_params_old(args,pyargs):
    """
    args: settings for SWIG
    pyargs: settings from python
    """
    # -- set optional numeric vals --
    args.use_default = optional(pyargs,'use_default',True)

    args.ps_x = optional(pyargs,'ps_x',[-1,-1],np.int32)
    args.ps_t = optional(pyargs,'ps_t',[-1,-1],np.int32)
    args.tau = optional(pyargs,'tau',[0.,400.],np.float32)
    args.num_patches = optional(pyargs,'num_patches',[-1,-1],np.int32)
    args.sizeSearchWindow = optional(pyargs,'sizeSearchWindow',[27,27],np.int32)
    args.sizeSearchTimeFwd = optional(pyargs,'sizeSearchTimeFwd',[6,6],np.int32)
    args.sizeSearchTimeBwd = optional(pyargs,'sizeSearchTimeBwd',[6,6],np.int32)

    args.rank = optional(pyargs,'rank',[-1,-1],np.int32)
    args.thresh = optional(pyargs,'thresh',[-1.,-1.],np.float32)
    args.beta = optional(pyargs,'beta',[-1.,-1.],np.float32)

    args.flat_areas = optional(pyargs,'flat_areas',[False,True],np.bool_)
    args.couple_ch = optional(pyargs,'couple_ch',[False,False],np.bool_)
    args.aggreBoost = optional(pyargs,'aggre_boost',[True,True],np.bool_)
    args.procStep = optional(pyargs,'procStep',[-1,-1],np.int32)

    # args.use_clean = not(optional(pyargs,'clean',None) is None)
    # use_flow = not(type(optional(pyargs,'fflow',None)) == type(None))
    # use_flow = use_flow and not(type(optional(pyargs,'bflow',None)) == type(None))
    # args.use_flow = use_flow

    args.testing = optional(pyargs,'testing',False)
    args.var_mode = optional(pyargs,'var_mode',False) # T == Soft, F == Hard
    args.verbose = optional(pyargs,'verbose',False)
    args.print_params = optional(pyargs,'print_params',0)

def get_param_translations():
    names = {}
    names['sigma'] = ['std']
    names['sigmaBasic'] = ['std_basic','sigma_basic']
    names['sizePatch'] = ['ps','ps_x','patchsize']
    names['sizePatchTime'] = ['ps_t']
    names['nSimilarPatches'] = ['num_patches','npatches']
    names['sizeSearchWindow'] = ['npsearch','npatch_search']
    names['sizeSearchTimeFwd'] = ['nsearch_fwd']
    names['sizeSearchTimeBwd'] = ['nsearch_bwd']
    names['flatAreas'] = ['flat_areas']
    names['gamma'] = []
    names['coupleChannels'] = ['couple_channels']
    names['variThres'] = ['thresh','vari_thresh','variThresh']
    names['rank'] = []
    names['beta'] = []
    names['tau'] = []
    names['isFirstStep'] = []
    names['procStep'] = ['proc_step']
    names['aggreBoost'] = ['aggre_boost']
    names['onlyFrame'] = []
    names['var_mode'] = []
    names['verbose'] = []
    names['testing'] = []
    names['nThreads'] = ['nthreads']
    names['nParts'] = ['nparts']

    names['use_imread'] = []
    names['set_sizePatch'] = []
    names['set_sizePatchTime'] = []
    names['set_nSim'] = []
    names['set_rank'] = []
    names['set_aggreBoost'] = []
    names['set_procStep'] = []
    names['set_nThreads'] = []
    names['set_nParts'] = []


    return names

def get_param_type():
    types = edict()
    types['sigma'] = np.float32
    types['sigmaBasic'] = np.float32
    types['sizePatch'] = np.uint32
    types['sizePatchTime'] = np.uint32
    types['nSimilarPatches'] = np.uint32
    types['sizeSearchWindow'] = np.uint32
    types['sizeSearchTimeFwd'] = np.uint32
    types['sizeSearchTimeBwd'] = np.uint32
    types['flatAreas'] = np.bool_
    types['gamma'] = np.float32
    types['coupleChannels'] = np.bool_
    types['variThres'] = np.float32
    types['rank'] = np.uint32
    types['beta'] = np.float32
    types['tau'] = np.float32
    types['isFirstStep'] = np.bool_
    types['procStep'] = np.uint32
    types['aggreBoost'] = np.bool_
    types['onlyFrame'] = np.int32
    types['verbose'] = np.bool_
    types['testing'] = np.bool_
    types['var_mode'] = np.int32
    types['nThreads'] = np.int32
    types['nParts'] = np.uint32

    types['use_imread'] = np.bool_
    types['set_sizePatch'] = np.bool_
    types['set_sizePatchTime'] = np.bool_
    types['set_nSim'] = np.bool_
    types['set_rank'] = np.bool_
    types['set_aggreBoost'] = np.bool_
    types['set_procStep'] = np.bool_
    types['set_nThreads'] = np.bool_
    types['set_nParts'] = np.bool_

    return types

def get_defaults():
    defaults = edict()
    defaults.sigma = [0.,0.]
    defaults.sigmaBasic = [0.,0.]
    defaults.sizePatch = [0,0]
    defaults.sizePatchTime = [0,0]
    defaults.nSimilarPatches = [0,0]
    defaults.sizeSearchWindow = [27,27]
    defaults.sizeSearchTimeFwd = [6,6]
    defaults.sizeSearchTimeBwd = [6,6]
    defaults.flatAreas = [False,True]
    defaults.gamma = [-1.,-1.]
    defaults.coupleChannels = [False,False]
    defaults.variThres = [-1.,-1.]
    defaults.rank = [0,0]
    defaults.beta = [-1.,-1.]
    defaults.tau = [0.,400.]
    defaults.isFirstStep = [True,False]
    defaults.procStep = [-1,-1]
    defaults.aggreBoost = [True,True]
    defaults.onlyFrame = [-1,-1]
    defaults.var_mode = [svnlb.CLIPPED,svnlb.CLIPPED]
    defaults.verbose = [False,False]
    defaults.testing = [False,False]
    defaults.nThreads = [0,0]
    defaults.nParts = [0,0]

    # -- must be set using "handle_set_bools" --
    defaults.use_imread = False
    defaults.set_sizePatch = None
    defaults.set_sizePatchTime = None
    defaults.set_nSim = None
    defaults.set_rank = None
    defaults.set_aggreBoost = None
    defaults.set_procStep = None
    defaults.set_nThreads = None
    defaults.set_nParts = None

    return defaults

def handle_set_bools(pydict):
    """
    On the c++ side, we don't know
    if an input bool should be used to
    overwrite the output or now.

    This extra set of bools tells us
    whether or not we should use the
    additional bool

    """
    # -- setup function for bool --
    pyfields = set(pydict.keys())
    translate = get_param_translations()
    def check_any_exists(field):
        fields = set(translate[field] + [field])
        any_bool = len(list(set(fields) & pyfields)) > 0
        return [any_bool,any_bool]

    # -- enumerate "set" bool --
    pydict['set_sizePatch'] = check_any_exists("sizePatch")
    pydict['set_sizePatchTime'] = check_any_exists("sizePatchTime")
    pydict['set_nSim'] = check_any_exists("nSimilarPatches")
    pydict['set_rank'] = check_any_exists("rank")
    pydict['set_aggreBoost'] = check_any_exists("aggreBoost")
    pydict['set_procStep'] = check_any_exists("procStep")
    pydict['set_nThreads'] = check_any_exists("nThreads")
    pydict['set_nParts'] = check_any_exists("nParts")


def get_param_fields():
    names = get_param_translations()
    return list(names.keys())

def dict2params(pydict):
    fields = get_param_fields()
    params = svnlb.nlbParams()
    for field in fields:
        setattr(params,field,pydict[field])
    return params

def params2dict(params,step):
    fields = get_param_fields()
    pydict = {}
    for field in fields:
        # if not(field in pydict):
        #     pydict[field] = [None,None]
        pydict[field] = getattr(params,field)
    return pydict

def reindex_and_fill_dict(pyargs,step):
    """
    pyargs: settings from python in (k,v) dictionary
    Example:
    pyargs: (k_1,(v_{1,1},v_{1,2})),(k_2,(v_{2,1},v_{2,2})),...(k_N,(v_{N,1},v_{N,2}))
    args:
    [(k'_1,v_{1,step}),(k'_2,v_{2,step}),...,(k'_N,v_{N,step}),...,(k_M,v_M)]

    1. a diction of tuples -> a pair of dictionary with single elements
    2. a partial list of parameters for vnlb -> a full list of vnlb params
    """
    # -- initialize --
    params = edict()
    types = get_param_type()
    defaults = get_defaults()
    translate = get_param_translations()
    for field,default_pair in defaults.items():
        fields = translate[field] + [field]
        value = optional(pyargs,fields,default_pair,types[field])
        if hasattr(value,"__getitem__"):
            try: value = value[step].item()
            except: value = value[step]
        params[field] = value
    return params

def reindex_params_to_py(params,pyargs,overwrite=False,use_pyfield=False):
    """
    some key names used as inputs didn't match the C++ keys.

    a.) modify keys
    We translate the keys in the python-dictionary BACK
    into the keys used as the inputs

    b.) [optional]: modify values
    We also, optionally, overwrite the python-dictionary
    VALUES if we want "full, explicit control" of the settings

    """
    py_keys = set(pyargs.keys())
    translate = get_param_translations()
    for cpp_field,_py_fields in translate.items():
        py_fields = set(_py_fields + [cpp_field])
        field = list(py_fields & py_keys)
        param_field = field if use_pyfield else cpp_field
        if len(field) > 1: # just pick cpp one
            field = cpp_field
        elif len(field) == 0: continue
        else: field = field[0]
        if overwrite:
            params[param_field] = pyargs[field]
        else:
            params[param_field] = params[cpp_field]
        if param_field != cpp_field:
            del params[cpp_field]


#
# -- Tensor Parser Variables --
#

def np_zero_tensors(t,c,h,w):
    tensors = edict()
    tensors.fflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.bflow = np.zeros((t,2,h,w),dtype=np.float32)
    tensors.oracle = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.clean = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.basic = np.zeros((t,c,h,w),dtype=np.float32)
    tensors.denoised = np.zeros((t,c,h,w),dtype=np.float32)
    return tensors

def set_tensors(args,pyargs,tensors):

    # -- set tensors --
    args.fflow = optional(pyargs,'fflow',tensors.fflow)
    args.bflow = optional(pyargs,'bflow',tensors.bflow)
    args.oracle = optional(pyargs,'oracle',tensors.oracle)
    args.clean = optional(pyargs,'clean',tensors.clean)
    args.basic = optional(pyargs,'basic',tensors.basic)
    args.denoised = optional(pyargs,'denoised',tensors.denoised)

    # -- set bools --
    args.use_flow = check_flows(pyargs)
    args.use_clean = check_none(optional(pyargs,'clean',None),'neq')
    args.use_oracle = check_none(optional(pyargs,'oracle',None),'neq')

def combine_dicts(params_1,params_2):
    py_params = {}
    for key in params_1.keys():
        v1 = params_1[key]
        v2 = params_2[key]
        v1_i = hasattr(v1,"__getitem__")
        v2_i = hasattr(v2,"__getitem__")
        both = v1_i and v2_i
        neither = (not v1_i) and (not v2_i)
        if both:
            assert np.sum(np.abs(v1_i-v2_i)) < 1e-12
            py_params[key] = v1
        elif neither:
            py_params[key] = [v1,v2]
        else:
            raise ValueError("Can't combine iterable and non-iterable values.")
    return py_params

#
# -- Main Parser --
#

def parse_args(noisy,sigma,tensors,pyargs):


    # -- init empty dicts --
    if tensors is None: tensors = {}
    if pyargs is None: pyargs = {}

    # -- verbose & testing --
    verbose = optional(pyargs,'verbose',False)
    if not(hasattr(verbose,"__len__")): pyargs['verbose'] = [verbose,verbose]
    testing = optional(pyargs,'testing',False)
    if not(hasattr(testing,"__len__")): pyargs['testing'] = [testing,testing]
    verbose = np.any(verbose)
    testing = np.any(testing)

    # -- sigma --
    sigma = optional(pyargs,'sigma',sigma)
    if sigma is None:
        sigma = est_sigma(noisy)

    # -- run parsers --
    params,swig_params = parse_params(noisy.shape,sigma,pyargs)
    tensors,swig_tensors = parse_tensors(noisy,tensors,verbose)

    return params,swig_params,tensors,swig_tensors

def parse_params(shape,sigma,pyargs=None):
    """
    Parse parameters for function
    """

    # -- set python parameters --
    pyargs = edict(pyargs)
    if not('sigma' in pyargs):
        pyargs.sigma = sigma if hasattr(sigma,'__getitem__') else [sigma,sigma]
    handle_set_bools(pyargs) # set bools before copying over
    params_1 = reindex_and_fill_dict(pyargs,0)
    params_2 = reindex_and_fill_dict(pyargs,1)

    # -- dict -> nlbParams --
    swig_params_1 = dict2params(params_1)
    swig_params_2 = dict2params(params_2)

    # -- create tensors --
    py_targs = edict()
    py_targs.t,py_targs.c,py_targs.h,py_targs.w = shape
    swig_tensors = svnlb.VnlbTensors()
    assign_swig_args(py_targs,swig_tensors)

    # -- set params --
    svnlb.setVnlbParamsCpp(swig_params_1,swig_tensors,1)
    svnlb.setVnlbParamsCpp(swig_params_2,swig_tensors,2)

    # -- nlbParams -> dict --
    params_1 = params2dict(swig_params_1,0)
    params_2 = params2dict(swig_params_2,1)

    # -- reindex params back to user-input names --
    overwrite_cpp = optional_pair(pyargs,'overwrite_cpp',[False,False],np.bool_)
    reindex_params_to_py(params_1,pyargs,overwrite=overwrite_cpp[0])
    reindex_params_to_py(params_2,pyargs,overwrite=overwrite_cpp[1])

    # -- combine two dicts of values into one dict of pairs --
    py_params = combine_dicts(params_1,params_2)
    py_params = edict(py_params)

    # -- [after overwrite] keep a swig-ready set of nlbParams --
    params_1 = reindex_and_fill_dict(py_params,0)
    params_2 = reindex_and_fill_dict(py_params,1)
    swig_params_1 = dict2params(params_1)
    swig_params_2 = dict2params(params_2)
    swig_params = [swig_params_1,swig_params_2]

    return py_params,swig_params

def parse_tensors(noisy,py_tensors,verbose=False):
    """
    Parse image tensors data
    """

    # -- extract info --
    dtype = noisy.dtype
    t,c,h,w  = noisy.shape

    # -- format noisy image --
    noisy = ndarray_ctg_dtype(noisy,np.float32,verbose)

    # -- format flows for c++ (t-1 -> t) --
    if check_flows(py_tensors): check_and_expand_flows(py_tensors,t)

    # -- set tensors vars --
    tensors = edict()
    tensors.w = w
    tensors.h = h
    tensors.c = c
    tensors.t = t
    tensors.noisy = noisy
    ztensors = np_zero_tensors(t,c,h,w)
    set_tensors(tensors,py_tensors,ztensors)

    # -- copy to swig --
    swig_tensors = svnlb.VnlbTensors()
    assign_swig_args(tensors,swig_tensors)

    return tensors, swig_tensors
