
# -- python imports --
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
from easydict import EasyDict as edict

def create_named_meshgrid(lists,names):
    named_mesh = []
    mesh = create_meshgrid(lists)
    for elem in mesh:
        named_elem = edict(OrderedDict(dict(zip(names,elem))))
        named_mesh.append(named_elem)
    return named_mesh

def create_meshgrid(lists):
    # -- num lists --
    L = len(lists)

    # -- tokenize each list --
    codes,uniques = [],[]
    for l in lists:
        l_codes,l_uniques = pd.factorize(l)
        codes.append(l_codes)
        uniques.append(l_uniques)

    # -- meshgrid and flatten --
    lmesh = np.meshgrid(*codes)
    int_mesh = [grid.ravel() for grid in lmesh]

    # -- convert back to tokens --
    mesh = [uniques[i][int_mesh[i]] for i in range(L)]

    # -- "transpose" the axis to iter goes across original lists --
    mesh_T = []
    L,M = len(mesh),len(mesh[0])
    for m in range(M):
        mesh_m = []
        for l in range(L):
            elem = mesh[l][m]
            if isinstance(elem,np.int64):
                elem = int(elem)
            mesh_m.append(elem)
        mesh_T.append(mesh_m)

    return mesh_T

def apply_mesh_filters(mesh,filters,ftype="keep"):
    filtered_mesh = mesh
    for mfilter in filters:
        filtered_mesh = apply_mesh_filter(filtered_mesh,mfilter,ftype=ftype)
    return filtered_mesh

def apply_mesh_filter(mesh,mfilter,ftype="keep"):
    filtered_mesh = []
    fields_str = list(mfilter.keys())[0]
    values = mfilter[fields_str]
    field1,field2 = fields_str.split("-")
    for elem in mesh:
        match_any = False
        match_none = True
        for val in values:
            eq1 = (elem[field1] == val[0])
            eq2 = (elem[field2] == val[1])
            if eq1 and eq2:
                match_any = True
                match_none = False
        if ftype == "keep":
            if match_any: filtered_mesh.append(elem)
        elif ftype == "remove":
            if match_none: filtered_mesh.append(elem)
        else: raise ValueError(f"[vnlb.exps.mesh] Uknown ftype [{ftype}]")
    return filtered_mesh


def create_list_pairs(fields):
    pairs = []
    for f1,field1 in enumerate(fields):
        for f2,field2 in enumerate(fields):
            if f1 >= f2: continue
            pairs.append([field1,field2])
    return pairs


