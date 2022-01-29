
import svnlb

def flat_areas_parser(groupNoisy,groupBasic,flatAreas,nSimP,c):
    # -- create swig --
    params = vnlb.swig.FlatParams()
    params.groupNoisy = vnlb.swig.swig_ptr(groupNoisy)
    params.groupBasic = vnlb.swig.swig_ptr(groupBasic)
    params.nSimP = nSimP
    params.c = c
    params.flatAreas = flatAreas
    return params
