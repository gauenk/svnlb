
import pyvnlb
from ..utils import optional

def mask_parser(mask,vnlb_params,info=None):

    # -- unpack --
    t,h,w = mask.shape

    # -- create swig --
    params = pyvnlb.MaskParams()

    # -- create mask --
    params.mask = pyvnlb.swig_ptr(mask)

    # -- create shapes --
    params.nframes = t
    params.width = w
    params.height = h

    params.origin_t = optional(info,'origin_t',0)
    params.origin_h = optional(info,'origin_h',0)
    params.origin_w = optional(info,'origin_w',0)

    params.ending_t = optional(info,'origin_t',t)
    params.ending_h = optional(info,'origin_h',h)
    params.ending_w = optional(info,'origin_w',w)

    params.step_t = optional(info,'step_t',1)
    params.step_h = optional(info,'procStep',1)
    params.step_w = optional(info,'procStep',1)

    params.ps = vnlb_params['sizePatch']
    params.ps_t = vnlb_params['sizePatchTime']
    sWt_f = vnlb_params['sizeSearchTimeFwd']
    sWt_b = vnlb_params['sizeSearchTimeBwd']
    params.sWt = sWt_f + sWt_b + 1
    params.sWx = vnlb_params['sizeSearchWindow']

    params.ngroups = 0


    return params
