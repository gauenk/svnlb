
def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None,step=0):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- extract info for explicit call --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    onlyFrame = params['onlyFrame'][step]
    aggreBoost =  params['aggreBoost'][step]

    # -- exec search --
    nmasked = exec_aggregation(deno,group,indices,weights,mask,
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

def exec_aggregation(deno,group,indices,weights,mask,
                     ps,ps_t,onlyFrame,aggreBoost):
    # -- init --
    nmasked = 0

    # -- update [deno,weights,mask] --



    return nmasked

