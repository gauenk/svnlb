
# -- python imports --
import cv2
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- pytorch imports --
import torch
import torchvision.utils as tvUtils


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  
#       Read VNLB Results
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def read_vnlb_results(vnlb_path,fstart,nframes):

    # -- load dict --
    results = edict()
    results.clean = read_result(vnlb_path,"%05d.jpg",fstart,nframes)
    results.noisy = read_result(vnlb_path,"%03d.tif",fstart,nframes)
    results.fflow = read_result(vnlb_path,"tvl1_%03d_f.flo",fstart,nframes)
    results.bflow = read_result(vnlb_path,"tvl1_%03d_b.flo",fstart,nframes)
    results.basic = read_result(vnlb_path,"bsic_%03d.tif",fstart,nframes)
    results.denoised = read_result(vnlb_path,"deno_%03d.tif",fstart,nframes)
    results.std = np.loadtxt(str(vnlb_path/"sigma.txt")).item()

    # -- reshape --
    for key,val in results.items():
        if key == "std": continue
        results[key] = rearrange(val,'t h w c -> t c h w')
    return results

def read_file(filename):
    if filename.suffix == ".flo":
        return read_flo_file(filename)
    else:
        img = cv2.imread(str(filename),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)        
        return img

def read_result(vnlb_path,fmt,fstart,nframes):
    agg = []
    for t in range(fstart,fstart+nframes):
        path = vnlb_path / (fmt % t)
        if not path.exists(): return None
        data = read_file(path)
        agg.append(data)
    agg = np.stack(agg)
    return agg

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1).item()
        h = np.fromfile(f, np.int32, count=1).item()
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  
#             Misc
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def save_images(tensor,fn,imax=255.):
    # -- swap string and tensor --
    if isinstance(tensor,str):
        tmp = tensor
        tensor = fn
        fn = tmp

    # -- save torch image --
    tensor = torch.FloatTensor(tensor.copy())/imax
    tvUtils.save_image(tensor,fn)

