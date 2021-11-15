"""
Test execution of the VNLB library

"""

# -- python imports --
from PIL import Image
import numpy as np
import numpy.random as npr
from einops import rearrange,repeat

# -- pytorch imports --
import torch
import torchvision.utils as tvUtils


# -- import vnlb --
import vnlb.pylib as pyvnlb


# -- small dyanmics lib for testing --

def rand_dir(mrange):
    grid = np.arange(mrange**2)
    grid = np.unravel_index(grid,(mrange,mrange))
    grid = np.stack(grid).T
    idx = npr.choice(mrange)
    return grid[idx]
    
def gen_jitter(mrange,nframes):
    flow = np.zeros((nframes,2),np.int32)
    for t in range(nframes):
        flow[t,:] = rand_dir(mrange)
    flow[nframes//2,:] = 0
    return flow
        
def add_dynamics(img,mrange,nframes,h,w):

    # -- create flow --
    flow = gen_jitter(mrange,nframes)
    
    # -- pick origin --
    origin = int(2*mrange)

    # -- create burst --
    burst = []
    for t in range(nframes):
        top,left = flow[t]+origin
        frame = img[top:top+h,left:left+w]
        burst.append(frame)
    burst = np.stack(burst,axis=0)
    burst = rearrange(burst,'t h w c -> c t h w')
        
    return burst
        
def th_save_image(burst,fn):
    burst = torch.FloatTensor(burst/255.)
    burst = rearrange(burst,'c t h w -> t c h w')
    tvUtils.save_image(burst,fn)

def test_exec_vnlb_color():
    exec_vnlb(3)

def test_exec_vnlb_bw():
    exec_vnlb(1)

def exec_vnlb(c):
    # -- load image --
    mrange = 3
    t,h,w = 5,64,64 # fixed order by user
    itype = "RGB" if c == 3 else "L"
    img = np.array(Image.open("./tests/image.jpg").convert(itype))
    if c == 1: img = img[:,:,None]
    burst = add_dynamics(img,mrange,t,h,w)
    th_save_image(burst,"burst.png")

    # -- add noise --
    std = 10.
    noise = np.random.normal(0,scale=std,size=(c,t,h,w))
    noisy = np.clip(burst + noise,0,255.)
    th_save_image(noisy,"noisy.png")

    # -- denoise --
    result = pyvnlb.runPyVnlb(noisy,std)
    denoised = result['denoised']
    th_save_image(denoised,"denoised.png")
