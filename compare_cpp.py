"""
Compare the Python API with the C++ Results

"""

# -- python imports --
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- this package --
import vnlb.pylib as pyvnlb

# -- local imports --
from data_loader import load_dataset
from file_io import save_images

# ----------------------------------
# 
#     Compare Results with C++
# 
# ----------------------------------

# -- load c++ results --
res_vnlb = load_dataset("davis_pariasm_vnlb")
clean,noisy,std = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std

# -- exec python --
pyargs = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,"nscales":100,
          "fscale":1,"zfactor":0.5,"nwarps":5,"epsilon":0.01,
          "verbose":False,"testing":False,'bw':True}
fflow,bflow = pyvnlb.runPyFlow(noisy,std,pyargs)
pyargs = {'fflow':fflow,'bflow':bflow,'testing':True}
# pyargs['fflow'] = np.ascontiguousarray(res_vnlb.fflow)
# pyargs['bflow'] = np.ascontiguousarray(res_vnlb.bflow)
res_pyvnlb = pyvnlb.runPyVnlb(noisy,std,pyargs)
# res_pyvnlb = {}

# -- prepare dict --
res_pyvnlb['fflow'] = fflow
res_pyvnlb['bflow'] = bflow
pyvnlb.expand_flows(res_pyvnlb,axis=0) # nflows must match nframes

#
# -- compare outputs --
#

# fields = ['fflow','bflow']
fields = ['denoised','fflow','bflow',"basic"]

maxes = {}
for field in fields:
    pymax = np.abs(res_pyvnlb[field]).max()
    cppmax = np.abs(res_vnlb[field]).max()
    fmax = max(pymax,cppmax)
    maxes[field] = fmax if fmax < 255. else 255.
print(maxes)

for field in fields:
    cppField = res_vnlb[field]
    pyField = res_pyvnlb[field]
    print(cppField.shape,pyField.shape)
    psnrs = np.mean(pyvnlb.compute_psnrs(cppField,pyField,maxes[field]))
    print(f"PSNR for [{field}]: %2.2f" % psnrs)
    if field in ['fflow','bflow']:
        cppField = rearrange(cppField,'c t h w -> t c h w')
        pyField = rearrange(pyField,'c t h w -> t c h w')
        save_images(f"cpp_{field}_0.png",cppField[:,[0]],imax=1.)
        save_images(f"py_{field}_0.png",pyField[:,[0]],imax=1.)
        save_images(f"cpp_{field}_1.png",cppField[:,[1]],imax=1.)
        save_images(f"py_{field}_1.png",pyField[:,[1]],imax=1.)
    else:
        psnrs = np.mean(pyvnlb.compute_psnrs(cppField,clean))
        print(f"Denoising PSNR [CPP,{field}]: %2.3f" % psnrs)
        psnrs = np.mean(pyvnlb.compute_psnrs(pyField,clean))
        print(f"Denoising PSNR [Py,{field}]: %2.3f" % psnrs)
        save_images(f"cpp_{field}.png",cppField)
        save_images(f"py_{field}.png",pyField)
