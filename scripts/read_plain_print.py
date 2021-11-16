
import numpy as np
import torch
import torchvision.utils as tvUtils
from einops import rearrange

def th_save_image(burst,fn):

    burst = torch.FloatTensor(burst)/255.
    if burst.dim == 4:
        burst = rearrange(burst,'t c h w -> t c h w')
    tvUtils.save_image(burst,fn)


if False:
    print("Read input noisy image.")
    cppFile = "/home/gauenk/Documents/packages/vnlb/build/bin/plain_print.txt"
    pyFile = "/home/gauenk/Documents/packages/pyvnlb/plain_print.txt"
    cppData = np.loadtxt(cppFile).reshape(3,3,64,64)
    pyData = np.loadtxt(pyFile).reshape(3,3,64,64)
    print(np.sum(np.abs(cppData - pyData)))
    th_save_image(cppData,"cpp_pp.png")
    th_save_image(pyData,"py_pp.png")
    
if False:
    print("Read VNLB mask.")
    cppFile = "/home/gauenk/Documents/packages/vnlb/build/bin/plain_print_char.txt"
    pyFile = "/home/gauenk/Documents/packages/pyvnlb/plain_print_char.txt"
    cppData = np.loadtxt(cppFile,dtype=np.int32).reshape(3,64,64)
    pyData = np.loadtxt(pyFile,dtype=np.int32).reshape(3,64,64)
    print(np.sum(np.abs(cppData - pyData)))
    th_save_image(cppData,"cpp_pp_char.png")
    th_save_image(pyData,"py_pp_char.png")

print("Read VNLB mask.")
cppFile = "/home/gauenk/Documents/packages/vnlb/build/bin/plain_print_burst.txt"
pyFile = "/home/gauenk/Documents/packages/pyvnlb/plain_print_burst.txt"
cppData = np.loadtxt(cppFile).reshape(8,1,64,64)
pyData = np.loadtxt(pyFile).reshape(8,1,64,64)
pyData = np.concatenate([pyData[4:],pyData[:4]])
print("Shape [cpp,py]")
print(cppData.shape,pyData.shape)

print("Locs of Nan [cpp,py]")
print(np.where(np.isnan(cppData)))
print(np.where(np.isnan(pyData)))

dstack = np.stack([cppData,pyData],axis=-1)
print(dstack)

delta = np.abs(cppData - pyData)
print(np.mean(np.abs(cppData - pyData)/cppData))
th_save_image(cppData,"cpp_pp_burst.png")
th_save_image(pyData,"py_pp_burst.png")
th_save_image(delta,"delta_pp_burst.png")


