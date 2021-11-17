
import numpy as np
from pathlib import Path
from einops import rearrange

import vnlb.pylib as pyvnlb
from vnlb.benchmarks.utils import th_save_image
from vnlb.benchmarks.create_noisy_burst import get_vnlb_burst

# print(clean.shape) # (nframes,channels,height,width)


# -- parms --
std,fstart,nframes = 20,0,5
ipath = Path("../vnlb/data/davis_baseball_64x64/")
opath = Path(f"../vnlb/output/davis_baseball_64x64_{std}/")
vnlb_path = opath / "./vnlb/"
pyvnlb_path = opath / f"./pyvnlb_vnlb.pkl"
pyflow_path = opath / f"./pyvnlb_flow.pkl"

# -- get data --
clean,noisy,npaths = get_vnlb_burst(ipath,vnlb_path,fstart,nframes)
clean = rearrange(clean,'t h w c -> t c h w')
noisy = rearrange(noisy,'t h w c -> t c h w')

# -- add noise --
std = 10.
noisy = np.random.normal(clean,scale=std)

# -- TV-L1 Optical Flow --
fflow,bflow = pyvnlb.runPyFlow(noisy,std)

# -- Video Non-Local Bayes --
result = pyvnlb.runPyVnlb(noisy,std,{'fflow':fflow,'bflow':bflow})
denoised = result['denoised']

# -- compute denoising quality --
noisy_psnrs = pyvnlb.compute_psnrs(clean,noisy)
print("Starting PSNRs:")
print(noisy_psnrs)

psnrs = pyvnlb.compute_psnrs(clean,denoised)
print("Denoised PSNRs:")
print(psnrs)

# -- save images --
th_save_image(clean,"clean.png",imax=255.)
th_save_image(noisy,"noisy.png",imax=255.)
th_save_image(denoised,"denoised.png",imax=255.)
fflow_img = pyvnlb.flow2img(fflow)
bflow_img = pyvnlb.flow2img(bflow)
th_save_image(fflow,"fflow.png",imax=1.)
th_save_image(bflow,"bflow.png",imax=1.)
