

# -- imports --
import numpy as np
import vnlb.pylib as pyvnlb
from data_loader import load_dataset
from file_io import save_images

# -- get data --
clean = load_dataset("davis_64x64")

# -- add noise --
std = 20.
noisy = np.random.normal(clean,scale=std)

# -- TV-L1 Optical Flow --
fflow,bflow = pyvnlb.runPyFlow(noisy,std)

# -- Video Non-Local Bayes --
result = pyvnlb.runPyVnlb(noisy,std,{'fflow':fflow,'bflow':bflow})
denoised = result['denoised']

# -- compute denoising quality --
psnrs = pyvnlb.compute_psnrs(clean,denoised)
print("Denoised PSNRs:")
print(psnrs)

# -- compare with original  --
noisy_psnrs = pyvnlb.compute_psnrs(clean,noisy)
print("Starting PSNRs:")
print(noisy_psnrs)

# -- save images --
save_images(clean,"clean.png",imax=255.)
save_images(noisy,"noisy.png",imax=255.)
save_images(denoised,"denoised.png",imax=255.)
# save_images(pyvnlb.flow2img(fflow),"fflow.png",imax=1.)
# save_images(pyvnlb.flow2img(bflow),"bflow.png",imax=1.)
