

# -- imports --
import numpy as np
from pathlib import Path
import vnlb.pylib as pyvnlb
from data_loader import load_dataset
from file_io import save_images

# -- get data --
clean = load_dataset("davis_64x64")[0]['clean']

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
save_outputs = True
if save_outputs:
    output = Path("./output/")
    if not output.exists(): output.mkdir()
    save_images(clean,output/"clean.png",imax=255.)
    save_images(noisy,output/"noisy.png",imax=255.)
    save_images(denoised,output/"denoised.png",imax=255.)
    save_images(pyvnlb.flow2burst(fflow),"output/fflow.png")
    save_images(pyvnlb.flow2burst(bflow),"output/bflow.png")
