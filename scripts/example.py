

# -- python imports --
import numpy as np
from pathlib import Path

# -- this package --
import pyvnlb

# -- helper imports --
from pyvnlb.pylib.tests.data_loader import load_dataset
from pyvnlb.pylib.tests.file_io import save_images

#
# -- load & denoise a video --
#

# -- check omp --
print("Running example script.")
pyvnlb.check_omp_num_threads()

# -- get data --
clean = load_dataset("davis_64x64",vnlb=False)[0]['clean']

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

# -- compare with original  --
noisy_psnrs = pyvnlb.compute_psnrs(clean,noisy)

#
# -- report outputs --
#

print_report = True
if print_report:

    # -- print psnrs --
    print("Denoised PSNRs:")
    print(psnrs)

    print("Starting PSNRs:")
    print(noisy_psnrs)

    # -- save images --
    output = Path("./output/")
    if not output.exists(): output.mkdir()
    save_images(clean,output/"clean.png",imax=255.)
    save_images(noisy,output/"noisy.png",imax=255.)
    save_images(denoised,output/"denoised.png",imax=255.)
    save_images(pyvnlb.flow2burst(fflow),"output/fflow.png")
    save_images(pyvnlb.flow2burst(bflow),"output/bflow.png")

#
# -- show vnlb's timing --
#

show_cpp_timing = False
if show_cpp_timing:
    print("The decrease in speed seems to be due to slower C++ functions.")
    pyvnlb.runVnlbTimed(noisy,std,{'fflow':fflow,'bflow':bflow})
    msg = "In the original C++ Code, the elapsed time "
    msg += "is ~2.5 and ~3.1 seconds, respectively."
    print(msg)
