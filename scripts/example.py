

# -- python imports --
import numpy as np
from pathlib import Path

# -- this package --
import svnlb

# -- helper imports --
from svnlb.testing.data_loader import load_dataset
from svnlb.testing.file_io import save_images

#
# -- load & denoise a video --
#

print("Running example script.")
svnlb.check_omp_num_threads()

# -- get data --
clean = load_dataset("davis_64x64",vnlb=False)[0]['clean'].copy()[:3]

# -- add noise --
np.random.seed(123)
std = 20.
noisy = np.random.normal(clean.copy(),scale=std)

# -- TV-L1 Optical Flow --
fflow,bflow = svnlb.swig.runPyFlow(noisy,std)

# -- Video Non-Local Bayes --
result = svnlb.swig.runPyVnlb(noisy,std,{'fflow':fflow,'bflow':bflow,'nThreads':4})
basic = result['basic']
denoised = result['denoised']

# -- Video Non-Local Bayes [very slow] --
# result = svnlb.cpu.runPythonVnlb(noisy,std,{})
# basic = result['basic']
# denoised = result['denoised']

# -- compute denoising quality --
psnrs = svnlb.utils.compute_psnrs(clean,denoised)
psnrs_basic = svnlb.utils.compute_psnrs(clean,basic)

# -- compare with original  --
noisy_psnrs = svnlb.utils.compute_psnrs(clean,noisy)

#
# -- report outputs --
#

print_report = True
if print_report:

    # -- print psnrs --
    print("Denoised PSNRs:")
    print(psnrs)

    print("Basic PSNRs:")
    print(psnrs_basic)

    print("Starting PSNRs:")
    print(noisy_psnrs)

    # -- save images --
    output = Path("./output/")
    if not output.exists(): output.mkdir()
    print(f"\nSaving Example Images at directory [{str(output)}]")
    save_images(clean,output/"clean.png",imax=255.)
    save_images(noisy,output/"noisy.png",imax=255.)
    save_images(denoised,output/"denoised.png",imax=255.)
    save_images(svnlb.utils.flow2burst(fflow),"output/fflow.png")
    save_images(svnlb.utils.flow2burst(bflow),"output/bflow.png")

#
# -- show svnlb's timing --
#

show_cpp_timing = True
if show_cpp_timing:
    print("\nExecuting a timed VNLB.")
    svnlb.swig.runPyVnlbTimed(noisy,std,{'fflow':fflow,'bflow':bflow})
    msg = "In the original C++ Code, the elapsed time "
    msg += "is ~2.5 and ~3.1 seconds, respectively."
    print(msg)
