
# -- standard python imports --
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange

# -- local imports --
from file_io import read_vnlb_results

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  
#       Menu to Selet Images
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def load_dataset(name,**kwargs):
    menu = ["davis","gmobile","davis_pariasm_vnlb"]
    if name == "davis_64x64":
        kwargs['small'] = True
        return load_data_davis(**kwargs)
    elif name == "davis":
        kwargs['small'] = False
        return load_data_davis(**kwargs)
    elif name == "gmobile":
        return load_data_gmobile(**kwargs)
    elif name == "davis_pariasm_vnlb":
        return load_davis_pariasm_vnlb(**kwargs)
    else:
        print("Options include:")
        print(menu)
        raise ValueError(f"Uknown dataset name {name}")

def load_data_davis(small=True,nframes=5):

    # -- check if path exists --
    if small == True:
        fmax = 5
        path = Path("data/davis_baseball_64x64/")
    else:
        fmax = 10
        path = Path("data/davis_baseball/")        
    if not path.exists():
        print("Please download the davis baseball file from the git repo.")
    
    # -- max frames --
    if nframes <= 0: nframes = 2
    if nframes > fmax: nframes = 5

    # -- read files using PIL --
    burst = []
    for i in range(nframes):
        fn = path / ("%05d.jpg" % i)
        print(fn)
        if not fn.exists():
            print(f"Error: the file {fn} does not exist.")
            sys.exit(1)
        frame = cv2.imread(str(fn),-1)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        burst.append(frame)

    # -- format --
    burst = np.stack(burst)
    burst = rearrange(burst,'t h w c -> t c h w')
    burst = np.ascontiguousarray(burst.copy())
    
    return burst

def load_data_gmobile(nframes=5):

    # -- check if path exists --
    path = Path("data/gmobile/")
    if not path.exists():
        print("Please run the following commands")
        print("mkdir data/gmobile/")
        print("cd data/gmobile/")
        print("wget http://dev.ipol.im/~pariasm/video_nlbayes/videos/gmobile.avi")
        print("ffmpeg -i gmobile.avi -f image2 %03d.png")
    
    # -- max frames --
    if nframes <= 0: nframes = 2
    if nframes > 300: nframes = 300

    # -- read files using PIL --
    burst = []
    for i in range(nframes):
        fn = path / ("%03d.png" % i)
        if not fn.exists():
            print(f"Error: the file {fn} does not exist.")
            sys.exit(1)
        frame = cv2.imread(str(fn),-1)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        burst.append(frame)

    # -- format --
    burst = np.stack(burst)
    burst = rearrange(burst,'t h w c -> t c h w')
    burst = np.ascontiguousarray(burst.copy())
    print("burst.shape: ",burst.shape)
    
    return burst

def load_davis_pariasm_vnlb(nframes=5):

    # -- check if path exists --
    path = Path("/home/gauenk/Documents/packages/vnlb/")
    path = path / Path("output/davis_baseball_64x64_20/vnlb/")
    # path = Path("data/example_pariasm_vnlb/")
    if not path.exists():
        print("Please download the example results from the git repo.")

    # -- max frames --
    if nframes <= 0: nframes = 2
    if nframes > 5: nframes = 5

    return read_vnlb_results(path,0,nframes)


