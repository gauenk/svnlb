
# -- python imports --
import cv2
import sys
import glob
import numpy as np
from pathlib import Path


def load_burst(path,fstart,nframes):

    # -- ensure path exists --
    if not path.exists():
        print(f"Ensure the path [{path}] exists.")

    # -- collect files --
    gpath = str(path / "./*")
    files = []
    for fn in glob.glob(gpath):
        files.append(fn)
        
    # -- read burst --
    burst = []
    for t in range(fstart,fstart+nframes):
        frame = cv2.imread(files[t])
        burst.append(frame)
    burst = np.stack(burst)
    return burst

def save_burst(burst,path,fstart):

    # -- ensure path exists --
    if not path.exists():
        path.mkdir(parents=True)
    
    # -- write each frame --
    npaths = []
    for fid,frame in enumerate(burst):
        fid = path / ("frame_%03d.tif" % (fid+fstart))
        frame = np.clip(frame,0,255)
        frame = frame.astype(np.uint8)
        cv2.imwrite(str(fid),frame)
        npaths.append(fid)
    return npaths

def load_noisy(path,fstart,nframes):

    # -- write each frame --
    burst,npaths = [],[]
    for fid in range(fstart,fstart+nframes):
        fid = path / ("frame_%03d.tif" % fid)
        if not fid.exists(): return None
        npaths.append(fid)
        frame = cv2.imread(str(fid))
        burst.append(frame)
    burst = np.stack(burst)
    return burst,npaths

def get_vnlb_burst(ipath,vnlb_path,fstart,nframes,prefix=None):
    clean,_ = get_vnlb_burst_at_path(ipath,fstart,nframes,"jpg","%05d",prefix)
    noisy,npath = get_vnlb_burst_at_path(vnlb_path,fstart,nframes,"tif","%03d",prefix)
    return clean,noisy,npath

def get_vnlb_burst_at_path(vnlb_path,fstart,nframes,ext,fmt="%03d",prefix=None):
    # -- write each frame --
    burst,npaths = [],[]
    for fid in range(fstart,fstart+nframes):
        if prefix: ffmt = f"{prefix}_{fmt}.%s"
        else: ffmt = f"{fmt}.%s"
        fid = vnlb_path / (ffmt % (fid,ext))
        if not fid.exists(): return None,None
        npaths.append(fid)
        frame = cv2.imread(str(fid),-1)
        burst.append(frame)
    burst = np.stack(burst)
    return burst,npaths


def get_noisy_burst(ipath,opath,std,fstart,nframes):

    print("Loading noisy burst [if exists].")
    noisy = load_noisy(ipath,fstart,nframes)
    if not(noisy is None): return noisy

    print("Noisy burst does not exist. Creating...")
    print("Reading burst.")
    burst = load_burst(ipath,fstart,nframes)
    print("Adding noise.")
    noise = np.random.normal(0,scale=std/255.,size=burst.shape)
    noisy = burst + noise
    print("Saving burst.")
    npaths = save_burst(burst,opath,fstart)
    print("Complete.")
    return noisy,npaths
    
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Please specify an integer noise level.")
        print("Examples: 10, 20, ..., 50")
        sys.exit(0)
    std = int(sys.argv[1])
    ipath = Path("data/davis_baseball/")
    opath = Path(f"output/davis_baseball_{std}/")

    print(f"Creating a noisy burst with noise level {std}")
    get_noisy_burst(ipath,opath,std,0,20)
