
import cv2
import numpy as np

def read_result(vnlb_path,fmt,fstart,nframes,direction="fwd"):
    agg = []

    # -- which ids to load --
    if direction == "fwd":
        fids = np.arange(fstart,fstart+nframes-1)
    elif direction == "bwd":
        fids = np.flip(np.arange(fstart+nframes-1,fstart,-1),axis=0)
    else:
        fids = np.arange(fstart,fstart+nframes)

    nfids = len(fids)
    for _t in range(nfids):
        t = fids[_t]
        path = vnlb_path / (fmt % t)
        print(path)
        if not path.exists(): return None
        data = read_file(path)
        agg.append(data)
    agg = np.stack(agg)
    return agg

def read_file(filename):
    if filename.suffix == ".flo":
        return read_flo_file(filename)
    else:
        img = cv2.imread(str(filename),-1)
        return img

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1).item()
        h = np.fromfile(f, np.int32, count=1).item()
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d
