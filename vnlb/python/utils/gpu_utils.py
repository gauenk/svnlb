
import numpy as np
import torch as th

def yuv2rgb_cpp(burst):
    # burst.shape = (...,c,h,w) in YUV

    # -- n-dim -> 4-dim --
    shape = burst.shape
    shape_rs = (-1,shape[-3],shape[-2],shape[-1])
    burst = burst.reshape(shape_rs)

    # -- color convert --
    burst = apply_yuv2rgb(burst)

    # -- 4-dim -> n-dim --
    burst = burst.reshape(shape)
    return burst

def apply_yuv2rgb(burst):
    """
    rgb -> yuv [using the "cpp repo" logic]
    """
    burst_rgb = []
    # burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):

        # -- init --
        image = burst[ti]
        image_rgb = th.zeros_like(image)
        weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
        w = weights

        # -- rgb -> yuv --
        image_rgb[0] = w[0] * image[0] + w[1] * image[1] + w[2] * 0.5 * image[2]
        image_rgb[1] = w[0] * image[0] - w[2] * image[2]
        image_rgb[2] = w[0] * image[0] - w[1] * image[1] + w[2] * 0.5 * image[2]

        # -- append --
        burst_rgb.append(image_rgb)
    burst_rgb = th.stack(burst_rgb)

    return burst_rgb

def rgb2yuv_cpp(burst):
    """
    rgb -> yuv [using the "cpp repo" logic]
    """
    burst_yuv = []
    # burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):

        # -- init --
        image = burst[ti]
        image_yuv = th.zeros_like(image)
        weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]

        # -- rgb -> yuv --
        image_yuv[0] = weights[0] * (image[0] + image[1] + image[2])
        image_yuv[1] = weights[1] * (image[0] - image[2])
        image_yuv[2] = weights[2] * (.25 * image[0] - 0.5 * image[1] + .25 * image[2])

        # -- append --
        burst_yuv.append(image_yuv)
    burst_yuv = th.stack(burst_yuv)

    return burst_yuv

def apply_color_xform_cpp(burst):
    """
    rgb -> yuv [using the "cpp repo" logic]
    """
    burst_yuv = []
    # burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):

        # -- init --
        image = burst[ti]
        image_yuv = th.zeros_like(image)
        weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]

        # -- rgb -> yuv --
        image_yuv[0] = weights[0] * (image[0] + image[1] + image[2])
        image_yuv[1] = weights[1] * (image[0] - image[2])
        image_yuv[2] = weights[2] * (.25 * image[0] - 0.5 * image[1] + .25 * image[2])

        # -- append --
        burst_yuv.append(image_yuv)
    burst_yuv = th.stack(burst_yuv)

    return burst_yuv

def apply_color_xform(burst):
    """
    rgb -> yuv
    """
    burst_yuv = []
    burst = rearrange(burst,'t c h w -> t h w c')
    t,h,w,c = burst.shape
    for ti in range(t):
        image_yuv = cv2.cvtColor(burst[ti], cv2.COLOR_RGB2YUV)
        burst_yuv.append(image_yuv)
    burst_yuv = th.stack(burst_yuv)
    burst_yuv = rearrange(burst_yuv,'t h w c -> t c h w')
    return burst_yuv
