"""Utils for monoDepth.
"""
import sys
import re
import numpy as np
import cv2
import torch
import h5py
import matplotlib.pyplot as plt



def write_depth(path, depth, bits=1):
    """Write depth map to pfm and png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    #write_pfm(path + ".pfm", depth.astype(np.float32))
    # hf = h5py.File(path + ".h5", 'w')
    # hf.create_dataset('depth', data=depth.astype(np.float16))
    # hf.close()


    depth_min = depth.min()
    depth_max = depth.max()
    # import pdb
    # pdb.set_trace()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
    save_path_rgb = path.replace('jpg', 'depth_epoch0_v51.png')

    if bits == 1:
        cv2.imwrite(save_path_rgb , out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(save_path_rgb , out.astype("uint16"))


def write_curv_rgb(path, descs_curv, dsize,  bits=2):
    curv_val = (2**(8*bits))-1
    descs_curv = np.abs(descs_curv)
    curv_min = descs_curv.min()
    curv_max = descs_curv.max()
    out = curv_val * (descs_curv - curv_min) / (curv_max - curv_min)
    out = (cv2.resize(out, dsize=dsize, interpolation=cv2.INTER_CUBIC)).astype("uint16")
    save_path_rgb = path.replace('jpg', 'curv_epoch0_v51.png')
    cv2.imwrite(save_path_rgb , out)


def write_depth_color(path, depth, bits=2):
    """Write depth map to pfm and png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    depth_min = depth.min()
    depth_max = depth.max()

    # max_val = (2**(8*bits))-1
    max_val = 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = 1 - max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 1 - np.zeros(depth.shape, dtype=depth.type)
    save_path_rgb = path.replace('jpg', 'depth_new_llllllllast.png')

    plt.imsave(save_path_rgb, out, cmap='gray')

def write_curv_color(path, descs_curv, dsize,  bits=2):
    curv_val = (2**(8*bits))-1
    descs_curv = np.abs(descs_curv)
    curv_min = descs_curv.min()
    curv_max = descs_curv.max()
    out = curv_val * (descs_curv - curv_min) / (curv_max - curv_min)
    out = (cv2.resize(out, dsize=dsize, interpolation=cv2.INTER_CUBIC)).astype("uint16")
    save_path_rgb = path.replace('jpg', 'curv_no_train_sign.png')

    out_new = (out - np.min(out)) / (np.max(out) - np.min(out))
    # plt.imsave(save_path_rgb, out_new, cmap='viridis')
    plt.imsave(save_path_rgb, out_new, cmap='viridis')

def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)