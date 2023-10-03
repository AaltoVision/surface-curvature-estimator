import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.DPT.utils import ellipsoid_plot
from matplotlib import pyplot as plt

import numpy as np

def extract_curv_tensor(z_unfold, _device):
    bz, kpts, groups = z_unfold.shape[0], z_unfold.shape[1], z_unfold.shape[2]
    # set matrix C
    k = 4
    C = torch.zeros((bz, kpts, 6, 6), device = _device)
    D = torch.zeros((bz, kpts, 10, groups),device = _device)
    Coff = torch.zeros((bz, kpts, 4, 4),device = _device)
    T = torch.zeros((bz, kpts, 4, 4),device = _device)

    C[:,:,0,0] = -1
    C[:,:,1,1] = -1
    C[:,:,2,2] = -1
    C[:,:,3,3] = -k
    C[:,:,4,4] = -k
    C[:,:,5,5] = -k
    C[:,:,0, 1] = k/2 - 1
    C[:,:,0, 2] = k/2 - 1
    C[:,:,1, 0] = k/2 - 1
    C[:,:,1, 2] = k/2 - 1
    C[:,:,2, 0] = k/2 - 1
    C[:,:,2, 1] = k/2 - 1

    inv_C = torch.linalg.pinv(C)

    p1 = z_unfold[:,:,:,0]
    p2 = z_unfold[:,:,:,1]
    p3 = z_unfold[:,:,:,2]

    D[:,:,0,:] = p1 * p1
    D[:,:,1,:] = p2 * p2
    D[:,:,2,:] = p3 * p3
    D[:,:,3,:] = 2 * p1 * p2
    D[:,:,4,:] = 2 * p1 * p3
    D[:,:,5,:] = 2 * p2 * p3
    D[:,:,6,:] = 2 * p1
    D[:,:,7,:] = 2 * p2
    D[:,:,8,:] = 2 * p3

    D_T = D.transpose(3,2)
    S = torch.einsum('bkmn,bknp->bkmp', D, D_T)

    S11 = S[:,:, 0:6, 0:6]
    S12 = S[:,:, 0:6, 6:10]
    S22 = S[:,:, 6:10, 6:10] 
    A = S11 - torch.einsum('bpij,bpjk,bpkm->bpim', S12, torch.linalg.inv(S22 + 1e-5 * torch.rand((bz, kpts, 4, 4), device = _device).float()), S12.transpose(3,2))

    M =  torch.einsum('bpij,bpjk->bpik',inv_C, A)  + 1e-4 * torch.rand((bz, kpts, 6, 6), device = _device).float()
    eigvals, eigvecs = torch.linalg.eig(M)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    idx = torch.argmax(eigvals,dim=2)
    vecs1 = torch.zeros_like(eigvals)
    for i in torch.arange(bz):
        vecs1[i,:,:] = eigvecs[i, torch.arange(kpts), :, idx[i,:]]

    vecs2 = - torch.einsum('bpij, bpjk,bpk->bpi',torch.linalg.inv(S22+1e-5 * torch.rand((bz, kpts, 4, 4), device = _device).float()), S12.transpose(3,2), vecs1)

    vecs = torch.cat((vecs1, vecs2),2)
    Coff[:,:,0,0] = vecs[:,:,0]
    Coff[:,:,1,1] = vecs[:,:,1]
    Coff[:,:,2,2] = vecs[:,:,2]
    Coff[:,:,3,3] = vecs[:,:,9]

    Coff[:,:,0,1] = Coff[:,:,1,0] = vecs[:,:,3]
    Coff[:,:,0,2] = Coff[:,:,2,0] = vecs[:,:,4]
    Coff[:,:,0,3] = Coff[:,:,3,0] = vecs[:,:,6]
    Coff[:,:,1,2] = Coff[:,:,2,1] = vecs[:,:,5]
    Coff[:,:,1,3] = Coff[:,:,3,1] = vecs[:,:,7]
    Coff[:,:,2,3] = Coff[:,:,3,2] = vecs[:,:,8]

    Coff  = Coff + 1e-7 * torch.rand(bz, kpts,4,4, device = _device)
    centre = torch.linalg.solve(-Coff[:,:,0:3, 0:3], vecs[:,:,6:9])


    T[:,:,0,0] = T[:,:,1,1]=T[:,:,2,2] = T[:,:,3,3] = 1
    T[:,:,3, 0:3] = centre
    R = torch.einsum('bpij,bpjk,bpkm->bpim', T, Coff, T.transpose(3,2))
    evals, evecs = torch.linalg.eig(R[:,:,0:3, 0:3] / -R[:,:,3, 3][:,:,None,None])
    evecs = evecs.transpose(3,2)
    radii = torch.sqrt(1 / torch.clamp(torch.abs(evals), min=1e-9))
    return centre, evecs, radii, vecs

class GetCurvature(nn.Module):
    def __init__(self, img_w, img_h):
        super().__init__()
        x = torch.linspace(0,img_w-1, img_w)
        y = torch.linspace(0,img_h-1, img_h)
        [yy, xx]=torch.meshgrid(y,x)
        xx_unfold1 = F.unfold(xx[None, None, :,:], kernel_size=(8, 8), stride=8, padding=1)[:,None,:,:]
        yy_unfold1 = F.unfold(yy[None, None, :,:], kernel_size=(8, 8), stride=8, padding=1)[:,None,:,:]
        self.register_buffer('xx_unfold1', xx_unfold1, persistent=False)
        self.register_buffer('yy_unfold1', yy_unfold1, persistent=False)

    def get_curv_descs(self, depth):
        _device = depth.device
        # depth = depth + 1e-5 * torch.rand(depth.shape, device = _device).float()
        d_unfold1 =  F.unfold(depth[None, None, :,:], kernel_size=(8, 8), stride=8, padding=1)[:,None,:,:]

        z_unfold1 = torch.cat((self.xx_unfold1, self.yy_unfold1, d_unfold1), axis=1).permute(0,3,2,1)
        center_points =  z_unfold1[:,:, 36, :][:,:,None,:].repeat(1,1,64,1)
        norm_points = z_unfold1 - center_points
        num_kpts = norm_points.shape[1]
        descs_curv = torch.zeros((1, num_kpts , 1), device = _device)
        z_unfolds = [norm_points]
        for i, z_unfold in enumerate(z_unfolds):
            center, evecs, radii, vecs = extract_curv_tensor(z_unfold, _device)
            # # viz
            # ctr_points = norm_points[:,:1,:,:].reshape(-1,3).cpu().numpy()
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(ctr_points[:,0], ctr_points[:,1], ctr_points[:,2], marker='o', color='g')
            # ax.scatter(0,0,0, marker='o', color='r')
            # ellipsoid_plot(center[:,:1,:].reshape(-1,3).cpu().numpy().squeeze(), radii[:,:1,:].reshape(-1,3).cpu().numpy().squeeze(), evecs[:,:1,:,:].reshape(3,3).cpu().numpy(), ax=ax, plot_axes=True, cage_color='g')
            # plt.show()
            # # end viz
            simi_curv = torch.div(torch.min(radii, 2).values, torch.max(radii,2).values)
            descs_curv[:,:, i] = simi_curv
        return descs_curv

    def get_curv_descs_cos_similarity(self, depth):
        _device = depth.device
        # depth = depth + 1e-5 * torch.rand(depth.shape, device = _device).float()
        d_unfold1 =  F.unfold(depth[None, None, :,:], kernel_size=(8, 8), stride=8, padding=1)[:,None,:,:]

        z_unfold1 = torch.cat((self.xx_unfold1, self.yy_unfold1, d_unfold1), axis=1).permute(0,3,2,1)
        center_points =  z_unfold1[:,:, 36, :][:,:,None,:].repeat(1,1,64,1)
        norm_points = z_unfold1 - center_points
        num_kpts = norm_points.shape[1]
        descs_curv = torch.zeros((1, num_kpts , 1), device = _device)
        z_unfolds = [norm_points]
        for i, z_unfold in enumerate(z_unfolds):
            center, evecs, radii, vecs = extract_curv_tensor(z_unfold, _device)
            # # viz
            # ctr_points = norm_points[:,:1,:,:].reshape(-1,3).cpu().numpy()
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(ctr_points[:,0], ctr_points[:,1], ctr_points[:,2], marker='o', color='g')
            # ax.scatter(0,0,0, marker='o', color='r')
            # ellipsoid_plot(center[:,:1,:].reshape(-1,3).cpu().numpy().squeeze(), radii[:,:1,:].reshape(-1,3).cpu().numpy().squeeze(), evecs[:,:1,:,:].reshape(3,3).cpu().numpy(), ax=ax, plot_axes=True, cage_color='g')
            # plt.show()
            # # end viz
            modulus_radii = torch.linalg.vector_norm(radii, dim=2, keepdim=True)
            descs_curv = radii / modulus_radii.repeat(1,1,3)
        return descs_curv

from pdb import set_trace as bb

@torch.no_grad()
def compute_patch_convexity(depth_map, patch_size=8):
    # Extract patches using unfold
    patches = depth_map.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    
    # Compute the first derivatives
    dx_filter = torch.Tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]).view(1, 1, 3, 3).to(depth_map.device)
    dy_filter = torch.Tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]).view(1, 1, 3, 3).to(depth_map.device)
    
    dx = F.conv2d(patches.unsqueeze(1), dx_filter, padding=1)
    dy = F.conv2d(patches.unsqueeze(1), dy_filter, padding=1)
    
    # Compute the second derivatives
    dxx = F.conv2d(dx, dx_filter, padding=1)
    dyy = F.conv2d(dy, dy_filter, padding=1)
    dxy = F.conv2d(dx, dy_filter, padding=1)
    
    # Compute Gaussian curvature at the centers
    center_idx = patch_size // 2
    K = (dxx[:, 0, center_idx, center_idx] * dyy[:, 0, center_idx, center_idx] - dxy[:, 0, center_idx, center_idx]**2) / (1 + dx[:, 0, center_idx, center_idx]**2 + dy[:, 0, center_idx, center_idx]**2)**2
    K_sign = torch.sign(K)
    return K_sign




