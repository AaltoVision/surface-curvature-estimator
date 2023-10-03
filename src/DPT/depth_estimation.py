from src.DPT.midas.dpt_depth import DPTDepthModel
# from src.DPT.midas.midas_net import MidasNet
from src.DPT.curvature_extraction import GetCurvature, compute_patch_convexity
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from src.DPT.utils import write_depth_color, write_curv_color
import os
from pdb import set_trace as bb



class MiDasDepth(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['trainer']['testing']:
            model_path = None
        else:
            model_path = 'weights/dpt_large-midas-2f21e586.pt'
        print('load_depth_model_from', model_path)
        model = DPTDepthModel(
                        path= model_path, # load the pretrained model for fine-tune
                        backbone="vitl16_384",
                        non_negative=True,
                    )

        model.to(memory_format=torch.channels_last)
        self.model = model.train()
        if config['dataset']['test_data_source'] == 'MegaDepth' or config['dataset']['trainval_data_source'] == 'MegaDepth' or config['dataset']['test_data_source'] == 'YFCC':
            img_resize = config['dataset']['mgdpt_img_resize']
            self.getcurvature = GetCurvature(img_resize, img_resize)
        elif config['dataset']['test_data_source'] == 'ScanNet' or config['dataset']['trainval_data_source'] == 'ScanNet':
            self.getcurvature = GetCurvature(640, 480)
        else:
            raise NotImplementedError()

        self.get_curv_map = None # or L2
        self.get_curv_desc = None # or 'cosine_similarity'
        self.temperature = 1
        self.viz = False
        self.scale_factor = 100
        self.sign = config['loftr']['match_coarse']['curv_sign'] # consider if the surface is convex or concave

    def forward(self, data):
        dataset_name = data['dataset_name'][0]
        img0 = data['img2depth0']
        img1 = data['img2depth1']
        sample0 = img0.to(memory_format=torch.channels_last)
        sample1 = img1.to(memory_format=torch.channels_last)
        sample = torch.cat((sample0, sample1), 0)
        prediction = self.model.forward(sample)
        depth0 = prediction[0,:,:]
        depth1 = prediction[1,:,:]
        if dataset_name == 'ScanNet':
            depth0 = (
                        torch.nn.functional.interpolate(
                            depth0[None,None,:,:],
                            size=(480,640),
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze(0)
                    )

            depth1 = (
                torch.nn.functional.interpolate(
                    depth1[None,None,:,:],
                    size=(480,640),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
            )

        else:
            img0_size = (data['image0'].shape[2], data['image0'].shape[3])
            img1_size = (data['image1'].shape[2], data['image1'].shape[3])
            depth0 = (
                        torch.nn.functional.interpolate(
                            depth0[None,None,:,:],
                            size=img0_size,
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze(0)
                    )

            depth1 = (
                torch.nn.functional.interpolate(
                    depth1[None,None,:,:],
                    size=img1_size,
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze(0)
            )

        # print(depth0.min(), depth0.median(),depth0.max())
        # print(data['pair_names'])
        if self.get_curv_desc == 'cosine_similarity':
            descs_curv0 = self.getcurvature.get_curv_descs_cos_similarity(depth0.squeeze() * self.scale_factor)
            descs_curv1 = self.getcurvature.get_curv_descs_cos_similarity(depth1.squeeze() * self.scale_factor)
            if self.get_curv_map == 'L2':
                curv_map = torch.cdist(descs_curv0, descs_curv1, p=2)
                norm_curv_map = 1 - (curv_map - curv_map.min())/(curv_map.max() - curv_map.min())
            else:
                norm_curv_map = torch.einsum("nlc,nsc->nls", descs_curv0, descs_curv1) / self.temperature

        else:
            descs_curv0 = self.getcurvature.get_curv_descs(depth0.squeeze() * self.scale_factor)
            descs_curv1 = self.getcurvature.get_curv_descs(depth1.squeeze() * self.scale_factor)
            if self.sign:
                with torch.no_grad():
                    sign0 = compute_patch_convexity(depth0.squeeze())
                    sign1 = compute_patch_convexity(depth1.squeeze())
                curv_map = torch.cdist(descs_curv0*sign0 , descs_curv1*sign1, p=2)
            else:
                curv_map = torch.cdist(descs_curv0 , descs_curv1, p=2)
            norm_curv_map = 1 - (curv_map - curv_map.min())/(curv_map.max() - curv_map.min())

        data.update({'curv_map': norm_curv_map})

        # if self.viz:
        #     depth0 = depth0.squeeze().cpu().numpy()
        #     depth1 = depth1.squeeze().cpu().numpy()
        #     if dataset_name == 'ScanNet':
        #         save_path0 = os.path.join('data/scannet/test', data['pair_names'][0][0])
        #         save_path1 = os.path.join('data/scannet/test', data['pair_names'][1][0])
                
        #     elif dataset_name == 'MegaDepth':
        #         save_path0 = os.path.join('data/megadepth/test', data['pair_names'][0][0])
        #         save_path1 = os.path.join('data/megadepth/test', data['pair_names'][1][0])
        #     else:
        #         raise NotImplementedError()
        #     # write_depth_color(save_path0, depth0, bits=2)
        #     # write_depth_color(save_path1, depth1, bits=2)
        #     # add curvature representation
        #     if dataset_name == 'ScanNet':
        #         dsize = (640, 480)
        #         descs_curv0 = descs_curv0.squeeze().cpu().numpy().reshape(60, 80)
        #         descs_curv1 = descs_curv1.squeeze().cpu().numpy().reshape(60, 80)
        #     elif dataset_name == 'MegaDepth':
        #         dsize = (data['image0'].shape[2], data['image0'].shape[3])
        #         descs_curv0 = descs_curv0.squeeze().cpu().numpy().reshape(104, 104)
        #         descs_curv1 = descs_curv1.squeeze().cpu().numpy().reshape(104, 104)
        #     write_curv_color(save_path0, descs_curv0 , dsize, bits=2)
        #     write_curv_color(save_path1, descs_curv1 , dsize, bits=2)               
