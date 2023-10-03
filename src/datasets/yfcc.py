import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
import deepdish as dd

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth, read_image_for_depth_megadepth

# change the format from megadepth to yfcc
class YFCCDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        # self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0

        # with open(test_path, 'r') as f:
        #     self.pair_infos = [l.split() for l in f.readlines()]

        self.scene_info = np.load(npz_path)
        self.pair_infos = self.scene_info['pair_infos'].tolist().copy()
        # del self.scene_info['pair_infos']
        # self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)



    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        pairs = self.pair_infos[idx]
        name0, name1 = pairs[0], pairs[1]
        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, name0)
        img_name1 = osp.join(self.root_dir, name1)
        
        # reuse the megadepth code for image reading and resize YFCC images to 1600
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # recuse the megadepth code for YFCC image
        img2depth0 = read_image_for_depth_megadepth(img_name0, self.img_resize, self.df, self.img_padding, None)
        img2depth1 = read_image_for_depth_megadepth(img_name1, self.img_resize, self.df, self.img_padding, None)

        # # read depth. shape: (h, w)
        # if self.mode in ['train', 'val']:
        #     depth0 = read_megadepth_depth(
        #         osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
        #     depth1 = read_megadepth_depth(
        #         osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        # else:
        # no gt depth for test
        depth0 = depth1 = torch.tensor([])
        # curv0_path = osp.join(self.root_dir, self.scene_info['depth_paths'][idx0])
        # desc_curv0 = dd.io.load(curv0_path.replace('.h5', '.curv.h5'))
        # curv1_path = osp.join(self.root_dir, self.scene_info['depth_paths'][idx1])
        # desc_curv1 = dd.io.load(curv1_path.replace('.h5', '.curv.h5'))


        # read intrinsics of original size
        K_0 = torch.from_numpy(np.array(pairs[4:13]).astype(float).reshape(3, 3))
        K_1 = torch.from_numpy(np.array(pairs[13:22]).astype(float).reshape(3, 3))
        T_0to1 = torch.from_numpy(np.array(pairs[22:]).astype(float).reshape(4, 4))

        # read and compute relative poses
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'img2depth0' : img2depth0,
            'image1': image1,
            'depth1': depth1,
            'img2depth1' : img2depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'YFCC',
            'scene_id': idx,
            'pair_id': idx,
            'pair_names': (name0, name1),
        }

        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        return data
