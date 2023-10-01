# [ICCV 2023] [Guiding Local Feature Matching with Surface Curvature](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Guiding_Local_Feature_Matching_with_Surface_Curvature_ICCV_2023_paper.pdf)
[Shuzhe Wang](https://scholar.google.com/citations?user=Kzq9fl4AAAAJ&hl=en&oi=ao), [Juho Kannala](https://scholar.google.com/citations?user=c4mWQPQAAAAJ&hl=en), [Marc Pollefeys](https://scholar.google.com/citations?user=YYH0BjEAAAAJ&hl=en), [Daniel Barath](https://scholar.google.com/citations?user=U9-D8DYAAAAJ&hl=en)



We propose a new method, called curvature similarity extractor (CSE), for improving local feature matching across images. CSE calculates the curvature of the local 3D surface patch for each detected feature point in a viewpoint-invariant manner via fitting quadrics to predicted monocular depth maps. This curvature is then leveraged as an additional signal in feature matching with off-the-shelf matchers like SuperGlue and LoFTR. Additionally, CSE enables end-to-end joint training by connecting the matcher and depth predictor networks. Our experiments demonstrate on large-scale real-world datasets that CSE consistently improves the accuracy of state-of-the-art methods. Fine-tuning the depth prediction network further enhances the accuracy. The proposed approach achieves state-of-the-art results on the ScanNet dataset, showcasing the effectiveness of incorporating 3D geometric information into feature matching.

![](images/main_figure_curvature.svg)

## Environment Setup

Our curvature similarity extractor is an add-on component for advanced matchers. Here we consider the [QuadTree](https://github.com/Tangshitao/QuadTreeAttention) as the matcher and [DPT](https://github.com/isl-org/DPT) for the depth estimation. Please consider setup the [QuadTree](https://github.com/Tangshitao/QuadTreeAttention) environment with the comments:

```
git@github.com:Tangshitao/QuadTreeAttention.git
cd QuadTreeAttention&&python setup.py install
```

Download our CSE module.

```
cd ..
git@github.com:AaltoVision/surface-curvature-estimator.git
cd surface-curvature-estimator
conda env create -f environment.yaml
conda activate 
```

## Dataset 

Please refer to the [LoFTR](https://github.com/zju3dv/LoFTR) for dataset setup.



