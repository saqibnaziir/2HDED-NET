# 2HDED-NET
Depth estimation and image restoration by deep learning from defocused images:

Monocular depth estimation and image deblurring are two fundamental tasks in computer vision, given their crucial role in understanding 3D scenes. Performing any of them by relying on a single image is an ill-posed problem. The recent advances in the field of Deep Convolutional Neural Networks (DNNs) have revolutionized many tasks in computer vision, including depth estimation and image deblurring. When it comes to using defocused images, the depth estimation and the recovery of the All-in-Focus (Aif) image become related problems due to defocus physics. Despite this, most of the existing models treat them separately. There are, however, recent models that solve these problems simultaneously by concatenating two networks in a sequence to first estimate the depth or defocus map and then reconstruct the focused image based on it. We propose a DNN that solves the depth estimation and image deblurring in parallel. Our Two-headed Depth Estimation and Deblurring Network (2HDED:NET) extends a conventional Depth from Defocus (DFD) networks with a deblurring branch that shares the same encoder as the depth branch. The proposed method has been successfully tested on two benchmarks, one for indoor and the other for outdoor scenes: NYU-v2 and Make3D. Extensive experiments with 2HDED:NET on these benchmarks have demonstrated superior or close performances to those of the state-of-the-art models for depth estimation and image deblurring.

[Paper](https://ieeexplore.ieee.org/abstract/document/10158786)

## Repository Requirements

## Data Organization

## Usage

## Cite

Please consider citing our work if you find it useful:

> @article{nazir2023depth,
  title={Depth estimation and image restoration by deep learning from defocused images},
  author={Nazir, Saqib and Vaquero, Lorenzo and Mucientes, Manuel and Brea, V{\'\i}ctor M and Coltuc, Daniela},
  journal={IEEE Transactions on Computational Imaging},
  volume={9},
  pages={607--619},
  year={2023},
  publisher={IEEE}
}




> @inproceedings{nazir20222hded,
  title={2HDED: Net for joint depth estimation and image deblurring from a single out-of-focus image},
  author={Nazir, Saqib and Vaquero, Lorenzo and Mucientes, Manuel and Brea, V{\'\i}ctor M and Coltuc, Daniela},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={2006--2010},
  year={2022},
  organization={IEEE}
}
