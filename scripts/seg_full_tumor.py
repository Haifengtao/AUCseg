#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   seg_full_tumor.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/8 20:57   Botao Zhao      1.0         None
'''

# import lib
import sys
import os

sys.path.append('../')
from utils import file_io
from utils import img_utils
import numpy as np


def seg_pipeline(flair_dir, save_dir, f_cluster=5,  c_method='Kmeans', ):
    """
    segment the full tumor by clustering.
    :param flair_dir: flair image.
    :param save_dir: the dir to save the segmentation result.
    :param f_cluster: The cluster number of clustering.
    :param c_method: The cluster method, such as Kmeans, GMM, minibatchkmeans et al.
    :return: None
    """
    assert os.path.isfile(flair_dir), print('There is not a file at %s' % flair_dir)
    # Read data
    flair_img, affine_flair, hdr_flair = file_io.read_nii(flair_dir)

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], int(f_cluster), method=c_method)
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_full(full_tumor)
    final_mask = img_utils.crop2raw(flair_img.shape, full_tumor, st1, et1)
    if save_dir is not None:
        file_io.save_nii(final_mask, affine_flair, hdr_flair, save_dir)


if __name__ == '__main__':
    import argparse
    long_description = "Full brain tumor segmentation engine"
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-flair', '--flair_dir', nargs='?',
                        default='None',
                        help='the path of flair image!')
    parser.add_argument('-s', '--save_dir', nargs='?',
                        default='None',
                        help='the path of segmentation image!')
    parser.add_argument('-fc', '--full_seg_cluster', nargs='?',
                        default='5',
                        help='full tumor segmentation clustering number')
    parser.add_argument('-mode', '--cluster_mode', default='Kmeans', nargs='?',
                        help='enhancing segmentation clustering number')
    args = parser.parse_args()
    seg_pipeline(args.flair_dir, args.save_dir,
                 f_cluster=args.full_seg_cluster,
                 c_method=args.cluster_mode)
