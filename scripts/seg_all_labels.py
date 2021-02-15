#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   seg_all_labels.py
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/1 22:08   Botao Zhao      1.0         None
'''

# import lib
import sys

sys.path.append('../')
from utils import file_io
import os
from utils import img_utils
import numpy as np


def seg_pipeline(t1ce_dir, flair_dir, save_dir, bbox=None, nc_seg_mode='cc', t2_dir=None, t2_cluster=3, f_cluster=5,
                 ec_cluster=4,
                 c_method='Kmeans', ):
    assert os.path.isfile(t1ce_dir), print('There is not a file at %s' % t1ce_dir)
    assert os.path.isfile(flair_dir), print('There is not a file at %s' % flair_dir)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(t1ce_dir)
    flair_img, affine_flair, hdr_flair = file_io.read_nii(flair_dir)
    if bbox is not None:
        st1, et1 = bbox[0], bbox[1]
    else:
        st1, et1 = img_utils.get_bbox(flair_img)

    # Preprocess
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], int(f_cluster), method=c_method)
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_full(full_tumor)
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], int(ec_cluster), method=c_method)

    # necrotic segmentation
    if nc_seg_mode == 'cc':
        nc_tumor = img_utils.segment_necrotic(ec_tumor, 3, 1)
        # integrate mask
        final_mask = np.zeros(full_tumor_roi.shape)
        final_mask[full_tumor_roi == 1] = 2
        final_mask[ec_tumor == 1] = 4
        final_mask[nc_tumor == 1] = 1
    elif nc_seg_mode == 't2':
        # core segmentation
        assert os.path.isfile(t2_dir), print('There is not a file at %s' % t2_dir)
        t2_img, affine_t2, hdr_t2 = file_io.read_nii(t2_dir)
        t2_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t2_img, st1, et1), st_2, et_2))
        t2_roi[full_tumor_roi != 1] = 0
        core_tumor = img_utils.clustering_img([t2_roi, ], t2_cluster, method=c_method)
        # integrate mask
        final_mask = np.zeros(full_tumor_roi.shape)
        final_mask[full_tumor_roi == 1] = 2
        final_mask[core_tumor == 1] = 1
        final_mask[ec_tumor == 1] = 4
    else:
        raise Exception('This necrotic region segmentation method have not been defined!')
    final_mask = img_utils.crop2raw(flair_img.shape, img_utils.crop2raw(full_tumor.shape, final_mask, st_2, et_2), st1,
                                    et1)
    if save_dir is not None:
        file_io.save_nii_sitk(final_mask, save_dir)
    return final_mask


if __name__ == '__main__':
    import argparse
    long_description = "Brain Tumor Segmentation Engine"
    parser = argparse.ArgumentParser(description=long_description)
    parser.add_argument('-t1ce', '--t1ce_dir', nargs='?',
                        default='None',
                        help='The path of t1ce image!')
    parser.add_argument('-flair', '--flair_dir', nargs='?',
                        default='None',
                        help='The path of flair image!')
    parser.add_argument('-s', '--save_dir', nargs='?',
                        default='None',
                        help='The path to save the segmented image!')
    parser.add_argument('-fc', '--full_seg_cluster', nargs='?',
                        default='5',
                        help='Full tumor segmentation clustering number')
    parser.add_argument('-ec', '--ec_seg_cluster', default='3', nargs='?',
                        help='Enhancing segmentation clustering number')
    parser.add_argument('-cm', '--cluster_method', default='Kmeans', nargs='?',
                        help='Clustering method, including Kmeans, MiniBatchKMeans, GMM, FCM')
    parser.add_argument('-bbox', '--bbox', default='None', nargs='?',
                        help='The bounding box to crop the raw data!')
    parser.add_argument('-nc_seg_mode', '--nc_seg_mode', default='cc', nargs='?',
                        help='The necrotic region segmentation pipeline,including cc,t2. if you choose t2, the input '
                             't2 should not be None!')
    parser.add_argument('-t2', '--t2_dir', default='None', nargs='?',
                        help='The path of t2 image!')
    parser.add_argument('-t2_n', '--t2_n_cluster', default='None', nargs='?',
                        help='The path of t2 image!')
    args = parser.parse_args()
    seg_pipeline(args.t1ce_dir, args.flair_dir, args.save_dir,
                 f_cluster=args.full_seg_cluster,
                 ec_cluster=args.ec_seg_cluster,
                 c_method=args.cluster_method,
                 bbox=args.bbox,
                 nc_seg_mode=args.nc_seg_mode,
                 t2_dir=args.t2_dir,
                 t2_cluster=args.t2_n_cluster,)
