#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   scripts.py
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/1 22:08   Botao Zhao      1.0         None
'''

# import lib
from utils import file_io
import os
from utils import img_utils
import numpy as np
from utils import metrics
from utils import logger
from tqdm import tqdm


def seg_pipeline1(rootdir, uid, save_name):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], 5, method='MiniBatchKMeans')
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_image_z(full_tumor)
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)

    # core tumor segmentation
    t2_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t2_img, st1, et1), st_2, et_2))
    core_tumor = img_utils.clustering_img([t2_roi, ], 3, method='MiniBatchKMeans')

    # enhancing tumor
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], 3, method='MiniBatchKMeans')

    # Calculate metrics
    final_mask = np.zeros(full_tumor_roi.shape)
    final_mask[full_tumor_roi == 1] = 2
    final_mask[core_tumor == 1] = 1
    final_mask[ec_tumor == 1] = 4
    final_mask = img_utils.crop2raw(seg_img.shape, img_utils.crop2raw(full_tumor.shape, final_mask, st_2, et_2), st1,
                                    et1)
    ec_result = metrics.test_result(final_mask, seg_img, mode='ec')
    core_result = metrics.test_result(final_mask, seg_img, mode='core')
    full_result = metrics.test_result(final_mask, seg_img, mode='full')

    if save_name is not None:
        file_io.save_nii(final_mask, affine_seg, hdr_seg, os.path.join(rootdir, uid, uid + save_name))
    return ec_result, core_result, full_result


def seg_pipeline2(rootdir, uid, save_name):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], 5, method='MiniBatchKMeans')
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_image_z(full_tumor)
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], 3, method='MiniBatchKMeans')


if __name__ == '__main__':
    # import argparse
    # long_description = "brain tumor segmentation engine"
    #
    # parser = argparse.ArgumentParser(description=long_description)
    # parser.add_argument('-fc', '--full_seg_cluster', nargs='?',
    #                     default='5',
    #                     help='full tumor segmentation clustering number')
    # parser.add_argument('-cc', '--core_seg_cluster', default='3', nargs='?',
    #                     help='core segmentation clustering number')
    # parser.add_argument('-ec', '--ec_seg_cluster', default='3', nargs='?',
    #                     help='enhancing segmentation clustering number')
    # args = parser.parse_args()
    rootdir = '/share/inspurStorage/home1/zhaobt/data/BraTS2018_training/HGG'
    # rootdir = 'D:\jupyter\giloma_seg_pipeliine\data'
    case_list = os.listdir(rootdir)
    print(len(case_list))
    ec_saver = logger.Logger_csv(['uid', 'dice', 'IoU', 'FP', 'FN'], './ec_result.csv')
    core_saver = logger.Logger_csv(['uid', 'dice', 'IoU', 'FP', 'FN'], './core_result.csv')
    full_saver = logger.Logger_csv(['uid', 'dice', 'IoU', 'FP', 'FN'], './full_result.csv')
    for case in tqdm(case_list):
        ec_result, core_result, full_result = seg_pipeline1(rootdir, case, save_name='_pred.nii.gz')
        ec_msg = {'uid': case, 'dice': ec_result[0], 'IoU': ec_result[1], 'FP': ec_result[2], 'FN': ec_result[3]}
        core_msg = {'uid': case, 'dice': core_result[0], 'IoU': core_result[1], 'FP': core_result[2],
                    'FN': core_result[3]}
        full_msg = {'uid': case, 'dice': full_result[0], 'IoU': full_result[1], 'FP': full_result[2],
                    'FN': full_result[3]}
        ec_saver.update(ec_msg)
        core_saver.update(core_msg)
        full_saver.update(full_msg)
        # print(ec_msg)
        # print(core_msg)
        # print(full_msg)
