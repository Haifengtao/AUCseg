#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pipeline2.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/7 12:35   Botao Zhao      1.0         None
'''

# import lib
from utils import file_io
import os
from utils import img_utils
import numpy as np
from utils import metrics
from utils import logger
from tqdm import tqdm
import time


def seg_pipeline2(rootdir, uid, save_name, mode):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], 5, method=mode)
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_full(full_tumor)
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], 3, method=mode)

    # necrotic segmentation
    nc_tumor = img_utils.segment_necrotic(ec_tumor, 3, 1)

    # Calculate metrics
    final_mask = np.zeros(full_tumor_roi.shape)
    final_mask[full_tumor_roi == 1] = 2
    final_mask[ec_tumor == 1] = 4
    final_mask[nc_tumor == 1] = 1
    final_mask = img_utils.crop2raw(seg_img.shape, img_utils.crop2raw(full_tumor.shape, final_mask, st_2, et_2), st1,
                                    et1)
    ec_result = metrics.test_result(final_mask, seg_img, mode='ec')
    core_result = metrics.test_result(final_mask, seg_img, mode='core')
    full_result = metrics.test_result(final_mask, seg_img, mode='full')

    if save_name is not None:
        file_io.save_nii_sitk(final_mask,  os.path.join(rootdir, uid, uid + save_name))
    return ec_result, core_result, full_result


def seg_pipeline3(rootdir, uid, save_name):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], 5, method='Kmeans')
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    full_tumor = img_utils.fill_full(full_tumor)
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], 3, method='Kmeans')

    # core segmentation
    t2_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t2_img, st1, et1), st_2, et_2))
    t2_roi[full_tumor_roi != 1] = 0
    core_tumor = img_utils.clustering_img([t2_roi, ], 3, method='MiniBatchKMeans')

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


def seg_pipeline4(rootdir, uid, save_name):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))

    # full tumor segmentation
    full_tumor = img_utils.clustering_img([flair_roi], 5, method='Kmeans')
    full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
    nc_tumor_1 = img_utils.get_holes(full_tumor)
    full_tumor = img_utils.fill_full(full_tumor, )
    st_2, et_2 = img_utils.get_bbox(full_tumor)
    full_tumor_roi = img_utils.crop_img(full_tumor, st_2, et_2)
    nc_tumor_1 = img_utils.crop_img(img_utils.get_holes(nc_tumor_1), st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    ec_tumor = img_utils.clustering_img([t1ce_roi, ], 3, method='Kmeans')

    # necrotic segmentation
    nc_tumor_2 = img_utils.segment_necrotic(ec_tumor, 3, 1)
    nc_tumor = nc_tumor_1 + nc_tumor_2
    nc_tumor[nc_tumor == 2] = 1

    # Calculate metrics
    final_mask = np.zeros(full_tumor_roi.shape)
    final_mask[full_tumor_roi == 1] = 2
    final_mask[ec_tumor == 1] = 4
    final_mask[nc_tumor == 1] = 1
    final_mask = img_utils.crop2raw(seg_img.shape, img_utils.crop2raw(full_tumor.shape, final_mask, st_2, et_2), st1,
                                    et1)
    ec_result = metrics.test_result(final_mask, seg_img, mode='ec')
    core_result = metrics.test_result(final_mask, seg_img, mode='core')
    full_result = metrics.test_result(final_mask, seg_img, mode='full')

    if save_name is not None:
        file_io.save_nii(final_mask, affine_seg, hdr_seg, os.path.join(rootdir, uid, uid + save_name))
    return ec_result, core_result, full_result


if __name__ == '__main__':
    rootdir = '/share/inspurStorage/home1/zhaobt/data/BraTS2018_training/HGG'
    # rootdir = 'D:\jupyter\giloma_seg_pipeliine\data'
    # rootdir = 'D:\jupyter\giloma_seg_pipeliine\data\\'
    case_list = os.listdir(rootdir)
    # case_list = ['Brats18_CBICA_AUN_1']
    print('running seg_pipeline2')
    saver = logger.Logger_csv(
        ['uid', 'dice_full', 'IoU_full', 'FP_full', 'FN_full', 'dice_ec', 'IoU_ec', 'FP_ec', 'FN_ec', 'dice_core',
         'IoU_core', 'FP_core', 'FN_core', 'time'], './final_result/old_result_5_3_GMM_v2.csv')
    for case in tqdm(case_list):
        st = time.time()
        ec_result, core_result, full_result = seg_pipeline2(rootdir, case, save_name='_pred5_3_GMM.nii.gz', mode='GMM')
        msg = {'uid': case, 'dice_full': full_result[0], 'IoU_full': full_result[1],
               'FP_full': full_result[2], 'FN_full': full_result[3], 'dice_ec': ec_result[0], 'IoU_ec': ec_result[1],
               'FP_ec': ec_result[2],
               'FN_ec': ec_result[3], 'dice_core': core_result[0],
               'IoU_core': core_result[1], 'FP_core': core_result[2], 'FN_core': core_result[3],
               'time': time.time() - st}
        saver.update(msg)
        # print(ec_msg)
        # print(core_msg)
        # print(full_msg)
