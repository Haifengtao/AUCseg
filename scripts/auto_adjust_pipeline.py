#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   auto_adjust_pipeline.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/2/18 12:10   Botao Zhao      1.0         None
'''

# import lib
import sys

sys.path.append('../')
from utils import file_io
import os
from utils import img_utils
import numpy as np
from utils import metrics
from utils import logger
from tqdm import tqdm
import time


def seg_pipeline_adjusted(rootdir, uid, save_name, mode):
    # Read data
    t1ce_img, affine_t1ce, hdr_t1ce = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t1ce.nii.gz'))
    t2_img, affine_t2, hdr_t2 = file_io.read_nii(os.path.join(rootdir, uid, uid + '_t2.nii.gz'))
    flair_img, affine_flair, hdr_flair = file_io.read_nii(os.path.join(rootdir, uid, uid + '_flair.nii.gz'))
    seg_img, affine_seg, hdr_seg = file_io.read_nii(os.path.join(rootdir, uid, uid + '_seg.nii.gz'))

    # Preprocess
    st1, et1 = img_utils.get_bbox(flair_img)
    flair_roi = img_utils.normalize_0_1(img_utils.crop_img(flair_img, st1, et1))
    seg_roi = img_utils.crop_img(seg_img, st1, et1)

    # full tumor segmentation
    full_re = [None, -1, 0]
    for n_c in range(3, 25):
        full_tumor = img_utils.clustering_img([flair_roi], n_c, method=mode)
        full_tumor = img_utils.rm_small_cc(full_tumor.astype(np.uint8), rate=0.3)
        full_tumor = img_utils.fill_full(full_tumor)
        temp_seg = seg_roi.copy()
        temp_seg[seg_roi != 0] = 1
        full_re_metric = metrics.Seg_metric(temp_seg, full_tumor)
        # print('full_re_metric {}: dice:{}'.format(n_c, full_re_metric.dice()))
        if full_re_metric.dice() > full_re[-1]:
            full_re[0] = full_tumor
            full_re[1] = n_c
            full_re[2] = full_re_metric.dice()
        if full_re_metric.dice() < full_re[-1] - 0.1:
            break

    st_2, et_2 = img_utils.get_bbox(full_re[0])
    full_tumor_roi = img_utils.crop_img(full_re[0], st_2, et_2)

    # ec tumor segmentation
    t1ce_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t1ce_img, st1, et1), st_2, et_2))
    t1ce_roi[full_tumor_roi != 1] = 0
    EC_re = [None, -1, 0]
    for n_c in range(3, 6):
        ec_tumor = img_utils.clustering_img([t1ce_roi, ], n_c, method=mode)
        ec_tumor_de = img_utils.crop2raw(full_re[0].shape, ec_tumor, st_2, et_2)
        temp_seg = np.zeros(seg_roi.shape)
        temp_seg[seg_roi == 4] = 1
        ec_re_metric = metrics.Seg_metric(temp_seg, ec_tumor_de)
        # print('ec_re_metric {}: dice:{}'.format(n_c, ec_re_metric.dice()))
        if ec_re_metric.dice() >= EC_re[-1]:
            EC_re = [ec_tumor, n_c, ec_re_metric.dice()]
    ec_tumor = EC_re[0]

    # necrotic segmentation
    nc_tumor_1 = img_utils.segment_necrotic(ec_tumor, 3, 1)
    final_mask = np.zeros(full_tumor_roi.shape)
    final_mask[full_tumor_roi == 1] = 2
    final_mask[ec_tumor == 1] = 4
    final_mask[nc_tumor_1 == 1] = 1
    final_mask = img_utils.crop2raw(seg_img.shape, img_utils.crop2raw(full_re[0].shape, final_mask, st_2, et_2), st1,
                                    et1)
    core_re = [final_mask, 'cc', metrics.test_result(final_mask, seg_img, mode='core')[0]]
    # print('core_metric cc:dice:{}'.format(core_re[-1]))
    t2_roi = img_utils.normalize_0_1(img_utils.crop_img(img_utils.crop_img(t2_img, st1, et1), st_2, et_2))
    t2_roi[full_tumor_roi != 1] = 0
    for n_c in range(3, 6):
        core_tumor = img_utils.clustering_img([t2_roi, ], n_c, method=mode)
        final_mask = np.zeros(full_tumor_roi.shape)
        final_mask[full_tumor_roi == 1] = 2
        final_mask[core_tumor == 1] = 1
        final_mask[ec_tumor == 1] = 4
        final_mask = img_utils.crop2raw(seg_img.shape, img_utils.crop2raw(full_re[0].shape, final_mask, st_2, et_2),
                                        st1, et1)
        core_metric = metrics.test_result(final_mask, seg_img, mode='core')
        # print('core_metric {}:dice:{}'.format(n_c, core_metric[0]))
        if core_metric[0] >= core_re[-1]:
            core_re = [final_mask, n_c, core_metric[0]]

    final_mask = core_re[0]
    # Calculate metrics
    ec_result = metrics.test_result(final_mask, seg_img, mode='ec')
    core_result = metrics.test_result(final_mask, seg_img, mode='core')
    full_result = metrics.test_result(final_mask, seg_img, mode='full')

    if save_name is not None:
        file_io.save_nii_sitk(final_mask, os.path.join(rootdir, uid, uid + save_name))
    return ec_result, core_result, full_result, [full_re[1], EC_re[1], core_re[1]]


def run(threadid, case_list, saver, rootdir,):
    # print('running {}'.format(threadid))
    for case in tqdm(case_list):
        st = time.time()
        ec_result, core_result, full_result, hyper_param = seg_pipeline_adjusted(rootdir, case,
                                                                                 save_name='_pred_adjust_GMM.nii.gz',
                                                                                 mode='GMM')
        msg = {'uid': case, 'dice_full': full_result[0], 'IoU_full': full_result[1],
               'FP_full': full_result[2], 'FN_full': full_result[3], 'dice_ec': ec_result[0], 'IoU_ec': ec_result[1],
               'FP_ec': ec_result[2],
               'FN_ec': ec_result[3], 'dice_core': core_result[0],
               'IoU_core': core_result[1], 'FP_core': core_result[2], 'FN_core': core_result[3],
               'n_cluster1': hyper_param[0], 'n_cluster2': hyper_param[1], 'mode': hyper_param[2],
               'time': time.time() - st}
        # print('threadID: finish case {} : info {}'.format(threadid, case, msg))
        saver.update(msg)


if __name__ == '__main__':
    import threading, math

    rootdir = '/share/inspurStorage/home1/zhaobt/data/BraTS2018_training/HGG'
    # rootdir = 'D:\jupyter\giloma_seg_pipeliine\data'
    # rootdir = 'D:\jupyter\giloma_seg_pipeliine\data\\'
    caselist = os.listdir(rootdir)
    num_thread = 12

    # case_list = ['Brats18_CBICA_AUN_1']

    saver = logger.Logger_csv(
        ['uid', 'dice_full', 'IoU_full', 'FP_full', 'FN_full', 'dice_ec', 'IoU_ec', 'FP_ec', 'FN_ec', 'dice_core',
         'IoU_core', 'FP_core', 'FN_core', 'n_cluster1', 'n_cluster2', 'mode', 'time'],
        '../final_result/result_adjusted_GMM.csv')

    files_eachThread = math.floor(len(caselist) / num_thread)
    print('running seg_pipeline_adjusted')
    print(len(caselist))

    threads = []
    i = 0
    count = 1
    while count < num_thread:
        temp_t = threading.Thread(target=run, args=(
        count, caselist[i:i + files_eachThread], saver, rootdir))
        threads.append(temp_t)
        i += files_eachThread
        count += 1
    threads.append(threading.Thread(target=run, args=(
    num_thread, caselist[i:], saver, rootdir)))
    for x in threads:
        x.start()
    for x in threads:
        x.join()
