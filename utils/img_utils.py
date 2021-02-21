#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   img_utils.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/24 17:28   Botao Zhao      1.0    For array-type image.
'''

# import lib
import sklearn.cluster as cluster
from skimage import measure as ms
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import SimpleITK as sitk
import cv2 as cv
from skimage import morphology
from sklearn.mixture import GaussianMixture
from utils import FCM
from utils import file_io


def crop_img(img_arr, st, et):
    """
    crop img by voxel index;
    :param img_arr: img array;
    :param st: start point;
    :param et: end point;
    :return: new image
    """
    temp = img_arr.copy()
    return temp[st[0]:et[0], st[1]:et[1], st[2]:et[2]]


def clustering_img(img_arr_lists, num_clusters, chose_class='max', method='Kmeans'):
    """
    segment image by clustering method.
    :param img_arr_lists: a image list containing different images with the same shape.
    :param num_clusters: number of clustering subclass.
    :param chose_class: How to select the class.
    :param method: clustering method.
    :return: Selected class.
    """
    X = np.zeros((img_arr_lists[0].size, len(img_arr_lists)))
    for idx, img_arr in enumerate(img_arr_lists):
        X[:, idx] = img_arr.flatten()

    if method == 'Kmeans':
        model = cluster.KMeans(init="k-means++", n_clusters=num_clusters, n_init=4,
                               random_state=0).fit(X)

    elif method == 'MiniBatchKMeans':
        model = cluster.MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size=100,
                                        n_init=10, max_no_improvement=10, verbose=0,
                                        random_state=0).fit(X)

    elif method == 'FCM':
        model = FCM.FCM(X, num_clusters)
        model.fit()

    elif method == 'Birch':
        model = cluster.Birch(n_clusters=None)
        if chose_class == 'max':
            centroid = []
            for idx in range(np, max(model.labels_)):
                centroid.append(np.mean(X[model.labels_ == idx]))
            temp_label = np.zeros(img_arr_lists[0].size)
            temp_label[model.labels_ == np.argmax(np.array(centroid))] = 1
            return np.reshape(temp_label, img_arr.shape)
        else:
            raise ValueError('have not implement this method!')

    elif method == 'SpectralClustering':
        model = cluster.SpectralClustering(n_clusters=num_clusters,
                                           assign_labels="discretize",
                                           random_state=0).fit(X)
        if chose_class == 'max':
            centroid = []
            for idx in range(np, max(model.labels_)):
                centroid.append(np.mean(X[model.labels_ == idx]))
            temp_label = np.zeros(img_arr_lists[0].size)
            temp_label[model.labels_ == np.argmax(np.array(centroid))] = 1
            return np.reshape(temp_label, img_arr.shape)
        else:
            raise ValueError('have not implement this method!')

    elif method == 'GMM':
        gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(X)
        if chose_class == 'max':
            temp_label = np.zeros(img_arr_lists[0].size)
            temp_label[gmm.predict(X) == np.argmax(gmm.means_)] = 1
            return np.reshape(temp_label, img_arr.shape)
        else:
            raise ValueError('have not implement this method!')

    else:
        raise Exception('this clustering method did not contained in our model.')

    if chose_class == 'max':
        temp_label = np.zeros(img_arr_lists[0].size)
        temp_label[model.labels_ == np.argmax(model.cluster_centers_)] = 1
        temp_label = np.reshape(temp_label, img_arr.shape)
    else:
        raise ValueError('have not implement this method!')
    # print(np.unique(temp_label))
    return temp_label


def get_bbox(img_arr, ):
    """
    get the bounding box of a img_array.
    :param img_arr: image_array
    :return: start point;end point;
    """
    index_info = np.nonzero(img_arr)
    if len(img_arr.shape) < 3:
        st = [np.max([np.min(index_info[0]), 0]), np.max([np.min(index_info[1]), 0])]
        et = [np.min([np.max(index_info[0]), img_arr.shape[0]]), np.min([np.max(index_info[1]), img_arr.shape[1]])]
    elif len(img_arr.shape) == 3:
        st = [np.max([np.min(index_info[0]), 0]), np.max([np.min(index_info[1]), 0]),
              np.max([np.min(index_info[2]), 0])]
        et = [np.min([np.max(index_info[0]), img_arr.shape[0]]), np.min([np.max(index_info[1]), img_arr.shape[1]]),
              np.min([np.max(index_info[2]), img_arr.shape[2]])]
    else:
        raise ValueError('ERROR: we just support 2d or 3d image!')
    return st, et


def normalize_0_1(img_arr, min_intensity=None, max_intensity=None, ):
    """
    Normalize the image to 0-1 .
    :param img_arr: image_array;
    :param min_intensity: float
    :param max_intensity: float
    :return: new_image
    """
    if min_intensity is None:
        min_intensity = np.min(img_arr)
    if max_intensity is None:
        max_intensity = np.max(img_arr)
    img_arr[img_arr > max_intensity] = max_intensity
    img_arr[img_arr < min_intensity] = min_intensity
    return (img_arr - min_intensity) / max_intensity


def rm_small_cc(mask_array, rate=0.3):
    """
    remove small object
    :param mask_array: input binary image
    :param rate:size rate
    :return:binary image
    """
    sitk_mask_img = sitk.GetImageFromArray(mask_array)
    cc = sitk.ConnectedComponent(sitk_mask_img)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_mask_img)

    max_label = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            max_label = l
            maxsize = size

    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    label_img = sitk.GetArrayFromImage(cc)
    out_mask = label_img.copy()
    out_mask[label_img != max_label] = 0

    for i in range(len(not_remove)):
        out_mask[label_img == not_remove[i]] = 1
    return out_mask


def fill_image_2d(image, filled_number=1):
    contours, _ = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    num = len(contours)  # 轮廓的个数
    if num == 1:
        return image
    else:
        areas_contours = []
        fill_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            areas_contours.append(area)
        for idx, area in enumerate(areas_contours):
            if area <= np.max(areas_contours) * 0.3:
                # print("in...")
                fill_contours.append(contours[idx])
        cv.fillPoly(image, fill_contours, filled_number)
        return image


def fill_image_z(mask_arr, ):
    x, y, z = mask_arr.shape
    filled_img = np.zeros((x, y, z))
    for i in range(z):
        filled_img[:, :, i] = binary_fill_holes(mask_arr[:, :, i])
    return filled_img


def get_holes(mask_arr, ):
    x, y, z = mask_arr.shape
    filled_img = np.zeros((x, y, z))
    for i in range(z):
        temp = np.zeros((x, y))
        filled_2d = fill_image_2d(mask_arr[:, :, i].astype(np.uint8), filled_number=2)
        temp[filled_2d == 2] = 1
        filled_img[:, :, i] = temp
    return filled_img


def median_filter(mask_arr, radius=3):
    """
    median_filter for a 3d/2d image.
    :param mask_arr:
    :param radius:
    :return:
    """
    image = sitk.GetImageFromArray(mask_arr)
    sitk_median = sitk.MedianImageFilter()
    sitk_median.SetRadius(radius)
    sitk_median = sitk_median.Execute(image)
    median_array = sitk.GetArrayFromImage(sitk_median)
    return median_array


def dilation(data, kernel_size):
    """
    :param data:
    :param kernel_size:
    :return:
    """
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    enhance_data = morphology.dilation(data, kernel)  # dilation
    return enhance_data


def crop2raw(raw_shape, new_img, st, et):
    """
    crop the image roi to raw image.
    :param raw_shape: shape.
    :param new_img:
    :param st: start point.
    :param et: start point.
    :return: new image.
    """
    temp = np.zeros(raw_shape)
    temp[st[0]:et[0], st[1]:et[1], st[2]:et[2]] = new_img
    return temp


def segment_necrotic(data, kernel_size, connec, ):
    """
    segment the necrotic region by connected domain finding.
    :param data: a binary mask of enhancing region.
    :param kernel_size: dilation kernel size.
    :param connec: connectivity param.
    :return: necrotic array
    """
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    enhance_data = morphology.dilation(data, kernel)  # dilation to make sure that ec tumor is a cc.
    [heart_res, num] = ms.label(enhance_data, background=-1, connectivity=connec, return_num=True)
    areas = []
    region = ms.regionprops(heart_res)
    for i in range(num):
        areas.append(region[i].area)
    index = np.argsort(-np.array(areas))
    label_num1 = index[0] + 1
    necrotic = np.ones(data.shape)
    necrotic[heart_res == label_num1] = 0
    necrotic[enhance_data == 1] = 0
    necrotic = morphology.dilation(necrotic, kernel)
    return necrotic


def fill_full(data):
    """
    contain the largest connected domain .
    :param data: array 3d;
    :return:
    """
    c = np.ones((3, 3, 3))
    # dilation
    data = morphology.dilation(data, c)
    # find cc
    [heart_res, num] = ms.label(data, connectivity=3, background=-1, return_num=True)
    areas = []
    # fill the hole
    region = ms.regionprops(heart_res)
    for i in range(num):
        areas.append(region[i].area)
    label_num = np.argsort(-np.array(areas))
    for i in label_num[1:]:
        data[heart_res == (i + 1)] = 1
    return data


def inter_parts(parts, save_dir):
    """
    integrate the different parts of one case.
    :param parts: a list containing the segmentation results of every part.
    :param save_dir: the path to save the integrated image.
    :return:None
    """
    final_mask = np.zeros(parts[0].shape)
    for i in parts:
        final_mask += i
    if save_dir is not None:
        file_io.save_nii_sitk(final_mask, save_dir)
    return final_mask
