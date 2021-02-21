#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vis_tools.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/1/24 17:33   Botao Zhao      1.0         None
'''

# import lib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import color, img_as_float


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


def rotate(image, angle, center=None, scale=1.0):
    """
    rotate 2d image any angle.
    :param image: numpy array;
    :param angle: 0~360;
    :param center:
    :param scale:
    :return: new image.
    """
    image = image.astype(np.float64)
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def get_img_mask(img_arr, seg_arr, alpha, color_config_file=None):
    """

    :param img_arr:
    :param seg_arr:
    :param alpha:
    :param color_config_file:
    :return:
    """
    if color_config_file is None:
        color_config_file = {"1": [1, 0, 0], "2": [0, 1, 0], "3": [0, 0, 1], "4": [1, 1, 0], "5": [1, 0, 1]}
    img_arr = normalize_0_1(img_arr)
    img = img_as_float(img_arr)
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    for i in range(1, len(color_config_file) + 1):
        color_mask[seg_arr == i] = color_config_file[str(i)]

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def vis_3d_img(img_array, bins=5, cmap='gray'):
    """
    visualize the 3d image in 3 dim.
    :param img_array:
    :param bins: how many slices to show; int
    :param cmap: colormap
    :return: None
    """
    assert len(img_array.shape) == 3, print("The image dim must be 3!")
    figure, ax = plt.subplots(3, bins, figsize=(20, 10))
    x, y, z = img_array.shape
    for i in range(bins):
        temp1 = ax[0][i].imshow(rotate(img_array[int(x / (bins + 1)) * (i + 1), ...].T, 180), cmap=cmap)
        temp2 = ax[1][i].imshow(rotate(img_array[:, int(y / (bins + 1)) * (i + 1), :].T, 180), cmap=cmap)
        temp3 = ax[2][i].imshow(rotate(img_array[:, :, int(z / (bins + 1)) * (i + 1)].T, 180), cmap=cmap)
        figure.colorbar(temp1, ax=ax[0][i])
        figure.colorbar(temp2, ax=ax[1][i])
        figure.colorbar(temp3, ax=ax[2][i])
    plt.show()


def vis_3d_imgWithmask(img_array, mask_array, bbox, bins=5):
    """
    visualize the 3d image with mask in 3 dim.
    :param img_array:
    :param bins: how many slices to show; int
    :param cmap: colormap
    :return: None
    """
    assert len(img_array.shape) == 3, print("The image dim must be 3!")
    figure, ax = plt.subplots(3, bins, figsize=(20, 10))
    sp, ep = bbox[0], bbox[1]
    x, y, z = ep[0]-sp[0], ep[1]-sp[1], ep[2]-sp[2]
    for i in range(bins):
        masked_img1 = get_img_mask(rotate(img_array[int(x / (bins + 1)) * (i + 1)+ep[0], ...].T, 180),
                                   rotate(mask_array[int(x / (bins + 1)) * (i + 1)+ep[0], ...].T, 180), 0.95)
        masked_img2 = get_img_mask(rotate(img_array[:, int(y / (bins + 1)) * (i + 1)+ep[1], :].T, 180),
                                   rotate(mask_array[:, int(y / (bins + 1)) * (i + 1)+ep[1], :].T, 180), 0.95)
        masked_img3 = get_img_mask(rotate(img_array[:, :, int(z / (bins + 1)) * (i + 1)+ep[2]].T, 180),
                                   rotate(mask_array[:, :, int(z / (bins + 1)) * (i + 1)+ep[2]].T, 180), 0.95)

        temp1 = ax[0][i].imshow(masked_img1)
        temp2 = ax[1][i].imshow(masked_img2)
        temp3 = ax[2][i].imshow(masked_img3)
        figure.colorbar(temp1, ax=ax[0][i])
        figure.colorbar(temp2, ax=ax[1][i])
        figure.colorbar(temp3, ax=ax[2][i])
    plt.show()


def vis_2d_z(img_array, slices, cmap='gray'):
    plt.imshow(rotate(img_array[..., slices].T, 180), cmap=cmap)
    plt.xticks('')
    plt.show()
