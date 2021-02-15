import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np


def read_path(root):
    files = os.listdir(root)
    pathes = []
    for f in files:
        pathes.append(os.path.join(root, f))
    return pathes


def read_nii(path):
    """
    get the array data from nii file
    :param path: file path
    :return: data array
    """
    nib_data = nib.load(path)
    img = nib_data.get_data()
    new_img = img.copy()
    affine = nib_data.affine.copy()
    hdr = nib_data.header.copy()
    return new_img, affine, hdr


def read_spec_file(roots, postfix):
    r"""
    读取文件夹下特定后缀的所有文件，返回一个路径的列表；
    :param roots: a list that contained the root dirs
    :param postfix: the postfix of the file you want to find
    :return: list[path1,path2.......]
    """
    allfiles = []
    for i in roots:
        s = os.walk(i)
        length = -len(postfix)
        for dir, folder, files in s:
            # print(files)
            for j in files:
                if j[length:] == postfix:
                    allfiles.append(os.path.join(dir, j))
    return allfiles


def save_nii(arr, affine, hdr, save_name):
    new_nii = nib.Nifti1Image(arr, affine, hdr)
    nib.save(new_nii, save_name)


def save_nii_sitk(arr, save_name):
    arr = np.swapaxes(arr, 0, 2)
    new_nii = sitk.GetImageFromArray(arr)
    sitk.WriteImage(new_nii, save_name)
