import numpy as np
import os
import nibabel as nib
from skimage import measure as ms
import time
from skimage import morphology
import pickle
from sklearn.cluster import KMeans
import FCM

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


def readfile(roots, postfix):
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


class based_kmeans():
    def __init__(self, t1ce, flair, ):
        self.t1ce = t1ce
        self.flair = flair
        self.shape = t1ce.shape
        self.full_label = np.zeros(self.shape)
        self.full_label_bbox = None
        self.all_label = np.zeros(self.shape).astype("uint8")
        self.enhance_label = np.zeros(self.shape)
        self.necrotic_label = np.zeros(self.shape)
        self.enhance_label_bbox = None

    def element_Choose(self, data):
        # kernal = np.ones((3, 3, 3))
        # enhance_data = morphology.opening(data, kernal)  # dilation
        [heart_res, num] = ms.label(data, connectivity=3, return_num=True)
        areas = []
        region = ms.regionprops(heart_res)
        for i in range(num):
            areas.append(region[i].area)
        label_num = areas.index(max(areas)) + 1
        data[heart_res != label_num] = 0
        data[heart_res == label_num] = 1
        full_label_bbox = region[label_num - 1].bbox
        return data, full_label_bbox

    def segment_necrotic(self, data):
        kernal = np.ones((3, 3, 3))
        enhance_data = morphology.dilation(data, kernal)  # dilation

        [heart_res, num] = ms.label(enhance_data, background=-1, connectivity=1, return_num=True)
        areas = []
        region = ms.regionprops(heart_res)
        for i in range(num):
            areas.append(region[i].area)
        index = np.argsort(-np.array(areas))
        label_num1 = index[0] + 1
        necrotic = np.ones(data.shape)
        necrotic[heart_res == label_num1] = 0
        necrotic[enhance_data == 1] = 0
        necrotic = morphology.dilation(necrotic, kernal)
        return necrotic

    def fill_full(self,data):
        c = np.ones((3, 3, 3))
        data = morphology.dilation(data, c)  # dilation
        [heart_res, num] = ms.label(data, connectivity=3, background=-1, return_num=True)
        areas = []
        region = ms.regionprops(heart_res)
        for i in range(num):
            areas.append(region[i].area)
        label_num = np.argsort(-np.array(areas))
        for i in label_num[1:]:
            data[heart_res == (i+1)] = 1
        return data

    def Kmeans_segment(self, ):
        # segment full tumor
        X = (self.flair-self.flair.mean())/self.flair.std()
        model = FCM.siFCM(X, 5)
        model.FCM_cluster()
        full_label = np.zeros(X.shape)
        full_label[model.result == np.argmax(model.C)] = 1
        full_tumor = np.reshape(full_label, self.shape)

        # 填充连通域,获取boundingbox
        self.full_label, self.full_label_bbox = self.element_Choose(full_tumor)
        self.full_label = self.fill_full(self.full_label)
        # 新的分割掩膜

        self.t1ce[self.full_label == 0] = 0
        self.t1ce = self.t1ce[self.full_label_bbox[0]:self.full_label_bbox[3],
                              self.full_label_bbox[1]:self.full_label_bbox[4], self.full_label_bbox[2]:self.full_label_bbox[5]]

        # segment the enhance tumor
        X2 = (self.t1ce - self.t1ce.mean()) / self.t1ce.std()
        model2 = FCM.siFCM(X2, 3)
        model2.FCM_cluster()
        enhance_tumor = np.zeros(X2.shape)
        enhance_tumor[model2.result == np.argmax(model2.C)] = 1
        enhance_tumor = np.reshape(enhance_tumor, self.t1ce.shape)
        # 找到最大连通域的b_box
        enhance_label, _ = self.element_Choose(enhance_tumor)
        self.enhance_label[self.full_label_bbox[0]:self.full_label_bbox[3],
        self.full_label_bbox[1]:self.full_label_bbox[4],
        self.full_label_bbox[2]:self.full_label_bbox[5]] = enhance_label

        # segment the necrotic
        self.necrotic_label = self.segment_necrotic(self.enhance_label)

        self.all_label[self.full_label == 1] = 2
        self.all_label[self.enhance_label == 1] = 4
        self.all_label[self.necrotic_label == 1] = 1


def calcuR_(myseg, groudtruth, label):
    temp1 = np.ones(myseg.shape)
    temp2 = np.zeros(myseg.shape)
    temp3 = np.zeros(myseg.shape)
    temp4 = np.zeros(myseg.shape)
    temp1[myseg == label] = 2
    temp3[myseg == label] = 1
    km = np.sum(temp3)
    temp2[groudtruth == label] = 2
    temp3[groudtruth == label] = 1
    temp4[groudtruth == label] = 1
    gt = np.sum(temp4)
    union = np.sum(temp3)
    intersection = np.sum(temp1 == temp2)
    return [2*intersection/(km+gt), intersection/union, (km-intersection)/gt, (gt-intersection)/gt, intersection/gt]


def calcuResult(myseg, groudtruth):
    dic = {"whole tumor": [], "core tumor": [], "enhancing tumor": [], "necrotic": []}
    # dic = {"dice": 0, "JI": 0, "fp": 0, "fn": 0, "sensitive": 0}

    # enhancing tumor
    dic["necrotic"] = calcuR_(myseg, groudtruth, 1)
    dic["enhancing tumor"] = calcuR_(myseg, groudtruth, 4)

    temp_my = np.zeros(myseg.shape)
    temp_gt = np.zeros(myseg.shape)
    temp_my[(myseg == 1) != (myseg == 4)] = 2
    temp_gt[(groudtruth == 1) != (groudtruth == 4)] = 2
    dic["core tumor"] = calcuR_(temp_my, temp_gt, 2)

    temp_km = np.zeros(myseg.shape)
    temp_gt = np.zeros(myseg.shape)
    temp_km[myseg != 0] = 2
    temp_gt[groudtruth != 0] = 2
    dic["whole tumor"] = calcuR_(temp_km, temp_gt, 2)
    return dic

def run(start,end):
    label_path = readfile(["./data"], "seg.nii")
    t1ce_path = readfile(["./data"], "t1ce.nii")
    flair_path = readfile(["./data"], "flair.nii")
    print("running fcm: ", (start,end))
    print(len(label_path),len(t1ce_path),len(flair_path))
    allresult = {}
    for i in range(start, end):
        print(label_path[i])
        t1ce_img, affine_t1ce, hdr_t1ce = read_nii(t1ce_path[i])
        flair_img, affine_flair, hdr_flair = read_nii(flair_path[i])
        seg_img, affine_seg, hdr_seg = read_nii(label_path[i])

        # segment
        start = time.clock()
        model = based_kmeans(t1ce_img, flair_img)
        model.Kmeans_segment()
        time_consumed = time.clock() - start
        print("time consumed: ", time.clock() - start)
        # 形成新的nii文件
        new_nii = nib.Nifti1Image(model.all_label, affine_seg, hdr_seg)
        target_path = label_path[i][:-7] + "seg_5_fcm.nii"
        nib.save(new_nii, target_path)

        # dice index
        print(calcuResult(model.all_label, seg_img))
        allresult[label_path[i]] = calcuResult(model.all_label, seg_img)
        allresult[label_path[i]]["time_consumed"] = time_consumed
    path = "./result_5_fcm"+str(start)+"_"+str(end)+".pkl"
    with open(path, "wb") as f:
        pickle.dump(allresult, f)

if __name__ == '__main__':
    run(0,30)

