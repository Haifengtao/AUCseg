# AUCseg:An automatically unsupervised clustering toolbox for 3D-segmentation of High-grade gliomas on multi-parametric MR images 
## 1. Introduction
This is an automatically unsupervised clustering toolbox for 3D-segmentation of High-grade gliomas on multi-parametric MR images. 

## 2. Preprocess 
### 2.1 Skull stripping.
Skull will impact the segmentation result.You can do this step by FSL using the folowing command. 
```shell
bet <input> <output> [options]
```

### 2.2 Registration 
Our method will use the multi-parametric MR images, so we need register them. In theory, linear registration is enough.

## 3. Requirements
You can install the external package by the folowing command.
```shell
cd project_path
pip install -r requirements.txt
```
## 4. How to use
you can use the folowing command to get the user guide.
```shell
seg_all_labels.py --help
```
You can run the examples by the folowing command to segment the whole tumor, tumor core, and enhancing tumor.
```shell
python seg_all_labels.py -t1ce ./data/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1_t1ce.nii.gz -flair ./data/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1_flair.nii.gz -s ./data/Brats18_TCIA02_151_1/seg_all.nii.gz -fc 4 -ec 3 -nc_seg_mode t2 -t2 ./data/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1_t2.nii.gz -t2_n 5

python seg_all_labels.py -t1ce ./data/Brats18_TCIA02_171_1/Brats18_TCIA02_171_1_t1ce.nii.gz -flair ./data/Brats18_TCIA02_171_1/Brats18_TCIA02_171_1_flair.nii.gz -s ./data/Brats18_TCIA02_171_1/seg_all.nii.gz -fc 9 -ec 3 -nc_seg_mode cc

python seg_all_labels.py -t1ce ./data/Brats18_TCIA01_231_1/Brats18_TCIA01_231_1_t1ce.nii.gz -flair ./data/Brats18_TCIA01_231_1/Brats18_TCIA01_231_1_flair.nii.gz -s ./data/Brats18_TCIA01_231_1/seg_all.nii.gz -fc 3 -ec 3 -nc_seg_mode cc
```

if you just need the whole tumor
```shell
python seg_wt.py -flair ./data/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1_flair.nii.gz -s ./data/Brats18_TCIA02_151_1/wt.nii.gz -fc 4
```

## 5. How to cite
None


