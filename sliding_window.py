#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:18:30 2020

@author: kasia
"""
import numpy as np 
import os   
import nibabel as nib
import matplotlib as plt
from nilearn.input_data import NiftiMasker

text_file = open("/Users/kasia/Documents/Topology/list.txt", "r")
mask1='/Users/kasia/Documents/Topology/dlPFC_both.nii'
mask2='/Users/kasia/Documents/Topology/dACC_both.nii'
list1 = text_file.readlines()
list2=[]
i = 0

# for j in range(5):
#     i += 1
for numb  in list1:
    i +=1   
    epi_dir=os.path.join('/Users/kasia/Documents/Topology',numb,'functional_to_standard/_scan_func-1/_selector_M-SDBVC_BP-B0.009-T0.08/bandpassed_demeaned_filtered_antswarp.nii.gz')
    epi=epi_dir.replace('\n', "")
    # epi=nib.load(epi)
    # epi_data = epi.get_fdata()
    # n_voxels = np.prod(epi_data.shape[:-1])
    # vol_shape = epi_data.shape[:-1]
    # n_voxels = np.prod(vol_shape)
        
    # voxel_by_time = epi_data.reshape(n_voxels, epi_data.shape[-1])
    # X=voxel_by_time.T
        
    masker = NiftiMasker(
    mask1, 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=0.72,
    memory="nilearn_cache")
    X = masker.fit_transform(epi)
    
    masker2 = NiftiMasker(
        mask2, 
        standardize=True, detrend=True, smoothing_fwhm=4.0,
        low_pass=0.09, high_pass=0.008, t_r=0.72,
        memory="nilearn_cache")
    X2 = masker2.fit_transform(epi)
        
    globals()["mask1_sub" + str(i)] = X
    globals()["mask2_sub" + str(i)] = X2
    print(numb)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

list=mask1_sub1.T
list2=mask2_sub1.T
result=rolling_window(list, 10)
result2=rolling_window(list2, 10)
see=np.concatenate(result, axis=1)
see2=np.concatenate(result2, axis=1)
try1=np.diff(see)
try2=np.diff(see2)

from sklearn.decomposition import PCA
pca = PCA(1, random_state=1)
tran1=pca.fit_transform(try1)
tran2=pca.fit_transform(try2)

import matplotlib.pyplot as plt
plt.plot(tran1, '*')
plt.plot(tran2, '*')


