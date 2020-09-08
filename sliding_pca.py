#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:23:16 2020

@author: kasia
"""

import numpy as np 
import os   
import nibabel as nib
from nilearn.input_data import NiftiMasker

text_file = open("/Users/kasia/Documents/Topology/list.txt", "r")
mask1='/Users/kasia/Documents/Topology/dlPFC_both.nii'
mask2='/Users/kasia/Documents/Topology/dACC_both.nii'
list1 = text_file.readlines()
list2=[]
i = 0

for numb  in list1:
    i +=1   
    epi_dir=os.path.join('/Users/kasia/Documents/Topology',numb,'functional_to_standard/_scan_func-1/_selector_M-SDBVC_BP-B0.009-T0.08/bandpassed_demeaned_filtered_antswarp.nii.gz')
    epi=epi_dir.replace('\n', "")
        
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

ii=0
r=[]
r2=[]
step=10

from sklearn.decomposition import PCA
pca = PCA(1, random_state=1)

tran1=pca.fit_transform(see)
tran2=pca.fit_transform(see2)

import matplotlib.pyplot as plt
plt.plot(tran1, '*')
plt.plot(tran2, '^')


# for ii in range(395):
#     #aa=mask1_sub1[ii:ii+step,:]
#     r.append((np.array(pca.fit_transform((mask1_sub1[ii:ii+step,:])))))
#     r2.append((np.array(pca.fit_transform((mask2_sub1[ii:ii+step,:])))))
  
# final_mask1=np.concatenate(r, axis=1)
# final_mask2=np.concatenate(r2, axis=1)



