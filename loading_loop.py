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

    globals()["w" + str(i)] = X
    globals()["v" + str(i)] = X2
    print(i)
    
all_ACC=[]
all_PFC=[]

for ii in range(len(w1)):
       
    all_ACC.append(np.vstack((w1[ii,:], w2[ii,:], w3[ii,:], w4[ii,:], w5[ii,:])))
    all_PFC.append(np.vstack((v1[ii,:], v2[ii,:], v3[ii,:], v4[ii,:], v5[ii,:])))

#concatenated results
all_subjects_ACC=np.concatenate(all_ACC, axis=0)
all_subjects_PFC=np.concatenate(all_PFC, axis=0)

from sklearn.decomposition import PCA
pca = PCA(1, random_state=1)
tran1=pca.fit_transform(all_subjects_ACC)
tran2=pca.fit_transform(all_subjects_PFC)

import matplotlib.pyplot as plt
plt.plot(tran1, '^')
plt.plot(tran2, '*')
