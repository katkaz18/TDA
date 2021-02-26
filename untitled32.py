#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:03:55 2021

@author: kasia
"""

import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import nibabel as nib

epi2='/Users/kasia/Documents/Topology/brain_mag100_2reg.nii.gz'

#epi2='/Users/kasia/Documents/Topology/brain_mag100-2.nii'

epi = nib.load(epi2)

epi_data = epi.get_fdata()

epi_data.shape

from skimage.filters import gaussian


epi_data.shape

X=epi_data.T
for index, volume in enumerate(X):
    X[index]=gaussian(volume,sigma=4)

X=X.reshape(X.shape[0], -1)
X.shape

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
X=np.arcsinh(X/5)

time_range = np.zeros(len(X))
#cond_A = np.concatenate((np.arange(1,6), np.arange(23,29), np.arange(57,63)))
#cond_B = np.concatenate((np.arange(12,18), np.arange(34,40),np.arange(45,51)))
    
#cond_A = np.concatenate((np.arange(1,6), np.arange(34,40), np.arange(57,63)))
#cond_B = np.concatenate((np.arange(12,18), np.arange(23,29), np.arange(45,51)))

cond_A = np.concatenate((np.arange(1,6), np.arange(12,18), np.arange(23,29)))
cond_B = np.concatenate((np.arange(34,40), np.arange(46,52), np.arange(57,63)))

cond_A=cond_A+3
cond_B=cond_B+3

from sklearn.model_selection import train_test_split
A_train, A_test = train_test_split(cond_A, test_size=0.2)

A_train

x = cdist(X, np.mean(X[A_train], axis=0)[None,:])
y = cdist(X, np.mean(X[cond_B], axis=0)[None,:])

plt.scatter(x,y)
plt.scatter(x[A_test], y[A_test])

total_intensity = X.sum(axis=1)
time_range = np.zeros(len(X))
cond_A = np.concatenate((np.arange(1,6), np.arange(23,29), np.arange(57,63)))
cond_B = np.concatenate((np.arange(12,18), np.arange(34,40),
                         np.arange(45,51)))

time_A = time_range.copy()
time_A[cond_A] = 1
time_B = time_range.copy()
time_B[cond_B] = 1


cond_rest = np.arange(len(X))
cond_rest = np.setdiff1d(cond_rest, cond_A)
cond_rest = np.setdiff1d(cond_rest, cond_B)

plt.plot(total_intensity)
plt.plot(cond_A, total_intensity[cond_A])
plt.plot(cond_B, total_intensity[cond_B])