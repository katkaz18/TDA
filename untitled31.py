#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:47:43 2021

@author: kasia
"""


import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import nibabel as nib


epi2='/Users/kasia/Documents/Topology/brain_mag100-2.nii.gz'

epi = nib.load(epi2)

epi_data = epi.get_fdata()

epi_data.shape

X = epi_data.reshape((np.prod(epi_data.shape[:-1]), epi_data.shape[-1]))

X = X.T

dists = cdist(X, X)

reducer = umap.UMAP(metric='precomputed')

projection = reducer.fit_transform(dists)

total_intensity = X.sum(axis=1)

plt.scatter(*projection.T, c=total_intensity)

time_range = np.zeros(len(X))


plt.plot(np.arange(10), 0.5*np.arange(10))


#plt.plot(total_intensity * time_B)

cond_A = np.concatenate((np.arange(1,6), np.arange(34,40), np.arange(57,63)))
cond_B = np.concatenate((np.arange(12,18), np.arange(23,29),
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

dists[cond_rest][:, cond_rest].max()

dists[cond_A][:, cond_B].mean()

dists[cond_A][:, cond_A].mean()

plt.scatter(*projection.T)
plt.scatter(*projection[cond_A].T)
plt.scatter(*projection[cond_B].T)

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(*projection.T)

for i, point in enumerate(projection):
    ax.annotate(i, point)



projection[cond_A]


