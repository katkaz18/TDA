#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:45:11 2021

@author: kasia
"""

import numpy as np 
import pandas as pd
import networkx as nx

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt

epi2='/Users/kasia/Documents/Topology/brain_mag100-2.nii.gz'

masker = NiftiMasker(
    standardize=True, detrend=True, low_pass=0.09, high_pass=0.008,
     t_r=2, smoothing_fwhm=4.0,
    memory="nilearn_cache")
X = masker.fit_transform(epi2)

# Configure projection
pca2 = PCA(2, random_state=1)
umap2 = UMAP(n_components=2, init=pca2.fit_transform(X))

transform=umap2.fit_transform(X, y=None)
cond_A=np.concatenate((transform[1:6,:], transform[34:40,:],transform[57:63,:]))
cond_B=np.concatenate((transform[12:18,:], transform[23:29,:],transform[45:51,:]))
plt.scatter(*transform.T)
plt.scatter(*cond_A.T)
plt.scatter(*cond_B.T)

#, smoothing_fwhm=4.0 low_pass=0.09, high_pass=0.008,