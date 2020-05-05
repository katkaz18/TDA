#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:46:57 2020

@author: kasia

"""

import os
import nibabel as nib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import nilearn

from nilearn.datasets import fetch_haxby 
from nilearn.input_data import NiftiMasker

from kmapper import KeplerMapper, Cover
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from dyneusr import DyNeuGraph
from dyneusr.tools import visualize_mapper_stages
from dyneusr.mapper.utils import optimize_dbscan


base_dir ='/Users/kasia/Documents/Topology/'

df = pd.read_csv("directories.csv")

x = np.arange(len(df))

line='101915_ses-1'

#for n in x:
    #line=df.loc[n,"FolderName"]

#from nilearn import datasets

# By default 2nd subject will be fetched
#haxby_dataset = datasets.fetch_haxby()
epi = nib.load(os.path.join(base_dir, line,'functional_to_standard/_scan_func-1/_selector_M-SDBVC_BP-B0.009-T0.08/bandpassed_demeaned_filtered_antswarp.nii.gz'))
epi2='/Users/kasia/Documents/Topology/101915_ses-1/functional_to_standard/_scan_func-1/_selector_M-SDBVC_BP-B0.009-T0.08/bandpassed_demeaned_filtered_antswarp.nii.gz'

mask=nib.load(os.path.join(base_dir, line, 'anatomical_gm_mask/segment_seg_1_maths_maths.nii'))

# mask_data=mask.get_fdata()

epi_data = epi.get_fdata()

from nilearn.image import resample_to_img
resampled_stat_img = resample_to_img(epi, mask)

#4D array to 2D voxels by time
# n_voxels = np.prod(epi_data.shape[:-1])
# vol_shape = epi_data.shape[:-1]
# n_voxels = np.prod(vol_shape)
# voxel_by_time = epi_data.reshape(n_voxels, epi_data.shape[-1])

# X=voxel_by_time.T

gM=nilearn.masking.compute_gray_matter_mask(mask, threshold=0.5, connected=True, opening=2, memory=None, verbose=0)

masker = NiftiMasker(mask_strategy='template',detrend=True)
masker3 = NiftiMasker(
    mask, 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    t_r=0.72,
    memory="nilearn_cache")

masker = NiftiMasker(
    dataset.mask_vt[0], 
    standardize=True, detrend=True, smoothing_fwhm=4.0,
    low_pass=0.09, high_pass=0.008, t_r=0.72,
    memory="nilearn_cache")
X = masker.fit_transform(epi2)
X3=masker3.fit_transform(epi2)


from nilearn.plotting import plot_roi, plot_epi, show
plot_roi(masker3.mask_img_, mask, title='Mask')

#Generate a shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)
# Configure projection
pca2 = PCA(2, random_state=1)
umap2 = UMAP(n_components=2, init=pca2.fit_transform(X2))

#matplotlib.pyplot.scatter(*umap2.T)

# Construct lens and generate the shape graph
lens = mapper.fit_transform(
        umap2.fit_transform(X, y=None), 
        projection=[0, 1]) 
graph = mapper.map(
        lens, X=X, 
        cover=Cover(20, 0.5),
        clusterer=optimize_dbscan(X, k=3, p=100.0), )
    
    

# Convert to a DyNeuGraph
dG = DyNeuGraph(G=graph )

# Define some custom_layouts
dG.add_custom_layout(lens, name='lens')
dG.add_custom_layout(nx.spring_layout, name='nx.spring')
dG.add_custom_layout(nx.kamada_kawai_layout, name='nx.kamada_kawai')
dG.add_custom_layout(nx.spectral_layout, name='nx.spectral')
dG.add_custom_layout(nx.circular_layout, name='nx.circular')

# Configure some projections
pca = PCA(2, random_state=1)
tsne = TSNE(2, init='pca', random_state=1)
umap = UMAP(n_components=2, init=pca.fit_transform(X))

# Add projections as custom_layouts
dG.add_custom_layout(pca.fit_transform(X), name='PCA')
dG.add_custom_layout(tsne.fit_transform(X), name='TSNE')
dG.add_custom_layout(umap.fit_transform(X, y=None), name='UMAP')
#dG.add_custom_layout(umap.fit_transform(X, y=target), name='Supervised UMAP')

# Visualize 
dG.visualize(static=True, show=True)
    
   
