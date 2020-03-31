#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:09:18 2020

@author: kasia
"""

import os
import nibabel as nib
import numpy as np

base_dir = '/Users/kasia/Documents/Topology/100307_ses-1/functional_to_standard/_scan_func-1/_selector_M-SDBVC_BP-B0.009-T0.08'

epi = nib.load(os.path.join(base_dir,'bandpassed_demeaned_filtered_antswarp.nii.gz'))

#to check the size of the frames 
shape = epi.shape
print(shape)

#check the header information
header = epi.header
print(header)  

#matplotlib inline

import matplotlib.pyplot as plt
from nilearn.plotting import plot_epi

epi_data = epi.get_fdata()
print(epi_data.shape)

#average voxel signal intensity across the whole brain for 405 TRs
plt.plot(np.mean(epi_data,axis=(0,1,2)))

epi_mean=epi_data.mean()

# 4D array to 2D voxels by time
n_voxels = np.prod(epi_data.shape[:-1])
vol_shape = epi_data.shape[:-1]
n_voxels = np.prod(vol_shape)
voxel_by_time = epi_data.reshape(n_voxels, epi_data.shape[-1])

import networkx as nx
from kmapper import KeplerMapper, Cover
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

from dyneusr import DyNeuGraph
from dyneusr.tools import visualize_mapper_stages
from dyneusr.mapper.utils import optimize_dbscan

from dyneusr import DyNeuGraph 
from dyneusr.tools import visualize_mapper_stages
from dyneusr.mapper.utils import optimize_dbscan


# # Generate a shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)

# Configure projection
pca = PCA(2, random_state=1)
umap = UMAP(n_components=2, init=pca.fit_transform(voxel_by_time))

# # Construct lens and generate the shape graph
lens = mapper.fit_transform(umap.fit_transform(voxel_by_time, y=None), projection=[0, 1]) 
graph = mapper.map(lens, voxel_by_time=voxel_by_time, cover=Cover(20, 0.5), clusterer=optimize_dbscan(voxel_by_time, k=3, p=100.0), )

import kmapper as km

# Initialize
# mapper = km.KeplerMapper(verbose=1)

# # Fit to and transform the data
# projected_data = mapper.fit_transform(umap.fit_transform(voxel_by_time, y=None), projection=[0,1]) # X-Y axis

# # Create dictionary called 'graph' with nodes, edges and meta-information
# graph = mapper.map(projected_data, voxel_by_time, nr_cubes=10)

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html")