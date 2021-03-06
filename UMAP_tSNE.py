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


base_dir ='/Users/kasia/Documents/Topology/001'

#line='101309_ses-1'

#epi =nib.load(os.path.join(base_dir, '/Users/kasia/Documents/Topology/001/multi_scan.nii'))
epi2='/Users/kasia/Documents/Topology/brain_mag100-2.nii.gz'
epi=nib.load(epi2)

epi_data = epi.get_fdata()
n_voxels = np.prod(epi_data.shape[:-1])
vol_shape = epi_data.shape[:-1]
n_voxels = np.prod(vol_shape)
voxel= epi_data.reshape(n_voxels, epi_data.shape[-1])
voxels=voxel.T


import pandas as pd
dataset='/Users/kasia/Documents/Topology/labels2.csv'
df = pd.read_csv(dataset, sep=' ')
target, labels = pd.factorize(df.labels.values)
y = pd.DataFrame({l:(target==i).astype(int) for i,l in enumerate(labels)})



#Generate a shape graph using KeplerMapper
mapper = KeplerMapper(verbose=1)

# Configure projection
pca2 = PCA(2, random_state=1)
umap2 = UMAP(n_components=2, init=pca2.fit_transform(X))



umap3= UMAP()
umap4=umap2.fit_transform(X, y=None)


# Construct lens and generate the shape graph
lens = mapper.fit_transform(
        umap2.fit_transform(X, y=None), 
        projection=[0, 1]) 

#graph = mapper.map(lens, X, cover=Cover(20, 0.5), clusterer=DBSCAN(eps=20.))

graph = mapper.map(
        lens, X=X, 
        cover=Cover(46, 0.5),
        clusterer=optimize_dbscan(X, k=3, p=100.0), )
    
transform=umap2.fit_transform(X, y=None)
cond_A=np.concatenate((transform[1:6,:], transform[34:40,:],transform[57:63,:]))
cond_B=np.concatenate((transform[12:18,:], transform[23:29,:],transform[45:51,:]))
matplotlib.pyplot.scatter(*transform.T)
matplotlib.pyplot.scatter(*cond_A.T)
matplotlib.pyplot.scatter(*cond_B.T)



# matplotlib.pyplot.plot(YY, '*')

# Convert to a DyNeuGraph
dG = DyNeuGraph(G=graph)


# Visualize 
dG.visualize(static=True, show=True)
    
   
