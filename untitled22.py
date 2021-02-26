5
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:51:59 2020

@author: kasia
"""

import numpy as np 
import os

# path='/Users/kasia/Documents/Topology/'
path='/Users/kasia/Documents/Topology/100307_preproc/MNINonLinear/Results/tfMRI_WM_LR/'
epi=os.path.join(path, 'tfMRI_WM_LR.nii.gz')

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      epi)  # 4D data

from nilearn.regions import Parcellations

# Computing ward for the first time, will be long... This can be seen by
# measuring using time
import time
start = time.time()

# Agglomerative Clustering: ward

# We build parameters of our own for this object. Parameters related to
# masking, caching and defining number of clusters and specific parcellations
# method.
ward = Parcellations(method='ward', n_parcels=1000,
                     standardize=False, smoothing_fwhm=2.,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
# Call fit on functional dataset: single subject (less samples).
ward.fit(epi)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# We compute now ward clustering with 2000 clusters and compare
# time with 1000 clusters. To see the benefits of caching for second time.

# We initialize class again with n_parcels=2000 this time.
start = time.time()
ward = Parcellations(method='ward', n_parcels=2000,
                     standardize=False, smoothing_fwhm=1.,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
ward.fit(epi)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

ward_labels_img = ward.labels_img_

# Now, ward_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
ward_labels_img.to_filename('ward_parcellation.nii.gz')

from nilearn import plotting
from nilearn.image import mean_img, index_img

first_plot = plotting.plot_roi(ward_labels_img, title="Ward parcellation",
                               display_mode='xz')

# Grab cut coordinates from this plot to use as a common for all plots
cut_coords = first_plot.cut_coords

# Grab number of voxels from attribute mask image (mask_img_).

from nilearn.image import get_data
original_voxels = np.sum(get_data(ward.mask_img_))

# Compute mean over time on the functional image to use the mean
# image for compressed representation comparisons
mean_func_img = mean_img(epi)

# Compute common vmin and vmax
vmin = np.min(get_data(mean_func_img))
vmax = np.max(get_data(mean_func_img))

plotting.plot_epi(mean_func_img, cut_coords=cut_coords,
                  title='Original (%i voxels)' % original_voxels,
                  vmax=vmax, vmin=vmin, display_mode='xz')

# A reduced dataset can be created by taking the parcel-level average:
# Note that Parcellation objects with any method have the opportunity to
# use a `transform` call that modifies input features. Here it reduces their
# dimension. Note that we `fit` before calling a `transform` so that average
# signals can be created on the brain parcellations with fit call.
fmri_reduced = ward.transform(epi)

# Display the corresponding data compressed using the parcellation using
# parcels=2000.
fmri_compressed = ward.inverse_transform(fmri_reduced)

plotting.plot_epi(index_img(fmri_compressed, 0),
                  cut_coords=cut_coords,
                  title='Ward compressed representation (2000 parcels)',
                  vmin=vmin, vmax=vmax, display_mode='xz')
# As you can see below, this approximation is almost good, although there
# are only 2000 parcels, instead of the original 60000 voxels

# class/functions can be used here as they are already imported above.

# This object uses method='kmeans' for KMeans clustering with 10mm smoothing
# and standardization ON
start = time.time()
kmeans = Parcellations(method='kmeans', n_parcels=50,
                       standardize=True, smoothing_fwhm=10.,
                       memory='nilearn_cache', memory_level=1,
                       verbose=1)
# Call fit on functional dataset: single subject (less samples)
kmeans.fit(epi)

kmeans_labels_img = kmeans.labels_img_

plotting.plot_roi(kmeans_labels_img, mean_func_img,
                  title="KMeans parcellation",
                  display_mode='xz')

# kmeans_labels_img is a Nifti1Image object, it can be saved to file with
# the following code:
kmeans_labels_img.to_filename('kmeans_parcellation.nii.gz')

start = time.time()
rena = Parcellations(method='rena', n_parcels=5000, standardize=False,
                     smoothing_fwhm=2., scaling=True)

rena.fit_transform(epi)
print("ReNA 5000 clusters: %.2fs" % (time.time() - start))

rena_labels_img = rena.labels_img_

# Now, rena_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
rena_labels_img.to_filename('rena_parcellation.nii.gz')

plotting.plot_roi(ward_labels_img, title="ReNA parcellation",
                  display_mode='xz', cut_coords=cut_coords)

# Display the original data
plotting.plot_epi(mean_func_img, cut_coords=cut_coords,
                  title='Original (%i voxels)' % original_voxels,
                  vmax=vmax, vmin=vmin, display_mode='xz')

# A reduced data can be created by taking the parcel-level average:
# Note that, as many scikit-learn objects, the ReNA object exposes
# a transform method that modifies input features. Here it reduces their
# dimension.
# However, the data are in one single large 4D image, we need to use
# index_img to do the split easily:
fmri_reduced_rena = rena.transform(epi)

# Display the corresponding data compression using the parcellation
compressed_img_rena = rena.inverse_transform(fmri_reduced_rena)

plotting.plot_epi(index_img(compressed_img_rena, 0), cut_coords=cut_coords,
                  title='ReNA compressed representation (5000 parcels)',
                  vmin=vmin, vmax=vmax, display_mode='xz')

