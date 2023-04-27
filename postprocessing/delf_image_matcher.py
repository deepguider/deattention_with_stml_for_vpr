#!/usr/bin/env python
# coding: utf-8
# ##### Copyright 2018 The TensorFlow Hub Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# # How to match images using DELF and TensorFlow Hub
# TensorFlow Hub (TF-Hub) is a platform to share machine learning expertise packaged in reusable resources, notably pre-trained **modules**.
# In this colab, we will use a module that packages the [DELF](https://github.com/tensorflow/models/tree/master/research/delf) neural network and logic for processing images to identify keypoints and their descriptors. The weights of the neural network were trained on images of landmarks as described in [this paper](https://arxiv.org/abs/1612.06321).

#get_ipython().system('pip install scikit-image')

from absl import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen

from ipdb import set_trace as bp
from tictoc import tic,toc

# ## The data
# In the next cell, we specify the URLs of two images we would like to process with DELF in order to match and compare them.

def download_and_resize(name=None, url=None, new_width=256, new_height=256):
    # Download, resize, save and display the images.
    if url is not None:
        path = tf.keras.utils.get_file(url.split('/')[-1], url)
    else:
        path = name
    image = Image.open(path)
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image

#@title Choose images
def get_sample():
    images = "Bridge of Sighs" #@param ["Bridge of Sighs", "Golden Gate", "Acropolis", "Eiffel tower"]
    if images == "Bridge of Sighs":
        # from: https://commons.wikimedia.org/wiki/File:Bridge_of_Sighs,_Oxford.jpg
        # by: N.H. Fischer
        IMAGE_1_URL = 'https://upload.wikimedia.org/wikipedia/commons/2/28/Bridge_of_Sighs%2C_Oxford.jpg'
        # from https://commons.wikimedia.org/wiki/File:The_Bridge_of_Sighs_and_Sheldonian_Theatre,_Oxford.jpg
        # by: Matthew Hoser
        IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/c/c3/The_Bridge_of_Sighs_and_Sheldonian_Theatre%2C_Oxford.jpg'
    elif images == "Golden Gate":
        IMAGE_1_URL = 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Golden_gate2.jpg'
        IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/3/3e/GoldenGateBridge.jpg'
    elif images == "Acropolis":
        IMAGE_1_URL = 'https://upload.wikimedia.org/wikipedia/commons/c/ce/2006_01_21_Ath%C3%A8nes_Parth%C3%A9non.JPG'
        IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/5/5c/ACROPOLIS_1969_-_panoramio_-_jean_melis.jpg'
    else:
        IMAGE_1_URL = 'https://upload.wikimedia.org/wikipedia/commons/d/d8/Eiffel_Tower%2C_November_15%2C_2011.jpg'
        IMAGE_2_URL = 'https://upload.wikimedia.org/wikipedia/commons/a/a8/Eiffel_Tower_from_immediately_beside_it%2C_Paris_May_2008.jpg'

    return IMAGE_1_URL, IMAGE_2_URL

def show_two_inputs(self, image1, image2):
    plt.subplot(1,2,1)
    plt.imshow(image1)
    plt.subplot(1,2,2)
    plt.imshow(image2)
    plt.pause(0.001)


class delf_image_matcher():
    def __init__(self, device='/device:GPU:0'):
        self.device = device
        #self.strategy = tf.distribute.MirroredStrategy()
        with tf.device(self.device):
        #with self.strategy.scope():
            #tf.compat.v1.disable_eager_execution()   # it doesn't work fine. for multi-gpu usage, https://lv99.tistory.com/12
            #self.delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
            ## Download delf_1.tar.gz from https://tfhub.dev/google/delf/1 , and unzip it into your local path
            self.delf = hub.load('/home/ccsmm/workdir/delf_using_tfLib/hub_data/delf_1').signatures['default']
            #tf.config.optimizer.set_jit(True) # Enable XLA. : Do not use this because run_delf() will be slower and it need more g-memory. 
            tf.config.optimizer.set_jit(False) # Disable XLA. Default.

    def run_delf(self, image):
        # ## Apply the DELF module to the data
        # The DELF module takes an image as input and will describe noteworthy points with vectors. The following cell contains the core of this colab's logic.
        with tf.device(self.device):
        #with self.strategy.scope():
            np_image = np.array(image)
            float_image = tf.image.convert_image_dtype(np_image, tf.float32)
            result = self.delf(
                image=float_image,
                score_threshold=tf.constant(100.0),
                image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
                max_feature_num=tf.constant(1000))
        return result

    def set_image1(self, image1):
        self.image1 = image1
        self.result1 = self.run_delf(image1)

    def set_image2(self, image2):
        self.image2 = image2
        self.result2 = self.run_delf(image2)

    def match_images(self, image1=None, image2=None, knn_mode='kdtree', ransac_en=True, disp_en=False, verbose=False):
        if image1 == None:
            result1 = self.result1
            image1 = self.image1
        else:
            result1 = self.run_delf(image1)

        if image2 == None:
            image2 = self.image2
            result2 = self.result2
        else:
            result2 = self.run_delf(image2)

        # 1 means db's descriptors, 2 means query's descriptors
        #title TensorFlow is not needed for this post-processing and visualization
        distance_threshold = 0.8
        
        # Read features.
        num_features_1 = result1['locations'].shape[0]
        if verbose: print("Loaded image 1's %d features" % num_features_1)
        
        num_features_2 = result2['locations'].shape[0]
        if verbose: print("Loaded image 2's %d features" % num_features_2)
        
        # Find nearest-neighbor matches using a KD tree.
        ## result1['descriptors'].shape : [233 , 40], [num_features, depth_of_feature]
        ## result2['descriptors'].shape : [262 , 40]
        if knn_mode.lower() == 'kdtree':
            d1_tree = cKDTree(result1['descriptors'])
            _, indices = d1_tree.query( result2['descriptors'], distance_upper_bound=distance_threshold)
        elif knn_mode.lower() == 'knn':
            knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
            knn.fit(result1['descriptors'])
            distance , indices = knn.kneighbors(result2['descriptors'])
            indices[distance > distance_threshold] = num_features_1
            indices = indices.reshape(-1)
        
        # Select feature locations for putative matches.
        locations_2_to_use = np.array([ result2['locations'][i,] for i in range(num_features_2) if indices[i] != num_features_1 ])
        locations_1_to_use = np.array([ result1['locations'][indices[i],] for i in range(num_features_2) if indices[i] != num_features_1 ])

        # Perform geometric verification using RANSAC.
        inliers = None
        if ransac_en == True:
            try:
                _, inliers = ransac(
                    (locations_1_to_use, locations_2_to_use),
                    AffineTransform,
                    min_samples=3,
                    residual_threshold=20,
                    max_trials=1000)
                    # skimage.transform.AffineTransform
                    #   Bases: skimage.transform._geometric.ProjectiveTransform
                    #       2D affine transformation of the form:
                    #           X = a0*x + a1*y + a2 =
                    #             = sx*x*cos(rotation) - sy*y*sin(rotation + shear) + a2
                    #           Y = b0*x + b1*y + b2 =
                    #             = sx*x*sin(rotation) + sy*y*cos(rotation + shear) + b2
                    #           where sx and sy are zoom factors in the x and y directions,
                    #           and the homogeneous transformation matrix is:
                    #           [[a0  a1  a2]
                    #            [b0  b1  b2]
                    #            [0   0    1]]
            except:
                try:
                    if len(locations_2_to_use) > 0 and len(locations_1_to_use) > 0 :
                        inliers = locations_2_to_use[:,0]==locations_2_to_use[:,0]  # [True, True, ... , True]
                except:
                    bp()
        else:
            try:
                if len(locations_2_to_use) > 0 and len(locations_1_to_use) > 0 :
                    inliers = locations_2_to_use[:,0]==locations_2_to_use[:,0]  # [True, True, ... , True]
            except:
                bp()

        if inliers is not None:
            if verbose: print('Found %d inliers' % sum(inliers))
            if (disp_en == True):
                # Visualize correspondences.
                inlier_idxs = np.nonzero(inliers)[0]

                fig = plt.figure(1)
                fig.clf()
                ax1 = fig.add_subplot(211)
                plot_matches(
                    ax1,
                    image1,
                    image2,
                    locations_1_to_use,
                    locations_2_to_use,
                    np.array([[0,0]]),
                    keypoints_color='b',
                    matches_color='gray')
                ax1.axis('off')
                ax1.set_title('DELF keypoints')

                ax2 = fig.add_subplot(212)
                plot_matches(
                    ax2,
                    image1,
                    image2,
                    locations_1_to_use,
                    locations_2_to_use,
                    np.column_stack((inlier_idxs, inlier_idxs)),
                    keypoints_color='b',
                    matches_color='g')
                ax2.axis('off')
                ax2.set_title('DELF correspondences')

                plt.draw()
                plt.pause(0.1)
            return sum(inliers)
        else:
            return 0


if __name__ == '__main__':

    imatch = delf_image_matcher(device='/device:GPU:3')

    db_url, q_url = get_sample()
    db_img = download_and_resize(None, db_url)
    q_img = download_and_resize(None, q_url)
    bp()
    #show_inputs(db_img, q_img)

    # ## Use the locations and description vectors to match the images
    for i in range(1):
        tic()
        imatch.match_images(db_img, q_img, 'kdtree', disp_en=True)
        toc()
        bp()
