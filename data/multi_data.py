import os
import sys
import json
import fnmatch
import tarfile
import scipy.io as sio
import cv2

from PIL import Image
from glob import glob
import glog as log
from tqdm import tqdm
from six.moves import urllib

import numpy as np
import struct

from utils import loadmat, imwrite, imread

def normalize(arr):
    maxval = np.amax(arr)
    arr = maxval - arr

    minval = np.min(arr[np.nonzero(arr)])
    maxval = np.amax(arr)

    arr -= minval
    arr *= ((maxval)/(maxval-minval))
    minval = np.amin(arr)
    maxval = np.amax(arr)

    log.info("MIN: "+str(minval))
    log.info("MAX: "+str(maxval))

    arr_idx = arr <= minval
    arr[arr_idx] =  1.0

    return arr

class DataLoader(object):
    def __init__(self, config, rng=None):
        self.rng = np.random.RandomState(1) if rng is None else rng

        self.data_path = os.path.join(config.data_dir, 'multi')
        self.hand_data_path = os.path.join(self.data_path, config.hand_data_dir)
        self.joint_data_path = os.path.join(self.hand_data_path, config.joint_data_dir)
        self.sample_path = os.path.join(self.data_path, config.sample_dir)
        self.batch_size = config.batch_size
        self.debug = config.debug

        #self.real_data, synthetic_image_path = load(config, self.joint_data_path, rng)
        #===============================cropped real/synthetic images====================
        #read real image data paths and store them into array
        self.bin_real_path = os.path.join(self.hand_data_path, "test")
        self.real_data_paths = np.array(glob(os.path.join(self.bin_real_path, '*.bin')))
        #self.real_data_dims = list(imread(self.real_data_paths[0]).shape) + [1]
        #read synthetic image data paths and store them into array
        self.bin_synth_path = os.path.join(self.hand_data_path, "test")
        self.synthetic_data_paths = np.array(glob(os.path.join(self.bin_synth_path, '*.bin')))
        print(self.synthetic_data_paths[0])
        self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0]).shape) + [1]
        #read joint information in the matrix

        self.joint_mat = np.arange(len(self.synthetic_data_paths))
        sio.savemat(self.joint_data_path,{'vect':self.joint_mat})

        print "Processing real hand dataset..."
        self.real_data = []
        self.new_datapath = os.path.join(self.hand_data_path, "new_real_data")
        if not os.path.exists(self.new_datapath):
            os.mkdir(self.new_datapath)
            counter = 0
            #for img_path in tqdm(self.real_data_paths):
            for img_path in self.real_data_paths:
              im_new = imread(img_path,length=11,real=False)
              im = normalize(im_new)
              imwrite(self.new_datapath + "/data_" + str(counter) + ".bin",im)
              cv2.imshow("real",im)
              cv2.waitKey(1)
              counter += 1
              self.real_data.append(im)
              if counter == 1000:
                break
        else:
            for img_path in self.real_data_paths:
                im = imread(img_path)
                #im = normalize(imread(img_path))
                self.real_data.append(im)
                break
        self.real_data_dims = list(self.real_data[0].shape) + [1]


        print "Processing synthetic hand dataset..."
        self.new_datapath = os.path.join(self.hand_data_path,"new_synt_data")
        if not os.path.exists(self.new_datapath):
            os.mkdir(self.new_datapath)
            counter = 0
            for img_path in self.synthetic_data_paths:
              im_new = imread(img_path)
              im = normalize(im_new)
              #print self.new_datapath + "/data_" + str(counter) + ".bin"
              imwrite(self.new_datapath + "/data_" + str(counter) + ".bin",im)
              print "writing to path: " + str(img_path)
              cv2.imshow("syn",im)
              cv2.waitKey(1)
              counter += 1
              if counter == 1000:
                break

        #print self.new_datapath
        self.synthetic_data_paths = np.array(glob(os.path.join(self.new_datapath, '*.bin')))
        #print self.synthetic_data_paths
        #input("pause")
        tmp = imread(self.synthetic_data_paths[0])
        print(tmp.shape)
        self.synthetic_data_dims = list(tmp.shape) + [1]
        self.synthetic_data_paths.sort()

        #if np.rank(self.real_data) == 3:
        if np.ndim(self.real_data) == 3:
            self.real_data = np.expand_dims(self.real_data, -1)

        self.real_p = 0

        print '[*] # of real data : {}, # of synthetic data : {}'. \
            format(len(self.real_data), len(self.synthetic_data_paths))


    def get_observation_size(self):
        return self.real_data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.real_p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size#batch_size set to 50 for hand and 512 for eyes

        if self.real_p == 0:
            inds = self.rng.permutation(self.real_data.shape[0])
            self.real_data = self.real_data[inds]

        if self.real_p + n > self.real_data.shape[0]:
            self.reset()

        x = self.real_data[self.real_p: self.real_p + n]
        self.real_p += self.batch_size

        return x

    next = __next__
