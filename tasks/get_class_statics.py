from __future__ import print_function
from math import ceil
from os.path import join, exists
from os import makedirs
import os
import numpy as np
import h5py
import faiss
import pickle

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F


from utils import misc  # This code is called by ../main.py. So working directory is ../

from ipdb import set_trace as bp
import sys;sys.path.insert(0,'/home/ccsmm/workdir/ccsmmutils');import torch_img_utils as tim; tim.init()
import matplotlib.pyplot as plt


'''
data=torch.Tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]).unsqueeze(0)
ipdb> data
tensor([[ 1.,  2.,  3.,  4.,  5.],
        [ 6.,  7.,  8.,  9., 10.],
        [11., 12., 13., 14., 15.]])
ipdb> data.view(-1)
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
        15.])

ipdb> data.shape
torch.Size([1, 3, 5])

    data.view(-1)[7] : 8
    data[0, 1, 2] : 8
    data[0, y, x], 

    rand_num = 7
    y = int(7/5) = 1
    x = 7 - y*5 = 2
    data[0, y, x]
'''

def save_data(opt, dbClass_simple, dbClass_category_id, dbClass_id, fname="dbClass.pickle"):
    import pickle
    data = {
            "dbClass_simple":dbClass_simple,
            "dbClass_category_id":dbClass_category_id,
            "dbClass_id":dbClass_id
            }
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_data(opt, fname="dbClass.pickle"):
    import pickle
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        dbClass_simple = data["dbClass_simple"]
        dbClass_category_id = data["dbClass_category_id"]
        dbClass_id = data["dbClass_id"]

    except:
        dbClass_simple, dbClass_category_id, dbClass_id = [], [], []
    return dbClass_simple, dbClass_category_id, dbClass_id

def class_statics(opt, seg_model, mDatasetStruct):
    _, _, _, cluster_set, _ = mDatasetStruct.get_dataset()
    cuda, device = misc.get_device(opt)
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    with torch.no_grad():
        print('   ===> Extracting Classes')
        dbClass_simple = np.ones(nDescriptors)*99
        dbClass_category_id = np.ones(nDescriptors)*99
        dbClass_id = np.ones(nDescriptors)*99
        
        for iteration, (input, indices) in enumerate(data_loader, 1):
            input = input.to(device)
            seg_simple, seg_category_id, seg_id = seg_model.get_seg_class(input.detach()) # Ground Truth of segmentation, image shape [batchsize, h, w]

            seg_simple = seg_simple.view(seg_simple.size(0), -1)
            seg_category_id = seg_category_id.view(seg_category_id.size(0), -1)
            seg_id = seg_id.view(seg_id.size(0), -1)

            batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
            for ix in range(seg_simple.size(0)):
                # sample different location for each image in batch
                sample = np.random.choice(seg_simple.size(1), nPerImage, replace=False)
                startix = batchix + ix*nPerImage
                dbClass_simple[startix:startix+nPerImage] = seg_simple[ix, sample].detach().cpu().numpy()
                dbClass_category_id[startix:startix+nPerImage] = seg_category_id[ix, sample].detach().cpu().numpy()
                dbClass_id[startix:startix+nPerImage] = seg_id[ix, sample].detach().cpu().numpy()

            if iteration % 50 == 0 or len(data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, ceil(nIm/opt.cacheBatchSize)), flush=True)
            del input, seg_simple

    dataset = cluster_set.dataset
    print('   ===> Indexing class information : {}'.format(dataset))
    print('   ===> Done!')
    return dbClass_simple, dbClass_category_id, dbClass_id

def histogram(dbClass, color="green"):
    cls = dbClass.copy()
    cls_bins = np.unique(cls)
    cls_bins_p1 = np.append(cls_bins, cls_bins[-1]+1)
    '''
    All but the last (righthand-most) bin is half-open. In other words, if bins is:
        [1, 2, 3, 4]
    then the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3).
    The last bin, however, is [3, 4], which includes 4.
    So we need to extra last one element which is larger than last one.
    '''
    ## Histogram using numpy
    freq, bins = np.histogram(dbClass, cls_bins_p1-0.5)
    bins = cls_bins[:-1]

    ## Histogram using matplotlib
    plt.hist(dbClass, cls_bins_p1-0.5, rwidth = 0.8, color = color, alpha = 0.5)
    #plt.bar(freq, bins, rwidth = 0.8, color = 'green', alpha = 0.5)
    plt.grid()
    plt.xlabel('Class ID', fontsize = 14)
    plt.ylabel('Freq.', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.draw(); plt.pause(0.001)
    return freq, bins

def class_name(seg_model):
    classes = seg_model.get_classes()
    cls_dict = {}
    for i, cls in enumerate(classes):
        cls_dict[cls.category_id] = cls.category
    cls_list = []
    for i in range(len(cls_dict)):
        cls_list.append(cls_dict[i])
    return cls_dict, cls_list

def normalize(freq, sigmoid=False):
    ## Normalize
    freq_norm = freq/freq.sum()
    freq_norm = freq_norm/freq_norm.max()

    if sigmoid == True:
        ## Sigmoid
        freq_sigmoid = 1/(1+np.exp(-freq_norm))
        freq_norm = freq_sigmoid/freq_sigmoid.max()
    ## Softmax
    #z = torch.FloatTensor(freq_norm)
    #freq_norm = F.softmax(z, dim=0)
    return freq_norm

def get_class_freq(cls_dict, freq):
    cls_freq_dict = {}
    for i in range(len(freq)):  # len(freq) is less one than len(cls_dict)
        cls_freq_dict[cls_dict[i+1]] = freq[i]
    return cls_freq_dict

def get_class_statics(opt, seg_model, mDatasetStruct):
    ## Load pre-done info.
    _, _, _, cluster_set, _ = mDatasetStruct.get_dataset()
    dataset = cluster_set.dataset
    fname = "dbClass_{}.pickle".format(dataset)
    dbClass_simple, dbClass_category_id, dbClass_id = load_data(opt, fname)

    if len(dbClass_simple) == 0: # If not available loading data, do it
        dbClass_simple, dbClass_category_id, dbClass_id = class_statics(opt, seg_model, mDatasetStruct)
        save_data(opt, dbClass_simple, dbClass_category_id, dbClass_id, fname)

    #freq, bins = histogram(dbClass_simple, "red")
    ## all classes
    plt.figure(100, figsize=(20,7))
    print("id: 7 ~ 33, with no void(0~6) ")
    freq, bins = histogram(dbClass_id, "black")

    ## Category (groupped classes)
    plt.figure(101, figsize=(10,7))
    freq, bins = histogram(dbClass_category_id, "green")

    ## Draw histogram with class name
    plt.figure(102, figsize=(10,7))
    cls_dict, cls_list = class_name(seg_model)
    #plt.grid()
    plt.bar(cls_list[1:], freq)
    plt.xlabel('Class Name', fontsize = 14)
    plt.ylabel('Freq.', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.draw(); plt.pause(0.001)

    ## Draw histogram (normalized) with class name
    plt.figure(103, figsize=(10,7))
    plt.bar(cls_list[1:], normalize(freq))
    plt.xlabel('Class Name', fontsize = 14)
    plt.ylabel('Freq.(normalized)', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.draw(); plt.pause(0.001)

    print(cls_dict)
    print("freq :", freq)
    print("freq_norm :", normalize(freq))
    #print(seg_model.get_classes())

    ## Draw histogram for all dataset
    freq1 = np.array([ 4530, 29957,   288,  8573,  5071,   135,  1446])  # for pitts30k train dataset
    freq2 = np.array([ 6782, 25612,   360, 10897,  4742,   150,  1457])  # for tokyotm train dataset
    freq3 = np.array([ 3287, 33138,   717,  6593,  4205,   276,  1784])  # for tokyo247 test dataset
    freq_sum = freq1 + freq2 + freq3

    freq_sum_norm = normalize(freq_sum, sigmoid=True)
    plt.figure(104, figsize=(10,7))
    plt.bar(cls_list[1:], freq_sum_norm)
    plt.xlabel('Class Name', fontsize = 14)
    plt.ylabel('Freq.(normalized)', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.draw(); plt.pause(0.001)
    print("freq(pitt30k+tokyotm+tokyo247) :", freq_sum)
    print("freq_norm(pitt30k+tokyotm+tokyo247) :", freq_sum_norm)

    print(get_class_freq(cls_dict, freq_sum_norm))
    ## norm : {'flat': 0.16457551264274523, 'construction': 1.0, 'object': 0.015387737157157834, 'nature': 0.2938099586278422, 'sky': 0.15802586041687802, 'human': 0.006324190875579154, 'vehicle': 0.052836867439999095}
    ## norm with sigmoid : {'flat': 0.7400928999587367, 'construction': 1.0, 'object': 0.6892017590821214, 'nature': 0.7836972764175455, 'sky': 0.7378676239332944, 'human': 0.6861023960478257, 'vehicle': 0.7020041343706618}

    bp()
    '''
    cls_dict = {0: 'void', 1: 'flat', 2: 'construction', 3: 'object', 4: 'nature', 5: 'sky', 6: 'human', 7: 'vehicle'}
    cls_list = ['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle']

    ## for pitts30k train dataset
    freq1 = np.array([ 4530, 29957,   288,  8573,  5071,   135,  1446])
    #normalize(freq1) : array([0.15121674, 1.        , 0.00961378, 0.28617685, 0.16927596,  0.00450646, 0.04826919])

    ## for tokyotm train dataset
    freq2 = np.array([ 6782, 25612,   360, 10897,  4742,   150,  1457])
    #normalize(freq2) : array([0.26479775, 1.        , 0.01405591, 0.42546463, 0.18514759, 0.00585663, 0.0568874 ])

    ## for tokyo247 test dataset
    freq3 = np.array([ 3287, 33138,   717,  6593,  4205,   276,  1784])
    #normalize(freq3) : array([0.09919126, 1.        , 0.02163679, 0.19895588, 0.1268936, 0.00832881, 0.05383548])

    ## Total :
    freq_sum = freq1 + freq2 + freq3 = array([14599, 88707,  1365, 26063, 14018,   561,  4687])
    #normalize(freq_sum) : np.array([0.16457551, 1.        , 0.01538774, 0.29380996, 0.15802586, 0.00632419, 0.05283687])
    '''

'''
    concern category in phd paper : ["nature", "sky", "human", "vehicle", "flat", "construction"]
    from /home/ccsmm/dg_git/image_retrieval_deatt/networks/MobileNet/cityscapes.py

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color',
                                                     'simple_category', 'simple_id', 'mask_id'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0), 'static', 0, 0),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81), 'static', 0, 1),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128), 'static',  0, 1),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232), 'static', 0, 1),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160), 'static', 0, 1),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140), 'static', 0, 1),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70), 'structure', 1, 1),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156), 'structure' , 1, 1),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153), 'structure', 1, 1),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180), 'static', 0, 1),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100), 'static', 0, 1),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90), 'static',0 ,1),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153), 'structure', 1, 1),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153), 'static', 0, 1),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30), 'structure', 1, 1),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0), 'structure', 1, 1),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35), 'nature', 2, 1),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152), 'nature', 2, 1),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180), 'nature', 2, 1),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60), 'moving', 3, 0),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0), 'moving', 3, 0),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142),'moving', 3, 0),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70),'moving',  3, 0),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100),'moving',  3, 0),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90),'static',  0, 0),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110),'static',  0, 0),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100),'moving',  3, 0),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230),'moving',  3, 0),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32),'moving',  3, 0),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142),'static',  0, 0),
    ]
'''

