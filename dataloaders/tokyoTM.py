#  # tokyo247 dataset is only for test
#  # tokyoTM dataset is for train/val
#  # coded by ccsmm@etri.re.kr

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
from os.path import join, exists
from scipy.io import loadmat, savemat
import numpy as np
import random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

import faiss

import cv2
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

from ipdb import set_trace as bp

root_dir = '/home/ccsmm/DB_Repo/TokyoTM/netvlad_v100_datasets'
# root_dir = './netvlad_v100_datasets_pitts/'  # you need this directory in the top.

if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets')  # dataset struture mat file locates at datasets, data set from google streetview (mat files)
db_dir = join(root_dir, 'tokyoTimeMachine', 'images')  # db locates at root_dir
queries_dir = join(root_dir, 'tokyoTimeMachine', 'images')  # query locates at queries_real, data set from real camera such as smartphone

def get_queries_dir():
    return queries_dir

def get_db_dir():
    return db_dir

def resize(img):
    return img.resize((640, 480))  # Width, Height

def faiss_radius_neighbors(Db, Q, radius=10):
    Db = Db.astype('float32')
    Q = Q.astype('float32')
    Index = faiss.IndexFlatL2(Db.shape[-1])  # The type of Db and Q should be 'float32'
    Index.add(Db)
    K = 100  # sufficient large number
    print("Searching all points within range: %f" % radius)
    distance, idx = Index.search(Q, K)
    l2norm = np.sqrt(distance)
    valid = l2norm < radius

    ret_idx = []
    ret_distance = []
    for i in range(len(Q)):
        a_idx = np.ndarray.tolist(idx[i][valid[i]])
        a_dist = np.ndarray.tolist(distance[i][valid[i]])
        ret_idx.append(a_idx)
        ret_distance.append(a_dist)

    # ret_idx = np.array(ret_idx, dtype=object)
    # ret_distance = np.array(ret_distance, dtype=object)

    return ret_distance, ret_idx


def squeeze_1d(data):
    while(True):
        if len(data.shape) > 1:
            data = data.squeeze()
        else:
            break
    return data


def faiss_knn(Db, Q, K, metric='l2'):
    if metric.lower() == 'l2':
        Index = faiss.IndexFlatL2(Db.shape[-1])  # L2 eculidian distance
    elif metric.lower() == 'ip':
        Index = faiss.IndexFlatIP(Db.shape[-1])  # Inner Product
    elif metric.lower() == 'cosine':
        Index = faiss.IndexFlatIP(Db.shape[-1])  # Inner Product
        faiss.normalize_L2(Db)
        Q=np.expand_dims(Q, axis=0)
        faiss.normalize_L2(Q)
        Q.squeeze()
    else:
        Index = faiss.IndexFlatL2(Db.shape[-1])  # L2 eculidian distance

    Index.add(Db)
    cost, idx = Index.search(Q.reshape(1, -1), K)

    if metric.lower() == 'l2':
        l2norm = np.sqrt(cost).squeeze()
        cost = l2norm

    return squeeze_1d(cost), squeeze_1d(idx)


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

## toky247.mat is only for validation of pretrained network. Not for train
#use_prebuilt_xordbq = True  # False for original dataset
use_prebuilt_xordbq = False  # False for original dataset

def get_whole_training_set(onlyDB=False):
    if use_prebuilt_xordbq:
        structFile = join(struct_dir, 'tokyoTM_train_xordbq.mat') # by ccsmm, query images were removed from db list.
    else:
        structFile = join(struct_dir, 'tokyoTM_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    if use_prebuilt_xordbq:
        structFile = join(struct_dir, 'tokyoTM_val_xordbq.mat') # by ccsmm, query images were removed from db list.
    else:
        structFile = join(struct_dir, 'tokyoTM_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_whole_test_set():
    if use_prebuilt_xordbq:
        structFile = join(struct_dir, 'tokyoTM_val_xordbq.mat') # by ccsmm, query images were removed from db list.
    else:
        structFile = join(struct_dir, 'tokyoTM_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    if use_prebuilt_xordbq:
        structFile = join(struct_dir, 'tokyoTM_train_xordbq.mat') # by ccsmm, query images were removed from db list.
    else:
        structFile = join(struct_dir, 'tokyoTM_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    if use_prebuilt_xordbq:
        structFile = join(struct_dir, 'tokyoTM_val_xordbq.mat') # by ccsmm, query images were removed from db list.
    else:
        structFile = join(struct_dir, 'tokyoTM_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

dbStructTM = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'utmDb', 'dateDb', 'qImage', 'utmQ', 'dateQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

dbStructTM_2015 = namedtuple('dbStruct', ['whichSet', 
    'dbImage', 'utmDb', 'dateDb', 'qImage', 'utmQ', 'dateQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_tokyoTM_dbStruct(path, remove_q_from_db_and_savemat=False):  # for original tokyoTM_().mat from netvlad paper
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'tokyoTM'  # tokyoTM

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]  # As is.
    # dbImage = [f[0].item().replace('.jpg', '.png') for f in matStruct[1]]  # Replace .jpg to .png
    utmDb = matStruct[2].T
    dateDb = matStruct[3].squeeze()

    qImage = [f[0].item() for f in matStruct[4]]
    utmQ = matStruct[5].T
    dateQ = matStruct[6].squeeze()

    numDb = matStruct[7].item()
    numQ = matStruct[8].item()

    posDistThr = matStruct[9].item()
    posDistSqThr = matStruct[10].item()
    nonTrivPosDistSqThr = matStruct[11].item()

    if remove_q_from_db_and_savemat:
        # Remove query images from db list only in tokyoTM dataset. by ccsmm.
        # The others datasets has no this kind situation (db == q) except tokyoTM
        rm_el = lambda x, i : np.delete(np.asarray(x), i, axis=0).tolist()  # remove i-th(s) elements from x
        for n, qImg in enumerate(qImage):
            ## Remove exactly same query image from db list
            idx = np.where(np.array(dbImage)==qImg)
            #print("Checking independence with removing query from db list) : {} / {} ...\r".format(n, len(qImage)), end='')
            if len(idx) > 0:
                dbImage = rm_el(dbImage, idx)
                utmDb = rm_el(utmDb, idx)
                dateDb = rm_el(dateDb, idx)
                numDb -= len(idx)

    dbStruct_tuple = dbStructTM(whichSet, dataset, dbImage, utmDb, dateDb, qImage,
            utmQ, dateQ, numDb, numQ, posDistThr,
            posDistSqThr, nonTrivPosDistSqThr)

    if remove_q_from_db_and_savemat:
        #dbStructTM_2015_tuple = dbStructTM_2015(whichSet, dbImage, utmDb, dateDb, qImage,
        #    utmQ, dateQ, numDb, numQ, posDistThr,
        #    posDistSqThr, nonTrivPosDistSqThr)
        #mat_save = {'dbStruct':dbStructTM_2015_tuple}
        mat_save = {'dbStruct':dbStruct_tuple}
        matpath = os.path.split(path)[0]
        matfile = os.path.split(path)[1].split('.')[0]+'_xordbq.'+os.path.split(path)[1].split('.')[1]
        matfname = os.path.join(matpath, matfile)
        savemat(matfname, mat_save)

    return dbStruct_tuple


def parse_tokyoTM_dbStruct_xordbq(path):  # Read mat file in which q images were removed from db list
    mat = loadmat(path)
    matStruct = mat['dbStruct'][0]

    whichSet = matStruct[0].item() # train, val, test
    dataset = matStruct[1].item() # train, val, test

    dbImage = [f for f in matStruct[2]]
    utmDb = matStruct[3]
    dateDb = matStruct[4].squeeze()

    qImage = [f for f in matStruct[5]]
    utmQ = matStruct[6]
    dateQ = matStruct[7].squeeze()

    numDb = matStruct[8].item()
    numQ = matStruct[9].item()

    posDistThr = matStruct[10].item()
    posDistSqThr = matStruct[11].item()
    nonTrivPosDistSqThr = matStruct[12].item()

    dbStruct_tuple = dbStructTM(whichSet, dataset, dbImage, utmDb, dateDb, qImage,
            utmQ, dateQ, numDb, numQ, posDistThr,
            posDistSqThr, nonTrivPosDistSqThr)

    return dbStruct_tuple

def parse_dbStruct(path):  # for pittsburgh, tokyo247
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'tokyo'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]  # As is.
    # dbImage = [f[0].item().replace('.jpg', '.png') for f in matStruct[1]]  # Replace .jpg to .png
    utmDb = matStruct[2].T

    qImage = [f[0].item() for f in matStruct[3]]
    utmQ = matStruct[4].T

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
            utmQ, numDb, numQ, posDistThr,
            posDistSqThr, nonTrivPosDistSqThr)

def exclude_same_date(distance, index, dateQ, dateDb, maxlen=0):  # ccsmm
    #  # Exclude image taken on same date from ground truth
    new_index = []
    new_distance = []
    for i in range(len(dateQ)):
        date_q = dateQ[i]
        date_db = dateDb[index[i]]
        valid_idx = (date_db != date_q)
        tmp_idx= []
        tmp_dist = []
        if valid_idx.sum() > 0:
            if maxlen == 0:  # bypass
                tmp_idx = index[i][valid_idx]
            else:  # trim array to top-k, k is maxlen.
                maxidx = np.where(np.cumsum(valid_idx)==maxlen+1)[0]
                if len(maxidx) == 0: 
                    tmp_idx = index[i][valid_idx]
                else:
                    try:
                        tmp_idx = index[i][0:maxidx[0]][valid_idx[:maxidx[0]]]
                    except:
                        bp()
            if distance is not None:
                tmp_dist = distance[i][valid_idx]
        new_index.append(np.asarray(tmp_idx))
        new_distance.append(np.asarray(tmp_dist))

    new_distance = np.asarray(new_distance, dtype='object')
    new_index = np.asarray(new_index, dtype='object')
    return new_distance, new_index

def exclude_same_db_q_set(distance, index, q_flist, db_flist):  # ccsmm
    #  # Exclude image taken on same date from ground truth
    new_index = []
    new_distance = []
    for i in range(len(q_flist)):
        name_q = q_flist[i]  # single q fname
        tmp_idx= []
        tmp_dist = []
        for j, db_idx in enumerate(index[i]):
            name_db = db_flist[db_idx]  # single db fname
            if name_db != name_q:  # valid
                tmp_idx.append(db_idx)
                if distance is not None:
                    tmp_dist.append(distance[i][j])
        new_index.append(np.asarray(tmp_idx))
        new_distance.append(np.asarray(tmp_dist))

    new_distance = np.asarray(new_distance, dtype='object')
    new_index = np.asarray(new_index, dtype='object')
    return new_distance, new_index

class WholeDatasetFromStruct(data.Dataset):  # Dataloader for test or validation
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        if 'naver' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        elif 'xordbq' in structFile.split('/')[-1]:
            self.dbStruct = parse_tokyoTM_dbStruct_xordbq(structFile)  # dbStructTM, modified mat in which q was removed from db.
        else:  # Default (tokyoTM_val.mat)
            self.dbStruct = parse_tokyoTM_dbStruct(structFile, remove_q_from_db_and_savemat=False)  # original mat file from paper author dbStructTM
            #self.dbStruct = parse_tokyoTM_dbStruct(structFile, remove_q_from_db_and_savemat=True)  # Not completed function yet.

        self.images = [join(db_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.negatives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = resize(img)

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives_ori(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if True:  # ori, use sklearn for knn
            if  self.positives is None:
                # knn = NearestNeighbors(n_jobs=-1)  # ori
                knn = NearestNeighbors(n_jobs=1)
                knn.fit(self.dbStruct.utmDb)

                self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                        radius=self.dbStruct.posDistThr)
        else:  # use faiss for knn by ccsmm
            if  self.positives is None:
                self.distances, self.positives = faiss_radius_neighbors(self.dbStruct.utmDb,
                        self.dbStruct.utmQ, radius=self.dbStruct.posDistThr)
        return self.positives

    def getPositives(self, radius=None, exclude_same_date_enable=True):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if radius is None:
            radius = self.dbStruct.posDistThr

        if True: # ori, use sklearn for knn
            if  self.positives is None:
                # knn = NearestNeighbors(n_jobs=-1)  # ori
                knn = NearestNeighbors(n_jobs=1)
                knn.fit(self.dbStruct.utmDb)
                self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=radius)
        else:  # use faiss for knn by ccsmm
            if  self.positives is None:
                self.distances, self.positives = faiss_radius_neighbors(self.dbStruct.utmDb, self.dbStruct.utmQ, radius=radius)

        #  # Exclude image taken on same date from ground truth by ccsmm
        if exclude_same_date_enable:
            self.distances, self.positives = exclude_same_date(self.distances, self.positives, self.dbStruct.dateQ, self.dbStruct.dateDb)

        return self.positives

    def getNegatives(self, radius=None):
        if  self.negatives is None:
            # potential negatives are those outside of posDistThr range
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.utmDb)
            self.negatives = []
            if radius is None:
                radius = self.dbStruct.posDistThr
            potential_positives = knn.radius_neighbors(self.dbStruct.utmQ, radius=radius, return_distance=False)
            for pos in potential_positives:
                potential_negatives = np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True)
                #self.negatives.append(np.random.choice(potential_negatives, 1))
                self.negatives.append(potential_negatives)
        return self.negatives

    def get_queries_dir(self):
        return queries_dir

    def get_db_dir(self):
        return db_dir

def worker_init_fn(worker_id):
    # This function ensures that the dataloader always returns the same data sequence for the'shuffle=Ture' option.
    torch_seed = torch.initial_seed()  # It may return 123 which is the setting value in main.py with torch.manual_seed(opt.seed)
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):  # Dataloader for train
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        if 'naver' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        elif 'xordbq' in structFile.split('/')[-1]:
            self.dbStruct = parse_tokyoTM_dbStruct_xordbq(structFile)  # dbStructTM
        else:
            self.dbStruct = parse_tokyoTM_dbStruct(structFile, remove_q_from_db_and_savemat=False)  # original mat file from paper author dbStructTM
            #self.dbStruct = parse_tokyoTM_dbStruct(structFile, remove_q_from_db_and_savemat=True)  # Not completed function yet.

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nNeg = nNeg  # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        # fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        if False:  # show potioninformation, ccsmm
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(self.dbStruct.utmDb[:, 0], self.dbStruct.utmDb[:, 1], c='tab:blue', s=10, label='Db', alpha=0.03)
            plt.scatter(self.dbStruct.utmQ[:, 0], self.dbStruct.utmQ[:, 1], c='tab:red', s=100, label='Q', alpha=0.03)
            plt.legend()
            plt.draw();plt.pause(0.001)

        # TODO use sqeuclidean as metric?
        
        radius = self.dbStruct.nonTrivPosDistSqThr**0.5  # 10 <== sqrt(100) for original
        # radius = 15  # 15 for tokyoTM, 10 for other dataset for nontrivial_positives
        if True:
            self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, radius=radius, return_distance=False))
            # radius returns unsorted, sort once now so we dont have to later
            for i, posi in enumerate(self.nontrivial_positives):
                self.nontrivial_positives[i] = np.sort(posi)
            # its possible some queries don't have any non trivial potential positives
            # lets filter those out
            _, self.nontrivial_positives = exclude_same_date(None, self.nontrivial_positives, self.dbStruct.dateQ, self.dbStruct.dateDb)

        else:  # (trial version) Increase radius to get valid nontrivial_positive.
            for trial in range(10):
                self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ, radius=radius, return_distance=False))
                # radius returns unsorted, sort once now so we dont have to later
                for i, posi in enumerate(self.nontrivial_positives):
                    self.nontrivial_positives[i] = np.sort(posi)
                # its possible some queries don't have any non trivial potential positives
                # lets filter those out
    
                try:
                    _, self.nontrivial_positives = exclude_same_date(None, self.nontrivial_positives, self.dbStruct.dateQ, self.dbStruct.dateDb)
                    break
                except:
                    radius += 2
                    print("*** Warning *** Increasing radius for nontrivial_positives is {} for tokyoTM, instead of {}.".format(radius, self.dbStruct.nonTrivPosDistSqThr**0.5))
                    continue

        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range , 25 meters
        self.potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr,
                return_distance=False)
        _, self.potential_positives = exclude_same_date(None, self.potential_positives, self.dbStruct.dateQ, self.dbStruct.dateDb)

        self.potential_negatives = []
        for pos in self.potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb), pos, assume_unique=True))

        self.cache = None  # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0, )) for _ in range(self.dbStruct.numQ)]
        self.violatingNeg_cnt = 0

    def __getitem__(self, index):
        index = self.queries[index]  # re-map index to match dataset
        if False:  # ori, use knn of sklearn
            with h5py.File(self.cache, mode='r') as h5:   #  # use knn of sklearn
                print("Use knn of sklearn")
                h5feat = h5.get("features")

                qOffset = self.dbStruct.numDb
                qFeat = h5feat[index+qOffset]

                posFeat = h5feat[self.nontrivial_positives[index].tolist()]
                knn = NearestNeighbors(n_jobs=1)  # TODO replace with faiss?
                knn.fit(posFeat)
                # Try to look for the positive image that is most similar to the query images
                dPos, posIndices = knn.kneighbors(qFeat.reshape(1, -1), 1)
                dPos = dPos.item()
                posIndex = self.nontrivial_positives[index][posIndices[0]].item()  # A nearest and most similar image.

                try:
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
                except:
                    print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()

                negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int32)

                negFeat = h5feat[negSample.tolist()]
                knn.fit(negFeat)

                # Try to look for the negative image that is most similar to the query images
                dNeg, negNN = knn.kneighbors(qFeat.reshape(1, -1), self.nNeg*10)  # to quote netvlad paper code: 10x is hacky but fine
                dNeg = dNeg.reshape(-1)
                negNN = negNN.reshape(-1)
                negNN_bakup = negNN.copy()  # debug

                # Try to find hard-negatives that are within margin, if there aren't any return none
                # ViolatingNeg means hard-negative image whose feature is more similar to query than postive image is.
                # violatingNeg = dNeg < dPos + self.margin**0.5  # ori
                violatingNeg = dNeg < dPos + self.margin

                if np.sum(violatingNeg) < 1:
                    # if none are violating then skip this query.
                    return None

                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices
                self.negIndices = negIndices
        else:  #  # use knn of faiss for speed up by ccsmm
            with h5py.File(self.cache, mode='r') as h5:  #  # use knn of faiss
                h5feat = h5.get("features")

                qOffset = self.dbStruct.numDb
                qFeat = h5feat[index+qOffset]

                posFeat = h5feat[self.nontrivial_positives[index].tolist()]

                dPos, posIndices = faiss_knn(posFeat, qFeat, K=100, metric='l2')
                self.dPos, self.posIndices = dPos, posIndices

                if False:  # In fact, it is violation that same image is included in db, query at the same time.
                    # But toykoTM dataset has a lot of iamges which meet these conditions. So I just treat them. by ccsmm.
                    very_similar_image_threshold = 0.8
                    valid_dPos_idx = dPos > very_similar_image_threshold  # array([False,  True,  True,  True, ...,  True])
                    # Filtering out db images captured in same date of query.
                    dPos = dPos[valid_dPos_idx]
                    posIndices = posIndices[valid_dPos_idx]
                    if len(dPos) == 0:
                        return None
                    dPos = dPos[0]
                    posIndices = posIndices[0]  # Get most similar image as a positive image

                dPos, posIndice = self.get_a_valid_positive(index, dPos, posIndices)

                posIndex = self.nontrivial_positives[index][posIndice].item()  # A nearest and most similar image.

                try:
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
                except:
                    print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()

                negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int32)
                negFeat = h5feat[negSample.tolist()]

                # We select an image that is geographically far from the query, but looks similar, as a negative sample.
                dNeg, negNN = faiss_knn(negFeat, qFeat, K=self.nNeg*10, metric='l2')
                negNN = negNN.reshape(-1)
                negNN_bakup = negNN.copy()  # debug

                # del knn

                # Try to find hard-negatives that are within margin, if there aren't any return none
                # ViolatingNeg means hard-negative image whose feature is more similar to query than postive image is.
                # violatingNeg = dNeg < dPos + self.margin**0.5  # ori
                violatingNeg = dNeg < dPos + self.margin
                if False:
                    self.imshow_qpn_pair(index, posIndex, negNN[0])

                if np.sum(violatingNeg) < 1:
                    # Violating, then skip this query.
                    self.violatingNeg_cnt += 1
                    #print("\nSkip {}-th query, {}/{}\n : Does not meet violatingNeg condition".format(index, self.violatingNeg_cnt, len(self.queries)))
                    if False:  # debug
                        self.imshow_qpn_pair(index, posIndex, negNN[0])
                        bp()
                    return None

                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices

        query = Image.open(join(queries_dir, self.dbStruct.qImage[index]))
        query = resize(query)
        positive = Image.open(join(db_dir, self.dbStruct.dbImage[posIndex]))
        positive = resize(positive)

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative = Image.open(join(db_dir, self.dbStruct.dbImage[negIndex]))
            negative = resize(negative)
            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)

    #  # Treat Time Machine dataset
    def get_a_valid_positive(self, qidx, dPos, posIndices, similar_th=1.0):
        qdate = self.dbStruct.dateQ[qidx]
        for i, dbidx in enumerate(posIndices, 0):
            posidx = self.nontrivial_positives[qidx][dbidx].item()
            posdate = self.dbStruct.dateDb[posidx]
            if dPos[i] < similar_th:  # diff(q, p) is very small, which means query and positive would be same image.
            # The more similar q and p are, the better, but not much more similar than the difference between q and n.
                continue
            if qdate == posdate:
                continue
            else:  # Good, Found db image captured on different date than capturing date of the query image.
                return dPos[i], posIndices[i]

        # If not found, Do nothing but to avoid to return exactly same image [0], return secondly similary image, [1]
        return dPos[1], posIndices[1]

    def imshow_qpn_pair(self, qidx, pidx, nidx):
       tim.cv2_imshow_images(
           [join(queries_dir, self.dbStruct.qImage[qidx]),
            join(db_dir, self.dbStruct.dbImage[pidx]),
            join(db_dir, self.dbStruct.dbImage[nidx])],
           imgnumlist=['q', 'p', 'n'])

    def get_queries_dir(self):
        return queries_dir

    def get_db_dir(self):
        return db_dir
