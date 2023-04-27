import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread/29172195#29172195
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
import h5py

import faiss
import cv2

from tictoc import tic, toc
from ipdb import set_trace as bp

root_dir = './netvlad_v100_datasets_dg/' #you need this directory in the top.

if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets') # For mat files in which list of image files are

## db_dir :  Use google streetview for DB images in trains [ 000 ~ 010 ] subdir
db_dir = join(root_dir, '.')

## queries_dir : Use real phone camera images for Query in trains and val/test [000 ~ 080] subdir
#queries_dir = join(root_dir, 'queries_real') # ori
queries_dir = join(root_dir, '.') # just for test (do not use this)

def faiss_radius_neighbors(Db, Q, radius=10):
    Db = Db.astype('float32')
    Q = Q.astype('float32')
    Index = faiss.IndexFlatL2(Db.shape[-1])  # The type of Db and Q should be 'float32'
    Index.add(Db)
    K = 100  # sufficient large number
    #print("Searching all points within range: %f" % radius)
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

    #ret_idx = np.array(ret_idx, dtype=object)
    #ret_distance = np.array(ret_distance, dtype=object)

    return ret_distance, ret_idx

def get_single_image(path):
    img = Image.open(path)
    return img.resize((640,480))

def faiss_knn(Db, Q, K): 
    Index = faiss.IndexFlatL2(Db.shape[-1])
    Index.add(Db)
    distance, idx = Index.search(Q.reshape(1,-1), K)
    l2norm = np.sqrt(distance).squeeze()
    return l2norm, idx 

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_training_set(onlyDB=False):  # Provide all DB and Q image
    #structFile = join(struct_dir, 'pitts30k_train.mat')
    structFile = join(struct_dir, 'dg_daejeon_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    structFile = join(struct_dir, 'dg_daejeon_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_whole_test_set(onlyQ=False):
    structFile = join(struct_dir, 'dg_daejeon_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyQ=onlyQ)

def get_training_query_set(margin=0.1, disp_en=False):  # Provide query and its corresponding positive/negative images
    structFile = join(struct_dir, 'dg_daejeon_train.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin, disp_en=disp_en)

def get_val_query_set():
    structFile = join(struct_dir, 'dg_daejeon_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

    whichSet = matStruct[0].item()

    dbImage = [f[0].item() for f in matStruct[1]]
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


def Load_dbStruct(fname):
    mat = loadmat(fname)
    matStruct = mat['dbStruct'][0]

    whichSet = matStruct[0].item()
    dataset = matStruct[1].item()

    dbImage = [f.item().strip() for f in matStruct[2]]
    utmDb = matStruct[3]
    
    qImage = [f.item().strip() for f in matStruct[4]]
    utmQ = matStruct[5]
    
    numDb = matStruct[6].item()
    numQ = matStruct[7].item()

    posDistThr = matStruct[8].item()
    posDistSqThr = matStruct[9].item()
    nonTrivPosDistSqThr = matStruct[10].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
            utmQ, numDb, numQ, posDistThr,
            posDistSqThr, nonTrivPosDistSqThr)


class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False, onlyQ=False):
        super().__init__()

        self.input_transform = input_transform

        if 'dg' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        else:
            self.dbStruct = parse_dbStruct(structFile)

        if onlyDB: # generate only db images for enumerate
            self.images = [join(db_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        elif onlyQ: # generate only q images for enumerate
            self.images = [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]
        else: # normal
            self.images = [join(db_dir, dbIm) for dbIm in self.dbStruct.dbImage]
            self.images += [join(queries_dir, qIm) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.negatives = None
        self.distances = None

    def __getitem__(self, index):
        img = get_single_image(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self, radius=None):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            #knn = NearestNeighbors(n_jobs=-1)
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.utmDb)
            if radius is None:
                radius = self.dbStruct.posDistThr
            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=radius)
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
                #self.negatives.append(np.random.choice(potential_negatives, 1000))
                self.negatives.append(potential_negatives)
        return self.negatives
        
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

class QueryDatasetFromStruct(data.Dataset):
    def disp_all_pts(self):
        plt.scatter(self.dbStruct.utmDb[:,0], self.dbStruct.utmDb[:,1], c='tab:blue', s=10, label='Db', alpha=0.03)
        plt.scatter(self.dbStruct.utmQ[:,0], self.dbStruct.utmQ[:,1], c='tab:red', s=100, label='Q', alpha=0.03)
        plt.legend()
        plt.draw();plt.pause(0.001)

    def disp_single_query(self, index):
        utm_x, utm_y = self.dbStruct.utmQ[index,0], self.dbStruct.utmQ[index,1]
        plt.clf()
        plt.scatter(self.dbStruct.utmDb[self.nontrivial_positives[index],0] - utm_x,
                self.dbStruct.utmDb[self.nontrivial_positives[index],1] - utm_y,
                c='tab:blue', s=10, label='Db', alpha=1)
        plt.scatter(0, 0, c='tab:red', s=100, label='Q', alpha=1)
        plt.legend()
        plt.draw();plt.pause(0.001)

    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None, disp_en=False):
        super().__init__()

        self.disp_en = disp_en
        self.input_transform = input_transform
        self.margin = margin

        if 'dg' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        else:
            self.dbStruct = parse_dbStruct(structFile)

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        #faiss_index = faiss.IndexFlatIP(self.dbStruct.utmDb.shape[-1])
        #faiss_index.add(self.dbStruct.utmDb)

        #knn = NearestNeighbors(n_jobs=-1)  # ori
        knn = NearestNeighbors(n_jobs=1) 
        knn.fit(self.dbStruct.utmDb)

        if True: # original code
            # TODO use sqeuclidean as metric?
            self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.nonTrivPosDistSqThr**0.5,  # 10 meters
                    return_distance=False))
            #lims, dist, index = faiss_knn.range_search(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)
        else:  # modified for DG's dbN_qR dataset by ccsmm
            self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.nonTrivPosDistSqThr,  # 100 meters
                    return_distance=False))

        if False: # debug : Show position information of all points, ccsmm
            self.disp_all_pts()

        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
            if False:  # debug : Show position information of signle point, ccsmm
                self.disp_single_query(index=i)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)
        #lims, dist, index = faiss_knn.range_search(self.dbStruct.utmQ, radius=self.dbStruct.nonTrivPosDistSqThr**0.5)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset

        if False:  # ori, use knn of sklearn
            with h5py.File(self.cache, mode='r') as h5:   ## use knn of sklearn
                #print("Use knn of sklearn")
                h5feat = h5.get("features")
    
                qOffset = self.dbStruct.numDb 
                qFeat = h5feat[index+qOffset]
    
                posFeat = h5feat[self.nontrivial_positives[index].tolist()]
                knn = NearestNeighbors(n_jobs=1) # TODO replace with faiss?
                knn.fit(posFeat)
                # Try to look for the positive image that is most similar to the query images
                dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
                dPos = dPos.item()
                posIndex = self.nontrivial_positives[index][posNN[0]].item()  # A nearest and most similar image.
    
                try:
                    np.random.seed(index)  # for re-producibility
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample).astype(np.int32)
                except:
                    #print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()
    
                negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int32)
    
                negFeat = h5feat[negSample.tolist()]
                knn.fit(negFeat)
    
                # Try to look for the negative image that is most similar to the query images
                dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
                dNeg = dNeg.reshape(-1)
                negNN = negNN.reshape(-1)
                negNN_bakup = negNN.copy()  # debug
    
                # Try to find hard-negatives that are within margin, if there aren't any return none
                # ViolatingNeg means hard-negative image whose feature is more similar to query than postive image is.
                #violatingNeg = dNeg < dPos + self.margin**0.5  #ori
                violatingNeg = dNeg < dPos + self.margin
         
                if np.sum(violatingNeg) < 1:
                    #if none are violating then skip this query.
                    return None
    
                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices
        else:  ## use knn of faiss for speed up by ccsmm
            with h5py.File(self.cache, mode='r') as h5:  ## use knn of faiss
                h5feat = h5.get("features")
    
                qOffset = self.dbStruct.numDb 
                qFeat = h5feat[index+qOffset]
    
                posFeat = h5feat[self.nontrivial_positives[index].tolist()]
    
                #knn = faiss.IndexFlatL2(posFeat.shape[-1])
                #if False :  # use gpu for faiss. Do not use this because it doesn't work well
                #    gpu_res = faiss.StandardGpuResources()  # use a single GPU
                #    knn = faiss.index_cpu_to_gpu(gpu_res, 0, knn)
                #knn.add(posFeat)
                #dPos, posNN = knn.search(qFeat.reshape(1,-1), 1)
                #dPos = np.sqrt(dPos)[0]

                dPos, posNN = faiss_knn(posFeat, qFeat, K=1)
                posIndex = self.nontrivial_positives[index][posNN[0]].item()  # A nearest and most similar image.
    
                try:
                    np.random.seed(index)  # for re-producibility
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample).astype(np.int32)
                except:
                    print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()
    
                negSample = np.unique(np.concatenate([self.negCache[index], negSample])).astype(np.int32)
                negFeat = h5feat[negSample.tolist()]

                #del knn

                #knn = faiss.IndexFlatL2(negFeat.shape[-1])
                #if False :  # use gpu for faiss. Do not use this because it doesn't work well
                #    gpu_res = faiss.StandardGpuResources()  # use a single GPU
                #    knn = faiss.index_cpu_to_gpu(gpu_res, 0, knn)
                #knn.add(negFeat)

                # Try to look for the negative image that is most similar to the query images
                #dNeg, negNN = knn.search(qFeat.reshape(1,-1), self.nNeg*10)
                #dNeg = np.sqrt(dNeg.reshape(-1))

                dNeg, negNN = faiss_knn(negFeat, qFeat, K=self.nNeg*10)
                negNN = negNN.reshape(-1)
                negNN_bakup = negNN.copy()  # debug
    
                #del knn

                # Try to find hard-negatives that are within margin, if there aren't any return none
                # ViolatingNeg means hard-negative image whose feature is more similar to query than postive image is.
                #violatingNeg = dNeg < dPos + self.margin**0.5  #ori
                violatingNeg = dNeg < dPos + self.margin
         
                if np.sum(violatingNeg) < 1:
                    #if none are violating then skip this query.
                    return None
    
                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices

        query = get_single_image(join(queries_dir, self.dbStruct.qImage[index]))
        positive = get_single_image(join(db_dir, self.dbStruct.dbImage[posIndex]))

        if self.disp_en:  # for debugging
            dbg_img = np.hstack([np.array(query), np.array(positive)])

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []
        disp_stack_cnt = 5
        for negIndex in negIndices:
            negative = get_single_image(join(db_dir, self.dbStruct.dbImage[negIndex]))
            if self.disp_en and (disp_stack_cnt > 0):  # debugging
                dbg_img = np.hstack([dbg_img, np.array(negative)])
                disp_stack_cnt -= 1

            if self.input_transform:
                negative = self.input_transform(negative)
            negatives.append(negative)

        if self.disp_en and (disp_stack_cnt == 0):  # for debugging
            dbg_img = cv2.cvtColor(dbg_img, cv2.COLOR_RGB2BGR)  #dbg
            dbg_img = cv2.resize(dbg_img, dsize=(0,0), fx=0.5, fy=0.5)
            cv2.imshow('Query(1st)_positive(2nd)_negatives(3rd_end)', dbg_img)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            plt.clf()
            plt_x = np.arange(1, min(20, dNeg.size+1))
            plt.bar(0, dPos, color='b')
            plt.bar(plt_x, dNeg[:plt_x.size], color='r')
            plt.plot(plt_x, violatingNeg[:plt_x.size], color='m')
            plt.draw()
            plt.pause(0.1)
            if False:
                negative_tmp1 = get_single_image(join(db_dir, self.dbStruct.dbImage[negSample[negNN_bakup[17]].astype(np.int32)]))
                negative_tmp2 = get_single_image(join(db_dir, self.dbStruct.dbImage[negSample[negNN_bakup[18]].astype(np.int32)]))
                cv2.imshow('good', cv2.cvtColor(np.array(negative_tmp1), cv2.COLOR_RGB2BGR))
                cv2.imshow('bad', cv2.cvtColor(np.array(negative_tmp2), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        # Number of queries with positives. Queries with no positives were filtered out.
        return len(self.queries)  # return number of valid queries that have Ground Truth.
