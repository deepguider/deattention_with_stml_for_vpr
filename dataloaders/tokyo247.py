## tokyo247 dataset is only for test
## tokyoTM dataset is for train/val

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image

from sklearn.neighbors import NearestNeighbors
import h5py

import faiss

from ipdb import set_trace as bp

root_dir = '/home/ccsmm/DB_Repo/Tokyo247/netvlad_v100_datasets'
#root_dir = './netvlad_v100_datasets_pitts/' #you need this directory in the top.

if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets') # dataset struture mat file locates at datasets, data set from google streetview (mat files)
db_dir = join(root_dir, 'db')  # db locates at root_dir
queries_dir = join(root_dir, 'query') # query locates at queries_real, data set from real camera such as smartphone

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

    #ret_idx = np.array(ret_idx, dtype=object)
    #ret_distance = np.array(ret_distance, dtype=object)

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
    cost, idx = Index.search(Q.reshape(1,-1), K)

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

def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'tokyo247.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)

def get_whole_val_set():
    structFile = join(struct_dir, 'tokyo247.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_whole_test_set():
    structFile = join(struct_dir, 'tokyo247.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1):
    structFile = join(struct_dir, 'tokyo247.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    structFile = join(struct_dir, 'tokyo247.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'tokyo'

    whichSet = matStruct[0].item()

    ## Todo : tokyo247 dataset downloaded has png extention, but tokyo247.mat of netvlad has .jpg extention.
    ##        So elements of dbImage need to be changed jpg to png
    ##        qImage is okay.

    #dbImage = [f[0].item() for f in matStruct[1]]  # As is.
    dbImage = [f[0].item().replace('.jpg', '.png') for f in matStruct[1]]  # Replace .jpg to .png
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

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        if 'naver' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        else:
            self.dbStruct = parse_dbStruct(structFile)

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
                self.negatives.append(np.random.choice(potential_negatives, 1)) 
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

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        if 'naver' in structFile.split('/')[-1]:
            self.dbStruct = Load_dbStruct(structFile)
        else:
            self.dbStruct = parse_dbStruct(structFile)

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.dbStruct.utmDb)

        if False: #show potioninformation, ccsmm
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(self.dbStruct.utmDb[:,0], self.dbStruct.utmDb[:,1], c='tab:blue', s=10, label='Db', alpha=0.03)
            plt.scatter(self.dbStruct.utmQ[:,0], self.dbStruct.utmQ[:,1], c='tab:red', s=100, label='Q', alpha=0.03)
            plt.legend()
            plt.draw();plt.pause(0.001)

        # TODO use sqeuclidean as metric?
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

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
                print("Use knn of sklearn")
                h5feat = h5.get("features")
    
                qOffset = self.dbStruct.numDb 
                qFeat = h5feat[index+qOffset]
    
                posFeat = h5feat[self.nontrivial_positives[index].tolist()]
                knn = NearestNeighbors(n_jobs=1) # TODO replace with faiss?
                knn.fit(posFeat)
                # Try to look for the positive image that is most similar to the query images
                dPos, posIndices = knn.kneighbors(qFeat.reshape(1,-1), 1)
                dPos = dPos.item()
                posIndex = self.nontrivial_positives[index][posIndices[0]].item()  # A nearest and most similar image.
    
                try:
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
                except:
                    print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()
    
                negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
    
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
                bp()
                violatingNeg = dNeg < dPos + self.margin
         
                if np.sum(violatingNeg) < 1:
                    #if none are violating then skip this query.
                    return None
    
                negNN = negNN[violatingNeg][:self.nNeg]
                negIndices = negSample[negNN].astype(np.int32)
                self.negCache[index] = negIndices
                self.negIndices = negIndices
        else:  ## use knn of faiss for speed up by ccsmm
            with h5py.File(self.cache, mode='r') as h5:  ## use knn of faiss
                h5feat = h5.get("features")
    
                qOffset = self.dbStruct.numDb 
                qFeat = h5feat[index+qOffset]
    
                posFeat = h5feat[self.nontrivial_positives[index].tolist()]
    
                dPos, posIndices = faiss_knn(posFeat, qFeat, K=5, metric='l2')
                self.dPos, self.posIndices = dPos, posIndices

                dPos = dPos[0]
                posIndices = posIndices[0]  # Get most similar image as a positive image
                posIndex = self.nontrivial_positives[index][posIndices].item()  # A nearest and most similar image.
                
                try:
                    negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
                except:
                    print("[Error] number of potential_negatives [must be > 0] = ", self.potential_negatives[index].size)
                    bp()
    
                negSample = np.unique(np.concatenate([self.negCache[index], negSample]))
                negFeat = h5feat[negSample.tolist()]

                dNeg, negNN = faiss_knn(negFeat, qFeat, K=self.nNeg*10, metric='l2')
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

    def get_queries_dir(self):
        return queries_dir

    def get_db_dir(self):
        return db_dir
