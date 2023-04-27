from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
import time

from tensorboardX import SummaryWriter
import numpy as np
import cv2
import imutils
import copy

from ipdb import set_trace as bp

from tasks.common import do_deattention, do_attention
from utils import misc, write_results
from utils.misc import add_clutter, save_feature, load_feature, get_pool_size_of_feat
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

from tasks.common import *

def convert_distance_to_confidence(distances=[], sigma=0.2):  # distances is list type
    confidences = []
    for dist in distances:
        conf = np.exp(-1*sigma*dist)
        confidences.append(conf)
    return confidences

def get_pool_size_of_feat(feat):
    return feat.shape[-1]

def indexing_descriptors(opt, dataset, dbFeat, qFeat, eval_set, epoch, writer, write_tboard, n_values=[1, 2, 3, 4, 5, 10, 15, 20, 25]):
    faiss_gpu = False  # False is defaults. Using GPU takes much time to copy data from mem to gpu mem.
    assert get_pool_size_of_feat(dbFeat) == get_pool_size_of_feat(qFeat)
    pool_size = get_pool_size_of_feat(dbFeat)
    if faiss_gpu == True:  # Use GPU for indexing with faiss. Do not use this due to low speed to copy data from memory to GPU.
        print('     Test[2/4] Building faiss index in GPU: ')
        ngpus = faiss.get_num_gpus()
        print("               Number of GPUs:", ngpus)
        cpu_index = faiss.IndexFlatL2(pool_size) # build a flat (CPU) index
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index.add(xb)         # add vectors to the index
        print('     Test[3/4] Indexing for calculating recall @ N in GPU: ')
        #n_values = [1, 5, 10, 20]
        #n_values = [1, 5, 10, 15, 20, 25]  #  for publish paper
        #n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]  #  for publish paper as like netvlad paper's figure 5.
    
        if opt.dataset.lower() == 'tokyotm':
            distances, predictions = gpu_index.search(qFeat, 200)  # 200 means large number
            distances, predictions = dataset.exclude_same_date(distances, predictions,
                    eval_set.dbStruct.dateQ, eval_set.dbStruct.dateDb, maxlen=max(n_values))
        else:
            distances, predictions = gpu_index.search(qFeat, max(n_values))
    else: # Use CPU for indexing with faiss
        print('     Test[2/4] Building faiss index : ')
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(dbFeat)
    
        print('     Test[3/4] Indexing for calculating recall @ N : ')
        #n_values = [1, 5, 10, 20]
        #n_values = [1, 5, 10, 15, 20, 25]  #  for publish paper
        #n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]  #  for publish paper as like netvlad paper's figure 5.
    
        if opt.dataset.lower() == 'tokyotm':
            distances, predictions = faiss_index.search(qFeat, 200)  # 200 means large number
            distances, predictions = dataset.exclude_same_date(distances, predictions,
                    eval_set.dbStruct.dateQ, eval_set.dbStruct.dateDb, maxlen=max(n_values))
        else:
            distances, predictions = faiss_index.search(qFeat, max(n_values))
    return distances, predictions

def acc_classify(conf, threshold=0.5):  # for acc.  [20]
    #return True  # Because recall score is very high, I will just return True which means it always succeess to find correct place
    conf_mean = conf.mean()
    if (conf_mean > threshold) :
        ret = True
    else:
        ret = False

    return ret

def acc_classify_to_do(confidences, qIx, top_k):  # for acc.  [20]
    #return True  # Because recall score is very high, I will just return True which means it always succeess to find correct place
    conf_k_std = confidences[qIx][:top_k].std()
    conf_25_std = confidences[qIx][:25].std()
    bp()
    if (conf_k_std >= conf_25_std*1.1) :
        ret = True
    else:
        ret = False

    return ret


def get_truth_for_recall(opt, pred, positive, n_idx, n_values, truth):
    truth = False
    matchedPos = np.intersect1d(pred[:n], positive)
    if len(matchedPos) > 0:   # TP : success to predict positive position
        recall_success = True
        recall_TP[i] += 1  # for report
    else:
        recall_success = False
        recall_FN[i] += 1  # for report

def get_conf_thre(opt, predictions, confidences, top_k=3):
    confidence_top_k = []
    if opt.conf_thre <= 0:
        for qIx, pred in enumerate(predictions):
            if len(confidence_top_k) == 0:
                confidence_top_k = confidences[qIx][:top_k]  # for debugging
            else:
                confidence_top_k = np.concatenate((confidence_top_k, confidences[qIx][:top_k]), axis=0)  # for debugging
        z = 2.58 # z = 1.96 for 95%,  z = 2.58 for 99% in normal distribution
        #conf_thre = confidence_top_k.mean() - z*confidence_top_k.std()/np.sqrt(len(predictions))  # https://m.blog.naver.com/iotsensor/222182891116
        conf_thre = confidence_top_k.mean() -  1.7*confidence_top_k.std() # https://m.blog.naver.com/iotsensor/222182891116
    else:
        conf_thre = opt.conf_thre
    return conf_thre, confidence_top_k

def get_recalls(opt, model, distances, predictions, dataset, dbFeat, qFeat, eval_set, epoch, writer, write_tboard, n_values=[1, 2, 3, 4, 5, 10, 15, 20, 25]):
    print('     Test[4/4] Calculating recall and accuracy @ N : ')

    confidences = convert_distance_to_confidence(distances)
    # For each query, get positive images's index within threshold distance
    if opt.recall_radius > 0.0:
        recall_radius = opt.recall_radius
    else:
        recall_radius = None  # Use default radius from dataset mat file
    gtPos = eval_set.getPositives(recall_radius)  # sample inside recall_radius
    correct_at_n = np.zeros(len(n_values))
    accuracy_at_n_nomi = np.zeros(len(n_values))  # for acc.

    recall_TP = np.zeros(len(n_values))  # for recall.
    recall_FN = np.zeros(len(n_values))  # for recall.
    accuracy_TP = np.zeros(len(n_values))  # for acc.
    accuracy_TN = np.zeros(len(n_values))  # for acc.
    accuracy_FP = np.zeros(len(n_values))  # for acc.
    accuracy_FN = np.zeros(len(n_values))  # for acc.

    ## Initialize Re-rank class
    if opt.rerank:
        from postprocessing.rerank_delf import rerank
        mRerank = rerank(eval_set, dataset)

    mSaveAVI = write_results.save_test_result_to_avi(opt, model, eval_set, dataset, fps=3)  # To do :change image name to opt.map_img

    #TODO can we do this on the matrix in one go?
    ## https://en.wikipedia.org/wiki/Precision_and_recall
    # TP : for actual positive gt(ground truth), correct prediction
    # FN : for actual positive gt, wrong prediction
    # FP : for actual negative gt, wrong prediction
    # TN : for actual negative gt, correct prediction

    conf_thre, confidence_top_k = get_conf_thre(opt, predictions, confidences, top_k=3)  # for report

    for qIx, pred in enumerate(predictions):
        if opt.rerank:
            t_start = time.time()
            pred, reranked = mRerank.run(qIx, pred, search_range=5, ratio=opt.rerank_ratio, disp_en=False)  # RE@1  0.87(base 0.85)
            #pred, reranked = mRerank.run(qIx, pred, search_range=20)  # RE@1 0.87(base 0.85)
            rerank_count = mRerank.get_reranked_cnt()
            t_end = time.time()
            print(" {} / {} , Reranked({}). It took {} sec.\r".format(qIx, len(predictions), rerank_count, t_end - t_start), end='')

        actual_positive = gtPos[qIx]

        ## For Recall
        ### True Positive (TP) : success quering for all query images
        recall_success = False  # recall success
        matched_n = 0
        for i,n in enumerate(n_values):
            TP = False
            matchedPos = np.intersect1d(pred[:n], actual_positive) #ccsmm, len(matched) > 0 : matched
            if len(matchedPos) > 0:   # TP : success to predict positive position
                recall_success = True
                recall_TP[i] += 1  # for report
                if matched_n == 0:
                    matched_n = n
            else:
                recall_success = False
                recall_FN[i] += 1  # for report

            if False :  # original code
                TP = recall_success
                if TP == True : # matched in top-1 ~ top-20
                    # If you hit in the top-n of n now, all cases greater than n are satisfied, so add 1 to all and break for loop.
                    correct_at_n[i:] += 1  # TP / (TP + FN) : correction positive prediction for all positive prediction (same as all here prediction)
                    break

        ## To do : Generate output images consitsting of q and top-k db images and their heatmap.
        mSaveAVI.save_to_avi(recall_success, qIx, gtPos, pred, dbFeat, qFeat, matched_at_top_k=matched_n, gt_max=1, sz_ratio=1.0)  # To do
        ## To do (end)

        ## For Accuracy
        ### True Positive (TP) : For a success in recall, when it is judged as a success
        ### True Positive (TN) : For a fail in recall, when it is judged as a fail
        recall_success = False  # recall success
        for i,n in enumerate(n_values):
            TP, FN, FP, TN = False, False, False, False
            matchedPos = np.intersect1d(pred[:n], actual_positive) #ccsmm, len(matched) > 0 : matched
            if len(matchedPos) > 0:
                recall_success = True
            else:
                recall_success = False

            acc_bool = acc_classify(confidences[qIx][:n], conf_thre)  # True or False
            #acc_bool = acc_classify(confidences, qIx, n) # True or False, for acc. It always uses only top-1.

            if recall_success == True:  # recall success
                if acc_bool == True:
                    TP = True
                    accuracy_TP[i] += 1  # for report
                else:
                    FN = True
                    accuracy_FN[i] += 1  # for report
            else:  # recall fail
                if acc_bool == True:
                    FP = True
                    accuracy_FP[i] += 1  # for report
                else:
                    TN = True
                    accuracy_TN[i] += 1  # for report

    recall_at_n = recall_TP / (recall_TP + recall_FN)  # TP / (TP + FN) 
    accuracy_at_n = (accuracy_TP + accuracy_TN) / (accuracy_TP + accuracy_TN + accuracy_FP + accuracy_FN)  # (TP+TN) / All, or (TP+TN) / (TP+TN+FP+FN)

    if False :  # original code
        recall_at_n = correct_at_n / eval_set.dbStruct.numQ  # TP / (TP + FN) 
        accuracy_at_n = accuracy_at_n_nomi / (eval_set.dbStruct.numQ)  # (TP+TN) / All, or (TP+TN) / (TP+TN+FP+FN)

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        if opt.report_verbose is not True:
            print("       Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            if writer is not None:
                writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    print("       -----------------------------")
    accs = {} #make dict for output
    if False:  # To do. Not implemented yet.
        for i,n in enumerate(n_values):
            accs[n] = accuracy_at_n[i]
            if opt.report_verbose is not True:
                print("       Accurcay@{}: {:.4f}".format(n, accuracy_at_n[i]))
            if write_tboard:
                if writer is not None:
                    writer.add_scalar('Val/Acc@' + str(n), accuracy_at_n[i], epoch)

    if opt.report_verbose is True:  # for project report
        if len(confidence_top_k) > 0:
            print("       Confidence (m, std) at top 3 : ({}, {})".format(confidence_top_k.mean(), confidence_top_k.std()))
        print("       Confidence threshold : {}".format(conf_thre))
        print("       Recall@{0:}, Accuracy@{1:}: {2:.4f}, {3:.4f}".format(n_values[0], n_values[2], recall_at_n[0], accuracy_at_n[2]))
        i = 0; TP, FN = recall_TP[i], recall_FN[i]
        print("       Recall@1   : TP, FN = {0:.4f}, {1:.4f}".format(TP, FN))
        print("                    100*TP / (TP+FN) =  100*{0:.4f} / {1:.4f} = {2:.4f}%".format( TP, (TP+FN), 100*TP / (TP+FN)))
        i = 2; TP, TN, FP, FN = accuracy_TP[i], accuracy_TN[i], accuracy_FP[i], accuracy_FN[i]
        print("       Accuracy@3 : TP, TN, FP, FN = {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}".format(TP, TN, FP, FN))
        print("                    100*(TP+TN) / (TP+TN+FP+FN) = 100*{0:.4f} / {1:.4f} = {2:.4f}%".format((TP+TN), (TP+TN+FP+FN), 100*(TP+TN)/(TP+TN+FP+FN)))

    return recalls, accs

def get_features(opt, model, seg_model, encoder_dim, eval_set):
    # TODO what if features dont fit in memory? 
    pool_size = encoder_dim
    if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters

    cuda, device = misc.get_device(opt)
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)
    model.eval()
    print('     Test[1/4] Extracting Features : DB[{}], Query[{}] with cachesize {}'.format(eval_set.dbStruct.numDb, eval_set.dbStruct.numQ, opt.cacheBatchSize))
    with torch.no_grad():
        dbFeat = np.empty((len(eval_set), pool_size))

        if opt.add_clutter_test_q:
            print("     Add clutter to query of testset. Iteration is {}".format(opt.add_clutter_iteration))
        if opt.add_clutter_test_db:
            print("     Add clutter to db of testset. Iteration is {}".format(opt.add_clutter_iteration))

        #import warnings
        #warnings.filterwarnings("ignore")  # To block matplotlib's warning messages
        enc_buf = None
        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            if seg_model is not None:
                do_add_clutter = False
                if iteration*opt.cacheBatchSize >= eval_set.dbStruct.numDb:  # for query
                    if opt.add_clutter_test_q:
                        do_add_clutter = True
                else:  # for db
                    if opt.add_clutter_test_db:
                        do_add_clutter = True
                if do_add_clutter == True:
                    input = add_clutter(opt, seg_model, input, indices, eval_set)

            if opt.do_not_process_only_for_writing_image:
                if iteration % 100 == 0:
                    print("(Just writing image mode) Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)
                continue

            input = misc.get_removed_clutter_input(opt, seg_model, input)  # Works when opt.remove_clutter_input is True
            input = misc.get_4ch_input(opt, seg_model, input, device)  # Works when opt.add_segmask_to_input_ch4 is True
            image_encoding = model.encoder(input)  # Local Feature Extractor
            write_results.display_result(opt, model, seg_model, input, image_encoding, iteration, indices, eval_set)

            image_encoding, _, _ = do_deattention(opt, model=model, query=input, image_encoding=image_encoding, run_mode="test")
            image_encoding, _ = do_attention(opt, model, image_encoding)  # tasks/common.py

            if False:  ## Accmulate local feature with ksize which means sequence of images
                if enc_buf is None:
                    ksize = 3  # For 1, it does not work. 1:0.64, 2:0.66, 3:0.65, 4:0.68
                    enc_buf = encoder_buf(image_encoding, ksize)
                image_encoding = enc_buf.make_seq_to_one(image_encoding)  # concatenate [B, C, H, W] to [B, C, H*ksize, W]

            descriptor = model.pool(image_encoding)   # descriptor is vlad with , descriptor_raw is vlad w/o norm.
            dbFeat[indices.detach().numpy(), :] = descriptor.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("       Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)
            del input, image_encoding, descriptor
    del test_data_loader
    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    return dbFeat, qFeat

class encoder_buf:
    def __init__(self, image_encoding, ksize=2, device=None):
        '''
        image_encoding's shape is [C,H,W]
        '''
        self.n_sample_count = 0 
        self.n_samples_max = 24  # maximum filter size of points for ransac
        if ksize < 1:
            self.ksize = 1  # minimum kernel_size (1 means bypass input to output)
        elif ksize > self.n_samples_max:
            self.ksize = self.n_samples_max  # maximum kernel_size
        else:
            self.ksize = ksize

        _, self.C, self.H, self.W = image_encoding.shape
        if device is None:
            self.device = image_encoding.device
        else:
            self.device = device
        self.data_buf = torch.zeros((self.n_samples_max, self.C, self.H, self.W))

    def make_seq_to_one(self, image_encoding):
        '''
        concatenate [B, C, H, W] to [B, C, H*ksize, W]
        '''
        B, C, H, W = image_encoding.shape
        H_new = H*self.ksize
        image_encoding_new = torch.empty(B, C, H_new, W)
        use_new_image_encoding = True
        for i in range(len(image_encoding)): # iter for batch size
            self.update_samples(image_encoding[i])
            image_encoding_seq = self.get_samples()
            if image_encoding_seq is None:
                use_new_image_encoding = False
                break
            image_encoding_concat = self.concat_enc(image_encoding_seq)
            if image_encoding_concat.shape[1] == image_encoding_new[i].shape[1]:
                image_encoding_new[i] = self.concat_enc(image_encoding_seq)
            else:
                use_new_image_encoding = False
                break
        if use_new_image_encoding == True:
            image_encoding = image_encoding_new

        if self.device is not None:
            image_encoding = image_encoding.to(self.device)

        return image_encoding

    def update_samples(self, image_encoding):
        '''
        remove an oldest sample and add a new image_encoding sample to self.data_buf
        image_encoding # C,H,W
        '''
        if self.n_sample_count < self.n_samples_max:
            self.n_sample_count += 1
        self.data_buf[:-1] = self.data_buf[1:].clone()  # shift data_buf :  data_buf[0,...,N-2] <<= data_buf[1,...,N-1]
        self.data_buf[-1,:] = image_encoding.clone() # add a new sample at the end of data_buf

    def get_samples(self):
        if self.n_sample_count <= 0:
            return None

        if self.n_sample_count < self.ksize:  # Data is not sufficient < ksize
            ksize = self.n_sample_count
        else:
            ksize = self.ksize

        if ksize == 0:
            return None
        else:
            return self.data_buf[-ksize:]

    def concat_enc(self, enc_seq):
        enc_concat = enc_seq[0]
        for i in range(1, len(enc_seq)): # concatenated ksize
            enc_concat = torch.cat((enc_concat, enc_seq[i]), dim=1)  # [ksize, 512, 30, 40] ==> [512, 30*ksize, 40]
        return enc_concat

def do_pca_whitening(opt, dbFeat, qFeat):
    if opt.pca_whitening == True:
        from postprocessing.netvladPCA import netvladPCA
        ## Initialize PCA class
        if opt.pca_whitening_mode == 0:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=False, inScale=False, outNorm=False)  #000
            print("PCA dimension reduction from {} to {}.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 1:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=False, inScale=False, outNorm=True)   #001
            print("PCA dimension reduction from {} to {}, with outNorm.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 10:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=False, inScale=True, outNorm=False)   #010
            print("PCA dimension reduction from {} to {}, with inScale.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 11:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=False, inScale=True, outNorm=True)    #011  # best recall
            print("PCA dimension reduction from {} to {}, with inScale, outNorm.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 100:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=True, inScale=False, outNorm=False)   #100
            print("PCA dimension reduction from {} to {}, with whiten.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 101:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=True, inScale=False, outNorm=True)    #101
            print("PCA dimension reduction from {} to {}, with whiten, outNorm.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 110:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=True, inScale=True, outNorm=False)    #110
            print("PCA dimension reduction from {} to {}, with whiten, inScale.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        elif opt.pca_whitening_mode == 111:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=True, inScale=True, outNorm=True)     #111
            print("PCA dimension reduction from {} to {}, with whiten, inScale, outNorm.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))
        else:
            mPCA = netvladPCA(dim=opt.pca_dim, whiten=False, inScale=False, outNorm=False)  #000
            print("PCA dimension reduction from {} to {}.".format(get_pool_size_of_feat(dbFeat), opt.pca_dim))

        ## Prepare PCA
        mPCA.fit(dbFeat)

        ## Do PCA
        dbFeat = mPCA.transform(dbFeat)
        qFeat = mPCA.transform(qFeat)

    return dbFeat, qFeat

def calc_time(opt, stime, stime_feat, dbFeat, qFeat):
    feat_time = stime_feat - stime
    recall_time = time.time() - stime_feat
    total_time = feat_time + recall_time #time.time() - stime

    numDb, numQ = len(dbFeat), len(qFeat)
    numTotal = numDb + numQ

    ## All process : [Sum(Db+Q), Sum(Db), Sum(Q), Single(Db), Single(Q)]
    t_time = total_time
    t_proc = [ t_time, t_time/numDb, t_time/numQ, t_time/numTotal]

    ## Feature
    t_time = feat_time
    t_feat = [ t_time, t_time/numDb, t_time/numQ, t_time/numTotal]

    ## Recall
    t_time = recall_time
    t_recall = [ t_time, t_time/numDb, t_time/numQ, t_time/numTotal]

    print("\n============= Elapsed time begin ============")
    print(" numDb, numQ, numTotal : {}, {}, {}".format(numDb, numQ, numTotal))
    print(" all_time, feat_time, recall_time : {}, {}, {} ".format(total_time, feat_time, recall_time))
    print(" db_feat_time, feat_time/image, recall_time/q_image : {}, {}, {}".format(numDb*(feat_time/numTotal), feat_time/numTotal, recall_time/numQ))
    #print(" Elapsed time :  Sum(Db+Q), Sum(Db), Sum(Q), Single(Db), Single(Q)")
    #print(" (all process)    {0:2.5f}, {1:2.5f}, {2:2.5f}, {3:2.5f}".format(t_proc[0], t_proc[1], t_proc[2], t_proc[3]))
    #print(" (Feat process)   {0:2.5f}, {1:2.5f}, {2:2.5f}, {3:2.5f}".format(t_feat[0], t_feat[1], t_feat[2], t_feat[3]))
    #print(" (Recall process) {0:2.5f}, {1:2.5f}, {2:2.5f}, {3:2.5f}".format(t_recall[0], t_recall[1], t_recall[2], t_recall[3]))
    #print("============= Elapsed time end   ============\n")

def test(opt, model, seg_model, encoder_dim, epoch, mDatasetStruct, writer, write_tboard=False, n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]):
    ## Prepare dataset
    dataset, _, _, _, eval_set = mDatasetStruct.get_dataset()

    ## Get descriptors (called features) of querys and databases
    valid_feature, dbFeat, qFeat, distances, predictions, _, _, _, _ = load_feature(opt)  # It works when opt.load_feature == True

    stime = time.time()

    if valid_feature == False:  # Default is False
        dbFeat, qFeat = get_features(opt, model, seg_model, encoder_dim, eval_set)
        distances, predictions = indexing_descriptors(opt, dataset, dbFeat, qFeat, eval_set, epoch, writer, write_tboard, n_values)
        save_feature(opt, dbFeat, qFeat, distances, predictions, eval_set)  # It works when opt.save_feature == True

    if opt.pca_whitening:
        dbFeat, qFeat = do_pca_whitening(opt, dbFeat, qFeat)
        distances, predictions = indexing_descriptors(opt, dataset, dbFeat, qFeat, eval_set, epoch, writer, write_tboard, n_values)
        fname = opt.feature_fname  # aaa/Feat.pickle
        fname = fname.replace(".pickle","_pcadim{}.pickle".format(get_pool_size_of_feat(dbFeat)))  # aaa/Feat_pcadim4096.pickle
        save_feature(opt, dbFeat, qFeat, distances, predictions, fname, eval_set)  # It works when opt.save_feature == True

    stime_feat = time.time()

    ## Calculate performance
    recalls, accs = get_recalls(opt, model, distances, predictions, dataset, dbFeat, qFeat, eval_set, epoch, writer, write_tboard, n_values) 

    calc_time(opt, stime, stime_feat, dbFeat, qFeat)

    return recalls, accs, n_values
