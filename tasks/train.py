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

from tensorboardX import SummaryWriter
import numpy as np
from utils import misc  # This code is called by ../main.py. So working directory is ../
from ipdb import set_trace as bp

from tasks.test import test
from tasks.common import do_deattention, do_attention
from utils import custom_loss
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

statistics = custom_loss.MeanStd()
import seaborn as sns
import pandas as pd

def train_single_tuple(opt, epoch, iteration, model, seg_model, optimizer, criterion, criterion_deatt, w_deatt_loss, epoch_loss, nBatches, query, positives, negatives, negCounts, indices, writer, device, training_data_loader):
    '''
    Core train process
    '''
    B, C, H, W = query.shape
    nNeg = torch.sum(negCounts)
    input = torch.cat([query, positives, negatives])
    input = input.to(device)

    if opt.add_clutter_train and (seg_model is not None):  # Do not use in cache building.
        _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation
        input = seg_model.add_clutter_to_cv_img(input.detach(), mask_gt, iteration=opt.add_clutter_iteration)

    input = misc.get_removed_clutter_input(opt, seg_model, input)  # Works when opt.remove_clutter_input is True
    input = misc.get_4ch_input(opt, seg_model, input, device)  # Works when opt.add_segmask_to_input_ch4 is True

    image_encoding = model.encoder(input)

    image_encoding, deatt_loss, deatt_loss_detach = do_deattention(opt, input, iteration, nBatches, model, seg_model, criterion_deatt, image_encoding, query, positives, negatives, indices, training_data_loader, save_interal_image=False)

    image_encoding, _ = do_attention(opt, model, image_encoding)

    vlad_encoding = model.pool(image_encoding) 
    vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])
    
    # calculate loss for each Query, Positive, Negative triplet
    # due to potential difference in number of negatives have to 
    # do it per query, per negative
    loss = 0
    for i, negCount in enumerate(negCounts):
        for n in range(negCount):
            negIx = (torch.sum(negCounts[:i]) + n).item()
            loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])
            statistics.push(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

    if nNeg > 0:
        loss /= nNeg.float().to(device) # normalise by actual number of negatives
    else:
        print("[Warning] At iteration : {}, nNeg = {}".format(iteration, nNeg))

    if opt.deattention or opt.deattention_auto:
        loss += (w_deatt_loss*deatt_loss)

    optimizer.zero_grad()  # optimizer.zero_grad() should run right before loss.backward()
    loss.backward()   # Differentiate : delta = (dL)/(dW),  (==dLoss/dWeight)
    optimizer.step()  # # Backpropagation(or Update weight) : w = w - a*delta,  where a is learning rate (lr)

    del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
    del query, positives, negatives

    batch_loss = loss.item()
    epoch_loss += batch_loss

    if iteration % 100 == 0 or nBatches <= 100:
        print("   ===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, nBatches, batch_loss), flush=True)
        if writer is not None:
            n_iter = (epoch-1) * nBatches + iteration
            ## Write Loss
            writer.add_scalar('Train/Loss', epoch_loss/iteration, n_iter)
            writer.add_scalar('Train/nNeg', nNeg, n_iter)
            writer.add_scalar('Train/Loss_DeAttx{}'.format(w_deatt_loss), deatt_loss_detach*w_deatt_loss, n_iter)

            ## Write dqp, dqn, dpn
            d_qp_m, d_qn_m, d_pn_m, angle_nqp_m = statistics.get_mean()
            d_qp_s, d_qn_s, d_pn_s, angle_nqp_s = statistics.get_std()
            if False:
                writer.add_scalar('Train/d_qP_mean', d_qp_m , n_iter)
                writer.add_scalar('Train/d_qN_mean', d_qn_m , n_iter)
                writer.add_scalar('Train/d_PN_mean', d_pn_m , n_iter)
                writer.add_scalar('Train/angle_NqP_mean', angle_nqp_m , n_iter)
                writer.add_scalar('Train/d_qP_std', d_qp_s , n_iter)
                writer.add_scalar('Train/d_qN_std', d_qp_s , n_iter)
                writer.add_scalar('Train/d_PN_std', d_pn_s , n_iter)
                writer.add_scalar('Train/angle_NqP_std', angle_nqp_s , n_iter)

            ## Group
            writer.add_scalars('Train/Loss_elements', {"loss": epoch_loss/iteration} , n_iter)
            writer.add_scalars('Train/Loss_elements', {"d_qp": d_qp_m} , n_iter)
            writer.add_scalars('Train/Loss_elements', {"d_qn": d_qn_m} , n_iter)
            writer.add_scalars('Train/Loss_elements', {"d_qp-d_qn": d_qp_m - d_qn_m} , n_iter)
            writer.add_scalars('Train/Loss_elements', {"d_margin": misc.get_margin(opt)}, n_iter)
            writer.add_scalars('Train/Loss_elements', {"d_pn": d_pn_m} , n_iter)
            writer.add_scalars('Train/d_qP', {"mean": d_qp_m} , n_iter)
            writer.add_scalars('Train/d_qP', {"std": d_qp_s} , n_iter)
            writer.add_scalars('Train/d_qN', {"mean": d_qn_m} , n_iter)
            writer.add_scalars('Train/d_qN', {"std": d_qn_s} , n_iter)
            writer.add_scalars('Train/d_PN', {"mean": d_pn_m} , n_iter)
            writer.add_scalars('Train/d_PN', {"std": d_pn_s} , n_iter)
            writer.add_scalars('Train/angle_NqP', {"mean": angle_nqp_m} , n_iter)
            writer.add_scalars('Train/angle_NqP', {"std": angle_nqp_s} , n_iter)

            #write_data_distribution(opt, iteration)

        #print('Allocated:', torch.cuda.memory_allocated())
        #print('Cached:', torch.cuda.memory_cached())

    del loss
    return epoch_loss

def write_data_distribution_bak(opt, iteration=0):
    if opt.write_data_distribution_in_train:
        if opt.tml2:
            ofname = "data_distribution_eTML_margin{}".format(misc.get_margin(opt))
        else:
            ofname = "data_distribution_TML_margin{}".format(misc.get_margin(opt))

        if not opt.resume:
            ofname = "{}_untrained".format(ofname)

        ofname_csv = "{}.csv".format(ofname)
        ofname_png1 = "{}_histo.png".format(ofname)
        ofname_png2 = "{}_histo2.png".format(ofname)
        ofname_png3 = "{}_kde.png".format(ofname)

        whole_d_qp, whole_d_qn, _, _ = statistics.get_whole_data()
        data = np.vstack((whole_d_qp, whole_d_qn)).T  # [N x 2]
        df = pd.DataFrame(data, columns=["d_qp", "d_qn"])
        df.to_csv(ofname_csv, index=False)

        if os.path.exists(ofname_png1):
            os.remove(ofname_png1)

        if os.path.exists(ofname_png2):
            os.remove(ofname_png2)

        if os.path.exists(ofname_png3):
            os.remove(ofname_png3)

        if True:  # csv
            data = pd.read_csv(ofname_csv)
        else:  # dict
            data = {"d_qn":whole_d_qn, "d_qp":whole_d_qp}

        fig = sns.displot(data, fill=True);fig.set_xlabels="distance"; fig.set_titles="Iter:{}".format(iteration)
        fig.savefig(ofname_png1, dpi=100)  # histo

        fig = sns.displot(data, kde=True, fill=True);fig.set_xlabels="distance"; fig.set_titles="Iter:{}".format(iteration)
        fig.savefig(ofname_png2, dpi=100)  # histo2

        fig = sns.displot(data, kind="kde", fill=True);fig.set_xlabels="distance"; fig.set_titles="Iter:{}".format(iteration)
        fig.savefig(ofname_png3, dpi=100)  # kde

def write_data_distribution(opt, iteration=0):
    if opt.write_data_distribution_in_train:
        out_dir = "graph/distance_distribution"
        if opt.tml2:
            ofname = "data_distribution_eTML_margin{}_dataloadermargin{}".format(misc.get_margin(opt), opt.dataloader_margin)
        else:
            ofname = "data_distribution_TML_margin{}_dataloadermargin{}".format(misc.get_margin(opt), opt.dataloader_margin)

        if not opt.resume:
            ofname = "{}_untrained".format(ofname)

        ofname_csv = os.path.join(out_dir, "{}.csv".format(ofname))
        whole_d_qp, whole_d_qn, _, _ = statistics.get_whole_data()

        misc.write_data_to_csv(ofname_csv, whole_d_qp, whole_d_qn)
        misc.write_png_from_csv(ofname_csv, min(6000, len(whole_d_qp)))

def evaluate(opt, model, seg_model, encoder_dim, epoch, mDatasetStruct, optimizer, best_score, not_improved, writer):
    if (epoch % opt.evalEvery) == 0:
        best_recall_at = 1  # key of recalls. one element of n_values. Recall @ 1
        print("   ===> Epoch[{}] Evaluating ...".format(epoch), flush=True)
        recalls, accs, n_values = test(opt, model, seg_model, encoder_dim, epoch, mDatasetStruct, writer, write_tboard=True)
        is_best = recalls[best_recall_at] > best_score
        if is_best:
            not_improved = 0
            best_score = recalls[best_recall_at]
        else:
            not_improved += 1

        misc.save_checkpoint({
               'epoch': epoch,
               'state_dict': model.state_dict(),
               'recalls': recalls,
               'best_score': best_score,
               'optimizer' : optimizer.state_dict(),
               'parallel' : misc.get_isParallel(opt),
               'deatt_category_list' : opt.deatt_category_list,
        }, is_best, opt.savePath)

    return best_score, not_improved, best_recall_at

def build_feature_cache(opt, model, seg_model, epoch, train_set, encoder_dim, train_set_for_cache, reuse_cache_for_debug=False):
    '''
    Build feature cache for all images which will be use to query postive and negavtive for given query image.
    '''
    backup = os.path.join(os.getcwd(), os.path.basename(train_set.cache))
    os.system("rm -f {}".format(train_set.cache))
    need_to_build_cache = True
    if reuse_cache_for_debug:
        print('   ===> Epoch[{}] : Reuse feature cache for debugging. [Warning] Do not use this option for normal train.'.format(epoch))
        print("                   To disable this, remove --reuse_cache_for_debug option in command line.")
        if os.path.exists(train_set.cache):
            need_to_build_cache = False
        elif os.path.exists(backup):
            try:
                os.system("ln -sf {} {}".format(backup, train_set.cache))
                need_to_build_cache = False
            except:
                print("You don't have {} for backup".format(backup))
                need_to_build_cache = True

    if need_to_build_cache:
        cuda, device = misc.get_device(opt)
        print('   ===> Epoch[{}] : Building Cache'.format(epoch))
        model.eval()
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", 
                    [len(train_set_for_cache.dataset.images), pool_size], 
                    dtype=np.float32)  # 17416 for pitts30k
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(train_set_for_cache, 1):
                    input = input.to(device)
                    input = misc.get_4ch_input(opt, seg_model, input, device)  # Works when opt.add_segmask_to_input_ch4 is True
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding) 
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding
        if os.path.exists(backup) == False:
             os.system("cp -f {} {}".format(train_set.cache, backup))  # for debugging purpose

def train_single_epoch(opt, model, seg_model, encoder_dim, optimizer, scheduler, criterion, criterion_deatt, w_deatt_loss, epoch, mDatasetStruct, writer):
    dataset, train_set_for_cache, train_set, _, _ = mDatasetStruct.get_dataset()
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging
    cuda, device = misc.get_device(opt)

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    cache_prefix = os.path.basename(os.getcwd())
    train_set.cache = join(opt.cachePath, cache_prefix + '_' + train_set.whichSet + '_feat_cache.hdf5')
    for subIter in range(subsetN):
        ## Build feature cache
        build_feature_cache(opt, model, seg_model, epoch, train_set, encoder_dim, train_set_for_cache, reuse_cache_for_debug=opt.reuse_cache_for_debug)

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])
        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=cuda)

        
        if opt.write_data_distribution_in_train:  # For analysis
            model.eval()
        else: # default
            model.train()
        statistics.clear_buffer();
        for iteration, (query, positives, negatives, negCounts, indices) in enumerate(training_data_loader, startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch
            epoch_loss = train_single_tuple(opt, epoch, iteration, model, seg_model, optimizer, criterion, criterion_deatt, w_deatt_loss, epoch_loss, nBatches, query, positives, negatives, negCounts, indices, writer, device, training_data_loader)

        startIter += len(training_data_loader)
        del training_data_loader
        torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    write_data_distribution(opt, iteration)
    avg_loss = epoch_loss / nBatches
    print("   ===> Epoch[{}] Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), flush=True)
    if writer is not None:
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def schedule_w_deatt_loss(opt, epoch, epoch_begin=1, w_deatt_loss_prev=0):
    if opt.w_deatt_loss == 0: # scheduling w_deatt_loss according to epoch
        if epoch <= epoch_begin + 3:  # epoch 1,2,3,4
            w_deatt_loss = 0.1  ## Focus on deattention strongly.
        else:  ## Focus on TML Loss 
            w_deatt_loss = 0.001  # 5,6, ...
        if w_deatt_loss_prev != w_deatt_loss:
            print(" *** Scheduling w_deatt_loss according to the epoch [{}] : {} ==> {}".format(epoch, w_deatt_loss_prev, w_deatt_loss))
    else:  # Fixed w_deatt_loss
        w_deatt_loss = opt.w_deatt_loss
    return w_deatt_loss

def train(opt, model, seg_model, encoder_dim, optimizer, scheduler, mDatasetStruct, writer):
    not_improved = 0
    best_score = 0
    criterion = misc.get_loss_function(opt)
    criterion_deatt = misc.get_loss_function_for_deatt(opt)
    w_deatt_loss = None

    statistics.set_keep_whole_data(opt.write_data_distribution_in_train)

    if opt.save_ckpt_without_train == True:  # To save weight(from pretrained by ImageNet) without any training.
        epoch = 1
        best_score, not_improved, best_recall_at = evaluate(opt, model, seg_model, encoder_dim, epoch, mDatasetStruct, optimizer, best_score, not_improved, writer)
        print("   ===> Best Recall@{0:d}: {1:.4f}".format(best_recall_at, best_score), flush=True)
        return

    if opt.write_data_distribution_in_train:
        opt.nEpochs = opt.start_epoch+1  # not for train. for writing data distribution png
    for epoch in range(opt.start_epoch+1, opt.nEpochs + 1): # opt.start_epoch is usually 0.
        w_deatt_loss = schedule_w_deatt_loss(opt, epoch, opt.start_epoch+1, np.copy(w_deatt_loss))
        train_single_epoch(opt, model, seg_model, encoder_dim, optimizer, scheduler, criterion, criterion_deatt, w_deatt_loss, epoch, mDatasetStruct, writer)
        best_score, not_improved, best_recall_at = evaluate(opt, model, seg_model, encoder_dim, epoch, mDatasetStruct, optimizer, best_score, not_improved, writer)
        if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
            print('   ===> Performance did not improve for', opt.patience, 'epochs. Stopping.')
            break

    print("   ===> Best Recall@{0:d}: {1:.4f}".format(best_recall_at, best_score), flush=True)
