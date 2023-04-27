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

import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

def do_attention(opt, model, image_encoding):
    attention_name = []
    if opt.ch_attention:
        image_encoding = model.ch_attention(image_encoding)
        attention_name = "ch_attention"

    if opt.ch_eca_attention:
        image_encoding = model.ch_eca_attention(image_encoding)
        attention_name = "ch_eca_attention"

    if opt.senet_attention:
        image_encoding = model.senet_attention(image_encoding)
        attention_name = "senet_attention"

    if opt.bam_attention:
        image_encoding = model.bam_attention(image_encoding)
        attention_name = "bam_attention"

    if opt.cbam_attention:
        image_encoding = model.cbam_attention(image_encoding)
        attention_name = "cbam_attention"

    if opt.crn_attention:
        image_encoding, _ = model.crn_attention(image_encoding)
        attention_name = "crn_attention"

    return image_encoding, attention_name

def do_deattention_auto(opt, input, iteration, nBatches, model, seg_model, criterion_deatt, image_encoding, query, positives, negatives, indices, training_data_loader, save_interal_image=False, run_mode="train"):
    '''
    for train mode:
        image_encoding, deatt_loss, deatt_loss_detach = do_deattention_auto(opt, input, iteration, nBatches, model, seg_model, criterion_deatt, image_encoding, query, positives, negatives, indices, training_data_loader, save_interal_image, run_mode)
    for test mode:
        image_encoding, _, _ = do_deattention_auto(opt, model=model, image_encoding=image_encoding, run_mode="test")
    '''
    dbg_display = False
    deatt_loss_detach = 0
    loss_tml_detach = 0
    deatt_loss = 0
    mask_pred, mask_gt = None, None
    vlad_encoding, vladQ, vladP, vladN = None, None, None, None
    if opt.deattention_auto:
        mask_pred, mask_weight = model.deattention_auto(image_encoding) # mask_pred is [w1, w2, w3, w4, w5, w6]
        if run_mode.lower() == "train":
            if dbg_display == True:  # debugging display
                tim.clf()
            mask_gt_dict = seg_model.preprocess_auto(input.detach()) # Ground Truth of segmentation
            key_list = [a for a in mask_gt_dict.keys()]
            for i, category_str in enumerate(key_list):  # key_list : ['nature', 'sky', 'human', 'vehicle', 'flat', 'construction']
                mask_gt = mask_gt_dict[category_str]
                mask_gt = torch.nn.functional.interpolate(mask_gt.float(), size=(30,40), mode="bicubic", align_corners=True)
                deatt_loss += criterion_deatt(mask_gt, mask_pred[i])
                if dbg_display == True:  # debugging display
                    row, col = 3, 6
                    mask_pred_detach = mask_pred[i].detach()
                    denorm = tim.Denormalize()
                    tim.imshow(mask_gt[0], (row,col,i+1), title='GT({})'.format(category_str), cmap='gray', dispEn=False)
                    tim.imshow(mask_pred_detach[0], (row, col, col+1+i), title='Pred({})'.format(category_str), cmap='gray', dispEn=False)
            if dbg_display == True:  # debugging display
                tim.imshow(img=denorm(query[0]), sp=(row, col, col*2+1), title='Query', dispEn=True)
                ws = mask_weight[0,:]
                tim.plot(y=ws, sp=(row, col, col*2+2), title="weight", dispEn=True)
                #tim.plt.savefig("a.png", dpi=300)
            deatt_loss_detach = deatt_loss.detach()
        if True:  # Learned weight
            # mask_pred[i] : [B,1,30,40] 
            # mask_weight  # [B, 6]
            # mask_weight[:,0].view(-1,1,1,1)  # [B, 1, 1, 1]
            w0 = mask_weight[:,0].view(-1,1,1,1)
            w1 = mask_weight[:,1].view(-1,1,1,1)
            w2 = mask_weight[:,2].view(-1,1,1,1)
            w3 = mask_weight[:,3].view(-1,1,1,1)
            w4 = mask_weight[:,4].view(-1,1,1,1)
            w5 = mask_weight[:,5].view(-1,1,1,1)
            deatt_mask_pred = mask_pred[0]*w0 + mask_pred[1]*w1 + mask_pred[2]*w2 + mask_pred[3]*w3 + mask_pred[4]*w4 + mask_pred[5]*w5 
        else:  # manual object weight
            ##           ["nature", "sky", "human", "vehicle", "flat", "construction"]
            w0,w1,w2,w3,w4,w5 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0  # To be learned, weight of ["nature", "sky", "human", "vehicle", "flat", "construction"] mask
            deatt_mask_pred = w0*mask_pred[0] + w1*mask_pred[1] + w2*mask_pred[2] + w3*mask_pred[3] + w4*mask_pred[4] + w5*mask_pred[5]
        deatt_mask_pred = torch.sigmoid(deatt_mask_pred)
        if False:  # debugging display for test mode
            tim.clf()
            denorm = tim.Denormalize()
            img_idx=0
            category = ["nature", "sky", "human", "vehicle", "flat", "construction"]
            for i in range(6):
                tim.imshow(mask_pred[i][img_idx], sp=331+i, title=category[i])
            tim.imshow(deatt_mask_pred[img_idx], sp=331+6, title="weighted_sum")
            tim.imshow(img=denorm(query[img_idx]), sp=331+7, title="input")
            bp()

        image_encoding = image_encoding * deatt_mask_pred

    return image_encoding, deatt_loss, deatt_loss_detach


#def do_deattention(opt, input=None, iteration, nBatches, model, seg_model, criterion_deatt, image_encoding, query, positives, negatives, indices, training_data_loader, save_interal_image=False, run_mode="train"):
def do_deattention(opt, input=None, iteration=None, nBatches=None, model=None, seg_model=None, criterion_deatt=None, image_encoding=None, query=None, positives=None, negatives=None, indices=None, training_data_loader=None, save_interal_image=False, run_mode="train"):
    deatt_loss_detach = 0
    loss_tml_detach = 0
    deatt_loss = 0
    mask_pred, mask_gt = None, None
    vlad_encoding, vladQ, vladP, vladN = None, None, None, None

    if opt.deattention_auto == True:
        image_encoding, deatt_loss, deatt_loss_detach = do_deattention_auto(opt, input, iteration, nBatches, model, seg_model, criterion_deatt, image_encoding, query, positives, negatives, indices, training_data_loader, save_interal_image, run_mode)

    elif opt.deattention == True:
        image_encoding, mask_pred = model.deattention(image_encoding)
        if run_mode.lower() == "train":
            _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation
            mask_gt = torch.nn.functional.interpolate(mask_gt.float(), size=(30,40), mode="bicubic", align_corners=True)
            deatt_loss = criterion_deatt(mask_gt, mask_pred)
            deatt_loss_detach = deatt_loss.detach()
            mask_pred_detach = mask_pred.detach()
            if (save_interal_image == True) and (iteration % 50 == 0 or nBatches <= 50) :  # debugging
                tim.clf()
                denorm = tim.Denormalize()
                tim.imshow(img=denorm(query[0]), sp=331, title='Query', dispEn=False)
                tim.imshow(img=denorm(positives[0]), sp=332, title='Pos', dispEn=False)
                tim.imshow(img=denorm(negatives[0]), sp=333, title='Neg', dispEn=False)
                tim.imshow(mask_gt[0], 334, title='SegMask-GT', cmap='gray', dispEn=False)
                tim.imshow(mask_pred_detach[0], 335, title='SegMask-pred', cmap='gray', dispEn=False)
                mask_diff = mask_gt[0] - mask_pred_detach[0]
                tim.imshow(mask_diff, 336, title='Error(GT-Pred)', cmap='gray', dispEn=False)
                imgpath = training_data_loader.dataset.dataset.dbStruct.qImage[indices[0]]  # I need to check variable, qImage ?
                parentname = os.path.dirname(imgpath)
                imgname = os.path.basename(imgpath)
                ofpath = os.path.join(opt.deatt_result_dir, parentname)
                if not os.path.exists(ofpath):
                    os.makedirs(ofpath)
                ofname = os.path.join(ofpath, "deatt_{}".format(imgname))
                tim.plt.savefig(ofname, dpi=300)
    return image_encoding, deatt_loss, deatt_loss_detach

