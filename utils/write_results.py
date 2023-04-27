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
import utm
import cv2
import copy

from tensorboardX import SummaryWriter
import numpy as np

from ipdb import set_trace as bp

from tasks.common import do_deattention, do_attention
from utils import misc
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )  # Supress warning of plt.subplot(sp) for r, g ,b color to draw GT.

def get_heatmap(cv_img, opt, model):  ## To do : not completed
    '''
    Shape of cv_img is 480 * 640 * 3 [H * W * C] dimensional.
    '''
    _, device = misc.get_device(opt)
    torch_img = tim.cv_to_tensor(cv_img).unsqueeze(0)  # 1*3*480*640
    torch_img = torch_img.to(device)
    normEn = False
    contrast = 1.0;brightness = 0.0

    image_encoding = model.encoder(torch_img)  # Local Feature Extractor
    if opt.deattention or opt.deattention_auto:
        image_encoding, mask_pred = model.deattention(image_encoding)  # image_encoding, mask_pred
    image_encoding, attention_name = do_attention(opt, model, image_encoding)  # tasks/common.py
    batch_idx = 0
    cv_img = np.float32(cv_img) / 255
    feat_norm, feat_heatmap, mask, img_jet = tim.heatmap(cv_img, image_encoding[batch_idx], ofname=None, brightness=brightness, contrast=contrast, normEn=normEn)  # without deattention

    return img_jet

def display_result(opt, model, seg_model, input, image_encoding, iteration, indices, eval_set):
    if opt.write_heatmap == True:
        if opt.deattention or opt.deattention_auto:
            image_encoding_ori = image_encoding.clone()
            image_encoding_deatt, mask_pred = model.deattention(image_encoding)  # image_encoding, mask_pred
            image_encoding_deatt_att, attention_name = do_attention(opt, model, image_encoding_deatt)  # tasks/common.py
            # for debugging#########
            #image_encoding_deatt2 = image_encoding_ori + image_encoding_deatt
            #bp()
            #############
            input = misc.get_3ch_input(opt, input)  # Works when opt.add_segmask_to_input_ch4 is True
            _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation
            mask_gt = torch.nn.functional.interpolate(mask_gt.float(), size=(30,40), mode="bicubic", align_corners=True)
            mask_pred_detach = mask_pred.detach()
            if (iteration % 1 == 0) :  # debugging
                imgidx=0
                tim.fig(100, figsize=(12,12))
                tim.clf()
                denorm = tim.Denormalize()
                ## Input
                tim.imshow(img=denorm(input[imgidx]), sp=331, title='Input', dispEn=False)

                ## Mask (Ground Truth)
                y1, y2, y3 = 15, 20, 25

                sp=334; title='SegMask-GT'
                tim.imshow(mask_gt[imgidx], sp, title=title, cmap='gray', dispEn=False)
                # plot line position on mask GT
                ymax, xmax = mask_gt.shape[-2:]
                y=y1;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="r:")  # 334,  sp=0 means same as previous
                y=y2;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="g:")  # sp=0 means same as previous
                y=y3;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="b:")  # sp=0 means same as previous
                tim.plt.title(title)
                tim.plt.ylim(0,ymax)
                tim.plt.gca().invert_yaxis()  # Fit plot axit to image's
                # plot data on mask GT
                y=y1;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=sp+3, color="r:")  # sp 337
                y=y2;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=0   , color="g:")  # sp=0 means same as previous
                y=y3;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=0   , color="b:")  # sp=0 means same as previous
                tim.plt.title(title)
                #tim.plt.grid()

                ## Mask (Prediction)
                sp=335; title='SegMask-pred'
                tim.imshow(mask_pred_detach[imgidx], sp, title=title, cmap='gray', dispEn=False)
                mask_diff = mask_gt[imgidx] - mask_pred_detach[imgidx]
                # plot line position on mask predection
                ymax, xmax = mask_gt.shape[-2:]
                y=y1;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="r")  # 335, sp=0 means same as previous
                y=y2;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="g")  # sp=0 means same as previous
                y=y3;tim.plot(np.arange(xmax).tolist(), (np.ones(xmax)*y).tolist(), sp=0, color="b")  # sp=0 means same as previous
                tim.plt.ylim(0,ymax)
                tim.plt.gca().invert_yaxis()  # Fit plot axit to image's
                tim.plt.title(title)
                # plot data on mask GT
                y=y1;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=sp+3, color="r:")   # 338, sp=0 means same as previous
                y=y2;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=0, color="g:")  # sp=0 means same as previous
                y=y3;tim.plot(range(mask_gt.shape[-1]), mask_gt[0,0,y,:], sp=0, color="b:")  # sp=0 means same as previous
                # plot data on mask prediction
                y=y1;tim.plot(range(mask_pred_detach.shape[-1]), mask_pred_detach[0,0,y,:], sp=0, color="r")  # 338,  sp=0 means same as previous
                y=y2;tim.plot(range(mask_pred_detach.shape[-1]), mask_pred_detach[0,0,y,:], sp=0, color="g")  # sp=0 means same as previous
                y=y3;tim.plot(range(mask_pred_detach.shape[-1]), mask_pred_detach[0,0,y,:], sp=0, color="b")  # sp=0 means same as previous
                tim.plt.ylim(-0.1, 1.1)
                #tim.plt.grid()
                tim.plt.title(title)
                
                ## Mask (Difference)
                tim.imshow(mask_diff, 336, title='Error(GT-Pred)', cmap='gray', dispEn=False)

                ## output filname
                imgpath = eval_set.images[indices[imgidx]]
                parentname = os.path.dirname(imgpath).split("/")[-1]
                imgname = os.path.basename(imgpath)
                ofpath = os.path.join(opt.heatmap_result_dir, parentname)
                if not os.path.exists(ofpath):
                    os.makedirs(ofpath)
                ofname = os.path.join(ofpath, "deatt_{}".format(imgname))
                ofname_heatmap_ori = os.path.join(ofpath, "deatt_heatmap0ori_{}".format(imgname))
                ofname_heatmap_deatt = os.path.join(ofpath, "deatt_heatmap1deatt_{}".format(imgname))
                ofname_heatmap_deatt_then_att = os.path.join(ofpath, "deatt_heatmap2deatt_then_att{}".format(imgname))

                ## Get attention map
                # Brightness and contrast were used to visualize the relative comparison of the two images.
                #contrast = 1.0;brightness = 0.3
                normEn = False
                contrast = 1.0;brightness = 0.0
                feat_norm_ori, feat_heatmap_ori, mask_ori, img_jet_ori = tim.heatmap(imgpath, image_encoding_ori[imgidx], ofname_heatmap_ori, brightness=brightness, contrast=contrast, normEn=normEn)  # without deattention
                contrast = 1.0;brightness = 0.0
                feat_norm_deatt, feat_heatmap_deatt, mask_deatt, img_jet_deatt = tim.heatmap(imgpath, image_encoding_deatt[imgidx], ofname_heatmap_deatt, brightness=brightness, contrast=contrast, normEn=normEn)  # deattention

                tim.imshow(img=tim.img_cv2_to_plt(img_jet_ori), sp=332, title='heatmap-input', cmap=None, dispEn=False)
                tim.imshow(img=tim.img_cv2_to_plt(img_jet_deatt), sp=333, title='heatmap-deatt', cmap=None, dispEn=False)

                if len(attention_name) > 0:
                    _, _, _, img_jet_deatt_then_att = tim.heatmap(imgpath, image_encoding_deatt_att[imgidx], ofname_heatmap_deatt_then_att, brightness=brightness, contrast=contrast, normEn=normEn)  # deattention
                    tim.imshow(img=tim.img_cv2_to_plt(img_jet_deatt_then_att), sp=339, title='heatmap-deatt-then-att', cmap=None, dispEn=False)

                tim.plt.savefig(ofname, dpi=300)
        else:  # without deattention
            if (iteration % 1 == 0) :  # debugging
                input = misc.get_3ch_input(opt, input)  # Works when opt.add_segmask_to_input_ch4 is True
                imgidx=0
                tim.clf()
                denorm = tim.Denormalize()
                tim.imshow(img=denorm(input[imgidx]), sp=131, title='Input', dispEn=False)
                ## output filname
                imgpath = eval_set.images[indices[imgidx]]
                parentname = os.path.dirname(imgpath).split("/")[-1]
                imgname = os.path.basename(imgpath)
                ofpath = os.path.join(opt.heatmap_result_dir, parentname)
                if not os.path.exists(ofpath):
                    os.makedirs(ofpath)
                ofname = os.path.join(ofpath, "baseline_{}".format(imgname))
                ofname_heatmap = os.path.join(ofpath, "baseline_heatmap_{}".format(imgname))

                ## Get attention map
                normEn = False
                image_encoding_ori = image_encoding.clone()
                _, _, _, img_jet = tim.heatmap(imgpath, image_encoding_ori[imgidx], ofname_heatmap, normEn=normEn)  # without deattention
                tim.imshow(img=tim.img_cv2_to_plt(img_jet), sp=132, title='heatmap-input', cmap=None, dispEn=False)

                image_encoding_att, attention_name = do_attention(opt, model, image_encoding_ori)  # tasks/common.py
                if len(attention_name) > 0:
                    _, _, _, img_jet = tim.heatmap(imgpath, image_encoding_att[imgidx], ofname_heatmap, normEn=normEn)  # without deattention
                    tim.imshow(img=tim.img_cv2_to_plt(img_jet), sp=133, title='heatmap-{}'.format(attention_name), cmap=None, dispEn=False)

                tim.plt.savefig(ofname, dpi=300)

def GetDbImg(filename):
    # filename : 'dbImg/200204/010690.jpg'
    basename = os.path.basename(filename).split('.')[0] # dbImg/200204/010690
    parent_dir = filename.split('/')[-2] # 200204
    fullname = os.path.join(dataset.db_dir, filename)
    Img = Image.open(fullname)
    return basename, parent_dir, fullname, Img

def GetQImg(filename):
    # filename : 'qImg/200204/010690.jpg'
    basename = os.path.basename(filename).split('.')[0] # qImg/200204/010690
    parent_dir = filename.split('/')[-2] # 200204
    fullname = os.path.join(dataset.queries_dir, filename)
    Img = Image.open(fullname)
    return basename, parent_dir, fullname, Img

def write_matched_image(N, matched, qIx, Preds, Confs, GTs, eval_set, writer, log_dir):
    start_time = time.time()

    fig = plt.figure(figsize=(16,12)) # (w, h) in inches
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    qFn, qPdn, qPn, qImg = GetQImg(eval_set.dbStruct.qImage[qIx]) # Query
    dbPredFn, dbPredPdn, dbPredPn, dbPredImg = GetDbImg(eval_set.dbStruct.dbImage[Preds[0]]) # Predicted DB
    dbConf = Confs[0]

    fig.add_subplot(3,3,1)
    write_all_pos_on_map(N, matched, qIx, Preds[0], GTs, eval_set, writer)
    plt.title("All pos.")

    fig.add_subplot(3,3,2)
    write_matched_pos_on_map(N, matched, qIx, Preds[0], GTs, eval_set, writer)
    #plt.title("Zoom-in pos.[meters]")

    fig.add_subplot(3,3,3)
    plt.plot(Confs, 'g*')
    plt.xlabel("Top-N")
    plt.ylabel("Similarity (%)")
    plt.ylim(0,100)
    plt.xlim(0,20)
    plt.grid()
    plt.title("Confidence\nMax : {}".format(Confs[0]))
    # plt.plot((Confs/np.max(Confs)))
    # plt.title("Confidence % to max\n{}".format(Confs[0]))

    if matched:
        title = "Matched at top-{}".format(N)
        fig.add_subplot(2,2,3)
        plt.imshow(qImg)
        plt.title('[Q] {} in {}'.format(qFn, qPdn))

        fig.add_subplot(2,2,4)
        plt.imshow(dbPredImg)
        plt.title('[DB-Matched] {} in {}\nConfidence : {}'.format(dbPredFn, dbPredPdn, dbConf))
    else:
        title = "Not matched in top-{}".format(N)
        fig.add_subplot(3,3,4)
        plt.imshow(qImg)
        plt.title('[Q] {} in {}'.format(qFn, qPdn))

        for i in range(min(len(Preds), int((9-4)/2))):
            dbGtFn, dbGtPdn, dbGtPn, dbGtImg = GetDbImg(eval_set.dbStruct.dbImage[GTs[i]]) # Ground Truth DB
            dbPredFn, dbPredPdn, dbPredPn, dbPredImg = GetDbImg(eval_set.dbStruct.dbImage[Preds[i]]) # Predicted DB
            dbConf = Confs[i]
            fig.add_subplot(3, 3, 5+i)
            plt.imshow(dbPredImg)
            plt.title('[DB-G.T.] {} in {}'.format(dbGtFn, dbPredPdn))
            #myiu.imshow(dbGtImg, sp, '[GT{}]{}'.format(i,dbGtFn), dispEn = False)
            fig.add_subplot(3, 3, 5+i+1)
            plt.imshow(dbPredImg)
            plt.title('[DB-Unmatched] {} in {}'.format(dbPredFn, dbPredPdn))
            dbPredImg.close()

    myiu.suptitle(title)

    fname = os.path.join(log_dir, 'Results_img_Qseq_{0:05d}.png'.format(qIx))
    plt.savefig(fname, bbox_inches='tight')
    #myiu.ImgFileResize(fname, h=1000, w=1000)

    #writer.add_figure('Matching between Q and DBs at {}'.format(opt.dataset), fig, global_step=qIx)

    qImg.close()
    dbPredImg.close()

    plt.close()
    print("===> Write Test result of {0} / {1} query into [ {2} ], for {3:.1f} seconds".format(qIx,
            eval_set.dbStruct.numQ, os.path.basename(fname), time.time() - start_time),
            end="\r", flush=True)

def write_all_pos_on_map(N, matched, qIx, pred, GTs, eval_set, writer):
    utmQ_x = eval_set.dbStruct.utmQ[:,0]
    utmQ_y = eval_set.dbStruct.utmQ[:,1]
    utmDb_x = eval_set.dbStruct.utmDb[:,0]
    utmDb_y = eval_set.dbStruct.utmDb[:,1]

    # all position
    plt.scatter(utmDb_x, utmDb_y, c='black', s=1, label='Db', alpha=0.5)
    plt.scatter(utmQ_x, utmQ_y, c='red', s=1, label='Q', alpha=0.5)

    # current position
    plt.scatter(utmDb_x[pred], utmDb_y[pred], c='blue', s=150, label='Db(Pred)', marker='o', alpha=0.7) # Predicted DB
    plt.scatter(utmQ_x[qIx], utmQ_y[qIx], c='red', s=130, label='Q', marker='x', alpha=0.9) # Query
    #plt.grid(color='b', linestyle='.', linewidth=0.5)
    plt.grid()

def scale_xy(src_x, src_y, cx, cy, scale_x, scale_y):
    tar_x = scale_x * src_x + (cx - scale_x * cx)
    tar_y = scale_y * src_y + (cy - scale_y * cy)
    return tar_x, tar_y

def write_matched_pos_on_map(N, matched, qIx, pred, GTs, eval_set, writer):
    utmQ_x0 = eval_set.dbStruct.utmQ[:,0]
    utmQ_y0 = eval_set.dbStruct.utmQ[:,1]
    utmDb_x0 = eval_set.dbStruct.utmDb[:,0]
    utmDb_y0 = eval_set.dbStruct.utmDb[:,1]

    # Set Query to origin to compare positions relatively
    utmQ_x = utmQ_x0 - utmQ_x0[qIx]
    utmQ_y = utmQ_y0 - utmQ_y0[qIx]

    utmDb_x = utmDb_x0 - utmQ_x0[qIx]
    utmDb_y = utmDb_y0 - utmQ_y0[qIx]

    # current position of zoom-in
    plt.scatter(utmDb_x[pred], utmDb_y[pred], c='blue', s=150, label='Db(Pred)', marker='o', alpha=0.5) # Predicted DB
    plt.scatter(utmQ_x[qIx], utmQ_y[qIx], c='red', s=130, label='Q', marker='x', alpha=0.9) # Query
    plt.legend()

    # display position information
    if 'pitts' in opt.dataset:
        utm_last = utm.from_latlon(40.4399486, -79.9976529)[-2:]
    elif 'dg_daejeon' in opt.dataset:
        utm_last = utm.from_latlon(36.3614846, 127.3386342)[-2:]
    else: # seoul
        utm_last = utm.from_latlon(37.514852, 127.0573766)[-2:]
    ux, uy = utmQ_x0[qIx], utmQ_y0[qIx]
    llx, lly = utm.to_latlon(ux, uy, utm_last[0], utm_last[1])
    txtl = "Origin(Lat, Lon) : ({}, {})".format(llx, lly)
    txtu = "Origin(utm) : ({}, {}, {}, {})".format(ux, uy, utm_last[0], utm_last[1])
    #plt.text(-22, -30, txtl, fontsize=7);
    #plt.text(-22, -32, txtu, fontsize=7);
    plt.title("Zoom-in [meters]\n{}\n{}".format(txtl, txtu))

    for i in range(min(len(GTs),1000)):
        plt.scatter(utmDb_x[GTs[i]], utmDb_y[GTs[i]], color='black', s=30, label='Db(GTs)', marker='.', alpha=0.9) # GT.
        if i == 0:
            plt.legend()

    plt.xlim(-50, 50) # meters
    plt.ylim(-50, 50) # meters
    plt.grid()

import sys;sys.path.insert(0, "/home/ccsmm/dg_git/dataset/ImageRetrievalDB/python");import dg_vps_dbmat_gen_ver2 as IRDB; #tim.init(plt_mode)

class save_test_result_to_avi():
    def __init__(self, opt, model, eval_set, dataset, map_path=None, fps=3, width=1280, top_k_show=5, img_w=640, img_h=480):
        '''
            show_map : display or not to display map to the right side of concatenated images
            top_k_show : int, the number of database images to be display with query.
        '''
        from utils.DispMap import DispMap
        self.opt = opt 
        self.avi_all = None
        self.avi_success = None
        self.avi_fail = None
        self.fps = fps 
        self.coord = "utm"
        self.show_map = self.opt.save_test_result_avi_with_map
        if self.show_map == True:
            self.map_img, self.map_ltop, self.map_rbottom = IRDB.get_map_image(opt)
            if self.map_img is not None:
                self.mMap = DispMap(self.map_img, ltop=self.map_ltop, rbottom=self.map_rbottom, incoord=self.coord, width=width)  # incoord is one of ["utm" , "latlon"]
                self.map_img = self.mMap.get_img()  # resized
                self.map_img_clean = copy.copy(self.map_img)  # backup clean map image for next frame
                self.show_map = True
            else:
                self.show_map = False
            if False:
                self.map_path = map_path
                if "pitts30k_val" in map_path:
                    self.ltop=(584220.9037397926, 4477372.690559181)
                    self.rbottom=(585279.1942611092, 4476587.527097304)
                    self.coord='utm'  # "latlon" or "utm"
                else:
                    self.ltop=(584220.9037397926, 4477372.690559181)
                    self.rbottom=(585279.1942611092, 4476587.527097304)
                    self.coord='utm'  # "latlon" or "utm"
                self.mMap = DispMap(self.map_path, ltop=self.ltop, rbottom=self.rbottom, incoord=self.coord, width=width)  # incoord is one of ["utm" , "latlon"]
                self.map_img = self.mMap.get_img()
        else:
            self.show_map = False

        self.color_red = (0, 0, 255)  #BGR
        self.color_blue = (255, 0, 0)  #BGR
        self.eval_set = eval_set
        self.dataset = dataset
        self.q_dir = dataset.queries_dir
        self.db_dir = dataset.db_dir
        self.qImage = self.eval_set.dbStruct.qImage
        self.dbImage = self.eval_set.dbStruct.dbImage
        self.utmQ = self.eval_set.dbStruct.utmQ
        self.utmDb = self.eval_set.dbStruct.utmDb
        self.top_k_show = top_k_show
        self.img_w, self.img_h = img_w, img_h
        self.dim = (img_w, img_h)
        self.model = model
        _, self.device = misc.get_device(opt)

    def reset_map_img(self):
        self.map_img = copy.copy(self.map_img_clean)

    def draw_point_on_map(self, xy=(0, 0), incoord=None,
            radius=5, color=(255, 0, 0),  # BGR
            thickness=-1  #  -1 : fill, positive: thick
            ):
        if incoord is None:
            incoord = self.coord
        self.mMap.set_img(self.map_img)
        self.map_img = self.mMap.draw_point_on_map(xy, incoord, radius, color, thickness)  # return cv_img
        return self.map_img

    def get_heatmap(self, cv_img, opt=None, model=None):  ## To do : not completed
        '''
        Get heatmap of single H*W*C dimensional cv2 image
        '''
        if opt is None:
            opt = self.opt
        if model is None:
            model = self.model
        return get_heatmap(cv_img, opt, model)  ## To do : not completed

    def imshow_results_heatmap(self, qIx, gt, pred, gt_max=5, sz_ratio=1.0):  # for top-1 prediction
        q_dir = self.q_dir
        db_dir = self.db_dir
        qname = os.path.join(q_dir, self.qImage[qIx])
        qimg = cv2.resize(cv2.imread(qname), self.dim)
        img_with_heatmap = self.get_heatmap(qimg)
        self.matched_images = copy.copy(img_with_heatmap)
        for dbidx in range(self.top_k_show):
            predname = os.path.join(db_dir, self.dbImage[pred[dbidx]])  # predname is predicted dbname
            #predimg = cv2.resize(cv2.imread(predname), (0,0), fx=sz_ratio, fy=sz_ratio)
            predimg = cv2.resize(cv2.imread(predname), self.dim)
            img_with_heatmap = self.get_heatmap(predimg)
            self.matched_images = cv2.hconcat((self.matched_images, img_with_heatmap))

        #if self.show_map:
        #    images = self.overlay_matched_image_on_map(qIx, pred[0])
        #else:
        images = self.matched_images

        if False:  # Display ground truth
            gtnames = []
            for i,gtidx in enumerate(gt[qIx],1):
                gtname = os.path.join(db_dir, self.dbImage[gtidx])
                gtnames.append(gtname)
                if i >= gt_max:
                    break
            gtimgs = concat_image.imgfile_concat(gtnames)
            images = cv2.hconcat((images, gtimgs))
            cv2.imshow('img', images)
            cv2.waitKey(1)

        return images, qname, predname

    def overlay_matched_image_on_map(self, qIx, dbIx, matched_images):
        self.reset_map_img()
        self.draw_point_on_map(xy=self.utmQ[qIx], incoord="utm", color=self.color_red)
        self.draw_point_on_map(xy=self.utmDb[dbIx], incoord="utm", color=self.color_blue)
        W, H = self.dim 
        patch = matched_images[:,:W*2,:]  # Use only query and top-1 image (width*2)
        patch = cv2.resize(patch, (0,0), fx=0.4, fy=0.4)
        mH, mW, _ = self.map_img.shape
        pH, pW, _ = patch.shape

        #self.map_img[:(mH-pH),:(mW-pW),:] = patch  # right aligned images
        #self.map_img[:pH,:pW,:] = patch  # Upper left aligned
        self.map_img[(mH-pH):, :pW,:] = patch  # Upper left aligned
        return self.map_img

    def imshow_results_image(self, qIx, gt, pred, gt_max=5, sz_ratio=1.0):  # for top-1 prediction
        q_dir = self.q_dir
        db_dir = self.db_dir
        qname = os.path.join(q_dir, self.qImage[qIx])

        qimg = cv2.resize(cv2.imread(qname), self.dim)
        self.matched_images = copy.copy(qimg)
        for dbidx in range(self.top_k_show):
            predname = os.path.join(db_dir, self.dbImage[pred[dbidx]])  # predname is predicted dbname
            #predimg = cv2.resize(cv2.imread(predname), (0,0), fx=sz_ratio, fy=sz_ratio)
            predimg = cv2.resize(cv2.imread(predname), self.dim)
            self.matched_images = cv2.hconcat((self.matched_images, predimg))

        #if self.show_map:
        #    images = self.overlay_matched_image_on_map(qIx, pred[0])
        #else:
        images = self.matched_images

        if False:  # Display ground truth
            gtnames = []
            for i,gtidx in enumerate(gt[qIx],1):
                gtname = os.path.join(db_dir, self.dbImage[gtidx])
                gtnames.append(gtname)
                if i >= gt_max:
                    break
            gtimgs = concat_image.imgfile_concat(gtnames)
            images = cv2.hconcat((images, gtimgs))
            cv2.imshow('img', images)
            cv2.waitKey(1)

        return images, qname, predname

    def save_to_avi(self, success, qIx, gt, pred, dbFeat, qFeat, matched_at_top_k=1, gt_max=1, sz_ratio=1.0):
        if self.opt.save_test_result_avi:  ## Save results into avi files.
            if True:  ## Concatenated input images (q + db1 + db2 + ... + dbN)
                out_cv, qname, predname = self.imshow_results_image(qIx, gt, pred, gt_max=1, sz_ratio=1.0)
                if self.opt.save_test_result_avi_disable_text == False:  # default is False
                    out_cv = cv2.putText(out_cv, qname, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    out_cv = cv2.putText(out_cv, predname, (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                #write_matched_pos_on_map(qIx, pred[0], gt, self.eval_set)
                #plt.pause(0.1)

            if True: ## To do : Heatmap for concatenated input images (q + db1 + db2 + ... + dbN)
                out_cv_heatmap, qname, predname = self.imshow_results_heatmap(qIx, gt, pred, gt_max=1, sz_ratio=1.0)
                if self.opt.save_test_result_avi_disable_text == False:  # default is False
                    out_cv_heatmap = cv2.putText(out_cv_heatmap, qname, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    out_cv_heatmap = cv2.putText(out_cv_heatmap, predname, (600, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                #write_matched_pos_on_map(qIx, pred[0], gt, self.eval_set)
                #plt.pause(0.1)

            ## Draw rectangle to image, where rect. around image is colored with blue for success and red for fail.
            for i in range(matched_at_top_k):
                if i == matched_at_top_k-1:
                    color = self.color_blue
                else:
                    color = self.color_red
                out_cv = cv2.rectangle(out_cv, (self.img_w*(i+1), 0), (self.img_w*(i+2), self.img_h), color, 5)
                out_cv_heatmap = cv2.rectangle(out_cv_heatmap, (self.img_w*(i+1), 0), (self.img_w*(i+2), self.img_h), color, 5)

            out_cv = cv2.vconcat((out_cv, out_cv_heatmap))

            if self.show_map:
                out_cv = self.overlay_matched_image_on_map(qIx, pred[0], out_cv)

            try:
                dataset_name = "_" + os.path.basename(os.path.realpath("netvlad_v100_datasets_dg"))
            except:
                dataset_name=""

            avi_prefix = '{}_{}{}'.format(self.opt.dataset, self.opt.split, dataset_name)

            #cv2.imshow('(dbg) q1-pred1-GTs', out_cv)
            #cv2.waitKey(1)

            if success:
                #cv2.imshow('(Success) q1-pred1-GTs', out_cv)
                #cv2.waitKey(1)
                if True:
                    if self.avi_success is None:
                        h, w = out_cv.shape[:2]
                        #self.avi_success = cv2.VideoWriter('Success_Q(red)-DB(blue).avi', 0x7634706d, self.fps, (w, h)) # write ad mp4
                        self.avi_success = cv2.VideoWriter('{}_Success.avi'.format(avi_prefix), 0x7634706d, self.fps, (w, h)) # write ad mp4
                    if self.avi_success is not None and out_cv is not None:
                        self.avi_success.write(out_cv)
            else:
                #cv2.imshow('(Fail) q1-pred1-GTs', out_cv)
                #cv2.waitKey(1)
                if True:
                    if self.avi_fail is None:
                        h, w = out_cv.shape[:2]
                        #self.avi_fail = cv2.VideoWriter('Fail_Q(red)-DB(blue).avi', 0x7634706d, self.fps, (w, h)) # write ad mp4
                        self.avi_fail = cv2.VideoWriter('{}_Fail.avi'.format(avi_prefix), 0x7634706d, self.fps, (w, h)) # write ad mp4
                    if self.avi_fail is not None and out_cv is not None:
                        self.avi_fail.write(out_cv)

            if self.avi_all is None:
                h, w = out_cv.shape[:2]
                #self.avi_all = cv2.VideoWriter('All_Q(red)-DB(blue).avi', 0x7634706d, self.fps, (w, h)) # write ad mp4
                self.avi_all = cv2.VideoWriter('{}_All.avi'.format(avi_prefix), 0x7634706d, self.fps, (w, h)) # write ad mp4
            if self.avi_all is not None and out_cv is not None:
                self.avi_all.write(out_cv)

            #cv2.imshow("matched(q,db1_dbN)", out_cv)
            #cv2.waitKey(1)
