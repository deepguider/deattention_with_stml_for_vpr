from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

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
import sys

import numpy as np

from networks.get_model import get_model
from utils import get_parser, misc
#from tasks.train import train
#from tasks.test import test
#from tasks.get_clusters import get_clusters

from ipdb import set_trace as bp

'''
Usage : Copy following two lines to your code and remove remark at the beginning of the line to choose the mode among "Interactive" and "Agg" :
        Mode of tim.init() in main.py will affect all sub-module where import tim. So make sure mode of main.py and do not call tim.init() in sub-modules
plt_mode="Interactive"  # Default, interactive mode in remote terminal with imshow, plot, etc. and save image also.
plt_mode="Agg"          # mode only to save figure to image in server without any of imshow, plot. It does not need remote terminal.
'''

#plt_mode="Interactive"  # For debugging, interactive mode in remote terminal with imshow, plot, etc. and save image also.
#plt_mode="Agg"          # Default, mode only to save figure to image in server without any of imshow, plot. It does not need remote terminal.
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim; #tim.init(plt_mode)

def set_plt_mode(opt, force_interactive=False):
    if opt.reuse_cache_for_debug == True:
         plt_mode="Interactive"  # debug mode, It need remote connection to display using tim.imshow()
    else:
         plt_mode="Agg"  # normal. It do not need remote connection.

    if force_interactive == True:
         plt_mode="Interactive"  # debug mode, It need remote connection to display using tim.imshow()

    if plt_mode.lower() == "interactive":
        print("")
        print(" [X] Warning : plt_mode is not Agg mode (in main.py) which may cause some error when calling tim.imshow() without remote PC connection.")
        print("               Use Interactive mode only when you debug code.")
        print("               Use Agg mode when you run it in screen -R without remote connection(Default).")
        print("")

    tim.init(plt_mode)
    return plt_mode

if __name__ == "__main__":
    ## Get command line parameters
    opt = get_parser.get_parser()
    plt_mode = set_plt_mode(opt, False)  # If you want 

    ## Fix all kinds of random seed to re-produce same results.
    misc.set_random_seed(opt)
    misc.get_random_seed(opt)

    ## Print all parameters of task
    print("Parameters begin {\n")
    print(opt)
    print("\n} Parameters end\n")


    ## Prepare training/testing dataset
    verbose = opt.verbose
    cuda, device = misc.get_device(opt)
    mDatasetStruct = misc.DatasetStruct(opt, verbose)
    dataset, train_set_for_cache, train_set, cluster_set, test_set = mDatasetStruct.get_dataset()

    ## Prepare CNN model
    model, encoder_dim = get_model(opt, mDatasetStruct, verbose=False)  # Set verbose True to print model summary
    model = misc.set_parallel(opt, model)
    isParallel = misc.get_isParallel(opt)

    ## Prepare optimizers, schedulers and loss functions
    optimizer, scheduler = misc.get_optimizer(opt, model)

    valid_ckpts, opt, model, optimizer = misc.resume_ckpts(opt, model, optimizer, verbose)
    assert valid_ckpts == True

    ## Do process : At first, get_cluseter() at once. Next train(), and lastly test()
    model = model.to(device)

    writer = misc.writer_init(opt)
    seg_model = misc.get_seg_model(opt)

    ## To do : Build dataset in which add_clutter is applied in advance.

    if opt.mode.lower() == 'cluster':
        from tasks.get_clusters import get_clusters
        print('   ===> Calculating descriptors and clusters')
        get_clusters(opt, model, seg_model, encoder_dim, mDatasetStruct)

    elif opt.mode.lower() == 'train':
        from tasks.train import train
        print('   ===> Training model')
        train(opt, model, seg_model, encoder_dim, optimizer, scheduler, mDatasetStruct, writer)

    elif opt.mode.lower() == 'test':
        from tasks.test import test
        print("   ===> Epoch[{}] Evaluating ...".format(opt.start_epoch), flush=True)
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 25]
        recalls, accs, n_values = test(opt, model, seg_model, encoder_dim, opt.start_epoch, mDatasetStruct, writer, write_tboard=False, n_values=n_values)

    elif opt.mode.lower() == 'class_statics':
        from tasks.get_class_statics import get_class_statics
        print('   ===> Get class statics of database')
        get_class_statics(opt, seg_model, mDatasetStruct)
        
    if writer is not None:
        writer.close()
