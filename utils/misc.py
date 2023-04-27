from __future__ import print_function
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
import torchvision.models as models
from datetime import datetime

from tensorboardX import SummaryWriter
import numpy as np
from utils import custom_loss

from ipdb import set_trace as bp

import seaborn as sns
import pandas as pd
import glob

def strip_module_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        key = key.replace('module.','')
        new_state_dict[key] = val
    return new_state_dict

def add_module_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        key = key.replace('encoder.','encoder.module.')
        key = key.replace('pool.','pool.module.')
        new_state_dict[key] = val
    return new_state_dict

def change_crm_to_crn(state_dict):  # Code for bugfix. It change old name of weight to new one it it has.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        key = key.replace('crm','crn')  # Fix name error of DeAttention class in networks/pool_model.py : crm to crn
        new_state_dict[key] = val
    return new_state_dict

def save_checkpoint(state, is_best, savePath):
    model_out_path = join(savePath, 'checkpoint.pth.tar')
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class DatasetStruct():
    def __init__(self, opt, verbose=False):
        ''' 
        whole_train_set :
            Used for Building Cache in which features of all image are caculated in advance for querying.
            It can get postive images with comparing utm distance.
        whole_train_set_dataloader : 
            (torch.utils.data)DataLoader of whole_train_set
        whole_train_set_can_query:
            whole_train_set_can_query can get postive and negative images with comparing utm distance and 
            feature similarity from cache built in whole_train_set
        whole_test_set:
            Used for testing.
            It can get postive images with comparing utm distance.
        '''
        self.dataset = None
        self.whole_train_set_dataloader = None
        self.whole_train_set_can_query = None
        self.whole_train_set = None
        self.whole_test_set = None
        self.verbose = verbose
        self.opt = opt
        self.load_dataset()

    def load_dataset(self):
        opt = self.opt
        verbose = self.verbose
        cuda, device = get_device(opt)

        if verbose:
            print('===> Loading dataset(s)')
    
        dataset = None
        whole_train_set_dataloader = None
        whole_train_set_can_query = None
        whole_train_set = None
        whole_test_set = None

        if opt.dataset.lower() == 'pittsburgh':
            from dataloaders import pittsburgh as dataset
        elif opt.dataset.lower() == 'pittsburgh_3k':
            from dataloaders import pittsburgh_3k as dataset
        elif opt.dataset.lower() == 'pittsburgh_6k':
            from dataloaders import pittsburgh_6k as dataset
        elif opt.dataset.lower() == 'pittsburgh_9k':
            from dataloaders import pittsburgh_9k as dataset
        elif opt.dataset.lower() == 'pittsburgh_12k':
            from dataloaders import pittsburgh_12k as dataset
        elif opt.dataset.lower() == 'pittsburgh_15k':
            from dataloaders import pittsburgh_15k as dataset
        elif opt.dataset.lower() == 'pittsburgh_18k':
            from dataloaders import pittsburgh_18k as dataset
        elif opt.dataset.lower() == 'pittsburgh_21k':
            from dataloaders import pittsburgh_21k as dataset
        elif opt.dataset.lower() == 'pittsburgh_24k':
            from dataloaders import pittsburgh_24k as dataset
        elif opt.dataset.lower() == 'pittsburgh_27k':
            from dataloaders import pittsburgh_27k as dataset
        elif opt.dataset.lower() == 'tokyo247':
            from dataloaders import tokyo247 as dataset
        elif opt.dataset.lower() == 'tokyotm':
            from dataloaders import tokyoTM as dataset
        elif (opt.dataset.lower() == 'rparis6k') or (opt.dataset.lower() == 'roxford5k'):
            from dataloaders import revisit_roxford5k_rparis6k as dataset
        elif opt.dataset.lower() == 'dg_daejeon':
            from dataloaders import dg_daejeon as dataset
        elif opt.dataset.lower() == 'dg_seoul':
            from dataloaders import dg_seoul as dataset
        elif opt.dataset.lower() == 'dg_bucheon':
            from dataloaders import dg_bucheon as dataset
        else:
            raise Exception('Unknown dataset')

        if opt.mode.lower() == 'train':
            if opt.split.lower() == 'train250k':
                whole_train_set = dataset.get_250k_whole_training_set(onlyDB=False)
                #whole_train_set_can_query = dataset.get_250k_training_query_set(get_margin(opt))  # ori
                whole_train_set_can_query = dataset.get_250k_training_query_set(opt.dataloader_margin)
            else:  # Default
                whole_train_set = dataset.get_whole_training_set(onlyDB=False)
                #whole_train_set_can_query = dataset.get_training_query_set(get_margin(opt))  # ori
                whole_train_set_can_query = dataset.get_training_query_set(opt.dataloader_margin)

            whole_train_set_dataloader = DataLoader(dataset=whole_train_set,
                    num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                    pin_memory=cuda, drop_last=True)
            if verbose:
                print('====> Training query set:', len(whole_train_set_can_query))
            whole_test_set = dataset.get_whole_val_set()
            if verbose:
                print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
        elif opt.mode.lower() == 'test':
            if opt.split.lower() == 'test':
                whole_test_set = dataset.get_whole_test_set()
                if verbose:
                    print('===> Evaluating on test set')
            elif opt.split.lower() == 'test250k':
                whole_test_set = dataset.get_250k_test_set()
                if verbose:
                    print('===> Evaluating on test250k set')
            elif opt.split.lower() == 'train':
                whole_test_set = dataset.get_whole_training_set()
                if verbose:
                    print('===> Evaluating on train set')
            elif opt.split.lower() == 'val':
                whole_test_set = dataset.get_whole_val_set()
                if verbose:
                    print('===> Evaluating on val set')
            else:
                raise ValueError('Unknown dataset split: ' + opt.split)
            if verbose:
                print('====> Query count:', whole_test_set.dbStruct.numQ)
        elif (opt.mode.lower() == 'cluster') or (opt.mode.lower() == 'class_statics') :
            whole_train_set = dataset.get_whole_training_set(onlyDB=True)

        self.dataset = dataset
        self.whole_train_set_dataloader = whole_train_set_dataloader
        self.whole_train_set_can_query = whole_train_set_can_query
        self.whole_train_set = whole_train_set 
        self.whole_test_set = whole_test_set 

    def get_dataset(self):
        '''
            Usage 1 :  dataset, train_set_for_cache, train_set, cluster_set, test_set = mDatasetStruct.get_dataset()
            Usage 2 :  dataset, train_set_for_cache, train_set, whole_train_set, test_set = mDatasetStruct.get_dataset()
        '''
        return self.dataset, self.whole_train_set_dataloader, self.whole_train_set_can_query, self.whole_train_set, self.whole_test_set 

def get_device(opt):
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device(opt.which_cuda.lower() if cuda else "cpu")  # "cuda" for using all cuda. "cuda:N" for using specific gpu
    return cuda, device

def get_seg_model(opt):
    from networks.Segmentation import Segmentation
    cuda, device = get_device(opt)
    if (opt.deattention == True) or (opt.deattention_auto == True)  or (opt.add_clutter_train == True) or (opt.add_clutter_test_q == True) or (opt.mode.lower() == "class_statics")\
        or (opt.add_clutter_test_db == True) or (opt.add_segmask_to_input_ch4 == True) or (opt.remove_clutter_input == True):
        seg_model = Segmentation(opt, device, deatt_category_list=opt.deatt_category_list)  # load MobileNet for segmentation, ["human", "vehicle"]
    else:
        seg_model = None
    return seg_model

def get_random_seed(opt):
    print("opt.seed is ", opt.seed)
    print("np.random.seed is ", np.random.get_state()[1][0])
    print("torch.manual_seed is ", torch.initial_seed())
    print("torch.cuda.manual_seed is ", torch.cuda.initial_seed())
    print("torch.backends.cudnn.benchmark is ", torch.backends.cudnn.benchmark)
    print("torch.backends.cudnn.deterministic is ", torch.backends.cudnn.deterministic)

def set_random_seed(opt):
    ## The seed setting command should be executed before netowrk initialization.
    random.seed(opt.seed)
    np.random.seed(opt.seed) # You can retrieve the current seed using np.random.get_state()[1][0] after it.
    torch.manual_seed(opt.seed)  # You can retrieve the current seed using torch.initial_seed() after it.
    cuda, device = get_device(opt)
    if cuda:
        torch.cuda.manual_seed(opt.seed)  # if you are using a GPU.
        torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True  # original is False
    torch.backends.cudnn.deterministic = True 

def set_parallel(opt, model):
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if (opt.mode.lower() != 'cluster') and (opt.mode.lower() != 'class_statics'):
            model.pool = nn.DataParallel(model.pool)
    return model

def get_isParallel(opt):
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        isParallel = True
    else:
        isParallel = False
    return isParallel

def get_optimizer(opt, model):
    optimizer = None
    scheduler = None
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)
    return optimizer, scheduler

def get_margin(opt):
    if opt.tml2:
        margin = opt.margin + opt.tml2_pn_margin # margin needs to larger than 1.5 (original is 0.1)
    else:
        margin = opt.margin  # 0.1

    return margin

def get_loss_function(opt):
    criterion = None
    cuda, device = get_device(opt)
    if opt.mode.lower() == 'train':
        margin = get_margin(opt)
        cuda, device = get_device(opt)
        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        if opt.tml2:
            # by ccsmm, for increasing the distance between positive and negative features. It helps train between DBs
            criterion = custom_loss.TripletMarginLoss2(margin=opt.margin, pn_margin=opt.tml2_pn_margin, version=opt.tml2_version, device=device, reduction='mean')
        else:
            if opt.deattention or opt.deattention_auto:  # deattention by ccsmm, it needs mean of loss for abruptly rejected region.
                criterion = nn.TripletMarginLoss(margin=opt.margin, p=2, reduction='mean').to(device) # by ccsmm, Batch size is no longer a hyper parameter.
            else:  # original paper
                criterion = nn.TripletMarginLoss(margin=opt.margin, p=2, reduction='sum').to(device)  # original github, It may make the batch size a hyper-parameter
    return criterion

def get_loss_function_for_deatt(opt):
    criterion = None
    cuda, device = get_device(opt)
    if opt.mode.lower() == 'train':
        cuda, device = get_device(opt)
        if opt.deatt_loss_is_bce == True:
            criterion = nn.BCELoss().to(device)  # Better for classification
        else:  # default
            criterion = nn.MSELoss(reduction='mean').to(device)  # Better for regression
    return criterion

def resume_ckpts(opt, model, optimizer, verbose=False):  # Resume model from check points
    valid = True
    ## "opt = get_parser.resume_parameters(opt, parser)" is called in utils/get_parser()
    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, opt.endPath, 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, opt.endPath, 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("===> Loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            if False:  # With False, I want to set up deatt_category_list independently in the test and train times.
                try:
                    opt.deatt_category_list = checkpoint['deatt_category_list']
                except:
                    opt.deatt_category_list = opt.deatt_category_list

            #model.load_state_dict(checkpoint['state_dict'])

            ## ccsmm added (debug for kind of pytorch bug).
            state_dict = checkpoint['state_dict']
            wasParallel = checkpoint['parallel']
            isParallel = get_isParallel(opt)
            if wasParallel == True:  # state_dict needs module keys, ex) 'encoder.module.', 'pool.modules.'
                if isParallel == True:
                    state_dict = state_dict
                else:  # False
                    state_dict = strip_module_state_dict(state_dict)  ## strip 'module' string at the end of encoder. and pool.
            else:  # wasParallel == False : state_dict dosen't need module keys
                if isParallel == True:
                    state_dict = add_module_state_dict(state_dict)  ## add 'module' string at the end of encoder. and pool.
                else:  # False
                    state_dict = state_dict

            state_dict = change_crm_to_crn(state_dict)
            model.load_state_dict(state_dict)
            cuda, device = get_device(opt)

            model = model.to(device)
            if opt.mode == 'train':
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    if opt.write_data_distribution_in_train:
                        print("Warning during loding optimizer parameters. But it's Okay when you are run train with --write_data_distribution_in_train which to test network in fact.")
                    else:
                        assert False, "Error during loading resume optimzer parameter : ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group"

            print(" [O] Loaded checkpoint '{}' (epoch {})".format(resume_ckpt, checkpoint['epoch']))
            valid = True
        else:
            print(" [X] No checkpoint found at '{}'".format(resume_ckpt))
            valid = False
    return valid, opt, model, optimizer

def makedir(pathdir):
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)

def writer_init(opt):
    common_string = datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling  # Apr27_20-57-56_vgg16_netvlad
    if opt.add_clutter_train == True:
        common_string = common_string + '_add_clutter{}_in_train'.format(opt.add_clutter_iteration)
    if opt.mode.lower() == 'train':
        if (opt.deattention == True) or (opt.deattention_auto == True):
            ckpt_path = join(opt.runsPath, common_string + '_deatt_w{}'.format(opt.w_deatt_loss))
        else:
            ckpt_path = join(opt.runsPath, common_string)
    elif opt.mode.lower() == 'test':
        if (opt.deattention == True) or (opt.deattention_auto == True):
            ckpt_path = join(opt.testPath, common_string + '_deatt_w{}'.format(opt.w_deatt_loss))
        else:
            ckpt_path = join(opt.testPath, common_string)
    else:  # clustering
        return None
    
    try:
        writer = SummaryWriter(log_dir=ckpt_path)
    except:  # permission error to directory
        print(" [X] Warning : Access error of SummaryWriter to : ", ckpt_path)
        return None

    opt.internal_result_path = ckpt_path
    if opt.write_heatmap:
        opt.heatmap_result_dir = os.path.join(opt.internal_result_path, opt.heatmap_result_dir)
        makedir(opt.heatmap_result_dir)

    ## write checkpoints in logdir
    logdir = writer.file_writer.get_logdir()
    opt.savePath = join(logdir, opt.endPath)
    makedir(opt.savePath)

    with open(join(opt.savePath, 'flags.json'), 'w') as f:
        f.write(json.dumps(
            {k:v for k,v in vars(opt).items()}
            ))
    print('   ===> It will save state to:', logdir)
    return writer

def get_3ch_input(opt, input):  # Works when opt.add_segmask_to_input_ch4 is True
    if opt.add_segmask_to_input_ch4 and (input.shape[1]==4):  # [B, 4, H, W]
        return input[:,:3,:,:]  # [B, 3, H, W], RGB only
    else:
        return input

def get_4ch_input(opt, seg_model, input, device=None):  #  input : [B, 3, H, W], Works when opt.add_segmask_to_input_ch4 is True
    if opt.add_segmask_to_input_ch4 and (seg_model is not None):
        _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation, [B, 1, H, W]
        input = torch.cat((input, mask_gt), dim=1)  # [B, 4, H, W]
        if device is not None:
            input = input.to(device)
    return input  # [B, 3, H, W] or [B, 4, H, W]


def get_removed_clutter_input(opt, seg_model, input):  #  input : [B, 3, H, W], Works when opt.add_segmask_to_input_ch4 is True
    if opt.remove_clutter_input and (seg_model is not None):  # Do not use in cache building.
        removed_clutter_input, mask_gt = seg_model.preprocess(input.detach(), opt.remove_clutter_mode) # Ground Truth of segmentation
        input = removed_clutter_input
    return input

def write_data_to_csv(csv_fname, whole_d_qp=None, whole_d_qn=None):
    if (whole_d_qp is None) or (whole_d_qn is None):
        whole_d_qp = 0+np.random.rand(10000)*9  # N
        whole_d_qn = 10+np.random.rand(10000)*9  # N

    data = np.vstack((whole_d_qp, whole_d_qn)).T  # [N x 2]
    df = pd.DataFrame(data, columns=["d_qp", "d_qn"])
    df.to_csv(csv_fname, index=False)

def write_png_from_csv(csv_fname, N=6000, verbose=True):
    import matplotlib.pyplot as plt
    data_ori = pd.read_csv(csv_fname)
    try:
        data = data_ori.sample(N, replace=False)
    except:  # Due to lack of data sample, it re-uses sampled ones
        data = data_ori.sample(N, replace=True)

    ofname_prefix = os.path.splitext(csv_fname)[0]  # execpt ext

    plt.clf()
    ofname = ofname_prefix + "_histo.png"
    if os.path.exists(ofname):
        os.remove(ofname)
    fig = sns.displot(data, fill=True);plt.set_xlabels="distance";plt.xlim(0.2, 1.6);plt.ylim(0,800);fig.savefig(ofname, dpi=100)  # histo

    ofname = ofname_prefix + "_histo2.png"
    if os.path.exists(ofname):
        os.remove(ofname)
    fig = sns.displot(data, kde=True, fill=True);plt.set_xlabels="distance";plt.xlim(0.2, 1.6);plt.ylim(0,800);fig.savefig(ofname, dpi=100)  # histo

    ofname = ofname_prefix + "_kde.png"
    if os.path.exists(ofname):
        os.remove(ofname)
    fig = sns.displot(data, kind="kde", fill=True);fig.set_xlabels="distance";plt.xlim(0.2, 1.6);plt.ylim(0,8);fig.savefig(ofname, dpi=100)  # histo

    if verbose:
        mean_p, mean_n = np.mean(data['d_qp']), np.mean(data['d_qn'])
        std_p, std_n = np.std(data['d_qp']), np.std(data['d_qn'])
        print("\n## Drawing : ", csv_fname)
        print("   - mean (d_qp, d_qn, diff) = ({}, {}, {})".format(mean_p, mean_n, np.abs(mean_n-mean_p)))
        print("   - std. (d_qp, d_qn, ) = ({}, {})".format(std_p, std_n))


def add_clutter(opt, seg_model, input, indices, eval_set):  # Called from test.py
    _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation
    if opt.write_add_clutter_image:
        imgidx = 0 
        imgpath = eval_set.images[indices[imgidx]]
        parentname = os.path.dirname(imgpath).split("/")[-1]
        imgname = os.path.basename(imgpath)
        ofpath = os.path.join("added_clutter_iteration_{}".format(opt.add_clutter_iteration), parentname)
        if not os.path.exists(ofpath):
            os.makedirs(ofpath)
        ofname = os.path.join(ofpath, "added_clutter_{}".format(imgname))
        input = seg_model.add_clutter_to_cv_img(input.detach(), mask_gt, iteration=opt.add_clutter_iteration, write_image=True, imgidx=imgidx, fname=ofname)
    else:
        input = seg_model.add_clutter_to_cv_img(input.detach(), mask_gt, iteration=opt.add_clutter_iteration)
    return input

#def save_feature(opt, dbFeat, qFeat, gt, n_values, predictions, distances, fname="Feat.pickle"):
def save_feature(opt, dbFeat, qFeat, distances, predictions, eval_set, fname=[]):  # Called by test.py
    '''
    eval_set should have followings
        eval_set.dbStruct.dbImage
        eval_set.dbStruct.qImage
        eval_set.dbStruct.utmDb
        eval_set.dbStruct.utmQ
        You can remake clear utmDb and utmQ np.array using following commands
        x = [dd[0][0] for dd in utmDb[:,0]]
        y = [dd[0][0] for dd in utmDb[:,1]]
        utmDb = np.stack((x,y), axis=1)  # shape : (3885, 2)
    '''
    import pickle
    if opt.save_feature == True:
        data = {
        'dbFeat':dbFeat,
        'qFeat':qFeat,
        'predictions':predictions,
        'distances': distances,
        'dbImage':eval_set.dbStruct.dbImage,
        'qImage':eval_set.dbStruct.qImage,
        'utmDb':eval_set.dbStruct.utmDb,
        'utmQ':eval_set.dbStruct.utmQ
        }
        ## save features
        if len(fname) == 0:
            fname = opt.feature_fname
        print("     Test[1/4] Saving Feature to {}".format(fname))
        with open(fname, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def get_pool_size_of_feat(feat):
    return feat.shape[-1]

def load_feature(opt, fname=[]):  # Called by test.py
    if len(fname) == 0:
        fname = opt.feature_fname
    dbFeat = []
    qFeat = []
    utmDb = []
    utmQ = []
    dbImage = [] 
    qImage = []
    valid_feature = False
    predictions = []
    distances = []
    if (os.path.exists(fname) == True) and (opt.load_feature == True) and (opt.mode.lower() != "train"):
        import pickle
        ## load
        print("     Test[1/4] Loading Features from {}".format(fname))
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            ## You can check key with data.keys() ==> dict_keys(['dbFeat', 'qFeat', 'predictions', 'distances', 'dbImage', 'qImage', 'utmDb', 'utmQ'])
            dbFeat = data['dbFeat']
            print("               ... dbFeat [{}].".format(len(dbFeat)))
            qFeat = data['qFeat']
            print("               ... qFeat [{}].".format(len(qFeat)))
            predictions = data['predictions']
            print("               ... predictions [{}].".format(len(predictions)))
            distances = data['distances']
            print("               ... distances [{}].".format(len(distances)))
            print("               ... feature dimension [{}].".format(get_pool_size_of_feat(dbFeat)))
            ## Misc for vps.py
            utmDb = data['utmDb']
            utmQ = data['utmQ']
            dbImage = data['dbImage']
            qImage = data['qImage']
        except:
            pass
        if len(dbFeat) > 0 and len(qFeat) > 0 and len(predictions)>0 and len(distances)>0:
            valid_feature = True
    return valid_feature, dbFeat, qFeat, distances, predictions, utmDb, utmQ, dbImage, qImage
