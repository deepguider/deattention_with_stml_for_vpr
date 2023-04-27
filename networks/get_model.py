import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import torchvision.models as models
import os
import h5py
from networks import netvlad  # This code is called by ../main.py. So working directory is ../
from networks import pool_model, deattention_model, senet, bam, cbam
from torchsummary import summary as summary

from ipdb import set_trace as bp


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) 

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim 

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def get_parameters_grad(layer):
    for p in layer.parameters():
        print(p, p.requires_grad)

def get_vgg16(pretrained=True):
    ## Input channel of vgg16 is 3
    model = models.vgg16(pretrained=pretrained)
    return model

def get_vgg16_diff_in_ch(in_ch=4, pretrained=True):
    ## I want to change it to 4
    model = models.vgg16(pretrained=pretrained)
    first_conv_layer = [nn.Conv2d(in_ch, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features= nn.Sequential(*first_conv_layer)
    return model

def get_model(opt, mDatasetStruct, verbose=False):
    _, _, train_set, _, test_set = mDatasetStruct.get_dataset()
    if verbose:
        print('===> Building model')
    layers = None
    pretrained = not opt.fromscratch

    ## Get pretrained Encoder
    encoder_dim = 512 
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256 
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512 
        #encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        #layers = list(encoder.features.children())[:-2]

        if opt.add_segmask_to_input_ch4 == False:  # baseline
            encoder = get_vgg16(pretrained)
            layers = list(encoder.features.children())[:-2]
            if pretrained:  
                # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                for l in layers[:-19]:  # Train from conv3 ~ to end, best recall according to Table 1 in the paper
                #for l in layers[:-5]:  # ori
                    for p in l.parameters():
                        p.requires_grad = False
        else:
            encoder = get_vgg16_diff_in_ch(4, pretrained)  # 4-ch data (RGB + SegMask) for input 
            layers = list(encoder.features.children())[:-2]
            if True:  # Fully re-train
                if pretrained:
                    for l in layers: 
                        for p in l.parameters():
                            p.requires_grad = True
            if False:  # Train new conv1 and original(conv5_)
                # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                if pretrained:
                    for l in layers[:-5]: 
                        for p in l.parameters():
                            p.requires_grad = False
                    for l in layers[:1]:   # Train first new conv
                        for p in l.parameters():
                            p.requires_grad = True
            if False:  # Train from new conv(4ch) to conv3 (128ch)
                # if using pretrained then only train conv5_1, conv5_2, and conv5_3
                if pretrained:
                    for l in layers[:-5]: 
                        for p in l.parameters():
                            p.requires_grad = False
                    for l in layers[:11]:   # Finetuning from new conv(4ch) to conv3 (128ch)
                        for p in l.parameters():
                            p.requires_grad = True

    ## ecnoder_dim = layers[-1].out_channels  # You can also get encoder_dim by this call.

    ## Add pooling layer to the end of Encoder
    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    if verbose:
        in_ch, in_h, in_w = 3, 480, 640
        fea_ch, fea_h, fea_w = 512, int(480/16), int(640/16)  # 512, 30, 40
        ## summary(models.vgg16().cuda(), (3,224,224))  # example
        print("summary(model.encoder.cuda(), ({},{},{}))".format(in_ch, in_h, in_w))
        summary(model.encoder.cuda(), (in_ch, in_h, in_w))

    if opt.deattention:
        if True: # default
            deattention = deattention_model.DeAttention(opt.deattention_version, opt.deatt_padding_mode)
        else:
            deattention = deattention_model.DeAttention_new(opt.deattention_version)
        model.add_module('deattention', deattention)

        if verbose:
            print("summary(model.deattention.cuda(), ({},{},{}))".format(fea_ch, fea_h, fea_w))
            summary(model.deattention.cuda(), (fea_ch, fea_h, fea_w))

    if opt.deattention_auto:
        if True: # default
            deattention_auto = deattention_model.DeAttention_auto(opt.deattention_version, opt.deatt_padding_mode)
        model.add_module('deattention_auto', deattention_auto)

    if opt.ch_attention:
        ch_attention = deattention_model.ch_attention()
        model.add_module('ch_attention', ch_attention)

    if opt.ch_eca_attention:
        ch_eca_attention = deattention_model.ch_eca_attention(opt.ch_eca_attention_k_size)
        model.add_module('ch_eca_attention', ch_eca_attention)

    if opt.senet_attention:
        senet_attention = senet.SELayer(channel=encoder_dim)
        model.add_module('senet_attention', senet_attention)
        if verbose:
            print("summary(model.senet_attention.cuda(), ({},{},{}))".format(fea_ch, fea_h, fea_w))
            summary(model.senet_attention.cuda(), (fea_ch, fea_h, fea_w))

    if opt.bam_attention:
        bam_attention = bam.BAM(gate_channel=encoder_dim)
        model.add_module('bam_attention', bam_attention)

    if opt.cbam_attention:
        cbam_attention = cbam.CBAM(gate_channels=encoder_dim)
        model.add_module('cbam_attention', cbam_attention)

    if opt.crn_attention:
        crn_attention = deattention_model.CRN()
        model.add_module('crn_attention', crn_attention)
        if verbose:
            print("summary(model.crn_attention.cuda(), ({},{},{}))".format(fea_ch, fea_h, fea_w))
            summary(model.crn_attention.cuda(), (fea_ch, fea_h, fea_w))

    if (opt.mode.lower() != 'cluster') and (opt.mode.lower() != 'class_statics'):
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume:
                if opt.mode.lower() == 'train':
                    initcache = os.path.join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = os.path.join(opt.dataPath, 'centroids', opt.arch + '_' + test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                    #initcache = os.path.join(opt.dataPath, 'centroids', opt.arch + '_' + test_set.dataset + '_' + str(opt.num_clusters) +'_des.hdf5')  # ori

                if not os.path.exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding : {}'.format(initcache))

                print("Use {} as centroid".format(initcache))
                with h5py.File(initcache, mode='r') as h5:
                    clsts = h5.get("centroids")[...]
                    clsts = clsts+1e-10  # prevent to divide by 0
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs)
                    del clsts, traindescs
            model.add_module('pool', net_vlad)
        elif opt.pooling.lower() == 'max':
            #global_pool = nn.AdaptiveMaxPool2d((1, 1)) # ori,  input : (24,512, 30 ,40) ==> global_pool(input) : (24, 512, 1, 1)
            global_pool = pool_model.MaxPooling() # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 512, 1, 1)
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()])) # Flatten() output : (24, 512)
        elif opt.pooling.lower() == 'avg':
            #global_pool = nn.AdaptiveAvgPool2d((1, 1)) # ori
            global_pool = pool_model.AvgPooling() # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 512, 1, 1)
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'gem':
            global_pool = pool_model.GeM(p=3) # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 512, 1, 1)
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()])) # Flatten() output : (24, 512)
        elif opt.pooling.lower() == 'tra':
            global_pool = pool_model.TopRankingAvg(top_ratio=opt.tra_ratio, p=3) # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 51
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()])) # Flatten() output : (24, 512)
        elif opt.pooling.lower() == 'trg':
            global_pool = pool_model.TopRankingGeM(top_ratio=opt.tra_ratio, p=3) # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 51
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()])) # Flatten() output : (24, 512)
        elif opt.pooling.lower() == 'prj':
            global_pool = pool_model.ProjectionPooling(H=30, W=40) # input : (24,512, 30 ,40) ==> global_pool(input) : (24, 512, 70)
            model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()])) # Flatten() output : (24, 512*70)
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

        if verbose:
            print("summary(model.pool.cuda(), ({},{},{}))".format(fea_ch, fea_h, fea_w))
            summary(model.pool.cuda(), (fea_ch, fea_h, fea_w))

    return model, encoder_dim
