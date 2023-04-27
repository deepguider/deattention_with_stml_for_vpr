import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as bp

## Usage : Copy following two lines to your code and remove remark at the beginning of the line to choose the mode among "Interactive" and "Agg" :
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

class ch_attention(nn.Module):
    """Constructs a ECA module.
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self):
        super(ch_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B, 512, 30, 40]
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  #[B, 512, 1, 1]

        # Multi-scale information fusion
        y = self.sigmoid(y)  # [B, 512, 1, 1]

        return x * y.expand_as(x)  #[B, 512, 30, 40]

class ch_eca_attention(nn.Module):
    """Constructs a ECA module.
    https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(ch_eca_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B, 512, 30, 40]
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  #[B, 512, 1, 1]

        # Two different branches of ECA module, conv1d([B, 1, 512])  ==> [B, 512, 1, 1], convolution in 512.
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)  # [B, 512, 1, 1]

        return x * y.expand_as(x)  #[B, 512, 30, 40]
        
class CRN(nn.Module):  # code by ccsmm
    def __init__(self, dim=512, version=1):
        super(CRN, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=1)
        self.version = version
        self.init_crn()

    def init_crn(self):
        dim = self.dim
        if self.version==3:
            d1, d2, d3 = 32, 32, 32
        else:  # CRN
            d1, d2, d3 = 32, 32, 20
        P = d1 + d2 + d3
        padding_mode = 'reflect'  # default is 'zeros', Refer to https://bo-10000.tistory.com/120
        self.crn_downsample = nn.Sequential(  # B, dim, 30, 40 ==> B, dim, 15, 20
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(dim),
                nn.MaxPool2d(2,2)
        )
        self.crn_conv1 = nn.Sequential(  # g, Multiscale Context Filters, kernel 3x3
                nn.Conv2d(dim, d1, 3, stride=1, padding=1, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d1)
        )
        self.crn_conv2 = nn.Sequential(  # g, Multiscale Context Filters, kernel 5x5
                nn.Conv2d(dim, d2, 5, stride=1, padding=2, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d2)
        )
        self.crn_conv3 = nn.Sequential(  # g, Multiscale Context Filters, kernel 7x7
                nn.Conv2d(dim, d3, 7, stride=1, padding=3, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d3)
        )
        self.crn_conv4 = nn.Sequential(  # w, Accumulation Weight
                nn.Conv2d(P, 1, 1, stride=1, padding=0, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(1)
        )

        self.crn_upsample = nn.Sequential(  # B, 1, 15, 20 ==> B, 1, 30, 40
                nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(dim),
        )
    
    def forward_crn_resize1(self, x):
        x = F.interpolate(x, (15,20), mode='bilinear', align_corners=True)
        x1 = self.crn_conv1(x)      # B, 512, 15, 20 ==> B, 32(d1), 15, 20
        x2 = self.crn_conv2(x)  # B, 512, 15, 20 ==> B, 32(d2), 15, 20
        x3 = self.crn_conv3(x)  # B, 512, 15, 20 ==> B, 20(d3), 15, 20
        x = torch.cat([x1,x2,x3], dim=1)  # concat, [B, 84, 15, 20]
        x = self.crn_conv4(x)  # B, 1, 15, 20
        x = F.interpolate(x, (30,40), mode='bilinear', align_corners=True)
        x = torch.sigmoid(x)
        #x = self.softmax(x) 
        return x

    def forward(self, x):  # x : [B, 512, 30, 40]
        x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        m1 = self.crn_conv1(x)  # B, 512, 30, 40 ==> B, 32(d1), 30, 40
        m2 = self.crn_conv2(x)  # B, 512, 30, 40 ==> B, 32(d2), 30, 40
        m3 = self.crn_conv3(x)  # B, 512, 30, 40 ==> B, 20(d3), 30, 40
        m = torch.cat([m1,m2,m3], dim=1)  # concat, [B, 84, 30, 40]
        m = self.crn_conv4(m)  # B, 1, 30, 40
        mask = torch.sigmoid(m)
        return x * mask.expand_as(x), mask  # x : [B, 512, 30, 40], mask : [B, 1, 30, 40]

class DeAttention(nn.Module):
    def __init__(self, version=1, padding_mode='zeros'):
        super(DeAttention, self).__init__()
        self.dim = 512 
        self.softmax = nn.Softmax(dim=1)
        self.version = version
        self.padding_mode = padding_mode  # default is 'zeros', Refer to https://bo-10000.tistory.com/120
        self.softmax = nn.Softmax(dim=-1)
        self.init_crn()

    def init_crn(self):
        dim = self.dim
        if self.version==3:
            d1, d2, d3 = 32, 32, 32
        else:  # CRN
            d1, d2, d3 = 32, 32, 20
        P = d1 + d2 + d3
        padding_mode = self.padding_mode
        self.crn_downsample = nn.Sequential(  # B, dim, 30, 40 ==> B, dim, 15, 20
                nn.Conv2d(dim, dim, 3, stride=1, padding=1, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(dim),
                nn.MaxPool2d(2,2)
        )
        self.crn_conv1 = nn.Sequential(  # g, Multiscale Context Filters, kernel 3x3
                nn.Conv2d(dim, d1, 3, stride=1, padding=1, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d1)
        )
        self.crn_conv2 = nn.Sequential(  # g, Multiscale Context Filters, kernel 5x5
                nn.Conv2d(dim, d2, 5, stride=1, padding=2, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d2)
        )
        self.crn_conv3 = nn.Sequential(  # g, Multiscale Context Filters, kernel 7x7
                nn.Conv2d(dim, d3, 7, stride=1, padding=3, padding_mode=padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(d3)
        )
        self.crn_conv4 = nn.Sequential(  # w, Accumulation Weight
                nn.Conv2d(P, 1, 1, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(1)
        )

        self.crn_upsample = nn.Sequential(  # B, 1, 15, 20 ==> B, 1, 30, 40
                nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(dim),
        )

    def forward_crn(self, x):
        m1 = self.crn_conv1(x)  # B, 512, 30, 40 ==> B, 32(d1), 30, 40
        m2 = self.crn_conv2(x)  # B, 512, 30, 40 ==> B, 32(d2), 30, 40
        m3 = self.crn_conv3(x)  # B, 512, 30, 40 ==> B, 20(d3), 30, 40
        m = torch.cat([m1,m2,m3], dim=1)  # concat, [B, 84, 30, 40]
        m = self.crn_conv4(m)  # B, 1, 30, 40
        mask = torch.sigmoid(m)
        return mask

    def normalize(self, x, eps=1e-9):  # transform data to [0 ~ 1] in 2d direction.
        B,C,H,W = x.shape
        x = x.view(B, C, -1)
        x = x-x.min(dim=2, keepdim=True)[0]
        x = x/(x.max(dim=2, keepdim=True)[0] + eps)
        x = x.view(B, C, H, W)
        return x

    def forward(self, x):  # x : [B, 512, 30, 40]
        x_norm = F.normalize(x, p=2, dim=1)  # across descriptor dim  # original
        mask = self.forward_crn(x_norm)
        if self.version == 2:
            x = x + x*mask
        else:  # default
            x = x*mask
        return x, mask  # [B, 1, 30, 40]

class DeAttention_new(nn.Module):  # DeAttention_new and DeAttention have same functionality, but DeAttention_new seems more concise.
    def __init__(self, version=1):
        super(DeAttention_new, self).__init__()
        self.dim = 512
        self.softmax = nn.Softmax(dim=1)
        self.version = version
        self.crn = CRN(self.dim, self.version)

    def forward(self, x):  # x : [B, 512, 30, 40]
        _, weight = self.crn(x)  # Attention weight
        if self.version == 2:
            x += x*weight
        else:  # default
            x = x*weight
        return x, weight  # [B, 1, 30, 40]

class DeAttention_auto(nn.Module):
    def __init__(self, version=1, concern_category_list=["nature", "sky", "human", "vehicle", "flat", "construction"]):
        super(DeAttention_auto, self).__init__()
        self.dim = 512
        self.softmax = nn.Softmax(dim=1)
        self.version = version
        self.concern_category_list = concern_category_list  # all category lists, ["nature", "sky", "human", "vehicle", "flat", "construction"] in cityscape.py
        self.num_of_net = len(self.concern_category_list)
        self.init_mask_pred()
        self.init_mask_weight()

    def init_mask_pred(self):
        self.crn1 = CRN(self.dim, self.version)  # nature
        self.crn2 = CRN(self.dim, self.version)  # sky
        self.crn3 = CRN(self.dim, self.version)  # human
        self.crn4 = CRN(self.dim, self.version)  # vehicle
        self.crn5 = CRN(self.dim, self.version)  # flat (road)
        self.crn6 = CRN(self.dim, self.version)  # construction (building)

    def init_mask_weight(self):
        dim = self.dim
        d1, d2, d3 = 32, 32, 20
        P = d1 + d2 + d3
        d4 = 32
        self.crn_conv1 = nn.Sequential(  # g, Multiscale Context Filters, kernel 3x3
                nn.Conv2d(dim, d1, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(d1)
        )
        self.crn_conv2 = nn.Sequential(  # g, Multiscale Context Filters, kernel 5x5
                nn.Conv2d(dim, d2, 5, stride=1, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(d2)
        )
        self.crn_conv3 = nn.Sequential(  # g, Multiscale Context Filters, kernel 7x7
                nn.Conv2d(dim, d3, 7, stride=1, padding=3),
                nn.ReLU(),
                nn.BatchNorm2d(d3)
        )
        self.crn_conv4 = nn.Sequential(  # w, Accumulation Weight
                nn.Conv2d(P, d4, 1, stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(d4)
        )

        oh, ow = 3, 4
        self.avgpool = nn.AdaptiveAvgPool2d((oh, ow))  # [B, c, h, w] ==> [ B, C, oh, ow]
        self.fc1 = nn.Linear(d4*oh*ow, 64)    # 32*3*4 (384) ==> 64
        self.fc2 = nn.Linear(64, 6)   # 64 ==> 6

    def forward_mask_weight(self, x):
        x1 = self.crn_conv1(x)  # B, 512, 30, 40 ==> B, 32(d1), 30, 40
        x2 = self.crn_conv2(x)  # B, 512, 30, 40 ==> B, 32(d2), 30, 40
        x3 = self.crn_conv3(x)  # B, 512, 30, 40 ==> B, 20(d3), 30, 40
        x = torch.cat([x1,x2,x3], dim=1)  # concat, [B, 84, 30, 40]
        x = self.crn_conv4(x)  # [B, 84, 30, 40] ==> [B, 32, 30, 40]
        x = self.avgpool(x)  # [B, 32, 30, 40] ==> [B, 84, 3, 4]
        x = x.view(x.size(0), -1)  # [B, 32, 3, 4] ==> [B, 384]
        x = self.fc1(x)  # [B, 384] ==> [B, 64]
        x = self.fc2(x)  # [B, 64] ==> [B, 6]
        mask_weight = torch.sigmoid(x)
        return mask_weight  #  [B, 6]

    def forward_mask_pred(self, x):
        _, mask_pred1 = self.crn1(x)  # [B, 1, 30, 40]
        _, mask_pred2 = self.crn2(x)  # [B, 1, 30, 40]
        _, mask_pred3 = self.crn3(x)  # [B, 1, 30, 40]
        _, mask_pred4 = self.crn4(x)  # [B, 1, 30, 40]
        _, mask_pred5 = self.crn5(x)  # [B, 1, 30, 40]
        _, mask_pred6 = self.crn6(x)  # [B, 1, 30, 40]
        mask_pred = [mask_pred1, mask_pred2, mask_pred3, mask_pred4, mask_pred5, mask_pred6]
        return mask_pred  # list of 6 tensor : [ [B,1,30,40], ... , [B,1,30,40]]

    def forward(self, x):  # x : [B, 512, 30, 40]
        mask_pred = self.forward_mask_pred(x)  # [B, 1, 30, 40] x 6
        mask_weight = self.forward_mask_weight(x)  # [mask_w0, ..., mask_w5] = [0.1, 1.0, ... , 0.9]
        return mask_pred, mask_weight   ## Result for mask of ["nature", "sky", "human", "vehicle", "flat", "construction"] in cityscape.py
