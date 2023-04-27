import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace as bp

## Usage : Copy following two lines to your code and remove remark at the beginning of the line to choose the mode among "Interactive" and "Agg" :
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

class GaussianPooling2d(nn.AvgPool2d):  # Not implemented yet.
    def __init__(self, num_features, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, hidden_node=None, stochasticity='HWCN', eps=1e-6):
        if stochasticity != 'HWCN' and stochasticity != 'CN' and stochasticity is not None:
            raise ValueError("gaussian pooling stochasticity has to be 'HWCN'/'CN' or None, "
                         "but got {}".format(stochasticity))
        if hidden_node is None:
            hidden_node = num_features // 2

        super(GaussianPooling2d, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad)
        self.eps = eps
        self.stochasticity = stochasticity

        self.ToHidden = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_features, hidden_node, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(hidden_node),
            nn.ReLU(False),
        )
        self.ToMean = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
        )
        self.ToSigma = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
            nn.Sigmoid()
        )
        self.activation = nn.Softplus()

    def forward(self, input):
        mu0 = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= F.avg_pool2d(input**2, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= torch.sqrt(torch.clamp(sig0 - mu0**2, self.eps))

        Z = self.ToHidden(input)
        MU = self.ToMean(Z)

        if self.training and self.stochasticity is not None:
            SIGMA = self.ToSigma(Z)
            if self.stochasticity == 'HWCN':
                size = sig0.size()
            else:
                size = [sig0.size(0), sig0.size(1), 1, 1]
            W = self.activation(MU + SIGMA *
                torch.randn(size, dtype=sig0.dtype, layout=sig0.layout, device=sig0.device))
        else:
            W = self.activation(MU)

        return mu0 + W*sig0

class HistogramPooling(nn.Module):  # Not implemented yet.
    def __init__(self):
        super(HistogramPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        N, C, H, W = x.shape  ## x : BatchSize x K x H x W, ex) 24 x 512 x 30 x 40
        x_norm = x/(torch.max(x,2)[0].unsqueeze(-1))

        return x

class MultiScaleMaxPooling(nn.Module):  # Not implemented.
    def __init__(self):
        super(MultiScaleMaxPooling, self).__init__()
        self.max_pool0 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool3 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        N, C, H, W = x.shape  ## x : BatchSize x K x H x W, ex) 24 x 512 x 30 x 40
        self.max_pool0

        return x

class ProjectionPooling(nn.Module):  # Not implemented.
    def __init__(self, H=3, W=4):  # feature size
        super(ProjectionPooling, self).__init__()
        self.H, self.W = H, W
        self.prj_to_hori = nn.LPPool2d(norm_type=1, kernel_size=(H,1))  # [30, 40] ==> [1, 40]
        self.prj_to_vert = nn.LPPool2d(norm_type=1, kernel_size=(1,W))  # [30, 40] ==> [30, 1]

    def resize(self, prediction : torch.Tensor):
        prediction = torch.nn.functional.interpolate(prediction,size=(self.H, self.W),
                mode="bicubic",align_corners=False)
        return prediction


    def forward(self, x:torch.Tensor):
        # x = torch.rand(1,512,H,W)  # feature size
        x = self.resize(x)

        prjx = self.prj_to_hori(x).squeeze(-2)  # [1,512,4]
        prjy = self.prj_to_vert(x).squeeze(-1)  # [1,512,3]

        x = torch.cat((prjx, prjy), dim=-1)  # [1,512,7]

        return x

class TopRankingGeM(nn.Module):  # TRA means Top Ranking Average.
    def __init__(self, top_ratio=0.03, eps=1e-6, p=3, H=30, W=40,
            intra_ch_normalize=False, inter_ch_normalize=False):
        super(TopRankingGeM, self).__init__()
        self.eps = eps
        self.p = p
        self.top_ratio = top_ratio
        self.inter_ch_normalize = inter_ch_normalize
        self.intra_ch_normalize = intra_ch_normalize
        self.global_feature_output_size = 1
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size = self.global_feature_output_size)
        self.top_k = int(H*W*self.top_ratio)  # 120, 1200 * 0.1

    def forward(self, x):  # x : local_feature
        N, C, H, W = x.shape  #  # x : BatchSize x C x H x W, ex) 24 x 512 x 30 x 40

        ## Flatten 2D space to serialized 1D space
        x = x.view(N, C, -1)  # Flatten, [N, C, H, W] ==> [N, C, H*W] Serialize 2D feature image to a 1D vector
        #x = x.clamp(min=self.eps)  # Clamping , do not clamping because x has negative value
        
        # Intra-channel normalization, L2 normalize across H*W space direction ( dim=2 )
        if self.intra_ch_normalize:
            x = F.normalize(x, p=2, dim=2)

        ## Use top-ranking features in every channel
        x = torch.topk(x, self.top_k).values  # [N, C, top_k]

        ## GeM 
        x = self.avgpool(x.clamp(min=self.eps).pow(self.p)).pow(1./self.p)

        if self.inter_ch_normalize:
            x = F.normalize(x, p=2, dim=1)  # Inter-channel normalization, L2 normalize across C direction ( dim=1 )

        return x # [B C 1] returned x is global feature

## We will get average of top 10%, 120 of 1200, feature pixels
class TopRankingAvg(nn.Module):  # TRA means Top Ranking Average.
    def __init__(self, top_ratio=0.03, eps=1e-6, p=3, H=30, W=40,
            intra_ch_normalize=False, inter_ch_normalize=False):
        super(TopRankingAvg, self).__init__()
        self.eps = eps
        self.p = p
        self.top_ratio = top_ratio
        self.inter_ch_normalize = inter_ch_normalize
        self.intra_ch_normalize = intra_ch_normalize
        self.global_feature_output_size = 1
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size = self.global_feature_output_size)
        self.top_k = int(H*W*self.top_ratio)  # 120, 1200 * 0.1

    def forward(self, x):  # x : local_feature
        N, C, H, W = x.shape  #  # x : BatchSize x C x H x W, ex) 24 x 512 x 30 x 40

        ## Flatten 2D space to serialized 1D space
        x = x.view(N, C, -1)  # Flatten, [N, C, H, W] ==> [N, C, H*W] Serialize 2D feature image to a 1D vector
        #x = x.clamp(min=self.eps)  # Clamping , do not clamping because x has negative value
        
        # Intra-channel normalization, L2 normalize across H*W space direction ( dim=2 )
        if self.intra_ch_normalize:
            x = F.normalize(x, p=2, dim=2)

        ## Use top-ranking features in every channel
        x = torch.topk(x, self.top_k).values  # [N, C, top_k]

        ## GeM 
        #global_feature = F.avg_pool1d(x.pow(self.p), kernel_size=(x.size(-1))).pow(1./self.p)  # kernel_size=(H, W)

        ## Average
        x = self.avgpool(x)  # kernel_size=(H, W)

        if self.inter_ch_normalize:
            x = F.normalize(x, p=2, dim=1)  # Inter-channel normalization, L2 normalize across C direction ( dim=1 )

        return x # [B C 1] returned x is global feature

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # N, C, H, W = x.shape  ## x : BatchSize x K x H x W, ex) 24 x 512 x 30 x 40
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        ## N, C, H, W ==> N, C, 1, 1
        global_feature = F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=(x.size(-2), x.size(-1))).pow(1./p)  # kernel_size=(H, W)
        return global_feature  # [N, C, 1, 1]  , squeezing will be conducted in Flatten() in main.py
        #return global_feature.squeeze().squeeze()

    def __repr__(self):return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x

class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_feature = torch.rand(24, 512, 30, 40).to(device)
    print(" Shape of local_feature : {}".format(local_feature.shape))

    if True:
        encoder = TopRankingAvg(top_ratio=0.03, p=3).to(device)
        global_feature = encoder(local_feature)
        print("[TopRankingAvg] Shape of global_feature : {}".format(global_feature.shape))

    if False:
        encoder = GeM(p=3).to(device)
        global_feature = encoder(local_feature)
        print("[GeM] Shape of global_feature : {}".format(global_feature.shape))
