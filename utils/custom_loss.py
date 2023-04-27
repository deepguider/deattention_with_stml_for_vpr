import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from ipdb import set_trace as bp


def l2norm(a, b):
    return ((a - b)**2).sum().sqrt()

class MeanStd():
    def __init__(self, keep_whole_data=False):
        self.d = self.l2norm
        self.keep_whole_data = keep_whole_data
        self.clear_whole_data()
        self.clear_buffer()
        self.clear_cache()

    def set_keep_whole_data(self, sw=True):
        self.keep_whole_data = sw

    def angle(self, a, b):
        '''
            Will return 1 for similar(0 degree), 0 different(+- 90 degree), -1 opposite (180 degree)
        '''
        cossim = F.cosine_similarity(a, b)
        degree = math.degrees(math.acos(cossim))
        return degree

    def clear_whole_data(self):
        self.whole_d_qn = np.empty(0)
        self.whole_d_qp = np.empty(0)
        self.whole_d_pn = np.empty(0)
        self.whole_angle_nqp = np.empty(0)

    def clear_buffer(self):
        self.d_qn = np.empty(0)
        self.d_qp = np.empty(0)
        self.d_pn = np.empty(0)
        self.angle_nqp = np.empty(0)

    def clear_cache(self):
        self.d_qn_cache = np.empty(0)
        self.d_qp_cache = np.empty(0)
        self.d_pn_cache = np.empty(0)
        self.angle_nqp_cache = np.empty(0)

    def l2norm(self, a, b):
        return ((a - b)**2).sum().sqrt()

    def push(self, anchor, positive, negative):
        q = anchor.detach().cpu()
        p = positive.detach().cpu()
        n = negative.detach().cpu()
        d_qp = self.d(q, p)
        d_qn = self.d(q, n)
        d_pn = self.d(p, n)
        angle_nqp = self.angle(q-n, q-p)
        ## Distance
        self.d_qp = np.append(self.d_qp, d_qp)
        self.d_qn = np.append(self.d_qn, d_qn)
        self.d_pn = np.append(self.d_pn, d_pn)
        ## Angle
        self.angle_nqp = np.append(self.angle_nqp, angle_nqp)

        if self.keep_whole_data:
            ## Distance
            self.whole_d_qp = np.append(self.whole_d_qp, d_qp)
            self.whole_d_qn = np.append(self.whole_d_qn, d_qn)
            self.whole_d_pn = np.append(self.whole_d_pn, d_pn)
            ## Angle
            self.whole_angle_nqp = np.append(self.whole_angle_nqp, angle_nqp)

    def push_cache(self, anchor, positive, negative):
        q = anchor.detach().cpu()
        p = positive.detach().cpu()
        n = negative.detach().cpu()
        self.d_qp_cache = np.append(self.d_qp_cache, self.d(q, p))
        self.d_qn_cache = np.append(self.d_qn_cache, self.d(q, n))
        self.d_pn_cache = np.append(self.d_pn_cache, self.d(p, n))
        ## Angle
        self.angle_nqp_cache = np.append(self.angle_nqp_cache, self.angle(q-n, q-p))

    def divide(self, div):
        if div > 0:
            self.d_qp = self.d_qp / div
            self.d_qn = self.d_qn / div
            self.d_pn = self.d_pn / div
            self.angle_nqp = self.angle_nqp / div

    def divide_cache(self, div):
        if div > 0:
            self.d_qp_cache = self.d_qp_cache / div
            self.d_qn_cache = self.d_qn_cache / div
            self.d_pn_cache = self.d_pn_cache / div
            self.angle_nqp_cache = self.angle_nqp_cache / div

    def flush_cache(self):
        self.d_qp = np.append(self.d_qp, self.d_qp_cache)
        self.d_qn = np.append(self.d_qn, self.d_qn_cache)
        self.d_pn = np.append(self.d_pn, self.d_pn_cache)
        self.angle_nqp = np.append(self.angle_nqp, self.angle_nqp_cache)
        self.clear_cache()

    def get_mean(self):
        return self.d_qp.mean(), self.d_qn.mean(), self.d_pn.mean(), self.angle_nqp.mean()

    def get_std(self):
        return self.d_qp.std(), self.d_qn.std(), self.d_pn.std(), self.angle_nqp.std()

    def get_data(self):
        return self.d_qp, self.d_qn, self.d_pn, self.angle_nqp

    def get_whole_mean(self):
        return self.whole_d_qp.mean(), self.whole_d_qn.mean(), self.whole_d_pn.mean(), self.whole_angle_nqp.mean()

    def get_whole_std(self):
        return self.whole_d_qp.std(), self.whole_d_qn.std(), self.whole_d_pn.std(), self.whole_angle_nqp.std()

    def get_whole_data(self):
        return self.whole_d_qp, self.whole_d_qn, self.whole_d_pn, self.whole_angle_nqp

class TripletMarginLoss2():
    def __init__(self, margin=0.1, pn_margin=1.4, version=1, device="cpu", reduction="mean"):
        self.margin = margin
        self.pn_margin = pn_margin
        self.device = device
        self.reduction = reduction
        self.version = version

    def l2norm(self, a, b):
        return ((a - b)**2).sum().sqrt()

    def set_margin(self, margin):
        self.margin = margin

    def get_margin(self, margin):
        return self.margin

    def __call__(self, anchor, positive, negative):
        distance_function = self.l2norm
        if self.version == 2:
            posneg_dist = distance_function(positive, negative)
            output = torch.clamp(self.pn_margin - posneg_dist, min=0.0).to(self.device)
        else:  # verwion == 1
            positive_dist = distance_function(anchor, positive)
            negative_dist = distance_function(anchor, negative)
            posneg_dist = distance_function(positive, negative)
            output = torch.clamp(positive_dist +self.margin - negative_dist + self.pn_margin - posneg_dist, min=0.0).to(self.device)

        return output

def TripletMarginLoss2_func(anchor, positive, negative, margin, device='cpu', reduction='mean'):
    """ anchor, positive, negative : shape of [1 , 32768]
    """
    distance_function = l2norm

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)
    posneg_dist = distance_function(positive, negative)

    output = torch.clamp(positive_dist - negative_dist + margin - posneg_dist, min=0.0)  # new

    print("TML2, output:", output)

    return output

def PNPairLoss(positive, negative, margin, device='cpu', reduction='mean'):
    """ positive, negative : shape of [1 , 32768]
    """
    distance_function = l2norm

    posneg_dist = distance_function(positive, negative)

    output = torch.clamp(margin - posneg_dist, min=0.0)  # new

    return output


def TripletMarginLoss(anchor, positive, negative, margin, device='cpu', reduction='mean'): 
    """ anchor, positive, negative : shape of [1 , 32768]
    """
    #distance_function = torch.nn.MSELoss(reduction=reduction).to(device)  # not same
    distance_function = l2norm

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)
    posneg_dist = distance_function(positive, negative)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)
    return output


if __name__ == '__main__':
    torch.manual_seed(100)
    Q = torch.rand(1, 4)  # batch, descriptor dim
    P = torch.rand(1, 4)  # batch, descriptor dim
    N = torch.rand(4, 4)  # batch, descriptor dim
    Margin = 1.0
    device='cpu'

    criterion = nn.TripletMarginLoss(margin=Margin, p=2, reduction='mean').to(device) # by ccsmm, Batch size is no longer a hyper parame
    criterion1 = TripletMarginLoss
    criterion2 = TripletMarginLoss2


    loss, loss1, loss2 = 0, 0, 0
    for i in range(len(N)):
        qe = Q
        pos = P
        neg = N[i].unsqueeze(0)
        loss += criterion(qe, pos, neg)
        loss1 += criterion1(qe, pos, neg, Margin)
        loss2 += criterion2(qe, pos, neg, Margin)
        bp()

    print(loss/len(N), loss1/len(N), loss2/len(N))
 
