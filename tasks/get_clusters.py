from __future__ import print_function
from math import ceil
from os.path import join, exists
from os import makedirs
import os
import numpy as np
import h5py
import faiss

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import misc  # This code is called by ../main.py. So working directory is ../

from ipdb import set_trace as bp
import sys;sys.path.insert(0,'/home/ccsmm/workdir/ccsmmutils');import torch_img_utils as tim; tim.init()


'''
data=torch.Tensor([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]).unsqueeze(0)
ipdb> data
tensor([[ 1.,  2.,  3.,  4.,  5.],
        [ 6.,  7.,  8.,  9., 10.],
        [11., 12., 13., 14., 15.]])
ipdb> data.view(-1)
tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
        15.])

ipdb> data.shape
torch.Size([1, 3, 5])

    data.view(-1)[7] : 8
    data[0, 1, 2] : 8
    data[0, y, x], 

    rand_num = 7
    y = int(7/5) = 1
    x = 7 - y*5 = 2
    data[0, y, x]
'''

def draw_point(img, X, Y, pixel=255):
    try:
        img[Y-5:Y+5,X-5:X+5] = pixel
    except:
        img[Y,X] = pixel
    return img


def get_pts(opt, img, mask_gt, sample, nPerImage, iteration, debug_en=False):
    ''' mask_gt  # [1, H, W]
        sample # [100]
    '''
    H, W = mask_gt.shape[-2:]
    if opt.arch=="vgg16":
        hh, ww = int(H/(2**4)), int(W/(2**4))
    else:
        bp()

    black1 = torch.zeros_like(mask_gt)[0]
    black2 = torch.zeros_like(mask_gt)[0]

    for rand_num in sample:
        y = int(rand_num/ww)
        x = rand_num - y*ww
        X = int(x*(W/ww))
        Y = int(y*(H/hh))
        if debug_en:
            black1 = draw_point(black1, X, Y, 255)
            black2 = draw_point(black2, X, Y, 255)

        if mask_gt[0][Y,X] < 0.5:  # Reject point
            sample = np.delete(sample, np.where(sample == rand_num))
            if debug_en:
                black2 = draw_point(black2, X, Y, 0)
    if debug_en:
        tim.clf()
        tim.imshow(sp=141, img=tim.Denormalize()(img), title="input", dispEn=False)
        tim.imshow(sp=142, img=mask_gt, title="mask", cmap="gray", dispEn=False)
        tim.imshow(sp=143, img=black1, title="sample", cmap="gray", dispEn=False)
        tim.imshow(sp=144, img=black2, title="sample(filtered)", dispEn=False, cmap="gray")
        ofdir="clustering_sample_points"
        ofname=os.path.join(ofdir, "sample_{0:05d}.png".format(iteration))
        if not exists(ofdir):
            os.makedirs(ofdir, exist_ok=True)
        tim.plt.savefig(ofname, dpi=100)

    return sample[:nPerImage]

def get_clusters(opt, model, seg_model, encoder_dim, mDatasetStruct):
    _, _, _, cluster_set, _ = mDatasetStruct.get_dataset()
    cuda, device = misc.get_device(opt)
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('   ===> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
                if opt.deattention:
                    _, mask_gt = seg_model.preprocess(input.detach()) # Ground Truth of segmentation

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    if opt.deattention:  # Since samples are expected to be masked out, larger numbers are used.
                        sample = np.random.choice(image_descriptors.size(1), nPerImage*2, replace=False)
                        sample = get_pts(opt, input[ix], mask_gt[ix], sample, nPerImage, iteration, debug_en=True)
                    else:  # normal
                        sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('   ===> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('   ===> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('   ===> Done!')
