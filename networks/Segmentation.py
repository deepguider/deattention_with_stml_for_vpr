from networks import MobileNet
import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import cv2

from ipdb import set_trace as bp
import sys;sys.path.insert(0,"/home/ccsmm/dg_git/ccsmmutils/torch_utils");import torch_img_utils as tim;

class Segmentation():
    def __init__(self, opt, device, deatt_category_list=["human", "vehicle"]):
        # Set up model
        model_map = {
            'deeplabv3_resnet50': MobileNet.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': MobileNet.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': MobileNet.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': MobileNet.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': MobileNet.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': MobileNet.deeplabv3plus_mobilenet
        }
        model_name = 'deeplabv3plus_mobilenet'
        num_classes = 19
        output_stride = 16
        val_batch_size = opt.batchSize
        num_worker = opt.threads
        ckpt = opt.seg_ckpt
        self.deatt_weighted_mask = opt.deatt_weighted_mask

        model = model_map[model_name](num_classes=num_classes, output_stride=output_stride)
        self.device = device

        if ckpt is not None and os.path.isfile(ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)

        model.eval()
        self.model = model
        ## You can get self.cls_freq_dict from task/get_class_statics.py by running ./0run_class_statics.sh
        if True:  # normalization with sigmoid
            self.cls_freq_dict = {'flat': 0.7400928999587367, 'construction': 1.0, 'object': 0.6892017590821214, 'nature': 0.7836972764175455, 'sky': 0.7378676239332944, 'human': 0.6861023960478257, 'vehicle': 0.7020041343706618}
        else:  # normalization only, recall at 1 is 0.87 in de-attention only, deatt_w = 0.001
            self.cls_freq_dict = {'flat': 0.16457551264274523, 'construction': 1.0, 'object': 0.015387737157157834, 'nature': 0.2938099586278422, 'sky': 0.15802586041687802, 'human': 0.006324190875579154, 'vehicle': 0.052836867439999095}
        self.initialize_Cityscapes_CustomImgLoader(deatt_category_list)

    def initialize_Cityscapes_CustomImgLoader(self, deatt_category_list):
        self.deatt_category_list = deatt_category_list
        self.CustomImgLoader = MobileNet.Cityscapes_CustomImgLoader()
        self.CustomImgLoader.keep_concern_category()  # Reset to use all category. In the following command, the only concern category will be choosed.
        self.concern_category_list = self.CustomImgLoader.concern_category  # all category lists, ["nature", "sky", "human", "vehicle", "flat", "construction"] in cityscape.py

        if self.deatt_weighted_mask == True:
            for category in self.concern_category_list:
                self.CustomImgLoader.set_train_id_to_mask_by_category(category, self.cls_freq_dict[category])
        else:
            for category in deatt_category_list:
                #print("\t- Set {} category as de-attention region.".format(category))
                self.CustomImgLoader.reject_train_id_to_mask_by_category(category)

        self.print_category()

    def print_category(self):
        categories = self.CustomImgLoader.get_train_id_to_category()
        mask_ids = self.CustomImgLoader.get_train_id_to_mask()
        names = self.CustomImgLoader.get_train_id_to_name()
        print("\tname (category) : 1 (keep) or 0 (deattention)")
        for idx, category in enumerate(categories):
            print("\t{} ({}) : {}".format(names[idx], category, mask_ids[idx]))

    def encoder(self, images):  # [Batch, C, H, W] of torch data
        preds = self.model(images)  # image.shape is [batch, class, H, W], ex) [ 24, 19, 480, 640]
        return preds

    def tonumpys(self, preds):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        segmentations = []
        masks = []
        for i, pred in enumerate(preds):
            segmentation, mask = self.CustomImgLoader.decode_simple(pred)
            segmentations.append(segmentation)
            masks.append(mask)
        return segmentations, masks  # list

    def tonumpy(self, preds, idx=0):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        pred = preds[idx]  #[480, 640]
        segmentation, mask = self.CustomImgLoader.decode_simple_old(pred)
        return segmentation, mask

    def tomask(self, preds):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        mask = self.CustomImgLoader.decode_mask(preds)
        return mask

    def tosimple(self, preds):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        mask = self.CustomImgLoader.decode_simple(preds)
        return mask

    def to_id(self, preds):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        mask = self.CustomImgLoader.decode_id(preds)
        return mask

    def get_classes(self):
        classes = self.CustomImgLoader.get_classes()
        return classes

    def to_category_id(self, preds):  # preds.shape is [batch, class, H, W]
        preds = preds.detach().max(dim=1)[1].cpu().numpy()  # converted size is [batch, 480, 640], ex) [24, 480, 640]
        mask = self.CustomImgLoader.decode_category_id(preds)
        return mask

    def get_result(self, image):
        return self.encoder(image) 

    def to_device(self, data):
        data = torch.from_numpy(data)
        data = data.unsqueeze(1)
        data = data.to(self.device) # In data, segmented region : 0, normal region : 1
        return data

    def get_seg_class(self, images, mode="simple"):
        self.images = images
        segmentation = self.encoder(images)  # output.shape is [batch, class, H, W], ex) [ 24, 19, 480, 640]
        seg_cls_simple = self.tosimple(segmentation)  # [24, 480, 640]
        seg_cls_category_id = self.to_category_id(segmentation)  # [24, 480, 640]
        seg_cls_id = self.to_id(segmentation)  # [24, 480, 640]
        return self.to_device(seg_cls_simple), self.to_device(seg_cls_category_id), self.to_device(seg_cls_id) # image shape, [batchsize, h, w]

    def preprocess_auto(self, images):
        self.images = images
        device = self.device
        segmentation = self.encoder(images)  # output.shape is [batch, class, H, W], ex) [ 24, 19, 480, 640]
        seg_mask_dict = {}

        ## Todo : change variuos weight
        for category in self.concern_category_list:  # all category lists, ["nature", "sky", "human", "vehicle", "flat", "construction"] in cityscape.py 
            if True:  # Reject all but keep only concern category
                self.CustomImgLoader.reject_concern_category()  # Reset to use all category. In the following command, the only concern category will be choosed.
                self.CustomImgLoader.keep_train_id_to_mask_by_category(category)
            else:  # Keep all but reject only concern category
                self.CustomImgLoader.keep_concern_category()  # Reset to use all category. In the following command, the only concern category will be choosed.
                self.CustomImgLoader.reject_train_id_to_mask_by_category(category)
            seg_mask = self.tomask(segmentation)  # [24, 480, 640]
            seg_mask = torch.from_numpy(seg_mask)
            seg_mask = seg_mask.unsqueeze(1)
            seg_mask = seg_mask.to(device) # In seg_mask, segmented region : 0, normal region : 1
            seg_mask_dict[category] = seg_mask
        return seg_mask_dict

    def preprocess(self, images, mode='random'):
        self.images = images
        device = self.device
        segmentation = self.encoder(images)  # output.shape is [batch, class, H, W], ex) [ 24, 19, 480, 640]
        seg_mask = self.tomask(segmentation)  # [24, 480, 640]
        seg_mask = torch.from_numpy(seg_mask)
        seg_mask = seg_mask.unsqueeze(1)

        seg_mask = seg_mask.to(device) # In seg_mask, segmented region : 0, normal region : 1
        mask_inv = (-1*seg_mask + 1)   # In mask_inv, segmented region : 1, normal region : 0
        seg_mask_shape = seg_mask.shape
    
        if mode.lower() == 'random':
            mask = 1*torch.rand(seg_mask_shape).to(device) * mask_inv
            images_att = images*seg_mask + mask
        elif mode.lower() == 'zero':  # The segmented regions are replaced with zero value
            images_att = images*seg_mask
        elif mode.lower() == 'one':  # ROI saturated
            one_mask = (1*torch.ones(seg_mask_shape)).to(self.device)*mask_inv
            images_att = images*seg_mask + one_mask.to(device)
        elif mode.lower() == 'mosaic':  # mosaic with gaussian blurring
            ## To do : make mosaic image
            images_att = torch.zeros_like(images)
            for batch_idx in range(len(images)):
                aimage = self.tensor_to_cv(images[batch_idx])
                amask_inv = self.tensor_to_cv(mask_inv[batch_idx]).squeeze()
                image_att_cv = self.get_contours_with_cv_img(aimage, amask_inv, sigma=5)
                # cv2.imshow('blurred', image_att_cv)
                # cv2.waitKey(1)
                images_att[batch_idx] = self.cv_to_tensor(image_att_cv)
            images_att = images_att.to(self.device)
        elif mode.lower() == 'softassign':  # return image as is, and seg_mask
            images_att = images
        else:
            images_att = images
        return images_att, seg_mask

    def cv_to_tensor(self, image):
        # image = np.random.rand(480,640,3)
        if type(image) is np.ndarray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv to PIL
            tensor_img = torch.Tensor(np.transpose(image,(2,0,1))).to(self.device)  # h,w,c ==> c,h,w
            return tensor_img
        else:
            return image

    def tensor_to_cv(self, image):
        # image = np.random.rand(3, 480,640)
        # image = torch.Tensor(image).to('cuda')
        image = image.cpu().numpy()
        # a_mask = np.array(mask.cpu(), dtype=np.uint8)  # [ch, h, w]
        cv_img = np.transpose(image,(1,2,0))  # c,h,w ==> h,w,c : 480*640*3
        if image.shape[0] == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # PIL to cv
        return cv_img

    def get_contours_with_cv_img(self, image, mask, sigma=1):  # The higher sigma is, The blurred image is.
        ## Refer : https://stackoverflow.com/questions/63928668/how-to-filter-contours-based-on-their-approximate-shape-in-a-binary-video-frames
        #find contours
        # mask.shape is [24, 1, 480, 640], which means [batch, ch, h, w]

        Binary = mask*255  # [480, 640]
        Binary = np.array(Binary, dtype=np.uint8)  # type casting for cv2.findContours()
        contours, hierarchy = cv2.findContours(Binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #create an empty image for contours
        img_contours = np.zeros(Binary.shape)
        # draw the contours on the empty image
        cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

        ## To do : make blurring
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            patch = image[y: y + h, x: x + w]
            patch_blurred = cv2.GaussianBlur(patch, ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
            image[y: y + h, x: x + w] = patch_blurred
            # image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
        return image

    def overwrite_patch(self, image, Binary, bbox, new_lt_pts): # bbox=(x,y,h,w), new_lt_pts=(new_x, new_y) of top-left point
        ''' Add patch of Binary into image at new_lt_pts, where bbox is the location of Binary '''
        if image.shape[-2:] != Binary.shape[-2:]:
            return image

        x,y,w,h = bbox
        new_x, new_y = new_lt_pts
        patch = image[:, y: y + h, x: x + w]
        patch_mask = Binary[y: y + h, x: x + w]  # background 0, object : 255

        _, patch_height, patch_width = patch.shape
        for py in range(patch_height):
            for px in range(patch_width):
                if patch_mask[py, px] > 0:
                    image[:, new_y + py, new_x + px] = patch[:, py, px]
        return image

    def remove_pepper_noise(self, img, N=2, ksize=3):
        img[img>0]=255
        img[img<255]=0

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        for i in range(N):  # remove pepper
            img = cv2.erode(img, k)
        
        for i in range(N):  # original size
            img = cv2.dilate(img, k)
        return img 

    def hole_filling(self, img, N=10, ksize=3):
        ## Hole filling
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        for i in range(N):  # hole filling
            img = cv2.dilate(img, k)
        
        for i in range(N):  # original size
            img = cv2.erode(img, k)
        return img 


    def add_clutter_to_single_cv_img(self, image, Binary, iteration=1, dispEn=False, write_image=False, fname=None):
        ##  Change that : mask=255, backgroud=0
        Binary = np.invert(Binary)
        
        Binary_ori = Binary.copy()
        ## Preprocessing, de-noising and hole filling
        Binary = self.remove_pepper_noise(Binary, N=3)
        Binary = self.hole_filling(Binary, N= 10)

        contours, hierarchy = cv2.findContours(Binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        C, H, W = image.shape

        if dispEn or write_image:
            tim.clf()
            tim.imshow(tim.Denormalize()(image), sp=141, title="input", dispEn=dispEn)
            tim.imshow(Binary_ori, sp=142, title="Mask_GT", dispEn=dispEn, cmap="gray")
            tim.imshow(Binary, sp=143, title="Mask_GT(preproc)", dispEn=dispEn, cmap="gray")

        ## To do : make blurring
        area_max = 0
        cnt_max = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w*h
            if area > W*H*0.7:  # Too large object
                continue
            if area < 20*20:  # Too small object
                continue
            if area > area_max:
                area_max = area
                cnt_max = cnt
            #image_clone = (tim.Denormalize()(image)).clone().to('cpu', dtype=torch.uint8)
            for i in range(iteration):
                patch = image[:, y: y + h, x: x + w]
                sf = 1.1 - 0.2*np.random.rand(1)  # 0.9 ~ 1.1
                patch_new = torch.nn.functional.interpolate(patch.unsqueeze(0), scale_factor=(sf)).squeeze(0)
                nh, nw = patch_new.shape[1:]
                #lt_x = np.random.randint(0, W-w+1)  # new ltop x
                #lt_y = np.random.randint(0, H-h+1)  # new ltop y 
                lt_x = np.random.randint(0, W-w+1)  # new ltop x
                lt_y = np.random.randint(int(y*0.7), H-h+1)  # upper limit is 70% of current y to avoid putting object into sky or building, new ltop y 

                image = self.overwrite_patch(image, Binary, (x,y,w,h), (lt_x, lt_y)) #  Add patch of Binary into image at new_lt_pts, where bbox is the location of Binary
                #try:
                #    image[:, lt_y: lt_y + nh, lt_x: lt_x + nw] = patch_new
                #except:
                #    image[:, lt_y: lt_y + h, lt_x: lt_x + w] = patch

                #image_clone[:, lt_y: lt_y + h, lt_x: lt_x + w] = patch
                #image_clone = self.draw_box(image_clone, [[x, y, x+w, y+h]])
                #image2 = cv2.rectangle(image_cv2, (x,y), (x+w,y+h), (255,0,0), 1)
                
        if dispEn or write_image:
            tim.imshow(tim.Denormalize()(image), sp=144, title="with_Clutter", dispEn=dispEn)

        if write_image and (fname is not None):
            tim.plt.savefig(fname, dpi=300)

        return image

    def draw_box(self, img, bbox):
        ''' 
            img = torch.rand((3,480,640)).to(dtype=torch.uint8)
            boxes = torch.Tensor([[10,10, 30, 40]])
            tim.imshow(draw_bounding_boxes(img, boxes))
        '''
        img = img.to(dtype=torch.uint8)
        boxes = torch.Tensor(bbox)
        return tim.draw_bounding_boxes(img, boxes)

    def add_clutter_to_cv_img(self, image, mask, iteration=1, dispEn=False, write_image=False, imgidx=0, fname=None):
        ## Refer : https://stackoverflow.com/questions/63928668/how-to-filter-contours-based-on-their-approximate-shape-in-a-binary-video-frames
        #find contours
        # mask.shape is [24, 1, 480, 640], which means [batch, ch, h, w]

        B = image.shape[0]
        Binary = mask*255  # [B, 1, 480, 640]
        Binary = np.array(Binary.cpu(), dtype=np.uint8)  # type casting for cv2.findContours()
        Binary = Binary.squeeze(1)  # [B, 480, 640]

        for i in range(B):
            if (i == imgidx) and (write_image == True):
                write_image_tmp = True
            else:
                write_image_tmp = False
            image[i] = self.add_clutter_to_single_cv_img(image[i], Binary[i], iteration, dispEn=dispEn, write_image=write_image_tmp, fname=fname)
        return image


def arg_parsing():
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--batchSize', type=int, default=4, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use, 0 means single process.')
    parser.add_argument('--segmentation', action='store_true', help='If you want to use Segmentation(DeepLabV3+), use this option')
    parser.add_argument('--seg_ckpt', type=str, default='./networks/MobileNet/pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth')
    return parser.parse_args()


if __name__ == '__main__':
    opt = arg_parsing()
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

#    seg_model = Segmentation(opt, device)  # load MobileNet for segmentation
#    seg_model = Segmentation(opt, device, [])  # load MobileNet for segmentation
#    seg_model = Segmentation(opt, device, [0])  # load MobileNet for segmentation
#
#    seg_model = Segmentation(opt, device, ["human"])  # rider, pedestrian(person)
#    seg_model = Segmentation(opt, device, ["vehicle"])  # car, truck, bus, caravan, trailer, train, motorcycle, bicyle
#    seg_model = Segmentation(opt, device, ["nature"])  # vegetation, terrain
#    seg_model = Segmentation(opt, device, ["sky"])  # load MobileNet for segmentation

    seg_model = Segmentation(opt, device, deatt_category_list=["human", "vehicle"])  # Default

#    seg_model = Segmentation(opt, device, ["human", "vehicle", "nature"])  # load MobileNet for segmentation
#    seg_model = Segmentation(opt, device, ["human", "vehicle", "sky"])  # load MobileNet for segmentation
#    seg_model = Segmentation(opt, device, ["human", "vehicle", "nature", "sky"])  # load MobileNet for segmentation
