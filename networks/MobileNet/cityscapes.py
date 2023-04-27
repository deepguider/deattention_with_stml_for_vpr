import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import numpy as np
import glob

from ipdb import set_trace as bp


class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                             self._get_target_suffix(self.mode, self.target_type))
                self.targets.append(os.path.join(target_dir, target_name))

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def get_classes(self):
        return self.classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)


class Denormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


class Cityscapes_CustomImgLoader(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color',
                                                     'simple_category', 'simple_id', 'mask_id'])
    classes = [
        CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0), 'static', 0, 0),
        CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0), 'static', 0, 0),
        CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81), 'static', 0, 1),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128), 'static',  0, 1),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232), 'static', 0, 1),
        CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160), 'static', 0, 1),
        CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140), 'static', 0, 1),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70), 'structure', 1, 1),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156), 'structure' , 1, 1),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153), 'structure', 1, 1),
        CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180), 'static', 0, 1),
        CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100), 'static', 0, 1),
        CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90), 'static',0 ,1),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153), 'structure', 1, 1),
        CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153), 'static', 0, 1),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30), 'structure', 1, 1),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0), 'structure', 1, 1),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35), 'nature', 2, 1),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152), 'nature', 2, 1),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180), 'nature', 2, 1),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60), 'moving', 3, 0),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0), 'moving', 3, 0),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142),'moving', 3, 0),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70),'moving',  3, 0),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100),'moving',  3, 0),
        CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90),'static',  0, 0),
        CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110),'static',  0, 0),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100),'moving',  3, 0),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230),'moving',  3, 0),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32),'moving',  3, 0),
        CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142),'static',  0, 0),
    ]


    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]

    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    ## used for segmentation masking by ccsmm, begin

    train_id_to_simple = [c.simple_id for c in classes if (c.train_id != -1 and c.train_id != 255)]

    train_id_to_mask = [np.float(c.mask_id) for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_category = [c.category for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_name = [c.name for c in classes if (c.train_id != -1 and c.train_id != 255)]

    train_id_to_category_id = [c.category_id for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_id = [c.id for c in classes if (c.train_id != -1 and c.train_id != 255)]

    train_id_to_simple = np.array(train_id_to_simple)
    train_id_to_mask = np.array(train_id_to_mask)
    train_id_to_id = np.array(train_id_to_id)
    train_id_to_category_id = np.array(train_id_to_category_id)


    ## used for segmentation masking by ccsmm, end
    
    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    # (R) means reject, (K) means keep, and (O) means optional 
    train_id_to_simple_details = {
            0 : 0,  # (R) Road to be rejected (TBR)
            1 : 0,  # (R) TBR, sidewalk
            2 : 1,  # (K) TBK, building
            3 : 1,  # (K) TBK, wall
            4 : 1,  # (K) TBK, fence
            5 : 1,  # (K) TBK, pole
            6 : 1,  # (K) TBK, traffic light
            7 : 1,  # (K) TBK, traffic sign
            8 : 2,  # (O) Nature, vegetation (Keeping or rejecting)
            9 : 2,  # (O) Nature, terrain
           10 : 2,  # (O) Nature, sky
           11 : 3,  # (R) moving object to be rejected (MOTBR), person
           12 : 3,  # (R) MOTBR, rider
           13 : 3,  # (R) MOTBR, car 
           14 : 3,  # (R) MOTBR, truck
           15 : 3,  # (R) MOTBR, bus  
           16 : 3,  # (R) MOTBR, train
           17 : 3,  # (R) MOTBR, motorcycle
           18 : 3,  # (R) MOTBR, bicycle 
           19 : 4,  # (O) Void
           255: 4   # (O) Void
           }

    train_id_to_mask_details = {
            0 : 0,  # (R) Road to be rejected (TBR)
            1 : 0,  # (R) TBR, sidewalk
            2 : 1,  # (K) TBK, building
            3 : 1,  # (K)  TBK, wall
            4 : 1,  # (K)  TBK, fence
            5 : 1,  # (K)  TBK, pole
            6 : 1,  # (K)  TBK, traffic light
            7 : 1,  # (K)  TBK, traffic sign
            8 : 1,  # (O) Nature, vegetation (Keeping or rejecting)
            9 : 1,  # (O) Nature, terrain
           10 : 1,  # (O) Nature, sky
           11 : 0,  # (R) moving object to be rejected (MOTBR), person
           12 : 0,  # (R) MOTBR, rider
           13 : 0,  # (R) MOTBR, car 
           14 : 0,  # (R) MOTBR, truck
           15 : 0,  # (R) MOTBR, bus  
           16 : 0,  # (R) MOTBR, train
           17 : 0,  # (R) MOTBR, motorcycle
           18 : 0,  # (R) MOTBR, bicycle 
           19 : 1,  # (O) Void
           255: 1   # (O) Void
           }

    def __init__(self,
                 root='img',
                 ext='jpg',
                 transform=None):

        self.images = glob.glob(os.path.join(root, '*.{}'.format(ext)))
        self.transform = transform
        self.concern_category = ["nature", "sky", "human", "vehicle", "flat", "construction"]  # flat : road, construction: building

    def get_classes(self):
        return self.classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img, _ = self.transform(img, img)

        return img

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def set_train_id_to_mask_by_category(self, category_str, value):
        for idx, category in enumerate(self.train_id_to_category):
            if category == category_str:
                self.train_id_to_mask[idx] = value

    def keep_train_id_to_mask_by_category(self, category_str):
        self.set_train_id_to_mask_by_category(category_str, 1)

    def reject_train_id_to_mask_by_category(self, category_str):
        self.set_train_id_to_mask_by_category(category_str, 0)

    def keep_concern_category(self):  # Reset all concern categories to one
        for category in self.concern_category:
            self.keep_train_id_to_mask_by_category(category) 

    def reject_concern_category(self): # Reset all concern categories to zero
        for category in self.concern_category:
            self.reject_train_id_to_mask_by_category(category) 

    def get_concern_category(self):
        return self.concern_category

    def get_train_id_to_mask(self):
        return self.train_id_to_mask

    def get_train_id_to_simple(self):
        return self.train_id_to_simple

    def get_train_id_to_category(self):
        return self.train_id_to_category

    def get_train_id_to_name(self):
        return self.train_id_to_name

    def get_train_id_to_color(self):
        return self.train_id_to_color

    @classmethod
    def decode_id(cls, target):
        return cls.train_id_to_id[target]

    @classmethod
    def decode_category_id(cls, target):
        return cls.train_id_to_category_id[target]

    @classmethod
    def decode_simple(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_simple[target]

    @classmethod
    def decode_mask(cls, target):  # semantic guidacne for de-attention module
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_mask[target]

    @classmethod
    def decode_simple_old(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        simple_ids = np.zeros_like(target)
        mask_ids = np.zeros_like(target)
        H, W = target.shape
        for h in range(H):
            for w in range(W):
                try:
                    simple_ids[h,w] = cls.train_id_to_simple_details[target[h,w]]
                    if simple_ids[h,w] in [0,1,2,4]:  # Keep : road, building, nature, mask : moving object including person
                        #mask_ids[h,w] = 255
                        mask_ids[h,w] = 1.0
                except:
                    bp()
        return simple_ids.astype(np.uint8), mask_ids.astype(np.uint8)

if __name__ == "__main__":
    loader = Cityscapes_CustomImgLoader()

    concern_category = loader.get_concern_category() #["nature", "sky", "human", "vehicle"]
    print(loader.get_train_id_to_mask())
    print(loader.get_train_id_to_category())

    print("Concern category : ", concern_category)

    print("\nKeep(1) concern category")
    loader.keep_concern_category()
    print(loader.train_id_to_mask)

    for category in concern_category:
        print("\nReject(0) {} category".format(category))
        loader.reject_train_id_to_mask_by_category(category)
        print(loader.get_train_id_to_mask())
