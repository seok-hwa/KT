import torch
from torch.utils import data
import os.path as osp
import numpy as np
import random
import cv2
from PIL import Image
import os


class CamvidTrainSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(360, 360), scale=True, mirror=True, ignore_label=-1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.is_scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if max_iters:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.img_ids = self.img_ids[:max_iters]

        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            # name = osp.splitext(osp.basename(label_path))[0]
            name = osp.splitext(osp.basename(label_path))[0] + '_L.png'
            img_file = osp.join(self.root, image_path)
            # label_file = osp.join(self.root, label_path)
            label_file = osp.join(self.root, 'trainannot', name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print('{} images are loaded!'.format(len(self.img_ids)))

        self.num_class = 11

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 15) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label


    def id2trainId(self, label):
        label_copy = label.copy().astype('int32')
        label_copy[label == 11] = -1
        return label_copy


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)

        size = image.shape
        name = datafiles["name"]
        if self.is_scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        img_h, img_w = label.shape

        
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(), label.copy(), name



class CamvidValSet(data.Dataset):
    def __init__(self, root, list_path, ignore_label=-1):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]

        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            # name = osp.splitext(osp.basename(label_path))[0]
            # img_file = osp.join(self.root, image_path)
            # label_file = osp.join(self.root, label_path)
            name = osp.splitext(osp.basename(label_path))[0] + '_L.png'
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, 'valannot', name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print('{} images are loaded!'.format(len(self.img_ids)))

        self.num_class = 11

    def __len__(self):
        return len(self.files)


    def id2trainId(self, label):
        label_copy = label.copy().astype('int32')
        label_copy[label == 11] = -1
        return label_copy


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)

        label = self.id2trainId(label)
        
        size = image.shape
        name = datafiles["name"]
        
        image = np.asarray(image, np.float32)
        image = image - np.array([104.00698793, 116.66876762, 122.67891434])
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        return image.copy(), label.copy(), name