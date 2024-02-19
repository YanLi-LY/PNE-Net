import os
import glob
import h5py
import random
from PIL import Image
from matplotlib import pyplot as plt

import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

from Utils.metrics import *
from Utils.option import *



class syn_Dataset(data.Dataset):
    def __init__(self, path, mode='train', size=opt.crop_size):
        super(syn_Dataset,self).__init__()
        self.size = size
        self.mode = mode
        self.format = format

        self.hazy_imgs_dir = os.path.join(path, 'hazy')
        self.hazy_imgs_list = os.listdir(self.hazy_imgs_dir)
        self.hazy_imgs = [os.path.join(self.hazy_imgs_dir, img) for img in self.hazy_imgs_list]
        self.gt_dir = os.path.join(path, 'clear')

        self.aux_dir = ''

        self.length = len(self.hazy_imgs_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            hazy_img_1 = Image.open(self.hazy_imgs[index])
            hazy_img_1_name = os.path.basename(self.hazy_imgs[index])
            name, extension = os.path.splitext(hazy_img_1_name)
            if '_' in hazy_img_1_name:
                id = hazy_img_1_name.split('_')[0]
            else:
                id, _ = os.path.splitext(hazy_img_1_name)
            gt_img_name = id + '.png'
            gt_img = Image.open(os.path.join(self.gt_dir, gt_img_name))
            gt_img = tfs.CenterCrop(hazy_img_1.size[::-1])(gt_img)

            hazy_img_path_list = glob.glob(os.path.join(self.hazy_imgs_dir, id + '_*' + '.png'))
            for i in range(len(hazy_img_path_list)):
                hazy_img_path_list[i] = os.path.basename(hazy_img_path_list[i])
            hazy_img_path_list.remove(hazy_img_1_name)
            hazy_img_2_name = random.choice(hazy_img_path_list)
            hazy_img_2 = Image.open(os.path.join(self.hazy_imgs_dir, hazy_img_2_name))
            hazy_img_2 = tfs.CenterCrop(hazy_img_1.size[::-1])(hazy_img_2)

            aux_negs = Image.open(os.path.join(self.aux_dir, hazy_img_1_name))
            aux_negs = tfs.CenterCrop(hazy_img_1.size[::-1])(aux_negs)

            if not isinstance(self.size, str):
                i, j, h, w = tfs.RandomCrop.get_params(hazy_img_1, output_size=(self.size, self.size))
                hazy_img_1 = FF.crop(hazy_img_1, i, j, h, w)
                hazy_img_2 = FF.crop(hazy_img_2, i, j, h, w)
                gt_img = FF.crop(gt_img, i, j, h, w)
                aux_negs = FF.crop(aux_negs, i, j, h, w)

            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            hazy_img_1 = self.augData_haze(hazy_img_1.convert("RGB"), rand_hor, rand_rot)
            hazy_img_2 = self.augData_haze(hazy_img_2.convert("RGB"), rand_hor, rand_rot)
            gt_img = self.augData_clear(gt_img.convert("RGB"), rand_hor, rand_rot)
            aux_negs = self.augData_clear(aux_negs.convert("RGB"), rand_hor, rand_rot)
            return hazy_img_1, hazy_img_2, gt_img, aux_negs, id, extension


    def augData_haze(self, haze, rand_hor, rand_rot):
        if self.mode == 'train':
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            if rand_rot:
                haze = FF.rotate(haze, 90*rand_rot)
        haze = tfs.ToTensor()(haze)
        haze = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(haze)
        return haze

    def augData_clear(self, clear, rand_hor, rand_rot):
        if self.mode == 'train':
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                clear = FF.rotate(clear, 90*rand_rot)
        clear = tfs.ToTensor()(clear)
        return clear

    def __len__(self):
        return self.length



class test_Dataset(data.Dataset):
    def __init__(self, path, mode='test', size=opt.crop_size):
        super(test_Dataset, self).__init__()
        self.size = size
        self.mode = mode
        self.format = format

        self.hazy_imgs_dir = os.path.join(path, 'hazy')
        self.hazy_imgs_list = os.listdir(self.hazy_imgs_dir)
        self.hazy_imgs = [os.path.join(self.hazy_imgs_dir, img) for img in self.hazy_imgs_list]
        self.gt_dir = os.path.join(path, 'clear')

        self.length = len(self.hazy_imgs_list)

    def __getitem__(self, index):
        if self.mode == 'test':
            hazy_img = Image.open(self.hazy_imgs[index])
            hazy_img_name = self.hazy_imgs[index].split('/')[-1]
            id = hazy_img_name.split('_')[0]
            gt_img_name = id + '.png'
            gt_img = Image.open(os.path.join(self.gt_dir, gt_img_name))
            gt_img = tfs.CenterCrop(hazy_img.size[::-1])(gt_img)

            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            hazy_img = self.augData_haze(hazy_img.convert("RGB"), rand_hor, rand_rot)
            gt_img = self.augData_clear(gt_img.convert("RGB"), rand_hor, rand_rot)
            return hazy_img, gt_img, hazy_img_name

    def augData_haze(self, haze, rand_hor, rand_rot):
        if self.mode == 'train':
            haze = tfs.RandomHorizontalFlip(rand_hor)(haze)
            if rand_rot:
                haze = FF.rotate(haze, 90*rand_rot)
        haze = tfs.ToTensor()(haze)
        haze = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(haze)
        return haze

    def augData_clear(self, clear, rand_hor, rand_rot):
        if self.mode == 'train':
            clear = tfs.RandomHorizontalFlip(rand_hor)(clear)
            if rand_rot:
                clear = FF.rotate(clear, 90*rand_rot)
        clear = tfs.ToTensor()(clear)
        return clear

    def __len__(self):
        return self.length




