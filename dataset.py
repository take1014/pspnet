#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import torch.utils.data as data
from PIL import Image

# dataset for using dataloader(pytorch)
class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)

        # pre proccesing
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        return img, anno_class_img
