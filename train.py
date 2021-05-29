#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os.path as osp
import config as cfg
import torch.utils.data as data

from data_transform import DataTransform
from dataset import VOCDataset

def make_datapath_list(rootpath, txt_path):
    img_list  = list()
    anno_list = list()

    # path template
    imgpath_template  = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')
    for line in open(txt_path):
        file_id = line.strip()
        img_path  = (imgpath_template  % file_id)
        anno_path = (annopath_template % file_id)
        img_list.append(img_path)
        anno_list.append(anno_path)

    return img_list, anno_list

def train():
    #===== make_datapath_list =====
    train_img_list, train_anno_list = make_datapath_list(rootpath=cfg.root_path, txt_path=cfg.train_txt_path)
    val_img_list,   val_anno_list   = make_datapath_list(rootpath=cfg.root_path, txt_path=cfg.val_txt_path)
    print( train_img_list, train_anno_list, val_img_list, val_anno_list )
    #==================================

    #===== dataset =====
    # data transform
    transform = DataTransform(input_size=cfg.input_image_size, color_mean=cfg.color_mean, color_std=cfg.color_std)
    # create dateset for pytorch's dataloader
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=transform)
    val_dataset   = VOCDataset(val_img_list, val_anno_list, phase='val', transform= transform)
    # for Debug
    print(val_dataset.__getitem__(0)[0].shape)
    print(val_dataset.__getitem__(0)[1].shape)
    print(val_dataset.__getitem__(0))
    #=======================

    #===== set dataloader =====
    train_dataloader = data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader   = data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}

    # for Debug
    batch_iterotor = iter(dataloaders_dict['val'])
    imges, anno_class_imges = next(batch_iterotor)
    print(imges.size())
    print(anno_class_imges.size())
    #====================

if __name__ == '__main__':
    train()
