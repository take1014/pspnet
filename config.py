#!/usr/bin/env python3
#-*- coding:utf-8 -*-

root_path      = "/home/take/fun/dataset/VOC2012/"
train_txt_path = root_path +'ImageSets/Segmentation/train.txt'
val_txt_path   = root_path +'ImageSets/Segmentation/val.txt'
input_image_size = 475
batch_size = 8
classes = 21
color_mean = (0.485, 0.456, 0.406)
color_std  = (0.229, 0.224, 0.225)
