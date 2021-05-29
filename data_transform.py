#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import utils.data_augumentation as da

class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
                'train': da.Compose([
                    da.Scale(scale=[0.5, 1.5]),
                    da.RandomRotation(angle=[-10,10]),
                    da.RandomMirror(),
                    da.Resize(input_size),
                    da.Normalize_Tensor(color_mean, color_std)
                    ]),
                'val': da.Compose([
                    da.Resize(input_size),
                    da.Normalize_Tensor(color_mean, color_std)
                    ])
                }
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)
