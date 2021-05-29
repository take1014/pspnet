#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import torch.nn as nn
import config as cfg

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = ConvBlock( in_channels, mid_channels,
                                kernel_size=1, stride=1, padding=0,
                                dilation=1, bias=False )

        self.cbr_2 = ConvBlock( mid_channels, mid_channels,
                                kernel_size=3, stride=stride, padding=dilation,
                                dilation=dilation, bias=False )

        self.cb_3 = Conv2DBatchNorm( mid_channels, out_channels,
                                     kernel_size=1, stride=1, padding=0,
                                     dilation=1, bias=False )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        # residual
        residual = x
        return self.relu(conv + residual)


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = ConvBlock( in_channels, mid_channels,
                                kernel_size=1, stride=1, padding=0,
                                dilation=1, bias=False )

        self.cbr_2 = ConvBlock( mid_channels, mid_channels,
                                kernel_size=3, stride=stride, padding=dilation,
                                dilation=dilation, bias=False )

        self.cb_3 = Conv2DBatchNorm( mid_channels, out_channels,
                                     kernel_size=1, stride=1, padding=0,
                                     dilation=1, bias=False )

        # residual
        self.cb_residual = Conv2DBatchNorm( in_channels, out_channels,
                                            kernel_size=1, stride=stride, padding=0,
                                            dilation=1, bias=False )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_1(x)
        conv = self.cbr_2(conv)
        conv = self.cb_3(conv)
        # residual
        residual = self.cb_residual(x)
        return self.relu(conv + residual)

class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d( in_channels, out_channels,
                               kernel_size, stride, padding,
                               dilation, bias=bias )
        self.batchnorm = nn.BacthNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return x

class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResudualBlockPSP, self):__init__()

        self.add_module( 'block1',
                         bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation) )
        for i in range(n_blocks-1):
            module_name = 'block' + str(i+2)
            self.add_module( module_name,
                             bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation) )

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(ConvBlock, self).__init__()
        self.conv2d_batchnorm = Conv2DBatchNorm( in_channels, out_channles,
                                                kernel_size, stride, padding,
                                                dilation, bias )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d_batchnorm(x)
        x = self.relu(x)
        return x

class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()
        self.convblock1 = ConvBlock( in_channels=3, out_channels=64,
                                     kernel_size=3, stride=2, padding=1,
                                     dilation=1, bias=False )

        self.convblock2 = ConvBlock( in_channels=64, out_channels=64,
                                     kernel_size=3, stride=1, padding=1,
                                     dilation=1, bias=False )

        self.convblock3 = ConvBlock( in_channels=64, out_channels=128,
                                     kernel_size=3, stride=1, padding=1,
                                     dilation=1, bias=False )

        self.maxpool = nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.maxpool(x)
        return x

class PSPNet(nn.Module):
    def __init__(self, n_classes):
        # override
        super(PSPNet, self).__init__()

        # set parameters
        block_config = [3, 4, 6, 3]
        img_size = cfg.input_image_size
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
                n_blocks=block_config[0], in_channels=128, mid_channels=64,
                out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(
                n_blocks=block_config[1], in_channels=256, mid_channels=128,
                out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
                n_blocks=block_config[2], in_channels=512, mid_channels=256,
                out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(
                n_blocks=block_config[2], in_channels=1024, mid_channels=512,
                out_channels=2048, stride=1, dilation=4)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6,3,2,1], height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(in_channels=1024, height=img_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        # AuxiliaryPSP
        output_aux = self.aux(x)

        # pyramid
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)
        return (output, output_aux)
