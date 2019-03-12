"""Ref:
 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
 https://arxiv.org/abs/1709.01507
 """


import torch
import torch.nn as nn
from torch.utils import model_zoo

import math
from collections import OrderedDict


__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):
    
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.se_module(x) + x


class Bottleneck(nn.Module):
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """ Bottleneck for SENet154"""
    expansion = 4

    def __init__(self, inplanes, filter, groups, reduction=16, stride=1, 
                downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, filter * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter * 2)
        
        self.conv2 = nn.Conv2d(filter * 2, filter * 4, kernel_size=3,
                    stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(filter * 4)

        self.conv3 = nn.Conv2d(filter * 4, filter * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter * 4)

        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(filter * 4, reduction=reduction)
        self.stride = stride
        self.downsample = downsample


class SEResNetBottleneck(Bottleneck):
    """ ResNet bottleneck with a se_module."""
    expansion = 4

    def __init__(self, inplanes, filter, groups, reduction=16,
                stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, filter, kernel_size=1,
                    bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(filter)

        self.conv2 = nn.Conv2d(filter, filter, kernel_size=3,
                    padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(filter)

        self.conv3 = nn.Conv2d(filter, filter * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(filter * 4, reduction=reduction)
        self.stride = stride
        self.downsample = downsample


class SEResNeXtBottleneck(Bottleneck):

    expansion = 4

    def __init__(self, inplanes, fiter, groups, reduction=16, stride=1,
                downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(filter * (base_width / 64)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1,
                        bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                        padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, filter * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter * 4)

        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(filter * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layer, groups, reduction=16, dropout_p=0.2,
                inplanes=128, input_3x3=True, downsample_kernel_size=3,
                downsample_padding=1, num_classes=1000):
        """SENet
        Args:
            block (nn.Module): Bottleneck class.
                - For SENet154: SEBottleneck
                - For SE-ResNet models: SEResNetBottleneck
                - For SE-ResNeXt models: SEResNeXtBottleneck
            layers (list of int): Number of groups for the 3x3 conv in each
                bottleneck block.
                - For SENet154: 64
                - For SE-ResNet models: 1
                - For SE-ResNeXt models: 32
            reduction (int) : Reduction ratio for Squeeze-and-Excitation modules.
                - For all models: 16 (cite paper)
            dropout_p (float or None): Drop probability for the Dropout layer.
                If `None`, it means you do not use the Dropout layer.
                - For SENet154: 0.2
                - For SE-ResNet models: None
                - For SE-ResNeXt models: None
            inplanes (int): Number of input channels for layer1.
                - For SENet154: 128
                - For SE-ResNet models: 64
                - For SE-ResNeXt models: 64
            input_3x3 (bool): if `True`, use three 3x3 convs instead of
                a single 7x7 conv in layer0.
                - For SENet154: True
                - For SE-ResNet models: False
                - For SE-ResNeXt models: False
            downsample_kernel_size (int): Kernel size for downsampling convs
                in layer2, layer3, and layer4.
                - For SENet154: 3
                - For SE-ResNet models: 1
                - For SE-ResNeXt models: 1
            downsample_padding (int): Padding for downsampling conv
                in layer2, layer3, and layer4.
                - For SENet154: 1
                - For SE-ResNet models: 0
                - For SE-ResNeXt models: 0
            num_classes (int): Number of outputs in `last_linear` layer.
                - For all models: 1000 (ImageNet classes)
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
            )
        
        self.layer1 = self._make_layer(
            block, planes=64, blocks=layer[0], groups=groups, reduction=reduction,
            downsample_kernel_size=1, downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block, planes=128, blocks=layer[1], stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block, planes=256, blocks=layer[2], stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block, planes=512, blocks=layer[3], stride=2, groups=groups, reduction=reduction,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_layer = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, groups, reduction=16, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample=None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=downsample_kernel_size, stride=stride,
                            padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                                downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)


    

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.last_layer(x)
        
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes shoudle be {}, but is {}'.format(settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3,8,36,3], groups=64, reduction=16,
                    dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3,4,6,3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0,
                    num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3,4,23,3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0,
                    num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3,8,36,3], groups=1, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0,
                    num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3,4,6,3], groups=32, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0,
                    num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3,4,23,3], groups=32, reduction=16,
                    dropout_p=None, inplanes=64, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0,
                    num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model
