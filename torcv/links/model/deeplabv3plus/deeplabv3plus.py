# Ref: https://github.com/jfzhang95/pytorch-deeplab-xception

import torch
import torch.nn as nn
import torch.nn.functional as F
from torcv.functions.SynchronizedBatchNorm import SynchronizedBatchNorm2d

from torcv.links.model.deeplabv3plus.aspp import build_aspp
from torcv.links.model.deeplabv3plus.decoder import build_decoder

from torcv.links.model.deeplabv3plus.backbone.mobilenet import mobilenetV2
from torcv.links.model.deeplabv3plus.backbone.drn import drn_a_resnet50, drn_c_26, drn_c_42, drn_c_58, drn_d_22, drn_d_24, drn_d_38, drn_d_40, drn_d_54, drn_d_105
from torcv.links.model.deeplabv3plus.backbone.xception import aligned_xception


def build_backbone(backbone='drn_d_54', pretrained=True, num_classes=1000):
    print('Pretrained model: {}'.format(pretrained))
    # xception
    if backbone == 'xception':
        return aligned_xception(pretrained=pretrained)
    # mobilenet
    elif backbone == 'mobilenet':
        return mobilenetV2(pretrained=pretrained)
    # drn
    elif backbone == 'drn_a_resnet50':
        return drn_a_resnet50(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_c_26':
        return drn_c_26(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_c_42':
        return drn_c_42(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_c_58':
        return drn_c_58(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_d_22':
        return drn_d_22(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_d_24':
        return drn_d_24(pretrained=pretrained, BatchNorm=nn.BatchNorm2d)
    elif backbone == 'drn_d_38':
        return drn_d_38(pretrained=pretrained, BatchNorm=SynchronizedBatchNorm2d)
    elif backbone == 'drn_d_40':
        return drn_d_40(pretrained=pretrained, BatchNorm=SynchronizedBatchNorm2d)
    elif backbone == 'drn_d_54':
        return drn_d_54(pretrained=pretrained, BatchNorm=SynchronizedBatchNorm2d)
    elif backbone == 'drn_d_105':
        return drn_d_105(pretrained=pretrained, BatchNorm=SynchronizedBatchNorm2d)


class DeepLabV3plus(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabV3plus, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone=backbone, pretrained=True)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

def deeplabV3plus(backbone='drn_d_54', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False):
    model = DeepLabV3plus(backbone=backbone,
                            output_stride=output_stride,
                            num_classes=num_classes,
                            sync_bn=sync_bn,
                            freeze_bn=freeze_bn)
    return model