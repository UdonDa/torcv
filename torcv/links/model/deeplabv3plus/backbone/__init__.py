# from torcv.links.model.resnet.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# from torcv.links.model.senet.senet import senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
# # from torcv.links.model.xception.xception import aligned_xception


# def build_backbone(backbone='resnet18', pretrained=True, num_classes=1000):
#     # ResNet
#     if backbone == 'resnet18':
#         return resnet18(pretrained=pretrained)
#     elif backbone == 'resnet34':
#         return resnet34(pretrained=pretrained)
#     elif backbone == 'resnet50':
#         return resnet50(pretrained=pretrained)
#     elif backbone == 'resnet101':
#         return resnet101(pretrained=pretrained)
#     elif backbone == 'resnet152':
#         return resnet152(pretrained=pretrained)
    
#     # SENet
#     elif backbone == 'senet154':
#         return senet154(num_classes=num_classes)

#     elif backbone == 'se_resnet50':
#         return se_resnet50(num_classes=num_classes)
#     elif backbone == 'se_resnet101':
#         return se_resnet101(num_classes=num_classes)
#     elif backbone == 'se_resnet152':
#             return se_resnet152(num_classes=num_classes)

#     elif backbone == 'se_resnext50_32x4d':
#             return se_resnext50_32x4d(num_classes=num_classes)
#     elif backbone == 'se_resnext101_32x4d':
#             return se_resnext101_32x4d(num_classes=num_classes)

#     # xception
#     elif backbone == 'xception':
#         return aligned_xception()
#     # mobilenet
#     elif backbone == 'mobilenet':
#             return se_resnet152(num_classes=num_classes)
    