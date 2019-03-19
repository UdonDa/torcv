import sys
sys.path.append('../../')


import argparse
import os
import numpy as np
from tqdm import tqdm

from torcv.links.model.deeplabv3plus.deeplabv3plus import deeplabV3plus
from torcv.links.model.deeplabv3plus.sync_batchnorm.replicate import patch_replication_callback

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torcv.utils.loss.segmentation_loss import SegmentationLosses

from torcv.utils.logger.saver import Saver
from torcv.utils.logger.summaries import TensorboardSummary
from torcv.utils.metrics.segmentation_evaluator import Evaluator
from torcv.solver.lr_scheduler.segmention_scheduler import LR_Scheduler

from torchvision.datasets import VOCSegmentation
from torchvision.utils import save_image

from sys import exit
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


def trainsforms_default():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def mask2png(mask):
    palette = get_palette(256)
    mask=Image.fromarray(mask.astype(np.uint8))
    mask.putpalette(palette)
    mask = np.asarray(mask)
    return mask


class Solver(object):

    def __init__(self, args):
        # TODO: augmentation.
        t_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.get_palette = get_palette(256)
        # Savar
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Tensorboard
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Dataloader
        kwargs = {'num_workers:': args.num_workers, 'pin_memory': True}
        # TODO: dataset download
        # self.train_loader, self.val_loader, self.test_loader, self.nclass = get_pascalvoc(args, base_dir=args.pascal_dataset_path ,transforms_train=t)
        t = trainsforms_default()
        self.train_loader = VOCSegmentation(root='./dataset/', year='2012',
                            image_set='train', download=False, transform=t, target_transform=t_val)
        
        self.val_loader = VOCSegmentation(root='./dataset/', year='2012',
                            image_set='val', download=False, transform=t, target_transform=t_val)

        # Dataset
        self.train_loader = DataLoader(self.train_loader, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_loader, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

        # Netwok
        self.model = deeplabV3plus(backbone=args.backbone,
                                output_stride=args.out_stride,
                                # num_classes=self.nclass,
                                num_classes=21,
                                sync_bn=args.sync_bn,
                                freeze_bn=args.freeze_bn).to(self.device)
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
        
        # Optimizer
        self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
        
        # Criterion
        # Wether to use class balanced weights.
        if args.use_balanced_weights:
            pass # TODO:
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=None).build_loss(mode=args.loss_type)

        # Cuda
        if args.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            patch_replication_callback(self.model)

        # Evaluator
        self.evaluator = Evaluator(21)
        # Lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError('no checkpoint found at: {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.model.module.load_state_dict(checkpoint['state_dict'])

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print('Loaded checkpoint: {} (epoch: {})'.format(args.resume, checkpoint['epoch']))
        
        if args.ft:
            args.start_epoch = 0

    
    # def training(self, epoch):
    #     train_loss = 0.0
    #     self.model.train()
    #     tbar = tqdm(self.train_loader)
    #     num_img_tr = len(self.train_loader)
    #     for i, (image, target) in enumerate(tbar):
    #         image, target = image.to(self.device), target.to(self.device)
    #         target = torch.squeeze(target)
    #         self.scheduler(self.optimizer, i, epoch, self.best_pred)
            
    #         self.optimizer.zero_grad()
    #         output = self.model(image)

    #         loss = self.criterion(output, target)
    #         loss.backward()
    #         self.optimizer.step()
    #         train_loss += loss.item()

    #         tbar.set_description('Train loss: {:.3f}'.format(train_loss / (i + 1)))
    #         self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

    #         # Show 10 * 3 inference results each epoch
    #         if i % (num_img_tr // 10) == 0:
    #             global_step = i + num_img_tr * epoch
    #             self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

    #     self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    #     print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
    #     print('Loss: %.3f' % train_loss)

    #     # if self.args.no_val:
    #     # save checkpoint every epoch
    #     is_best = False
    #     self.saver.save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': self.model.module.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'best_pred': self.best_pred,
    #     }, is_best)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (image, target) in enumerate(tbar):
            image, target = image.to(self.device), target.to(self.device)
            target = torch.squeeze(target)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        for i, (image, target) in enumerate(tbar):
            image, target = image.to(self.device), target.to(self.device)
            target_squeezed = torch.squeeze(target)
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target_squeezed)
            test_loss += loss.item()
            tbar.set_description('Test loss: {:.4f}'.format(test_loss / (i + 1)))
            pred_numpy = output.data.cpu().numpy()
            target_squeezed = target_squeezed.cpu().numpy()
            pred = np.argmax(pred_numpy, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_squeezed, pred)

            if i % 30 == 0:
                pred = pred[0, :, :]
                pred = mask2png(pred)

                pred = torch.from_numpy(pred)
                pred = torch.unsqueeze(pred, 0)

                image, target, pred = image[0,:,:,: ].cpu().float(), target[0,:,:,: ].cpu().float(), pred.float()
                target = torch.cat([target, target, target], dim=0)
                pred = torch.cat([pred, pred, pred], dim=0)
                concated = torch.cat([image.float(), target.float(), pred.float()], dim=2)
                save_image(concated, './tmp/{}_{}_mask.png'.format(epoch, i))


        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: {}, numImages: {}]'.format(epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: {:.4f}'.format(test_loss))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main(args):
    solver = Solver(args)
    print('Starting Epoch: {}'.format(solver.args.start_epoch))
    print('Total Epoch: {}'.format(solver.args.epochs))

    for epoch in range(solver.args.start_epoch, solver.args.epochs):
        solver.training(epoch)
        solver.validation(epoch)
    
    solver.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'drn_a_resnet50'],
                        help='backbone name (default: drn_d_54)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=True,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-parallel', action='store_true', default=True)
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # dataset path
    parser.add_argument('--pascal_dataset_path', type=str, default='./dataset/pascal/')
    parser.add_argument('--sbd_dataset_path', type=str, default='./dataset/sbd/')
    parser.add_argument('--cityscapes_dataset_path', type=str, default='./dataset/cityscapes/')
    parser.add_argument('--coco_dataset_path', type=str, default='./dataset/coco/')

    args = parser.parse_args()


    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    
    if args.batch_size is None:
        args.batch_size = int(4 * torch.cuda.device_count())
    
    if args.test_batch_size is None:
        args.test_batch_size = int(4 * torch.cuda.device_count())

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.07
        }

        args.lr = lrs[args.dataset.lower()] / (4 * torch.cuda.device_count()) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-{}'.format(str(args.backbone))
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print(args)

    main(args)