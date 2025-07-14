import argparse
import time
import datetime
import os
import shutil
import sys
import numpy as np
import random
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from losses import *
from models.model_zoo import get_segmentation_model
from models.tailoring import Tailoring

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from utils.flops import cal_multi_adds, cal_param_size

from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3', help='model name')
    parser.add_argument('--student-model', type=str, default='deeplabv3', help='model name')
    parser.add_argument('--student-backbone', type=str, default='resnet18', help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101', help='backbone name')
    parser.add_argument('--dataset', type=str, default='citys', help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/cityscapes/', help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N', help='ignore label')

    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False, help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')

    parser.add_argument("--kd-temperature", type=float, default=4.0, help="logits KD temperature")
    parser.add_argument("--feat-temperature", type=float, default=4.0, help="logits KD temperature")
    parser.add_argument("--lambda-task", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-mask", type=float, default=0., help="lambda_mask")
    parser.add_argument("--lambda-adv", type=float, default=0., help="lambda adversarial loss")
    parser.add_argument("--lambda-d", type=float, default=0., help="lambda discriminator loss")
    parser.add_argument("--lambda-custom", type=float, default=0., help="lambda reconstruction feature loss")
    parser.add_argument("--lambda-channel", type=float, default=0., help="lambda channel feature loss")
    parser.add_argument("--lambda-contrastive", type=float, default=0., help="lambda contrastive feature loss")
    parser.add_argument("--tuning-point", type=int, default=10000, help="change point of knowledge tailoring")

    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models', help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10, help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/', help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=40, help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800, help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800, help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None', help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None', help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None', help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None', help='pretrained seg model')
    parser.add_argument('--flag', type=int, default='None', help='environment flag')
    parser.add_argument('--save-img', type=bool, default=False, help='save img')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1, help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False, help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data, list_path='./dataset/list/cityscapes/train.lst',
                                          max_iters=args.max_iterations * args.batch_size,
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, list_path='./dataset/list/cityscapes/val.lst',
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt',
                                            max_iters=args.max_iterations * args.batch_size,
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters= args.max_iterations * args.batch_size,
                                        ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt',
                                           max_iters=args.max_iterations * args.batch_size,
                                           ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True,
                                           mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt',
                                                  max_iters=args.max_iterations * args.batch_size,
                                                  ignore_label=args.ignore_label,
                                                  crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler,
                                            num_workers=args.workers, pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset, batch_sampler=val_batch_sampler,
                                          num_workers=args.workers, pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model, backbone=args.teacher_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base='None', pretrained=args.teacher_pretrained, aux=True,
                                              norm_layer=nn.BatchNorm2d, num_class=train_dataset.num_class).to(
            self.args.local_rank)

        self.s_model = get_segmentation_model(model=args.student_model, backbone=args.student_backbone,
                                              local_rank=args.local_rank,
                                              pretrained_base=args.student_pretrained_base, pretrained='None',
                                              aux=args.aux,
                                              norm_layer=BatchNorm2d, num_class=train_dataset.num_class).to(self.device)
        self.tc_tail = Tailoring(128, 64).to(self.device)
        self.st_tail = Tailoring(256, 128).to(self.device)

        self.conv = nn.Conv2d(128, 256, kernel_size=1, bias=False).cuda()

        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        with torch.no_grad():
            logger.info('ST Params: %.2fM FLOPs: %.2fG'
                        % (cal_param_size(self.s_model) / 1e6, cal_multi_adds(self.s_model, (1, 3, 512, 1024)) / 1e9))
            logger.info('TC Params: %.2fM FLOPs: %.2fG'
                        % (cal_param_size(self.t_model) / 1e6, cal_multi_adds(self.t_model, (1, 3, 512, 1024)) / 1e9))

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        params_list = nn.ModuleList([])
        params_list.append(self.s_model)

        self.optimizer = torch.optim.SGD(params_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # create criterion
        x = torch.randn(1, 3, 512, 1024).cuda()
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)
        self.num_class = train_dataset.num_class

        self.st_conv = nn.Conv2d(s_channels, t_channels, kernel_size=1, bias=False).cuda()

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_FT = CriterionKT(s_channels, s_channels, norm_type='channel', divergence='kl',
                                                temperature=args.kd_temperature).to(self.device)
        self.criterion_contra = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.criterion_LT = CriterionKT(self.num_class, self.num_class, norm_type='channel', divergence='kl',
                                              temperature=args.kd_temperature).to(self.device)


        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, device_ids=[args.local_rank], output_device=args.local_rank)

        # evaluation metric
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr * ((1 - float(iter) / max_iter) ** (power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.s_model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):

            task_loss = torch.tensor(0.).cuda()
            output_loss = torch.tensor(0.).cuda()
            LT_loss = torch.tensor(0.).cuda()
            FT_loss = torch.tensor(0.).cuda()
            contrastive_loss = torch.tensor(0.).cuda()

            if iteration % 800 == 0:
                logger.info('========== epoch {:d} =========='.format(iteration // 800 + 1))
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.long().to(self.device)

            with torch.no_grad():
                t_outputs = self.t_model(images)
            s_outputs = self.s_model(images)

            # Create class-specific mask
            target_mask = targets.clone()
            target_mask[target_mask == -1] = self.num_class
            target_shape = target_mask.shape
            one_hot = torch.zeros((target_shape[0], self.num_class + 1) + target_shape[1:], device='cuda',
                                  dtype=torch.long)
            class_mask = one_hot.scatter_(1, target_mask.unsqueeze(1), 1.0).permute(1, 0, 2, 3).contiguous()
            class_mask = class_mask[:self.num_class].permute(1, 0, 2, 3).contiguous()
            non_class_mask = -(class_mask - 1)

            # Logit Tailoring
            tc_output = F.interpolate(t_outputs[0], args.crop_size, mode='nearest')
            st_output = F.interpolate(s_outputs[0], args.crop_size, mode='nearest')

            tc_class_masked_output = tc_output * class_mask
            tc_non_class_masked_output = tc_output * non_class_mask
            st_class_masked_output = st_output * class_mask
            st_non_class_masked_output = st_output * non_class_mask

            for i in range(0, tc_class_masked_output.shape[0]):
                for j in range(0, tc_class_masked_output.shape[1]):
                    tc_mean_value = tc_class_masked_output[i][j][tc_class_masked_output[i][j] > 0].mean()
                    st_mean_value = st_class_masked_output[i][j][st_class_masked_output[i][j] > 0].mean()
                    non_tc_mean_value = tc_non_class_masked_output[i][j].mean()
                    non_st_mean_value = st_non_class_masked_output[i][j].mean()
                    tc_class_masked_output[i][j][tc_class_masked_output[i][j] < tc_mean_value] = tc_mean_value
                    st_class_masked_output[i][j][st_class_masked_output[i][j] < st_mean_value] = st_mean_value
                    tc_non_class_masked_output[i][j][tc_non_class_masked_output[i][j] > non_tc_mean_value] = non_tc_mean_value
                    st_non_class_masked_output[i][j][st_non_class_masked_output[i][j] > non_st_mean_value] = non_st_mean_value
            tc_class_masked_output = tc_class_masked_output * class_mask
            st_class_masked_output = st_class_masked_output * class_mask
            tc_non_class_masked_output = tc_non_class_masked_output * non_class_mask
            st_non_class_masked_output = st_non_class_masked_output * non_class_mask
            tc_masked_output = tc_class_masked_output + tc_non_class_masked_output
            st_masked_output = st_class_masked_output + st_non_class_masked_output

            tc_features = t_outputs[-1]
            st_features = s_outputs[-1]

            high_tc_feature = []
            low_tc_feature = []
            high_st_feature = []
            low_st_feature = []

            t_n, t_c, t_h, t_w = tc_features.shape
            s_n, s_c, s_h, s_w = st_features.shape

            # Feature Tailoring
            if iteration >= self.args.tuning_point:
                for (tc_feature, st_feature) in zip(tc_features, st_features):
                    tmp_tc_feature = tc_feature
                    tmp_st_feature = st_feature
                    tc_feat_channel = F.normalize(tmp_tc_feature.reshape(t_c, -1), dim=1)
                    st_feat_channel = F.normalize(tmp_st_feature.reshape(s_c, -1), dim=1)

                    channel_corr_map = st_feat_channel.matmul(torch.transpose(tc_feat_channel, 0, 1).contiguous())
                    high_weighted_sum_tc_feature = channel_corr_map.matmul(tc_feature.reshape(t_c, -1)).reshape(s_c, t_h, t_w)
                    low_weighted_sum_tc_feature = (1-channel_corr_map).matmul(tc_feature.reshape(t_c, -1)).reshape(s_c, t_h, t_w)

                    high_tc_feature.append(high_weighted_sum_tc_feature.unsqueeze(dim=0))
                    low_tc_feature.append(low_weighted_sum_tc_feature.unsqueeze(dim=0))

                custom_high_tc_features = self.tc_tail(torch.cat(high_tc_feature, dim=0))
                custom_low_tc_features = self.tc_tail(torch.cat(low_tc_feature, dim=0))

            else:
                for (tc_feature, st_feature) in zip(tc_features, st_features):
                    tmp_tc_feature = tc_feature
                    tmp_st_feature = st_feature
                    tc_feat_channel = F.normalize(tmp_tc_feature.reshape(t_c, -1), dim=1)
                    st_feat_channel = F.normalize(tmp_st_feature.reshape(s_c, -1), dim=1)

                    channel_corr_map = tc_feat_channel.matmul(torch.transpose(st_feat_channel, 0, 1).contiguous())
                    high_weighted_sum_st_feature = channel_corr_map.matmul(st_feature.reshape(s_c, -1)).reshape(t_c, t_h, t_w)
                    low_weighted_sum_st_feature = (1-channel_corr_map).matmul(st_feature.reshape(s_c, -1)).reshape(t_c, t_h, t_w)

                    high_st_feature.append(high_weighted_sum_st_feature.unsqueeze(dim=0))
                    low_st_feature.append(low_weighted_sum_st_feature.unsqueeze(dim=0))

                custom_high_st_features = self.st_tail(torch.cat(high_st_feature, dim=0))
                custom_low_st_features = self.st_tail(torch.cat(low_st_feature, dim=0))

            if self.args.aux:
                task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)

            if self.args.lambda_kd != 0.:
                if self.args.aux:
                    output_loss = self.args.lambda_kd * (
                            self.criterion_kd(s_outputs[0], t_outputs[0])
                            + 0.4 * self.args.lambda_kd * self.criterion_kd(s_outputs[1], t_outputs[1]))
                else:
                    output_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], t_outputs[0])

            if self.args.lambda_mask != 0:
                LT_loss = self.args.lambda_mask * (
                        self.criterion_LT(st_output, tc_masked_output) +
                        self.criterion_LT(st_class_masked_output, tc_class_masked_output) +
                        self.criterion_LT(st_non_class_masked_output, tc_non_class_masked_output))

            if self.args.lambda_custom != 0:
                if iteration < self.args.tuning_point:
                    FT_loss = self.args.lambda_custom * \
                                  (self.criterion_FT(custom_high_st_features, tc_features))
                else:
                    FT_loss = self.args.lambda_custom * \
                                  (self.criterion_FT(st_features, custom_high_tc_features))

            if self.args.lambda_contrastive != 0:
                if iteration < self.args.tuning_point:
                    contrastive_loss = self.args.lambda_contrastive * \
                       (self.criterion_contra(self.st_conv(st_features), custom_high_st_features, custom_low_st_features))
                else:
                    contrastive_loss = self.args.lambda_contrastive * \
                       (self.criterion_contra(st_features, custom_high_tc_features, custom_low_tc_features))

            losses = task_loss + output_loss + LT_loss + FT_loss + contrastive_loss

            lr = self.adjust_lr(base_lr=args.lr, iter=iteration - 1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            output_loss_reduced = self.reduce_mean_tensor(output_loss)
            LT_loss_reduced = self.reduce_mean_tensor(LT_loss)
            FT_loss_reduced = self.reduce_mean_tensor(FT_loss)
            contrastive_loss_reduced = self.reduce_mean_tensor(contrastive_loss)

            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                if iteration % 800 == 0:
                    logger.info(
                        "Iters: 800/800 || Lr: {:.6f} || Task Loss: {:.4f} || Output Loss: {:.4f} || LT Loss: {:.4f} || FT Loss: {:.4f} || Contrastive Loss: {:.4f} "
                        "|| Cost Time: {} || Estimated Time: {}".format(
                            self.optimizer.param_groups[0]['lr'],
                            task_loss_reduced.item(),
                            output_loss_reduced.item(),
                            LT_loss_reduced.item(),
                            FT_loss_reduced.item(),
                            contrastive_loss_reduced.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string))
                else:
                    logger.info(
                        "Iters: {:d}/800 || Lr: {:.6f} || Task Loss: {:.4f} || Output Loss: {:.4f} || LT Loss: {:.4f} || FT Loss: {:.4f} || Contrastive Loss: {:.4f} "
                        "|| Cost Time: {} || Estimated Time: {}".format(
                            iteration % 800,
                            self.optimizer.param_groups[0]['lr'],
                            task_loss_reduced.item(),
                            output_loss_reduced.item(),
                            LT_loss_reduced.item(),
                            FT_loss_reduced.item(),
                            contrastive_loss_reduced.item(),
                            str(datetime.timedelta(seconds=int(time.time() - start_time))),
                            eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            if i % 5 == 0:
                logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).cuda().to(args.local_rank)
            sum_total_label = torch.tensor(self.metric.total_label).cuda().to(args.local_rank)
            sum_total_inter = torch.tensor(self.metric.total_inter).cuda().to(args.local_rank)
            sum_total_union = torch.tensor(self.metric.total_union).cuda().to(args.local_rank)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label)
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        logger.info("Best mIoU: {:.3f}".format(self.best_pred * 100))
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""
    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'KT_{}_{}_{}_{}.pth'.format(args.flag, args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module

    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'KT_{}_{}_{}_{}_best_model.pth'.format(args.flag, args.student_model,
                                                                     args.student_backbone,
                                                                     args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger(" ", args.log_dir, get_rank(), filename='KT_{}_{}_{}_{}_log.txt'.format(
        args.flag, args.student_model, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
