import sys
from tqdm.auto import tqdm
from test_utils import *

import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
#import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import math
import torch_utils
from torch.nn.utils import clip_grad_norm_

import warnings

warnings.filterwarnings("ignore")
from torch.nn import DataParallel
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchTrainMultiScale,DetBenchTrainMultiScaleV2
from effdet.efficientdet import HeadNet

def get_net(type_net='tf_efficientdet_d7',checkpoint_name='tf_efficientdet_d7_53-6d1d7a95.pth',resume=None):
    config = get_efficientdet_config(type_net)
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 1024 #
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    net = DataParallel(net)
    if resume:
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['model_state_dict'])
    return DetBenchTrain(net, config)

def get_net_multiscle(type_net='tf_efficientdet_d7',checkpoint_name='tf_efficientdet_d7_53-6d1d7a95.pth',resume=None,multiScale=[1.0,1.1,0.8]):
    config = get_efficientdet_config(type_net)
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 1024 #
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    net = DataParallel(net)
    if resume:
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['model_state_dict'])
    return DetBenchTrainMultiScaleV2(net, config,multiscale=multiScale) #DetBenchTrainMultiScaleV2


def collate_fn(batch):
    return tuple(zip(*batch))

def get_k_fols(config):
    marking = pd.read_csv(config.csv)

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    spike = pd.read_csv(config.csv.replace("train","train_spike_kaggle"))
    return marking,df_folds,spike

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from apex import amp      


class Fitter:
    
    def __init__(self, model, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 0
        self.grad_clip=0.1
        self.model = model
        self.ema = torch_utils.ModelEMA(model.model)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        self.optimizer = config.optimizer(pg0, lr=config.lr, momentum=0.937, nesterov=True)
#         self.optimizer = config.optimizer(pg0, lr=config.lr)
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.937, nesterov=True)
#         lf = lambda x: (((1 + math.cos(x * math.pi / 25)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
#         self.scheduler = config.scheduler(self.optimizer, lr_lambda=lf)
#         self.scheduler.last_epoch = -1
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        opt_level = 'O1'
        for m in self.model.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                print(m)
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg > self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                print(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                    os.remove(path)

#             if self.config.validation_scheduler:
#                 self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, validation_loader):
        self.model.eval()
        self.model.cuda()
        summary_loss = AverageMeter()
        t = time.time()
        all_predictions = []
        for images, targets, image_ids in tqdm(validation_loader, total=len(validation_loader)):
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.cuda().float()
                det = self.model.predict(images, torch.tensor([1]*images.shape[0]).float().cuda(),self.ema)
                for i in range(images.shape[0]):
                    boxes = det[i].detach().cpu().numpy()[:,:4]    
                    scores = det[i].detach().cpu().numpy()[:,4]
                    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                    all_predictions.append({
                        'pred_boxes': (boxes).clip(min=0, max=1023).astype(int),
                        'scores': scores,
                        'gt_boxes': (targets[i]['boxes'].cpu().numpy()).clip(min=0, max=1023).astype(int),
                        'image_id': image_ids[i],
                    })
        best_final_score, best_score_threshold = 0, 0
        for score_threshold in tqdm(np.arange(0.3, 0.7, 0.05), total=np.arange(0.3, 0.7, 0.05).shape[0], desc="OOF"):
            final_score = calculate_final_score(all_predictions, score_threshold)
            if final_score > best_final_score:
                best_final_score = final_score
                best_score_threshold = score_threshold
        print(          f'threshold: {best_score_threshold:.5f}, ' + \
                        f'score: {best_final_score:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}')
        summary_loss.update(best_final_score, 1)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        self.model.cuda()
        summary_loss = AverageMeter()
        t = time.time()
        self.optimizer.zero_grad()
#         p_bar = tqdm(train_loader)
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'lr: {lr:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = torch.stack(images)
            images = images.cuda().float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].cuda().float() for target in targets]
            labels = [target['labels'].cuda().float() for target in targets]
            loss, _, _ = self.model(images, boxes, labels)
            loss = loss/float(self.config.grad_step)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
#             loss.backward()
            summary_loss.update(loss.detach().item(), batch_size)
            if (step+1)%self.config.grad_step ==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema.update(self.model.model)
                self.scheduler.step()
        self.ema.update_attr(self.model.model)
        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'ema_state_dict':self.ema.ema.state_dict(),
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

        
