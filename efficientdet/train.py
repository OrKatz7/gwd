import train_utils
import datasets_utils
import sys
from tqdm.auto import tqdm
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
import torchvision
import argparse
import pprint


class Config:
    root_data = '../data/train/'
    csv = '../data/train.csv'
    fold_number = 0
    num_workers = 8
    batch_size = 8
    grad_step = 1
    n_epochs = 80
    optimizer = torch.optim.SGD #torch.optim.AdamW
    lr = 0.005
    SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_params = dict(
        T_max=500,
        )
    verbose = True
    verbose_step = 1
    TrainMultiScale = [1.0,0.5,0.625,0.75,0.875] # The first place must be 1.0
    net_name = 'tf_efficientdet_d7'
    checkpoint_name = '../efficientdet/tf_efficientdet_d7_53-6d1d7a95.pth'
    CocoFormat=False

class data_config:
    real = 0.5
    mosaic = 0.3
    cutmix = 0.1
    stylized = 0.2
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resume', type=str, help='resume training from last.pt',default=None)
    opt = parser.parse_args()
    config = Config()
    config.fold_number = opt.fold
    config.resume = opt.resume
    folder = 'effdet7_fold{}'.format(config.fold_number)
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.folder = folder
    
    if not config.CocoFormat:
        marking,df_folds,spike = train_utils.get_k_fols(config)


        train_dataset = datasets_utils.train_wheat(image_ids=df_folds[df_folds['fold'] != config.fold_number].index.values,
                                    marking=marking,
                                    data_config = data_config,
                                    transforms=datasets_utils.get_train_transforms(),
                                    test=False,
                                    TRAIN_ROOT_PATH=config.root_data)

        validation_dataset = datasets_utils.DatasetRetrieverTest(image_ids=df_folds[df_folds['fold'] == config.fold_number].index.values,
                                                  marking=marking,
                                                  transforms=datasets_utils.get_valid_transforms(),
                                                  test=True,
                                                  TRAIN_ROOT_PATH=config.root_data)
    else:
        train_csv = pd.read_csv("cocoFormat/tile_train.csv")
        pseudo_csv = pd.read_csv("cocoFormat/pseudo_train.csv")
        val_csv = pd.read_csv("cocoFormat/tile_val.csv")
        train_csv = pd.concat([train_csv,pseudo_csv])

        train_dataset = datasets_utils.train_wheat(image_ids=np.array(list(set(train_csv.image_id))),
                                    marking=train_csv,
                                    data_config=data_config,
                                    transforms=datasets_utils.get_train_transforms(),
                                    test=False,
                                    TRAIN_ROOT_PATH=config.root_data)

        validation_dataset = datasets_utils.DatasetRetrieverTest(image_ids=np.array(list(set(val_csv.image_id))),
                                                  marking=val_csv,
                                                  transforms=datasets_utils.get_valid_transforms(),
                                                  test=True,
                                                  TRAIN_ROOT_PATH=config.root_data)

    train_loader = torch.utils.data.DataLoader(
                                                    train_dataset,
                                                    batch_size=config.batch_size,
                                                    sampler=RandomSampler(train_dataset),
                                                    pin_memory=False,
                                                    drop_last=True,
                                                    num_workers=config.num_workers,
                                                    collate_fn=datasets_utils.collate_fn
                                                   )
    validation_loader = torch.utils.data.DataLoader(
                                                    validation_dataset,
                                                    batch_size=config.batch_size*4,
                                                    shuffle=False,
                                                    num_workers=2,
                                                    drop_last=False,
                                                    collate_fn=datasets_utils.collate_fn
                                                    )
    if len(config.TrainMultiScale)==1:
        net = train_utils.get_net(type_net=config.net_name,checkpoint_name=config.checkpoint_name,resume = config.resume)
    else:
        net = train_utils.get_net_multiscle(type_net=config.net_name,checkpoint_name=config.checkpoint_name,resume = config.resume,multiScale=config.TrainMultiScale)
    net.cuda()
    fitter = train_utils.Fitter(model=net, config=config)
    fitter.fit(train_loader, validation_loader)




