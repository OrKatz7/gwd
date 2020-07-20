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
    batch_size = 4
    n_epochs = 50
    optimizer = torch.optim.SGD #torch.optim.AdamW
    lr = 0.005
    verbose = True
    verbose_step = 1
    net_name = 'tf_efficientdet_d7'
    checkpoint_name = '../efficientdet/tf_efficientdet_d7_53-6d1d7a95.pth'
    lf = lambda x: (((1 + math.cos(x * math.pi / 25)) / 2) ** 1.0) * 0.9 + 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt',default=None)
    opt = parser.parse_args()
    config = Config()
    config.fold_number = opt.fold
    config.resume = opt.resume
    folder = '../effdet7_fold{}'.format(config.fold_number)
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.folder = folder
    marking,df_folds = train_utils.get_k_fols(config)
    
    train_dataset = datasets_utils.train_wheat(image_ids=df_folds[df_folds['fold'] != config.fold_number].index.values,
                                marking=marking,
                                transforms=datasets_utils.get_train_transforms(),
                                test=False,
                                TRAIN_ROOT_PATH=config.root_data)

    validation_dataset = datasets_utils.DatasetRetrieverTest(image_ids=df_folds[df_folds['fold'] == config.fold_number].index.values,
                                              marking=marking,
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
    
    net = train_utils.get_net(type_net=config.net_name,checkpoint_name=config.checkpoint_name,resume = config.resume)
    net.cuda()
    fitter = train_utils.Fitter(model=net, config=config)
    fitter.fit(train_loader, validation_loader)




