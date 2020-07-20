

import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from datetime import datetime
import time
import random
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import math
import torchvision
def collate_fn(batch):
    return tuple(zip(*batch))

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    #a = random.uniform(-degrees, degrees)
    a = random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    labels = np.ones((targets.shape[0],1))
    targets = np.concatenate([labels,targets],axis=1)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets[:, 1:5]

def get_train_transforms():
    return A.Compose(
        [
            A.RandomRotate90(),
            A.RandomSizedCrop(min_max_height=(900, 900), height=1024, width=1024, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=1024, width=1024, p=1),
            A.Cutout(num_holes=4, max_h_size=32, max_w_size=48, fill_value=0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_mosaic_transforms():
    return A.Compose(
        [
            A.RandomRotate90(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=1024, width=1024, p=1),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0.0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def collate_fn(batch):
    return tuple(zip(*batch))


class DatasetRetrieverTest(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False,TRAIN_ROOT_PATH='../data/train'):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.TRAIN_ROOT_PATH = TRAIN_ROOT_PATH

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image, boxes = self.load_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            sample = self.transforms(**{'image': image,'bboxes': target['boxes'],'labels': labels})
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
#           target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes




class train_wheat(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False,TRAIN_ROOT_PATH='../data/train'):
        super().__init__()
        self.root = TRAIN_ROOT_PATH
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        self.mosaic_transform = get_mosaic_transforms()

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        p_ratio = random.random()
        if self.test or p_ratio > 0.7:
            image, boxes = self.load_image_and_boxes(index)
        else:
            if p_ratio > 0.4:
                image, boxes = self.load_mosaic_image_and_boxes(index)
            elif p_ratio > 0.15:
                image, boxes = self.load_image_and_bboxes_with_cutmix(index)
            else:
                image, boxes = self.load_mixup_image_and_boxes(index)

        image = image.astype(np.uint8)
        if random.random() < 0.5:
            image, boxes = random_affine(image, boxes,
                                          degrees=0,
                                          translate=0,
                                          scale=0.45,
                                          shear=0)
        if random.random() < 0.5:
            augment_hsv(image, hgain=0.014, sgain=0.68, vgain=0.36)
        image = image.astype(np.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])


        if self.transforms:

            sample = self.transforms(**{'image': image,'bboxes': target['boxes'],'labels': labels})
            image = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.root}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        if self.mosaic_transform:
            sample = self.mosaic_transform(**{
                        'image': image,
                        'bboxes': boxes,
                        'labels': labels
                    })
            image = sample['image']*255
            boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).numpy()
        return image, boxes

    def load_mosaic_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of mosaic author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def load_image_and_bboxes_with_cutmix(self, index):
        image, bboxes = self.load_image_and_boxes(index)
        image_to_be_mixed, bboxes_to_be_mixed = self.load_image_and_boxes(
            random.randint(0, self.image_ids.shape[0] - 1))

        image_size = image.shape[0]
        cutoff_x1, cutoff_y1 = [int(random.uniform(image_size * 0.0, image_size * 0.49)) for _ in range(2)]
        cutoff_x2, cutoff_y2 = [int(random.uniform(image_size * 0.5, image_size * 1.0)) for _ in range(2)]

        image_cutmix = image.copy()
        image_cutmix[cutoff_y1:cutoff_y2, cutoff_x1:cutoff_x2] = image_to_be_mixed[cutoff_y1:cutoff_y2,
                                                                 cutoff_x1:cutoff_x2]

        # Begin preparing bboxes_cutmix.
        # Case 1. Bounding boxes not intersect with cut off patch.
        bboxes_not_intersect = bboxes[np.concatenate((np.where(bboxes[:, 0] > cutoff_x2),
                                                      np.where(bboxes[:, 2] < cutoff_x1),
                                                      np.where(bboxes[:, 1] > cutoff_y2),
                                                      np.where(bboxes[:, 3] < cutoff_y1)), axis=None)]

        # Case 2. Bounding boxes intersect with cut off patch.
        bboxes_intersect = bboxes.copy()

        top_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                 (bboxes[:, 2] > cutoff_x1) &
                                 (bboxes[:, 1] < cutoff_y2) &
                                 (bboxes[:, 3] > cutoff_y2))
        right_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                   (bboxes[:, 2] > cutoff_x2) &
                                   (bboxes[:, 1] < cutoff_y2) &
                                   (bboxes[:, 3] > cutoff_y1))
        bottom_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                    (bboxes[:, 2] > cutoff_x1) &
                                    (bboxes[:, 1] < cutoff_y1) &
                                    (bboxes[:, 3] > cutoff_y1))
        left_intersect = np.where((bboxes[:, 0] < cutoff_x1) &
                                  (bboxes[:, 2] > cutoff_x1) &
                                  (bboxes[:, 1] < cutoff_y2) &
                                  (bboxes[:, 3] > cutoff_y1))

        # Remove redundant indices. e.g. a bbox which intersects in both right and top.
        right_intersect = np.setdiff1d(right_intersect, top_intersect)
        right_intersect = np.setdiff1d(right_intersect, bottom_intersect)
        right_intersect = np.setdiff1d(right_intersect, left_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, top_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, left_intersect)
        left_intersect = np.setdiff1d(left_intersect, top_intersect)

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect = bboxes_intersect[np.concatenate((top_intersect,
                                                            right_intersect,
                                                            bottom_intersect,
                                                            left_intersect), axis=None)]

        # Case 3. Bounding boxes inside cut off patch.
        bboxes_to_be_mixed[:, [0, 2]] = bboxes_to_be_mixed[:, [0, 2]].clip(min=cutoff_x1, max=cutoff_x2)
        bboxes_to_be_mixed[:, [1, 3]] = bboxes_to_be_mixed[:, [1, 3]].clip(min=cutoff_y1, max=cutoff_y2)

        # Integrate all those three cases.
        bboxes_cutmix = np.vstack((bboxes_not_intersect, bboxes_intersect, bboxes_to_be_mixed)).astype(int)
        bboxes_cutmix = bboxes_cutmix[np.where((bboxes_cutmix[:, 2] - bboxes_cutmix[:, 0]) \
                                               * (bboxes_cutmix[:, 3] - bboxes_cutmix[:, 1]) > 500)]
        # End preparing bboxes_cutmix.

        return image_cutmix, bboxes_cutmix

    def load_mixup_image_and_boxes(self, index):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        return (image + r_image) / 2, np.vstack((boxes, r_boxes)).astype(np.int32)
