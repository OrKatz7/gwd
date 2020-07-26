""" PyTorch EfficientDet support benches

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
from .anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS
from .loss import DetectionLoss
import torch.nn.functional as F
import math
import numpy as np

def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        gs = 32  # (pixels) grid size
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes])
        for level in range(config.num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(config.num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all / config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all

class DetBenchEvalMultiScale(nn.Module):
    def __init__(self, model, config,multiscale=[]):
        super(DetBenchEvalMultiScale, self).__init__()
        self.config = config
        self.model = model
        self.multiscale = multiscale
        self.anchors = []
        for i,m in enumerate(multiscale):
            self.anchors.append(Anchors(
                config.min_level, config.max_level,
                config.num_scales, config.aspect_ratios,
                config.anchor_scale, int(config.image_size*m)).cuda())
        self.size = config.size

    def forward(self, x, image_scales):
        _,_,h,w = x.shape
        scale = self.size/h
        if scale not in self.multiscale:
            print("scale not in 0.5,0.625,0.75,0.875,1.0,1.125,1.25,1.375,1.5")
            return None
        s = np.where(np.array(self.multiscale)==scale)[0][0]
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors[s].boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)

class DetBenchEval(nn.Module):
    def __init__(self, model, config):
        super(DetBenchEval, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)

    def forward(self, x, image_scales):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors.boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)

class DetBenchTrainMultiScale(nn.Module):
    def __init__(self, model, config,multiscale=[1,1.1,0.9]):
        super(DetBenchTrainMultiScale, self).__init__()
        print(config)
        self.config = config
        self.model = model
        self.multiscale = multiscale
        self.anchors = []
        self.anchor_labeler = []
        for i,m in enumerate(multiscale):
            self.anchors.append(Anchors(
                config.min_level, config.max_level,
                config.num_scales, config.aspect_ratios,
                config.anchor_scale, int(config.image_size*m)).cuda())
            self.anchor_labeler.append(AnchorLabeler(self.anchors[i], config.num_classes, match_threshold=0.5).cuda())
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, gt_boxes, gt_labels):
        s = np.random.randint(len(self.multiscale))
        scale = self.multiscale[s]
        x = scale_img(x,scale)
        class_out, box_out = self.model(x)
        cls_targets = []
        box_targets = []
        num_positives = []
        # FIXME this may be a bottleneck, would be faster if batched, or should be done in loader/dataset?
        for i in range(x.shape[0]):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler[s].label_anchors(gt_boxes[i]*scale, gt_labels[i])
            cls_targets.append(gt_class_out)
            box_targets.append(gt_box_out)
            num_positives.append(num_positive)

        return self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
    
    def predict(self, x, image_scales,ema):
        ema.ema.eval()
        class_out, box_out = ema.ema(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors[0].boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)
    
    
class DetBenchTrainMultiScaleV2(nn.Module):
    def __init__(self, model, config,multiscale=[1,1.1,0.9]):
        super(DetBenchTrainMultiScaleV2, self).__init__()
        print(config)
        self.config = config
        self.model = model
        self.multiscale = multiscale
        self.anchors = []
        self.anchor_labeler = []
        self.anchors = Anchors(
                config.min_level, config.max_level,
                config.num_scales, config.aspect_ratios,
                config.anchor_scale, int(config.image_size))
        self.anchor_labeler = (AnchorLabeler(self.anchors[i], config.num_classes, match_threshold=0.5))
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, gt_boxes, gt_labels):
        s = np.random.randint(len(self.multiscale))
        scale = self.multiscale[s]
        x = scale_img(x,scale)
        class_out, box_out = self.model(x)
        box_out/=scale
        cls_targets = []
        box_targets = []
        num_positives = []
        # FIXME this may be a bottleneck, would be faster if batched, or should be done in loader/dataset?
        for i in range(x.shape[0]):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler.label_anchors(gt_boxes[i], gt_labels[i])
            cls_targets.append(gt_class_out)
            box_targets.append(gt_box_out)
            num_positives.append(num_positive)

        return self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
    
    def predict(self, x, image_scales,ema):
        ema.ema.eval()
        class_out, box_out = ema.ema(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors[0].boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)
    
class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(self.anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, gt_boxes, gt_labels):
        class_out, box_out = self.model(x)

        cls_targets = []
        box_targets = []
        num_positives = []
        # FIXME this may be a bottleneck, would be faster if batched, or should be done in loader/dataset?
        for i in range(x.shape[0]):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler.label_anchors(gt_boxes[i], gt_labels[i])
            cls_targets.append(gt_class_out)
            box_targets.append(gt_box_out)
            num_positives.append(num_positive)

        return self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
    
    def predict(self, x, image_scales):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)

        batch_detections = []
        # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
        for i in range(x.shape[0]):
            detections = generate_detections(
                class_out[i], box_out[i], self.anchors.boxes, indices[i], classes[i], image_scales[i])
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)
    
