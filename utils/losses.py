# -*- coding:utf-8 -*-
# author: Xinge


"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
from configparser import DuplicateSectionError
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import pdb


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

def jaccard_loss(probas, labels,ignore=None, smooth = 100, bk_class = None):
    """
    Something wrong with this loss
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    
    
    true_1_hot = torch.eye(vprobas.shape[1])[vlabels]
    
    if bk_class:
        one_hot_assignment = torch.ones_like(vlabels)
        one_hot_assignment[vlabels == bk_class] = 0
        one_hot_assignment = one_hot_assignment.float().unsqueeze(1)
        true_1_hot = true_1_hot*one_hot_assignment
    
    true_1_hot = true_1_hot.to(vprobas.device)
    intersection = torch.sum(vprobas * true_1_hot)
    cardinality = torch.sum(vprobas + true_1_hot)
    loss = (intersection + smooth / (cardinality - intersection + smooth)).mean()
    return (1-loss)*smooth

def hinge_jaccard_loss(probas, labels,ignore=None, classes = 'present', hinge = 0.1, smooth =100):
    """
    Multi-class Hinge Jaccard loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    C = vprobas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        if c in vlabels:
            c_sample_ind = vlabels == c
            cprobas = vprobas[c_sample_ind,:]
            non_c_ind =np.array([a for a in class_to_sum if a != c])
            class_pred = cprobas[:,c]
            max_non_class_pred = torch.max(cprobas[:,non_c_ind],dim = 1)[0]
            TP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max = hinge)+1.) + smooth
            FN = torch.sum(torch.clamp(max_non_class_pred - class_pred, min = -hinge)+hinge)
            
            if (~c_sample_ind).sum() == 0:
                FP = 0
            else:
                nonc_probas = vprobas[~c_sample_ind,:]
                class_pred = nonc_probas[:,c]
                max_non_class_pred = torch.max(nonc_probas[:,non_c_ind],dim = 1)[0]
                FP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max = hinge)+1.)
            
            losses.append(1 - TP/(TP+FP+FN))
    
    if len(losses) == 0: return 0
    return mean(losses)

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class DICELoss(nn.Module):
    
    def __init__(self, ignore_label=None, powerize=True, use_tmask=True):
        super(DICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize
        self.use_tmask = use_tmask

    def forward(self, output, target, softmax_input=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target = F.one_hot(target, num_classes=output.shape[1])
        if not softmax_input:
            output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)
        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)


class SoftDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = self.get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = self.get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)
        
    def get_soft(self, t_vector, eps=0.25):

        max_val = 1 - eps
        min_val = eps / (t_vector.shape[-1] - 1)

        t_soft = torch.empty(t_vector.shape)
        t_soft[t_vector == 0] = min_val
        t_soft[t_vector == 1] = max_val

        return t_soft


    def get_kitti_soft(self, t_vector, labels, eps=0.25):

        max_val = 1 - eps
        min_val = eps / (t_vector.shape[-1] - 1)

        t_soft = torch.empty(t_vector.shape)
        t_soft[t_vector == 0] = min_val
        t_soft[t_vector == 1] = max_val

        searched_idx = torch.logical_or(labels == 6, labels == 1)
        if searched_idx.sum() > 0:
            t_soft[searched_idx, 1] = max_val/2
            t_soft[searched_idx, 6] = max_val/2

        return t_soft