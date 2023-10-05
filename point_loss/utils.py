import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet import *
from torchvision.utils import save_image

def get_rand_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform
        
class F1_score(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-7
        self.tr = 0.5
        
    def forward(self, y_outputs, y_gt):
        y_pred = torch.softmax(y_outputs, dim=-1)
        y_pred = torch.where(y_pred>self.tr, 1, 0)
        
        TP = torch.sum(y_pred*y_gt)
        TN = torch.sum((1-y_pred)*(1-y_gt))
        FP = torch.sum(y_pred*(1-y_gt))
        FN = torch.sum((1-y_pred)*y_gt)
        
        precision = TP / (TP + FP + self.epsilon)
        recall = TP / (TP + FN + self.epsilon)
        
        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        return f1.item()
    
class img_aug_feature_sim(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([y,x])
        
    def get_centers(self, cams):
        return [(cam_i==torch.max(cam_i)).nonzero() for cam_i in cams]
    
    def make_heatmaps(self, centers, radi, sigma):
        heatmaps = []
        for center in centers:
            heatmap = torch.clone(self.base_heatmap)
            gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
            heatmap = (heatmap - gt)/radi
            heatmap = heatmap * heatmap
            # heatmap = heatmap * heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(3*sigma)**2
            heatmap = torch.exp(heatmap)
            heatmap = heatmap * heatmap
            heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)
        return heatmaps
    
    def forward(self, imgs, cams, gt_heatmaps, radi=128, sigma=1/2):
        centers = get_centers(cams)
        heatmaps = make_heatmaps(centers, radi, sigma)
        for i in range(len(gt_heatmaps)):
            if torch.sum(gt_heatmaps[i]) != 0:
                heatmaps[i] = gt_heatmaps[i]
        new_augs = cams * heatmaps**0.5 + heatmaps*heatmaps
        new_augs = torch.exp(new_augs)
        new_augs = new_augs - torch.min(new_augs)
        new_augs = new_augs / torch.max(new_augs)
        new_imgs = imgs * new_augs
        return new_imgs
        
class feature_aug_feature_sim(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([y,x])
        
    def get_centers(self, cam):
        return [(cam_i==torch.max(cam_i)).nonzero() for cam_i in cam]
    
    def make_heatmaps(self, centers, radi, sigma):
        heatmaps = []
        for center in centers:
            heatmap = torch.clone(self.base_heatmap)
            gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
            heatmap = (heatmap - gt)/radi
            heatmap = heatmap * heatmap
            # heatmap = heatmap * heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(3*sigma)**2
            heatmap = torch.exp(heatmap)
            heatmap = heatmap * heatmap
            heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)
        return heatmaps
    
    def forward(self, feature, cams, gt_heatmaps, radi=128, sigma=1/2):
        centers = get_centers(cams)
        heatmaps = make_heatmaps(centers, radi, sigma)
        for i in range(len(gt_heatmaps)):
            if torch.sum(gt_heatmaps[i]) != 0:
                heatmaps[i] = F.interpolate(gt_heatmaps[i], size=(radi*2, radi*2), mode='bilinear')
        new_augs = cams * heatmaps**0.5 + heatmaps*heatmaps
        new_augs = torch.exp(new_augs)
        new_augs = new_augs - torch.min(new_augs)
        new_augs = new_augs / torch.max(new_augs)
        new_feature = feature * new_augs
        return new_feature