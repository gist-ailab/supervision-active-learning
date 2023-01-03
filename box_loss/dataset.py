import os,sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import json
import csv
import xml.etree.ElementTree as ET
import cv2

class chestX(Dataset):
    def __init__(self, path, mode, selected_list):
        self.path = path
        self.selected_list = selected_list
        if mode == 'train':self.mode = 'train'
        elif mode == 'test':self.mode = 'test'
        self.json_path = os.path.join(self.path, 'annotations', f'chestx-det_{self.mode}.json')
        self.filedir_path = os.path.join(self.path, f'{self.mode}_data')
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            jsonfile = json.load(f)
        
        self.categories = jsonfile["categories"]
        self.images = jsonfile["images"]
        self.boxes = jsonfile["annotations"]
        self.downsample = transforms.Resize(256)
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([x,y])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.filedir_path, self.images[idx]["file_name"])
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = transforms.functional.to_tensor(image)
        
        # image = image.permute(2,0,1)
        label = self.boxes[idx]["category_id"]
        temp = torch.zeros(10)
        temp[label-1] = 1
        label = temp
        img_id = idx+1
        # print(self.selected_list)
        if img_id in self.selected_list:
            bbox_loc = self.boxes[idx]["bbox"]
            bbox_loc = torch.tensor(bbox_loc) * (256/1024)
            heatmap = torch.clone(self.base_heatmap)
            radi = min(bbox_loc[2]/2, bbox_loc[3]/2)
            sigma = radi/3
            gt = torch.tensor([bbox_loc[0]+bbox_loc[2]/2, bbox_loc[1]+bbox_loc[3]/2])
            
            heatmap = heatmap - gt
            heatmap = heatmap * heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(2*sigma)**2
            heatmap = torch.exp(heatmap)
        else:
            heatmap = torch.tensor([])
        image = self.downsample(image)
        
        # print(heatmap)
        return (image, label, heatmap, img_id)
    
class ilsvrc30(Dataset):
    def __init__(self, path, mode, selected_list):
        self.path = path
        self.selected_list = selected_list
        if mode == 'train':self.mode = 'train'
        elif mode == 'val':self.mode = 'val'
        self.data_path = os.path.join(self.path, '/ILSVRC/Data/CLS-LOC', mode)
        with open(os.path.join(self.path, f'LOC_{self.mode}_solution_30.csv'), newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            self.data_list = list(csv_reader)
        self.imagenet_transform()
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([x,y])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_id, label_string = self.data_list[idx]
        label, x_min, y_min, x_max, y_max = label_string.split()
        if self.mode=='train':
            image_path = os.path.join(self.data_path, self.mode, label, image_id)
        elif self.mode=='val':
            image_path = os.path.join(self.data_path, self.mode, image_id)
        image = cv2.imread(image_path)
        height,width,_ = image.shape
        image = self.transform(image)
        
        if idx in selected_list:
            rx_min, rx_max = 224*x_min/width, 224*x_max/width
            ry_min, ry_max = 224*y_min/height, 224*y_max/height
            heatmap = torch.clone(self.base_heatmap)
            radi = max((rx_max-rx_min)/2, (ry_max-ry_min)/2)
            sigma = radi/3
            gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
            heatmap = heatmap - gt
            heatmap = heatmap * heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(2*sigma)**2
            heatmap = torch.exp(heatmap)
        else:
            heatmap = torch.tensor([])
        return image, label, idx, heatmap
    
    def imagenet_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.functional.to_tensor(),
             normalize,
            ])