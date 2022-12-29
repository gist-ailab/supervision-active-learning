import os,sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import json
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