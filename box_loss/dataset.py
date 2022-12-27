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
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.filedir_path, self.images[idx]["file_name"])
        image = cv2.imread(file_path)
        image = torch.tensor(image)
        label = self.boxes[idx]["category_id"]
        id = idx+1
        if id in selected_list:
            bbox_loc = self.boxes[idx]["bbox"]
            heatmap = torch.zeros_like(image)
            radi = min(bbox_loc[2]/2, bbox_loc[3]/2)
            gt = (bbox_loc[0]+bbox_loc[2]/2, bbox_loc[1]+bbox_loc[3]/2)
            for x in range(bbox_loc[0], bbox_loc[0]+bbox_loc[2]):
                for y in range(bbox_loc[1], bbox_loc[1]+bbox_loc[3]):
                    heatmap = torch.exp(-((x-gt[0])**2+(y-gt[1])**2)/(2*(radi/3)**2))
        else:
            heatmap = None
        return (image, label, heatmap, id)