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
            bbox = self.boxes[idx]["bbox"]
        else:
            bbox = None
        return (image, label, bbox, id)