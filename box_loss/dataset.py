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
import numpy as np

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
        self.transform = self.imagenet_transform()
        self.downsample = transforms.Resize(256)
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([x,y])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.filedir_path, self.images[idx]["file_name"])
        image = cv2.imread(file_path)
        image = self.transform(image)
        
        # image = image.permute(2,0,1)
        label = self.boxes[idx]["category_id"]
        temp = torch.zeros(10)
        temp[label-1] = 1
        label = torch.argmax(temp)
        img_id = idx+1
        # print(self.selected_list)
        if img_id in self.selected_list:
            bbox_loc = self.boxes[idx]["bbox"]
            bbox_loc = torch.tensor(bbox_loc) * (256/1024)
            heatmap = torch.clone(self.base_heatmap)
            radi = torch.tensor([bbox_loc[2]/2, bbox_loc[3]/2])
            sigma = 1/2
            gt = torch.tensor([bbox_loc[0]+bbox_loc[2]/2, bbox_loc[1]+bbox_loc[3]/2])
            heatmap = (heatmap - gt)/radi
            heatmap = heatmap * heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(2*sigma)**2
            heatmap = torch.exp(heatmap)
            heatmap = torch.where(heatmap > 0.1, 1.0, 0.0)
        else:
            heatmap = torch.zeros((256,256))
        
        # print(heatmap)
        return (image, label, heatmap, img_id)
    
    def imagenet_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((512,512)),
             transforms.ToTensor(),
            ])
        return transform
    
class ilsvrc30(Dataset):
    def __init__(self, path, mode, selected_list):
        super(ilsvrc30, self).__init__()
        self.path = path
        self.selected_list = selected_list
        if mode == 'train':self.mode = 'train'
        elif mode == 'val':self.mode = 'val'
        self.data_path = os.path.join(self.path, 'ILSVRC/Data/CLS-LOC', mode)
        with open(os.path.join(self.path, f'LOC_{self.mode}_solution_30.csv'), newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            self.data_list = list(csv_reader)
            self.data_list = self.data_list[1:]
        self.transform = self.imagenet_transform()
        self.transform2 = self.imagenet_transform2()
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([y,x])
        self.label_class = {'n01440764' : 0,'n01443537' : 1,'n01484850' : 2,'n01491361' : 3,'n01494475' : 4,'n01496331' : 5,
                            'n01498041' : 6,'n01514668' : 7,'n01514859' : 8,'n01518878' : 9,'n01530575' : 10,'n01531178' : 11,
                            'n01532829' : 12,'n01534433' : 13,'n01537544' : 14,'n01558993' : 15,'n01560419' : 16,'n01580077' : 17,
                            'n01582220' : 18,'n01592084' : 19,'n01601694' : 20,'n01608432' : 21,'n01614925' : 22,'n01616318' : 23,
                            'n01622779' : 24,'n01629819' : 25,'n01630670' : 26,'n01631663' : 27,'n01632458' : 28,'n01632777' : 29}
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_id, label_string = self.data_list[idx]
        label_list = label_string.split()
        label = label_list[0]
        
        if self.mode=='train':
            image_path = os.path.join(self.data_path, label, image_id+'.JPEG')
        elif self.mode=='val':
            image_path = os.path.join(self.data_path, image_id+'.JPEG')
        image = cv2.imread(image_path)
        if type(image)==None:
            print(image_path)
        height,width,_ = image.shape
        # image2 = self.transform2(image)
        image = self.transform(image)
        
        if idx in self.selected_list:            
            final_heatmap = torch.zeros((256,256))
            for i in range(0,len(label_list),5):
                _, x_min, y_min, x_max, y_max = label_list[i:i+5]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                rx_min, rx_max = 256*x_min/width, 256*x_max/width
                ry_min, ry_max = 256*y_min/height, 256*y_max/height
                heatmap = torch.clone(self.base_heatmap)
                # radi = min((rx_max-rx_min)/2, (ry_max-ry_min)/2)
                sigma = 1/3
                x_r, y_r = (rx_max-rx_min)/2, (ry_max-ry_min)/2
                radi = torch.tensor([x_r,y_r])
                gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
                heatmap = (heatmap - gt)/radi
                heatmap = heatmap * heatmap
                heatmap = heatmap * heatmap
                heatmap = torch.sum(heatmap, dim=-1)
                heatmap = -1*heatmap/(2*sigma)**2
                heatmap = torch.exp(heatmap)
                heatmap = torch.where(heatmap > 0.1, 1.0, 0.0)
                final_heatmap = final_heatmap + heatmap
                # final_heatmap = torch.sigmoid(final_heatmap)
            # final_heatmap = final_heatmap/(len(label_list)/5)
            final_heatmap = torch.where(final_heatmap > 0, 1.0, 0.0)
            heatmap = final_heatmap
            # heatmap = heatmap[16:240,16:240]
        else:
            heatmap = torch.zeros((256,256))
        
        return (image, self.label_class[label], heatmap, idx)
    
    def imagenet_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((256,256)),
             transforms.ToTensor(),
             normalize,
            ])
        return transform
    
    def imagenet_transform2(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((256,256)),
             transforms.ToTensor(),
            ])
        return transform
    
class ilsvrc100(Dataset):
    def __init__(self, path, mode, selected_list):
        super(ilsvrc100, self).__init__()
        self.path = path
        self.selected_list = selected_list
        if mode == 'train':self.mode = 'train'
        elif mode == 'val':self.mode = 'val'
        self.data_path = os.path.join(self.path, 'ILSVRC/Data/CLS-LOC', mode)
        with open(os.path.join(self.path, f'LOC_{self.mode}_solution_100.csv'), newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            self.data_list = list(csv_reader)
            self.data_list = self.data_list[1:]
        self.transform = self.imagenet_transform()
        self.transform2 = self.imagenet_transform2()
        self.base_heatmap = torch.zeros((256,256,2))
        for x in range(256):
            for y in range(256):
                self.base_heatmap[x,y,:] = torch.tensor([y,x])
        
        self.label_class = dict()
        mapping = open(os.path.join(self.path, 'LOC_synset_mapping.txt'), 'r')
        self.map = mapping.readlines()
        idx = 0
        for line in self.map:
            label = line.split(' ')[0]
            self.label_class[label] = idx
            idx += 1
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image_id, label_string = self.data_list[idx]
        label_list = label_string.split()
        label = label_list[0]
        
        if self.mode=='train':
            image_path = os.path.join(self.data_path, label, image_id+'.JPEG')
        elif self.mode=='val':
            image_path = os.path.join(self.data_path, image_id+'.JPEG')
        image = cv2.imread(image_path)
        if type(image)==None:
            print(image_path)
        height,width,_ = image.shape
        # image2 = self.transform2(image)
        image = self.transform(image)
        
        if idx in self.selected_list:            
            final_heatmap = torch.zeros((256,256))
            for i in range(0,len(label_list),5):
                _, x_min, y_min, x_max, y_max = label_list[i:i+5]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                rx_min, rx_max = 256*x_min/width, 256*x_max/width
                ry_min, ry_max = 256*y_min/height, 256*y_max/height
                heatmap = torch.clone(self.base_heatmap)
                # radi = min((rx_max-rx_min)/2, (ry_max-ry_min)/2)
                sigma = 1/3
                x_r, y_r = (rx_max-rx_min)/2, (ry_max-ry_min)/2
                radi = torch.tensor([x_r,y_r])
                gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
                heatmap = (heatmap - gt)/radi
                heatmap = heatmap * heatmap
                heatmap = heatmap * heatmap
                heatmap = torch.sum(heatmap, dim=-1)
                heatmap = -1*heatmap/(2*sigma)**2
                heatmap = torch.exp(heatmap)
                heatmap = torch.where(heatmap > 0.1, 1.0, 0.0)
                final_heatmap = final_heatmap + heatmap
                # final_heatmap = torch.sigmoid(final_heatmap)
            # final_heatmap = final_heatmap/(len(label_list)/5)
            final_heatmap = torch.where(final_heatmap > 0, 1.0, 0.0)
            heatmap = final_heatmap
            # heatmap = heatmap[16:240,16:240]
        else:
            heatmap = torch.zeros((256,256))
        
        return (image, self.label_class[label], heatmap, idx)
    
    def imagenet_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((256,256)),
             transforms.ToTensor(),
             normalize,
            ])
        return transform
    
    def imagenet_transform2(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((256,256)),
             transforms.ToTensor(),
            ])
        return transform