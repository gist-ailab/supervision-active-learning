import os,sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms2
import transforms as T
import torchvision.transforms.functional as F2
from torchvision import datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import json
import csv
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
import copy
from glob import glob

class ISIC2017(Dataset):
    def __init__(self, path, mode, transforms=None):
        super(ISIC2017, self).__init__()
        self.path = path
        self.mode = mode
        if self.mode=='Test': self.mode='Test_v2'
        self.csv_path = os.path.join(self.path, self.mode, f'ISIC-2017_{self.mode}_Part3_GroundTruth.csv')
        self.img_path = os.path.join(self.path, self.mode, f'ISIC-2017_{self.mode}_Data')
        self.mask_path = os.path.join(self.path, self.mode, f'ISIC-2017_{self.mode}_Part1_GroundTruth')
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self.init_transforms()
        with open(self.csv_path, 'r', newline='') as f:
            self.gt = csv.reader(f)
            self.classes = next(iter(self.gt))[1:]
            self.gt = list(self.gt)
            
    def __len__(self):
        return len(self.gt)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.gt[index][0]+'.jpg'))
        if int(float(self.gt[index][1]))==1:
            label = 1
        elif int(float(self.gt[index][2]))==1:
            label = 2
        else:
            label = 0
        mask = Image.open(os.path.join(self.mask_path, self.gt[index][0]+'_segmentation.png'))
        
        img = self.transforms[0](img)
        mask = self.transforms[1](mask)
        # print(img.dtype)
        # print(type(label))
        # print(mask.dtype)
        # print(type(index))
        return img, label, mask, index
    
    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        if self.mode == 'Training':
            img_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomInvert(p=1.0),
                transforms.ToTensor(),
                normalize,
                ])
            mask_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        else:
            img_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                 transforms.RandomInvert(p=1.0),
                transforms.ToTensor(),
                normalize,
                ])
            mask_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        return img_transform, mask_transform

class ilsvrc30(Dataset):
    def __init__(self, path, mode):
        super(ilsvrc30, self).__init__()
        self.path = path
        self.heatmap_path = os.path.join(self.path, 'Heatmap_ILSVRC')
        if mode == 'train':self.mode = 'train'
        elif mode == 'val':self.mode = 'val'
        elif mode == 'test':self.mode = 'test'
        if self.mode == 'train' or self.mode == 'val':
            self.data_path = os.path.join(self.path, 'ILSVRC/Data/CLS-LOC', 'train')
        else:
            self.data_path = os.path.join(self.path, 'ILSVRC/Data/CLS-LOC', 'val')
        with open(os.path.join(self.path, f'new_LOC_{self.mode}_solution_30.csv'), newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            self.data_list = list(csv_reader)
            # self.data_list = self.data_list[1:]
        self.transform, _ = self.init_transforms()
        # self.transform2 = self.imagenet_transform2()
        self.base_heatmap = torch.zeros((224,224,2))
        for x in range(224):
            for y in range(224):
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
        
        if self.mode=='train' or self.mode=='val':
            image_path = os.path.join(self.data_path, label, image_id+'.JPEG')
        elif self.mode=='test':
            image_path = os.path.join(self.data_path, image_id+'.JPEG')
        # image = cv2.imread(image_path)
        image = Image.open(image_path).convert("RGB")
        if type(image)==None:
            print(image_path)
        width, height = image.size
        # image2 = self.transform2(image)
        image = self.transform(image)
        
        if os.path.isfile(os.path.join(self.heatmap_path, image_id+'.png')):
            heatmap = Image.open(os.path.join(self.heatmap_path, image_id+'.png'))
            heatmap = F2.to_tensor(heatmap)
        
        else:
            final_heatmap = torch.zeros((224,224))
            for i in range(0,len(label_list),5):
                _, x_min, y_min, x_max, y_max = label_list[i:i+5]
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                rx_min, rx_max = 224*x_min/width, 224*x_max/width
                ry_min, ry_max = 224*y_min/height, 224*y_max/height
                heatmap = torch.clone(self.base_heatmap)
                # radi = min((rx_max-rx_min)/2, (ry_max-ry_min)/2)
                sigma = 1/3
                x_r, y_r = (rx_max-rx_min)/2, (ry_max-ry_min)/2
                # radi = torch.tensor([x_r,y_r])
                radi = 112
                gt = torch.tensor([rx_min+(rx_max-rx_min)/2, ry_min+(ry_max-ry_min)/2])
                heatmap = (heatmap - gt)/radi
                heatmap = heatmap * heatmap
                heatmap = heatmap * heatmap
                heatmap = torch.sum(heatmap, dim=-1)
                heatmap = -1*heatmap/(3*sigma)**2
                heatmap = torch.exp(heatmap)
                heatmap = heatmap * heatmap
                # heatmap = torch.where(heatmap > 0.1, 1.0, 0.0)
                final_heatmap = final_heatmap + heatmap
                # final_heatmap = torch.sigmoid(final_heatmap)
            # final_heatmap = final_heatmap/(len(label_list)/5)
            # final_heatmap = torch.where(final_heatmap > 0, 1.0, 0.0)
            final_heatmap = (final_heatmap - torch.min(final_heatmap))/torch.max(final_heatmap)
            heatmap = final_heatmap
            heatmap = heatmap.unsqueeze(0)
            F2.to_pil_image(heatmap).save(os.path.join(self.heatmap_path, image_id+'.png'))
        
        return (image, self.label_class[label], heatmap, idx)
    
    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        if self.mode == 'train':
            img_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                normalize,
                ])
            mask_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        else:
            img_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
                ])
            mask_transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor()
                ])
        return img_transform, mask_transform
    
class BDD100K(Dataset):
    def __init__(self, path, mode):
        super(BDD100K, self).__init__()
        self.mode = mode
        self.img_root = os.path.join(path, mode)
        self.anno_path = os.path.join(path, f'bdd100k_labels_images_{mode}.json')
        self.classes = {'person':0, 'car':1, 'truck':2, 'bus':3, 'train':4, 'bike':5, 'motor':6, 'rider':7, 'traffic light':8, 'traffic sign':9}
        with open(self.anno_path, 'r') as f:
            self.annotation = json.load(f)
        self.t = self.transform() 
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        img_name = self.annotation[idx]['name']
        img = Image.open(os.path.join(self.img_root, img_name))
        boxes = []
        labels = []
        for label in self.annotation[idx]["labels"]:
            if label['category'] not in self.classes.keys():
                continue
            labels.append(self.classes[label['category']])
            boxes.append([label['box2d']['x1'],label['box2d']['y1'],label['box2d']['x2'],label['box2d']['y2']])
        if len(boxes)==0:
            print('NO OBJECTS')
            print(img_name)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(self.classes),), dtype=torch.int64)
        
        targets = {}
        targets["boxes"] = boxes
        targets["labels"] = labels
        targets["image_id"] = image_id
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        
        img, targets = self.t(img, targets)
        return img, targets
    
    def transform(self):
        normalize = transforms2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if self.mode=='train':
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(transforms2.ColorJitter(0.3,0.3,0.3))
        transforms.append(normalize)
        return T.Compose(transforms)
    
class AutomobileKorea(Dataset):
    def __init__(self, path, mode):
        super(AutomobileKorea, self).__init__()
        if mode=='train': self.mode='Training'
        if mode=='val': self.mode='Validation'
        self.img_root = os.path.join(path, 'daytime', 'data', self.mode, 'source')
        self.anno_root = os.path.join(path, 'daytime', 'data', self.mode, 'label')
        
        
    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
    
    def transform(self):
        normalize = transforms2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        if self.mode == 'train':
            img_transform = transforms2.Compose(
                [transforms2.Resize((224,224)),
                transforms2.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
                transforms2.ColorJitter(brightness=0.3, contrast=0.3),
                transforms2.ToTensor(),
                normalize,
                ])
        else:
            img_transform = transforms2.Compose(
                [transforms2.Resize((224,224)),
                transforms2.ToTensor(),
                normalize,
                ])
        return img_transform

if __name__ == '__main__':
    path = '/ailab_mat/dataset/ISIC_skin_disease'
    trainset = ISIC2017(path, 'Test', None)
    print(len(trainset))
    print(trainset[2][1])
    
