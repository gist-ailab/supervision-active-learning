import os,sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import torchvision.transforms.v2 as transforms2
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
import random
import shutil
import albumentations as A

class ISIC2017(Dataset):
    def __init__(self, path, mode):
        super(ISIC2017, self).__init__()
        self.path = path
        self.mode = mode
        self.classes = {'nv':2, 'mel':1, 'bkl':0}
        self.img_list = glob(os.path.join(self.path, self.mode, 'nv','*'))\
            + glob(os.path.join(self.path, self.mode, 'mel','*'))\
            + glob(os.path.join(self.path, self.mode, 'bkl','*'))
        self.imgt, self.maskt = self.init_transforms()
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        label = img_path.split('/')[-2]
        mask = Image.open(os.path.join(self.path, f'mask_{self.mode}', label, img_name +'_segmentation.png'))
        # print(label)
        img = self.imgt(img)
        mask = self.maskt(mask)
        label = self.classes[label]
        # p = random.random()
        # if p > 0.5 and self.mode=='train':
        #     img, mask = F2.hflip(img), F2.hflip(mask)
        return img, label, mask, idx
    
    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        img_transform = []
        mask_transform = []
        
        img_transform.append(transforms.Resize((256,256)))
        # if self.mode=='Training':
        if self.mode=='NONE':
            img_transform.append(transforms.CenterCrop((224,224)))
            # img_transform.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        else:
            img_transform.append(transforms.CenterCrop((224,224)))
        # img_transform.append(transforms.RandomInvert(p=1.0))
        img_transform.append(transforms.ToTensor())
        img_transform.append(normalize)
        
        mask_transform.append(transforms.Resize((256,256)))
        mask_transform.append(transforms.CenterCrop((224,224)))
        mask_transform.append(transforms.ToTensor())
        
        return transforms.Compose(img_transform), transforms.Compose(mask_transform)

class ISIC2017_2(Dataset):
    def __init__(self, path, mode):
        super(ISIC2017_2, self).__init__()
        self.path = path
        self.mode = mode
        self.classes = {'nv':2, 'mel':0, 'bkl':1}
        self.img_list = glob(os.path.join(self.path, self.mode, 'nv','*'))\
            + glob(os.path.join(self.path, self.mode, 'mel','*'))\
            + glob(os.path.join(self.path, self.mode, 'bkl','*'))
        # if self.mode=='train':
        #     self.img_list += glob(os.path.join(self.path, 'nv_add', '*'))
        #     self.img_list += glob(os.path.join(self.path, 'mel_add', '*'))
        #     self.img_list += glob(os.path.join(self.path, 'bkl_add', '*'))
        self.t = self.init_transforms()
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        label = img_path.split('/')[-2]
        if '_add' in label:
            label = label.split('_')[0]
        label = self.classes[label]
        try:
            mask = Image.open(os.path.join(self.path, f'mask_{self.mode}', label, img_name +'_segmentation.png'))
            transformed = self.t(image=img, mask=mask)
            t_img = transformed['image']
            t_mask = transformed['mask']
        except:
            mask = -1
            transformed = self.t(image=img)
            t_img = transformed['image']
        return img, label, mask, idx
    
    def init_transforms(self):
        t = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(10),
            A.Normalize()
        ])
        return t

class ISIC2017_3(Dataset):
    def __init__(self, path, mode, idx_list=[]):
        super(ISIC2017_3, self).__init__()
        self.path = path
        self.mode = mode
        self.classes = {'nv':1, 'mel':0, 'bkl':1}
        self.img_list = glob(os.path.join(self.path, self.mode, 'nv','*'))\
            + glob(os.path.join(self.path, self.mode, 'mel','*'))\
            + glob(os.path.join(self.path, self.mode, 'bkl','*'))
        self.imgt, self.maskt = self.init_transforms()
        self.idx_list = idx_list
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img_name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        label = img_path.split('/')[-2]
        mask = Image.open(os.path.join(self.path, f'mask_{self.mode}', label, img_name +'_segmentation.png'))
        # print(label)
        img = self.imgt(img)
        mask = self.maskt(mask)
        label = self.classes[label]
        p1 = random.random()
        p2 = random.random()
        if p1 > 0.5 and self.mode=='train':
            img, mask = F2.hflip(img), F2.hflip(mask)
        if p2 > 0.5 and self.mode=='train':
            img, mask = F2.vflip(img), F2.vflip(mask)
        if idx in self.idx_list:
            mask = mask.view(1, mask.shape[1], mask.shape[2])
            img_c = torch.cat([img, mask], dim=0)
        else:
            dummy = torch.ones([1, mask.shape[1], mask.shape[2]])
            img_c = torch.cat([img, dummy], dim=0)
        return img_c, label, mask, idx
    
    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        img_transform = []
        mask_transform = []
        
        img_transform.append(transforms.Resize((256,256)))
        img_transform.append(transforms.CenterCrop((224,224)))
        img_transform.append(transforms.ToTensor())
        img_transform.append(normalize)
        
        mask_transform.append(transforms.Resize((256,256)))
        mask_transform.append(transforms.CenterCrop((224,224)))
        mask_transform.append(transforms.ToTensor())
        
        return transforms.Compose(img_transform), transforms.Compose(mask_transform)

class HAM10000(Dataset):
    def __init__(self, path, mode):
        super(HAM10000, self).__init__()
        assert mode=='train' or mode=='test'
        self.path = path
        self.mode = mode
        # self.classes = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
        # self.classes = {'nv':0, 'mel':1, 'bkl':2, 'akiec':3, 'bcc':4, 'df':5, 'vasc':6}
        self.classes = {'nv':2, 'mel':1, 'bkl':0}
        self.img_list = glob(os.path.join(self.path, self.mode, 'nv','*'))\
            + glob(os.path.join(self.path, self.mode, 'mel','*'))\
            + glob(os.path.join(self.path, self.mode, 'bkl','*'))
        self.t = self.transform()
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.t(img)
        label = self.classes[img_path.split('/')[-2]]
        return img, label, torch.tensor([0]), idx
    
    def get_labels(self):
        labels = []
        for img_path in self.img_list:
            label = self.classes[img_path.split('/')[-2]]
            labels.append(label)
        return labels
    
    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_transform = []
        img_transform.append(transforms.Resize((224,224)))
        if self.mode=='train':
            img_transform.append(transforms.RandomHorizontalFlip())
            # img_transform.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        # img_transform.append(transforms.RandomInvert(p=1.0))
        img_transform.append(transforms.ToTensor())
        img_transform.append(normalize)
        return transforms.Compose(img_transform)

class HAM10000_origin(Dataset):
    def __init__(self, path, mode):
        super(HAM10000_origin, self).__init__()
        assert mode=='train' or mode=='test'
        self.path = path
        self.mode = mode
        # self.classes = {'akiec':0, 'bcc':1, 'bkl':2, 'df':3, 'mel':4, 'nv':5, 'vasc':6}
        self.classes = {'nv':2, 'mel':1, 'bkl':0, 'akiec':3, 'bcc':4, 'df':5, 'vasc':6}
        # self.classes = {'nv':2, 'mel':1, 'bkl':0}
        self.img_list = []
        for cls_name in os.listdir(os.path.join(self.path, self.mode)):
            self.img_list += glob(os.path.join(os.path.join(self.path, self.mode), cls_name, '*'))
        self.t = self.transform()
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.t(img)
        label = self.classes[img_path.split('/')[-2]]
        return img, label, torch.tensor([0]), idx

    def get_labels(self):
        labels = []
        for img_path in self.img_list:
            label = self.classes[img_path.split('/')[-2]]
            labels.append(label)
        return labels
    
    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img_transform = []
        img_transform.append(transforms.Resize((224,224)))
        if self.mode=='train':
            img_transform.append(transforms.RandomHorizontalFlip())
            img_transform.append(transforms.RandomRotation(15))
            # img_transform.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        # img_transform.append(transforms.RandomInvert(p=1.0))
        img_transform.append(transforms.ToTensor())
        img_transform.append(normalize)
        return transforms.Compose(img_transform)

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

class CUB200(Dataset):
    def __init__(self, path, mode, num_train=20):
        super(CUB200, self).__init__()
        self.path = path
        self.mode = mode
        self.img_path = os.path.join(path, mode)
        self.mask_path = os.path.join(path, mode+'_mask')
        self.img_list = []
        for cls_dir in os.listdir(self.img_path):
            label = int(cls_dir.split('.')[0])-1
            img_list = os.listdir(os.path.join(self.img_path, cls_dir))
            if self.mode=='train':
                random.shuffle(img_list)
                img_list = img_list[:num_train]
            for img_name in img_list:
                self.img_list.append((os.path.join(cls_dir, img_name), label))
        self.imgt, self.maskt = self.init_transforms()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx][0]
        label = self.img_list[idx][1]
        img = Image.open(os.path.join(self.img_path, img_path)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, img_path)).convert("L")
        # print("MAX : ", np.max(mask))
        img = self.imgt(img)
        mask = self.maskt(mask)
        p1 = random.random()
        p2 = random.random()
        if p1 > 0.5 and self.mode=='train':
            img, mask = F2.hflip(img), F2.hflip(mask)
        if p2 > 0.5 and self.mode=='train':
            img, mask = F2.vflip(img), F2.vflip(mask)
        return img, label, mask, idx

    def init_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        img_transform = []
        mask_transform = []
        
        img_transform.append(transforms.Resize((512,512)))
        if self.mode=='train':
            img_transform.append(transforms.CenterCrop((448,448)))
            img_transform.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        else:
            img_transform.append(transforms.CenterCrop((448,448)))
        img_transform.append(transforms.ToTensor())
        img_transform.append(normalize)
        
        mask_transform.append(transforms.Resize((512,512)))
        mask_transform.append(transforms.CenterCrop((448,448)))
        mask_transform.append(transforms.ToTensor())
        
        return transforms.Compose(img_transform), transforms.Compose(mask_transform)
        

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
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        if self.mode=='train':
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(transforms.ColorJitter(0.3,0.3,0.3))
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

if __name__ == '__main__':
    path = '/ailab_mat/dataset/ISIC_skin_disease'
    trainset = ISIC2017(path, 'Test', None)
    print(len(trainset))
    print(trainset[2][1])
    
