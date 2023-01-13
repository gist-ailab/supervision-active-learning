import os,sys
import cv2
from dataset import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

data_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
selected = [i for i in range(0,15849)]
trainset = ilsvrc30(data_path, 'train', selected)
train_loader = DataLoader(trainset, 1, drop_last=True, shuffle=True, num_workers=4)

pbar = tqdm(train_loader)
for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
    images = torch.cat((images[:,2,:], images[:,1,:,:], images[:,0,:,:]), dim=0)
    # print(images.shape)
    save_image(heatmaps, './temp_heatmap.png')
    save_image(images, './temp_img.png')
    break
