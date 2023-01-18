import os,sys
import cv2
from dataset import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from resnet import *

torch.random.manual_seed(0)

data_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
selected = [i for i in range(0,15849)]
trainset = ilsvrc30(data_path, 'train', selected)
train_loader = DataLoader(trainset, 1, drop_last=True, shuffle=True, num_workers=4)

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = '/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/all/seed0/loss4/31_55.304_model.pt'
model = ResNet18()
linear = Linear(num_classes=30)
decoder = Decoder(output_size=256)
model = model.to(device)
decoder = decoder.to(device)

model_para = torch.load(model_path)
model.load_state_dict(model_para['model'])
decoder.load_state_dict(model_para['decoder'])

pbar = tqdm(train_loader)
for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
    images2 = torch.cat((images[:,2,:], images[:,1,:,:], images[:,0,:,:]), dim=0)
    # print(images.shape)
    save_image(heatmaps, './temp_heatmap.png')
    save_image(images2, './temp_img.png')
    
    images = images.to(device)
    feature = model(images)
    output = decoder(feature)
    output = torch.sigmoid(output)
    feature = feature.squeeze()
    heatmap_output = torch.nn.functional.normalize(torch.sum(feature, dim=0), dim=-1)
    output= output.squeeze()
    print(output.shape)
    min_pred = torch.min(output)
    max_pred = torch.max(output)
    output = (output-min_pred)/(max_pred-min_pred)
    # save_image(heatmap_output, './temp_output_heatmap.png')
    save_image(output, './temp_output_heatmap.png')
    break



