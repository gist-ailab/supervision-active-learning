import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/yunjae_heo/SSD/yunjae.heo/chestx-det')
parser.add_argument('--save_path', type=str, default='/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/box_loss')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['loss'], default='loss')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = args.episode
if not os.path.isdir(args.data_path):
    os.mkdir(args.data_path)
if not args.seed==None:
    save_path = os.path.join(args.save_path, f'seed{args.seed}',args.query_algorithm)
else:
    save_path = os.path.join(args.save_path, 'current', args.query_algorithm)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
if __name__ == "__main__":
    selected = []
    trainset = chestX(args.data_path, 'train', selected)
    testset = chestX(args.data_path, 'test', [])
    trainloader = DataLoader(trainset, args.batch_size, drop_last=True, shuffle=True)
    testloader = DataLoader(testset, args.batch_size, drop_last=False, shuffle=False)
    
    model = ResNet18()
    Linear = Linear(num_classes=10)
    Decoder = Decoder(output_size=256)
    model = model.to(device)
    Linear = Linear.to(device)
    Decoder = Decoder.to(device)
    
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    Linear_optimizer = optim.Adam(Linear.parameters(), lr=args.lr)
    Decoder_optimizer = optim.Adam(Decoder.parameters(), lr=args.lr)
    
    classif_loss = nn.CrossEntropyLoss()
    heatmap_loss = heatmap_loss()
    
    #train-------------------------------------------------------------------
    def train(epoch):
        model.train()
        Linear.train()
        Decoder.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader)
        print(f'epoch : {epoch} _________________________________________________')
        for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
            images, labels, heatmaps = images.to(device), labels.to(device), heatmaps.to(device)
            model_optimizer.zero_grad()
            Linear_optimizer.zero_grad()
            Decoder_optimizer.zero_grad()
            feature = model(images)
            outputs = Linear(feature)
            pred_hmap = Decoder(feature)
            
            loss_cls = classif_loss(outputs, labels)
            loss_hmap = heatmap_loss(pred_hmap, heatmaps)
            loss = 0.1*loss_cls + loss_hmap
            loss.backward()
            model_optimizer.step()
            Linear_optimizer.step()
            Decoder_optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})
    
    #test---------------------------------------------------------------------
    def test(epoch):
        model.eval()
        Linear.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for idx, (images, labels, _, img_id) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = classif_loss(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
            acc = 100*correct/total
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(save_path,f'{epoch}_{acc:0.3f}_model.pt'))
    #------------------------------------------------------------------------------
    for i in range(args.epoch):
        train(i)
        test(i)