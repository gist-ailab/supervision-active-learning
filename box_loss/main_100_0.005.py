import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
from dataset import chestX, ilsvrc30
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/yunjae_heo/SSD/yunjae.heo/ILSVRC')
parser.add_argument('--save_path', type=str, default='/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/all')
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='4')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--query_algorithm', type=str, default='loss_01')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = args.episode
if not os.path.isdir(args.data_path):
    os.mkdir(args.data_path)
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

if not args.seed==None:
    save_path = os.path.join(args.save_path, f'seed{args.seed}')
else:
    save_path = os.path.join(args.save_path, 'current')
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
save_path = os.path.join(save_path, args.query_algorithm)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
if __name__ == "__main__":
    selected = [i for i in range(0,15849)]
    trainset = ilsvrc30(args.data_path, 'train', selected)
    testset = ilsvrc30(args.data_path, 'val', [])
    train_loader = DataLoader(trainset, args.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(testset, args.batch_size, drop_last=False, shuffle=False)
    
    model = ResNet18()
    linear = Linear(num_classes=30)
    decoder = Decoder(output_size=224)
    model = model.to(device)
    linear = linear.to(device)
    decoder = decoder.to(device)
    
    model_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    Linear_optimizer = optim.SGD(linear.parameters(), lr=args.lr)
    Decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)
    
    model_scheduler = MultiStepLR(model_optimizer, milestones=[50,100], gamma=0.1)
    linear_scheduler = MultiStepLR(Linear_optimizer, milestones=[50,100], gamma=0.1)
    Decoder_scheduler = MultiStepLR(Decoder_optimizer, milestones=[50,100], gamma=0.1)
    
    classif_loss = nn.CrossEntropyLoss()
    heatmap_loss = utils.heatmap_loss()
    
    #train-------------------------------------------------------------------
    def train(epoch):
        model.train()
        linear.train()
        decoder.train()
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
            outputs = linear(feature)
            pred_hmap = decoder(feature)
            pred_hmap = torch.sigmoid(pred_hmap)
            pred_hmap = pred_hmap.squeeze()
            
            loss_cls = classif_loss(outputs, labels) 
            loss_hmap = heatmap_loss(pred_hmap, heatmaps)
            # print(loss_cls, loss_hmap)
            loss = loss_cls + 0.005*loss_hmap
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
    def test(epoch, best_acc):
        model.eval()
        linear.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for idx, (images, labels, _, img_id) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                feature = model(images)
                outputs = linear(feature)
                loss = classif_loss(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
            acc = 100*correct/total
            if acc > best_acc:
                torch.save({'model':model.state_dict(), 'linear':linear.state_dict(), 'decoder':decoder.state_dict()}, os.path.join(save_path,f'{epoch}_{acc:0.3f}_model.pt'))
                best_acc = acc
            return best_acc 
    #------------------------------------------------------------------------------
    best_acc = 0
    for i in range(0, args.epoch):
        train(i)
        best_acc = test(i, best_acc)
        model_scheduler.step()
        linear_scheduler.step()
        Decoder_scheduler.step()