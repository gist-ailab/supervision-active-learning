import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
from dataset import chestX, ilsvrc30, ilsvrc100
import utils
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/yunjae_heo/SSD/yunjae.heo/ILSVRC')
parser.add_argument('--save_path', type=str, default='/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/loss_1500')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--epoch2', type=int, default=50)
parser.add_argument('--episode', type=int, default=5)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='6')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--query_algorithm', type=str, default='loss4')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--model_para', type=str, default='')

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

#train-------------------------------------------------------------------
def train(epoch, train_loader, AVG_Loss):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    alp = 10
    g = 0.15
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
        images, labels, heatmaps = images.to(device), labels.to(device), heatmaps.to(device)
        
        model_optimizer.zero_grad()
        outputs, _ = model(images)
        _, predicted = outputs.max(1)
        
        loss_cls = classif_loss(outputs, labels)
        loss = loss_cls        
        loss.backward()
        model_optimizer.step()
        
        train_loss += loss.item()
        if epoch == 0:
            AVG_Loss[img_id] = loss.item()
        else:
            temp_avg_loss = AVG_Loss[img_id].clone()
            temp_avg_loss = temp_avg_loss * (1-g) + loss.item() * g
            AVG_Loss[img_id] = temp_avg_loss
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})

def train2(epoch, train_loader):
    model.train()
    heatmap_model.eval()
    train_loss = 0
    correct = 0
    total = 0
    alp = 10
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
        images, labels, heatmaps = images.to(device), labels.to(device), heatmaps.to(device)
        # print(device)
        model_optimizer.zero_grad()
        outputs, acts = model(images)
        _, predicted = outputs.max(1)
        
        b,c,h,w = acts.shape
        weight = list(model.parameters())[-2].data
        beforDot = torch.reshape(acts, (b,c,h*w))
        weights = torch.stack([weight[i].unsqueeze(0) for i in labels], dim=0)
        # weights = torch.stack([weight[i].unsqueeze(0) for i in labels], dim=0)

        cam = torch.bmm(weights, beforDot)
        cam = torch.reshape(cam, (b, h, w))
        cam = torch.stack([cam[i]-torch.min(cam[i]) for i in range(b)], dim=0)
        cam = torch.stack([cam[i]/torch.max(cam[i]) for i in range(b)], dim=0)
        cam = cam.unsqueeze(dim=1)
        pred_hmap = F.interpolate(cam, size=(256,256))
        
        # print(outputs.shape, labels.shape)
        loss_cls = classif_loss(outputs, labels)
        #-----------------------------------------------
        # avg_loss[img_id] += loss_cls.item()
        #-----------------------------------------------
        with torch.no_grad():
            pseudo_heatmaps = heatmap_model(acts)
            pseudo_heatmaps = torch.stack([pseudo_heatmaps[i]-torch.min(pseudo_heatmaps[i]) for i in range(b)], dim=0)
            pseudo_heatmaps = torch.stack([pseudo_heatmaps[i]/torch.max(pseudo_heatmaps[i]) for i in range(b)], dim=0)
        loss_hmap = heatmap_loss(pred_hmap, heatmaps, pseudo_heatmaps)    
        if (idx+1)%10==0:
            alp = alp*0.9
        loss = loss_cls + alp*loss_hmap
        # print(loss_cls, alp*loss_hmap)
        loss.backward()
        model_optimizer.step()
        
        train_loss += loss.item()
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})
        
def train_heatmap(epoch, train_loader):
    model.eval()
    heatmap_model.train()
    train_loss = 0
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
        images, labels, heatmaps = images.to(device), labels.to(device), heatmaps.to(device)
        # print(device)
        gen_optimizer.zero_grad()
        _, acts = model(images)
        pseudo_heatmaps = heatmap_model(acts)
        
        b, h, w = pseudo_heatmaps.shape
        pseudo_heatmaps = torch.stack([pseudo_heatmaps[i]-torch.min(pseudo_heatmaps[i]) for i in range(b)], dim=0)
        pseudo_heatmaps = torch.stack([pseudo_heatmaps[i]/torch.max(pseudo_heatmaps[i]) for i in range(b)], dim=0)
        
        loss_hmap = gen_loss(pseudo_heatmaps, heatmaps, None)
        # loss_hmap = gen_loss(pseudo_heatmaps, heatmaps)  
        loss_hmap.backward()
        gen_optimizer.step()
        
        train_loss += loss_hmap.item()
        pbar.set_postfix({'loss':train_loss/len(train_loader)})
    if (epoch+1)%10 == 0:
        torch.save({'gen':heatmap_model.state_dict()}, os.path.join(save_path,f'epoch{epoch}_heatmap_model.pt'))
#test---------------------------------------------------------------------
def test(epoch, best_acc, test_loader, mode):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for idx, (images, labels, _, img_id) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            
            loss = classif_loss(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
        if acc > best_acc:
            torch.save({'model':model.state_dict()}, os.path.join(save_path,f'{mode}_{epoch}_{acc:0.3f}_model.pt'))
            best_acc = acc
        return best_acc 

#data selection---------------------------------------------------------------------
def select(AVG_Loss, K=150, mode='loss'):
    new_selected = torch.topk(AVG_Loss, K, dim=-1)
    return new_selected.numpy()

#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    selected = np.array([])
    unselected = np.array([i for i in range(14349)])
    
    model = ResNet18(num_classes=30)
    # model_para = torch.load(args.model_para)
    # model.load_state_dict(model_para)
    model = model.to(device)
        
    model_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model_scheduler = MultiStepLR(model_optimizer, milestones=[30,80], gamma=0.1)    
    
    classif_loss = nn.CrossEntropyLoss()
    heatmap_loss = utils.heatmap_loss4()
    
    trainset = ilsvrc30(args.data_path, 'train', selected)
    valset = ilsvrc30(args.data_path, 'val', [])
    train_loader = DataLoader(trainset, args.batch_size, drop_last=True, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, args.batch_size, drop_last=True, shuffle=False, num_workers=2)
    
    print("train base model -----------------------------------------------------")
    best_acc = 0
    AVG_Loss = torch.zeros(len(trainset))
    for i in range(0, args.epoch):
        train(i, train_loader, AVG_Loss)
        best_acc = test(i, best_acc, val_loader, 'base')
        model_scheduler.step()
        print(torch.argsort(AVG_Loss, dim=-1)[:30])
        print(torch.argsort(AVG_Loss, dim=-1)[-30:])
    
    torch.save(AVG_Loss[:150], os.path.join(save_path, f'seed{args.seed}_AVG_Loss.pt'))
        
    #------------------------------------------------------------------------------
    model = ResNet18(num_classes=30)
    model = model.to(device)
    
    para_list = os.listdir(save_path)
    base_list = [path for path in para_list if 'base' in path]
    base_list.sort(key=lambda x : int(x.split('_')[1]))
    best_para = base_list[1]
    print(best_para)
    model_para = torch.load(os.path.join(save_path,best_para))
    model.load_state_dict(model_para['model'])
    
    classif_loss = nn.CrossEntropyLoss()
    heatmap_loss = utils.heatmap_loss4()
    
    selected = torch.load(os.path.join(save_path, f'seed{args.seed}_AVG_Loss.pt'))
    print('selected : ',selected)
    trainset = ilsvrc30(args.data_path, 'train', selected)
    train_loader = DataLoader(trainset, int(args.batch_size/4), drop_last=True, shuffle=True, num_workers=2)
    
    print("Select targets for supervision --------------------------------------")
    # selected, unselected = select(episode, unselected, selected, train_loader, K=1500, mode='random')
    selected = select(AVG_Loss, K=150)
    selected = selected.tolist()
    # print(selected)
    
    model_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model_scheduler = MultiStepLR(model_optimizer, milestones=[30,80], gamma=0.1)    
     
    trainset = ilsvrc30(args.data_path, 'train', selected)
    valset = ilsvrc30(args.data_path, 'val', [])
    testset = ilsvrc30(args.data_path, 'test', [])
    
    train_loader = DataLoader(trainset, args.batch_size, drop_last=True, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, args.batch_size, drop_last=True, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, args.batch_size, drop_last=False, shuffle=False, num_workers=2)
    
    print("tuning models ---------------------------------------------------------")
    best_acc = 0
    for i in range(0, args.epoch2):
        train2(i, train_loader)
        best_acc = test(i, best_acc, val_loader, 'tuning')
        model_scheduler.step()
    #------------------------------------------------------------------------------
    best_acc = 0
    para_list = os.listdir(save_path)
    best_para = para_list[-1]
    print(best_para)
    model_para = torch.load(os.path.join(save_path,best_para))
    model.load_state_dict(model_para['model'])
    
    test(-1, best_acc, test_loader, 'test')