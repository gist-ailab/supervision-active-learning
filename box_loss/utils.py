import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet import *

def get_rand_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})

def test(epoch, model, test_loader, criterion, save_path, sign, device, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
        if not os.path.isdir(os.path.join(save_path,sign)):
            os.mkdir(os.path.join(save_path,sign))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_path,sign,'model.pt'))
    return acc

def query_test(epoch, model, test_loader, criterion, save_path, sign, device, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
        if not os.path.isdir(os.path.join(save_path,sign)):
            os.mkdir(os.path.join(save_path,sign))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_path,sign,'query_model.pt'))
    return acc
    
def query_algorithm(model, criterion, ulbl_loader, ulbl_idx, device, model_paths, K):
    model_dict = dict()
    for i in range(len(model_paths)):
        model_dict[i] = ResNet18().to(device)
        model_dict[i].load_state_dict(torch.load(model_paths[i]))
    
    conf_list = torch.tensor([]).to(device)
    with torch.no_grad():
        pbar = tqdm(ulbl_loader)
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            temp_tensor = torch.tensor([]).to(device)
            for j in range(len(model_dict)):
                outputs = model_dict[j](inputs)
                confidence = torch.max(F.softmax(outputs, dim=1),dim=1)
                if len(temp_tensor)==0:
                    temp_tensor = torch.cat((temp_tensor, confidence.values), 0)
                else:
                    temp_tensor = temp_tensor + confidence.values
            conf_list = torch.cat((conf_list,temp_tensor),0)
        arg = conf_list.argsort().cpu().numpy()
    return list(arg[:K])

class heatmap_loss(nn.Module):
    def __init__(self, N=1, a=2, b=4):
        super(heatmap_loss, self).__init__()
        self.N = N
        self.a = a
        self.b = b
        self.e = 1e-10
    
    def forward(self, Y_pred, Y_gt):
        total_loss = 0
        for b_idx in range(len(Y_gt)):
            if len(Y_gt[b_idx]) == 0:
                total_loss += 0
            else:
                gt = ((Y_gt[b_idx]==torch.max(Y_gt[b_idx])).nonzero())[0]
                Y_temp = torch.pow(1-Y_gt[b_idx],self.b)*torch.pow(Y_pred[b_idx],self.a)*(torch.log(1-Y_pred[b_idx]+self.e))
                Y_temp[gt[0],gt[1]] = ((1-Y_pred[b_idx][gt[0],gt[1]])**self.a)*(torch.log(1-Y_pred[b_idx][gt[0],gt[1]]+self.e))
                total_loss += torch.sum(Y_temp)
        return -1/self.N * total_loss
    
class heatmap_loss2(nn.Module):
    def __init__(self, N=1, a=2, b=4):
        super(heatmap_loss2, self).__init__()
        self.N = N
        self.a = a
        self.b = b
        self.e = 1e-10
    
    def forward(self, Y_pred, Y_gt):
        total_loss = 0
        for b_idx in range(len(Y_gt)):
            if len(Y_gt[b_idx]) == 0:
                total_loss += 0
            else:
                gt = ((Y_gt[b_idx]==torch.max(Y_gt[b_idx])).nonzero())[0]
                Y_temp = 0.01*torch.pow(1-Y_gt[b_idx],self.b)*(torch.log(1-Y_pred[b_idx]+self.e))
                Y_temp[gt[0],gt[1]] = ((1-Y_pred[b_idx][gt[0],gt[1]])**self.a)*(torch.log(1-Y_pred[b_idx][gt[0],gt[1]]+self.e))
                total_loss += torch.sum(Y_temp)
        return -1/self.N * total_loss
    
class heatmap_loss3(nn.Module):
    def __init__(self, a=1):
        super(heatmap_loss3, self).__init__()
        self.a = a
        self.e = 1e-6
        self.tr = 0.1
    
    def forward(self, Y_pred, Y_gt):
        total_loss = 0
        for b_idx in range(len(Y_gt)):
            if len(Y_gt[b_idx]) == 0:
                total_loss += 0
            else:
                gt = ((Y_gt[b_idx]==torch.max(Y_gt[b_idx])).nonzero())[0]
                min_pred = torch.min(Y_pred[b_idx])
                max_pred = torch.max(Y_pred[b_idx])
                Y_g = Y_gt[b_idx]
                Y_p = Y_pred[b_idx].clone()
                Y_p = (Y_pred - min_pred)/(max_pred-min_pred)
                Y_temp = Y_p * Y_g
                Y_temp = torch.where(Y_temp > self.tr, 1, 0)
                total_loss += torch.sum(Y_temp)
        return 1e7/(self.e+total_loss)
    
class heatmap_loss4(nn.Module):
    def __init__(self, a=1):
        super(heatmap_loss4, self).__init__()
        self.a = a
        self.e = 1e-6
        self.tr = 0.1
    
    def forward(self, Y_pred, Y_gt):
        total_loss = 0
        N = 0
        for b_idx in range(len(Y_gt)):
            if torch.max(Y_gt[b_idx])==0:
                total_loss += torch.tensor([0])
            else:
                N += 1
                Y_g = Y_gt[b_idx]
                Y_p = Y_pred[b_idx].clone()
                print(torch.max(Y_p))
                print(torch.min(Y_p[Y_p>0]))
                positive = torch.sum(Y_g*Y_p)
                negative = torch.sum((1-Y_g)*Y_p)
                total_loss += negative/(positive+self.e)
        if N == 0: N = 1
        if total_loss == 0:
            return 0.0
        else:
            return torch.log(total_loss)/N

def get_box_pos(heatmap, T):
    Y_g = torch.where(heatmap > T, heatmap, 0)
    for i in range(len(Y_g)):
        if torch.sum(Y_g[i]) > 0:
            left = i
            break
    for i in range(len(Y_g)):
        if torch.sum(Y_g[len(Y_g)-1-i]) > 0:
            right = i
            break
    for i in range(len(Y_g[0])):
        if torch.sum(Y_g[:,i]) > 0:
            up = i
            break
    for i in range(len(Y_g[0])):
        if torch.sum(Y_g[:,len(Y_g[0])-1-i]) > 0:
            down = i
            break
    x_r = (right-left)//2
    y_r = (down-up)//2
    radi = torch.tensor([x_r, y_r])
    gt = torch.tensor([left + x_r, up+y_r])
    return radi, gt

def make_heatmap(radi, gt, sigma = 1/3):
    heatmap = torch.zeros((256,256,2))
    for x in range(256):
        for y in range(256):
            heatmap[x,y,:] = torch.tensor([y,x])
    heatmap = (heatmap - gt)/radi
    heatmap = heatmap * heatmap
    heatmap = heatmap * heatmap
    heatmap = torch.sum(heatmap, dim=-1)
    heatmap = -1*heatmap/(2*sigma)**2
    heatmap = torch.exp(heatmap)
    heatmap = torch.where(heatmap > 0.1, 1.0, 0.0)
    return heatmap

class heatmap_loss5(nn.Module):
    def __init__(self, a=1):
        super(heatmap_loss5, self).__init__()
        self.a = a
        self.e = 1e-6
        self.tr = 0.1
    
    def forward(self, Y_pred, Y_gt):
        total_loss = 0
        N = 0
        for b_idx in range(len(Y_gt)):
            N += 1
            Y_p = Y_pred[b_idx].clone()
            
            if torch.max(Y_gt[b_idx])==0:
                radi, gt = get_box_pos(Y_p, T=0.5)
                Y_g = make_heatmap(radi, gt)
                save_image(Y_g, './Y_g_heatmap.png')
            else:
                Y_g = Y_gt[b_idx]
            
            positive = torch.sum(Y_g*Y_p)
            negative = torch.sum((1-Y_g)*Y_p)
            total_loss += negative/(positive+self.e)
        if N == 0: N = 1
        if total_loss == 0:
            return 0.0
        else:
            return torch.log(total_loss)/N