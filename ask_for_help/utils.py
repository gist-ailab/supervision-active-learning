import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import matplotlib.pyplot as plt
from skimage.io import imshow
import skimage
from tqdm import tqdm
from pycocotools import mask as pymask
import pickle
from scipy import ndimage
from torchvision.ops import masks_to_boxes
import random

def collate_fn(batch):
    return tuple(zip(*batch))

def train(epoch, model, loader, criterion, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    max_entropy = 0
    min_entropy = 100
    for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if type(outputs)==dict:
            outputs = outputs['fc']
        entropy = torch.sum(-1 * torch.log(torch.softmax(outputs, dim=-1)) * torch.softmax(outputs, dim=-1), dim=-1)
        temp_min = torch.min(entropy)
        temp_max = torch.max(entropy)
        
        if temp_max > max_entropy:
            max_entropy = temp_max
        if temp_min < min_entropy:
            min_entropy = temp_min
        
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss = running_loss / len(loader)
    total_acc = 100 * running_acc / total
    print(f'Max Entropy : {max_entropy}, Min Entropy : {min_entropy}')
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

#only origin img has L_cls
def supervision_train1(epoch, model, s_loader, u_loader, criterion1, criterion2, optimizer, device, base_heatmap, ratio):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    count = 0
    
    data_len = len(s_loader)+len(u_loader)
    s_loader = iter(s_loader)
    u_loader = iter(u_loader)
    for i in tqdm(range(data_len)):
        if i%int(1/ratio)==0:
            inputs, labels, masks, _ = next(s_loader)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            f1, f2, f3, f4, out = outputs['l1'],outputs['l2'],outputs['l3'],outputs['l4'],outputs['fc']
            # print(f1.shape,f2.shape,f3.shape,f4.shape)
            masked_inputs = masking_input(inputs, masks, base_heatmap, device, mode='mask')
            m_outputs = model(masked_inputs)
            
            mf1, mf2, mf3, mf4, m_out = m_outputs['l1'],m_outputs['l2'],m_outputs['l3'],m_outputs['l4'],m_outputs['fc']
            # print(mf1.shape,mf2.shape,mf3.shape,mf4.shape)
            loss2 = (4-criterion2(f1, mf1).mean()-criterion2(f2, mf2).mean()-criterion2(f3, mf3).mean()-criterion2(f4, mf4).mean())/4
            # loss3 = criterion1(m_out, labels)
            
            _, pred = torch.max(out, 1)
            total += out.size(0)
            running_acc += (pred == labels).sum().item()
            out = out.float()
            loss1 = criterion1(out, labels)
            
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            inputs, labels, _, _ = next(u_loader)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            f1, f2, f3, f4, out = outputs['l1'],outputs['l2'],outputs['l3'],outputs['l4'],outputs['fc']
            
            _, pred = torch.max(out, 1)
            total += out.size(0)
            running_acc += (pred == labels).sum().item()
            out = out.float()
            loss = criterion1(out, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
    total_loss = running_loss / data_len
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

# masked img also has L_cls
def supervision_train2(epoch, model, loader, s_loader, criterion, criterion2, optimizer, device, base_heatmap):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    
    for _, (inputs, labels, masks, index) in enumerate(tqdm(s_loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        f1, f2, f3, f4 = outputs['l1'],outputs['l2'],outputs['l3'],outputs['l4']
        # f4 = outputs['fc']
        masked_inputs = masking_input(inputs, masks, base_heatmap, device, mode='mask')
        m_outputs = model(masked_inputs)
        mf1, mf2, mf3, mf4 = m_outputs['l1'],m_outputs['l2'],m_outputs['l3'],m_outputs['l4']
        # mf4 = m_outputs['fc']
        # loss2 = criterion(mf4, labels)
        # loss3 = criterion(f4, labels)
        # f1, f2, f3, f4 = torch.sigmoid(f1), torch.sigmoid(f1), torch.sigmoid(f1), torch.sigmoid(f1)
        # mf1, mf2, mf3, mf4 = torch.sigmoid(mf1), torch.sigmoid(mf1), torch.sigmoid(mf1), torch.sigmoid(mf1)
        loss = (4-criterion2(f1, mf1).mean()-criterion2(f2, mf2).mean()-criterion2(f3, mf3).mean()-criterion2(f4, mf4).mean())/4
        # loss = (1-criterion2(f4,mf4).mean())/2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if type(outputs)==dict:
                outputs = outputs['fc']
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    total_loss = running_loss / len(loader)
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
    
def train_detector(epoch, model, loader, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    train_loss_list = []
    running_acc = 0.0
    total = 0
    for _, (images, targets) in enumerate(tqdm(loader)):
        images = list(img.to(device) for img in images)
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
        # print(targets[0])
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total += len(images)
        train_loss_list.append(loss_value)
        losses.backward()
        optimizer.step()
    # print(train_loss_list)
    # print(total)
    return sum(train_loss_list)/total
    
def test(epoch, model, loader, criterion, device, bestAcc, spath):
    print('\nEpoch: %d'%epoch)
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if type(outputs)==dict:
                outputs = outputs['fc']
            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        total_loss = running_loss / len(loader)
        total_acc = 100 * running_acc / total
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
        if total_acc > bestAcc:
            torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
            return total_acc
        else:
            return bestAcc
        
def val_detector(epoch, model, loader, device, bestAcc, spath):
    print('\nEpoch: %d'%epoch)
    # model.eval()
    val_loss_list = []
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        for _, (images, targets) in enumerate(tqdm(loader)):
            images = list(img.to(device) for img in images)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            # model.eval()
            # pred = model(images)
            # print(pred[0])
            # model.train()
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            total += len(images)
            val_loss_list.append(loss_value)
        return sum(val_loss_list)/total

def CAM(feats, weights, c_idx, height=224, width=224):
    h, w = 14,14
    feat = torch.reshape(feats, [-1, 2048, h*w])

    weight = weights[c_idx]
    weight = weight.unsqueeze(0).unsqueeze(0)
    cam = torch.bmm(weight, feat)
    cam = torch.reshape(cam, (1,h,w))
    cam = cam-torch.min(cam)
    cam = cam/torch.max(cam)
    cam = cam.unsqueeze(1)
    
    cam = F.interpolate(cam, size=(height,width), mode='bilinear')
    cam = cam.squeeze()
    return cam

def data_selection(model, loader, criterion, ratio=0.1, mode='random'):
    selected = []
    unselected = []
    for i, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
        if mode=='random':
            unselected.append(index)
    if mode=='random':
        random.shuffle(unselected)
        data_len = len(unselected)
        selected = unselected[:int(ratio*data_len)]
        unselected = unselected[int(ratio*data_len):]
        
    return selected, unselected

def masking_input(inputs, masks, base_heatmap, device, mode='point'):
    if mode=='point':
        masked_inputs = []
        for img,mask in zip(inputs,masks):
            mask_np = mask.numpy()
            center = ndimage.center_of_mass(mask_np)
            center = center[1:]
            # print(center)
            # print(torch.tensor(center, dtype=int).shape)
            heatmap = torch.clone(base_heatmap)
            heatmap = (heatmap - torch.tensor(center, dtype=int))/112
            heatmap = heatmap*heatmap
            heatmap = torch.sum(heatmap, dim=-1)
            heatmap = -1*heatmap/(1.5)**2
            heatmap = torch.exp(heatmap)
            heatmap = heatmap*heatmap
            heatmap = heatmap.to(device)
            masked_inputs.append(img*heatmap)
        masked_inputs = torch.stack(masked_inputs)
        
    elif mode=='box':
        boxes = masks_to_boxes(masks)
        cropped_imgs = []
        for img, box in zip(inputs, boxes):
            cropped_img = F2.crop(img, box[1],box[0],box[3]-box[1],box[2]-box[0])
            background = torch.zeros_like(img)
            background[:,box[1]:box[3],box[0]:box[2]] = cropped_img
            cropped_imgs.append(background)
        masked_inputs = torch.stack(cropped_imgs)
        
    elif mode=='mask':
        masks = masks.to(device)
        masks[masks==0] = 0.0
        # print(inputs.shape)
        # print(masks.shape)
        masked_inputs = inputs * masks
        # print(masked_inputs.shape)

    return masked_inputs