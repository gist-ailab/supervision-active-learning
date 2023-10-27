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
    # max_entropy = 0
    # min_entropy = 100
    for _, (inputs, labels, _, _) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if type(outputs)==dict:
            outputs = outputs['fc']
        # entropy = torch.sum(-1 * torch.log(torch.softmax(outputs, dim=-1)) * torch.softmax(outputs, dim=-1), dim=-1)
        # temp_min = torch.min(entropy)
        # temp_max = torch.max(entropy)
        
        # if temp_max > max_entropy:
        #     max_entropy = temp_max
        # if temp_min < min_entropy:
        #     min_entropy = temp_min
        
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
    # print(f'Max Entropy : {max_entropy}, Min Entropy : {min_entropy}')
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
def activation_map_matching(epoch, model, s_loader, criterion, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss = 0.0
    count = [0,0,0]
    pred_count = [0,0,0]
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs['fc']
        _, label = torch.max(outputs, 1)
        # print(preds.float())
        loss1 = criterion(outputs, labels)
        box_1x = [0,0,inputs.shape[2]-1,inputs.shape[3]-1]
        loss2 = 0
        for I, label, pred, mask in zip(inputs, labels, label, masks):
            # print(torch.max(mask))
            count[label] += 1
            pred_count[pred] += 1
            I = I.unsqueeze(0)
            mask = mask.squeeze()
            am = gen_am(I, model, device)
            
            model.eval()
            # with torch.no_grad():
            _, center = get_point(mask, am, tr=0.5)
            scaled_cropped_img_2x, box_2x = get_scaled_cropped_img(I, center, scale=1.2)
            scaled_cropped_img_3x, box_3x = get_scaled_cropped_img(I, center, scale=1.6)
            
            scaled_caam_2x = gen_am(scaled_cropped_img_2x, model, device)
            scaled_caam_3x = gen_am(scaled_cropped_img_3x, model, device)
            
            caams = [am, scaled_caam_2x,scaled_caam_3x]
            boxes = [box_1x,box_2x,box_3x]
            
            # caams = [am, scaled_caam_3x]
            # boxes = [box_1x,box_3x]
            
            concated_am = concat_ams(caams, boxes, device)
            concated_am = concated_am.clone().detach()
            loss2 += criterion2(concated_am, am)
        # print(loss1.item(), 10*loss2.item())
        # print(label[:5])
        loss = loss1 + loss2
        # loss = loss1
        # loss = loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('count : ',count)
    print('preds : ', pred_count)

def front_back_classification(epoch, model, s_loader, criterion, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss = 0.0
    count = [0,0,0]
    pred_count = [0,0,0]
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs['fc']
        feat = outputs['layer4']
        _, label = torch.max(outputs, 1)
        loss1 = criterion(outputs, labels)
        

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
            torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
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

def data_selection(model, loader, criterion, device, ratio=0.1, mode='random'):
    selected = []
    unselected = []
    entropy_list = []
    total = 0
    model.eval()
    for i, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
        total += 1
        if mode=='random':
            unselected.append(index)
        if mode=='wo_domi_cls':
            if labels[0] != 2:
                selected.append(index)
            else:
                unselected.append(index)
        inputs = inputs.to(device)
        output = model(inputs)
        output = output['fc']
        _, pred = torch.max(output, -1)
        entropy = criterion(output, pred)
        entropy = entropy.detach().cpu().item()
        entropy_list.append([entropy, index, labels])
    # print(total)
    if mode=='random':
        random.shuffle(unselected)
        data_len = len(unselected)
        selected = unselected[:int(ratio*data_len)]
        unselected = unselected[int(ratio*data_len):]
    if mode=='high_entropy':
        entropy_list.sort(key=lambda x : x[0])
        entropy_list.reverse()
        selected = [entropy_list[i][1] for i in range(int(len(entropy_list)*ratio))]
    if mode=='low_entropy':
        entropy_list.sort(key=lambda x : x[0])
        selected = [entropy_list[i][1] for i in range(int(len(entropy_list)*ratio))]
        return selected, entropy_list
    if mode=='class_balance':
        count = [0,0,0]
        entropy_list.sort(key=lambda x : x[0])
        for datas in entropy_list:
            idx, labels = datas[1], datas[2]
            # print(int(labels.item()))
            if count[int(labels.item())] < int(total*ratio/3):
                # 
                selected.append(idx)
                count[int(labels.item())] += 1
            if sum(count) == int(total*ratio):
                break
        print(count)
    return selected, entropy_list

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

def metric(model, loader, num_classes, device):
    model.eval()
    class_AP = dict()
    class_AR = dict()
    class_ARoverAP = dict()
    confusion_matrix = torch.zeros([num_classes, num_classes])
    for i, (imgs, labels, _, _) in enumerate(tqdm(loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        if type(outputs)==dict:
            outputs = outputs['fc']
        _, preds = torch.max(outputs, 1)
        
        for j in range(len(labels)):
            label = labels[j]
            pred = preds[j]
            confusion_matrix[label][pred] += 1
    
    for i in range(num_classes):
        TP = confusion_matrix[i][i]
        FP = torch.sum(confusion_matrix[:,i]) - TP
        FN = torch.sum(confusion_matrix[i]) - TP
        class_AP[i] = (100*TP / (TP + FP)).item()
        class_AR[i] = (100*TP / (TP + FN)).item()
        class_ARoverAP[i] = ((TP + FN) / (TP + FP)).item()
    print(confusion_matrix)
    # print(f'Class AP/AR: {class_ARoverAP},\n mAP/mAR : {torch.sum(torch.stack(list(class_ARoverAP.values())))/num_classes}')
    mAP = sum(class_AP.values())/num_classes+1e-6
    mAR = sum(class_AR.values())/num_classes+1e-6
    print('mAP : ',mAP)
    print('mAR : ', mAR)
    print('F1 : ', 2*(mAP * mAR)/(mAP + mAR))
    
#---------------------------------------------------------------------

def get_point(Mask, CAM, tr=0.7):
    # print(Mask.shape)
    # print(CAM.shape)
    assert Mask.shape==CAM.shape 
    binary_cam = CAM.clone()
    binary_cam[binary_cam>=tr] = 1
    binary_cam[binary_cam<tr] = 0
    remain = Mask * (1 - binary_cam)
    lbl_0 = skimage.measure.label(remain.detach().cpu().numpy())
    props = skimage.measure.regionprops(lbl_0)
    if len(props)==0:
        lbl_0 = skimage.measure.label(Mask.detach().cpu().numpy())
        props = skimage.measure.regionprops(lbl_0)
    max_contour = None
    max_area = 0
    for prop in props:
        area = prop.area
        if area > max_area:
            max_area = area
            max_contour = prop
            
    cy, cx = max_contour.centroid
    return binary_cam, (cx, cy)

# img, point, scale로 crop 된 이미지 생성
def get_scaled_cropped_img(img, point, scale=2.0):
    height, width = img.shape[-2], img.shape[-1]
    s_height, s_width = height/scale, width/scale
    lx, rx = int(point[0] - s_width/2), int(point[0] + s_width/2)
    ty, by = int(point[1] - s_height/2), int(point[1] + s_height/2)
    
    if lx <= 0: lx = 0
    if ty <= 0: ty = 0
    if rx >= width: rx = width-1
    if by >= height: by = height-1
    scaled_cropped_img = F2.resized_crop(img, ty, lx, by-ty, rx-lx, [height, width])
    # print(scaled_cropped_img.shape)
    return scaled_cropped_img, [ty, lx, by-ty, rx-lx]

def gen_caam(inputs, model, device):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    f4 = outputs['l4']

    height, width = inputs.shape[-2], inputs.shape[-1]
    feat = torch.reshape(f4, [-1,2048,7*7])
    weights = list(model.parameters())[-2]
    caam = torch.zeros((height, width)).to(device)
    for i in range(len(weights)):
        weight = weights[i]
        weight = weight.unsqueeze(0).unsqueeze(0)
        cam = torch.bmm(weight, feat)
        # print(cam.shape)
        cam = torch.reshape(cam, (1,7,7))
        cam = cam-torch.min(cam)
        cam = cam/torch.max(cam)
        # cam = torch.sigmoid(cam)
        cam = cam.unsqueeze(1)
        # print(caam[:1,:1])
        cam = F.interpolate(cam, size=(height, width), mode='bilinear')
        cam = cam.squeeze()
        caam += cam
        
    caam = caam-torch.min(caam)
    caam = caam/torch.max(caam)
    return caam

# Activation map
def gen_am(inputs, model, device):
    inputs = inputs.to(device)
    outputs = model(inputs)
    f4 = outputs['l4']

    height, width = inputs.shape[-2], inputs.shape[-1]
    AM = torch.avg_pool1d(f4.squeeze().permute(1,2,0), 2048)
    AM = AM.squeeze()
    AM = AM.unsqueeze(0).unsqueeze(0)
    AM = F.interpolate(AM, size=(height, width), mode='bilinear')
    
    AM = torch.softmax(AM, dim=-1)
    AM = AM-torch.min(AM)
    AM = AM/torch.max(AM)
    
    AM = AM.squeeze()
    # print(AM.shape)
    return AM

# zoom된 caam을 기존 caam에 concat하여 하나로 합침
def concat_ams(caams, boxes, device):
    height, width = caams[0].shape[-2], caams[0].shape[-1]
    final_caam = torch.zeros([height, width]).to(device)
    for box, caam in zip(boxes, caams):
        h, w = box[2], box[3]
        caam = F2.resize(caam.unsqueeze(0).unsqueeze(0), [h, w])
        caam = caam.squeeze()
        final_caam[box[0]:box[0]+box[2], box[1]:box[1]+box[3]] += caam
        final_caam = torch.clamp(final_caam, min=0, max=1.0)
    return final_caam

