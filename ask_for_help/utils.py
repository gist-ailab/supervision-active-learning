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

def activation_map_matching(epoch, model, s_loader, criterion, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch)
    running_loss1 = 0.0
    running_loss2 = 0.0
    total = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            feats = outputs['l4']
            norm_feats = []
            for I, feat, mask in zip(inputs, feats, masks):
                I = I.unsqueeze(0)
                mask = mask.squeeze()
                h, w = I.shape[-2], I.shape[-1]
                f_h, f_w = feat.shape[-2], feat.shape[-1]

                am = gen_am(I, model, device)
                _, center = get_point(mask, am, tr=0.5)
                encoded_center = [int(center[0]*7/h), int(center[1]*7/w)]
                
                mean_feat = torch.mean(feat, dim=0)
                # mean_feat = mean_feat.unsqueeze(0).unsqueeze(0)
                # mean_feat = F.interpolate(mean_feat, size=(5,5), mode='bilinear')
                norm_feat = (mean_feat-torch.min(mean_feat))/(torch.max(mean_feat)-torch.min(mean_feat))
                norm_feat = norm_feat.squeeze()
                norm_feat[norm_feat>0.5] = 1
                norm_feat[norm_feat<0.5] = 0
                norm_feat[encoded_center[1], encoded_center[0]] = 1
                norm_feats.append(norm_feat)
        norm_feats = torch.stack(norm_feats)
        norm_feats = norm_feats.to(device)
        # print("Shape : ",norm_feats.shape)
        
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        feats = outputs['l4']
        outputs = outputs['fc']
        pred_feats = []
        for feat in feats:
            mean_feat = torch.mean(feat, dim=0)
            # mean_feat = mean_feat.unsqueeze(0).unsqueeze(0)
            # mean_feat = F.interpolate(mean_feat, size=(5,5), mode='bilinear')
            mean_feat = mean_feat.squeeze()
            # mean_feat = torch.sigmoid(mean_feat)
            norm_feat = (mean_feat-torch.min(mean_feat))/(torch.max(mean_feat)-torch.min(mean_feat))
            pred_feats.append(norm_feat)
        pred_feats = torch.stack(pred_feats)
        pred_feats = pred_feats.to(device)

        _, label = torch.max(outputs, 1)
        # print(preds.float())
        loss1 = criterion(outputs, labels)
        loss2 = criterion2(pred_feats, norm_feats)
        
        # print(loss1.item(), loss2.item())
        # print(label[:5])
        loss = loss1 + 1000*loss2
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", 1000*running_loss2 / total)

def box_matching(epoch, model, model2, s_loader, criterion, criterion2, optimizer, device, device2):
    print('\nEpoch: %d'%epoch)
    model.train()
    model2.eval()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        enc_mask = F.interpolate(masks, size=(7,7), mode='bilinear').squeeze() # b,7,7
        outputs = model(inputs)
        feat = outputs['l4']
        outputs = outputs['fc']
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)
        loss2 = 0
        for I, label, mask in zip(inputs, labels, masks):
            bbox = gen_pseudo_box_by_CAM(I, mask.squeeze(0), model2, device2)
            cams = gen_softmax_cams(I, model, device)
            cam = cams[label]
            cam_lbl = torch.zeros_like(cam).to(device)
            cam_lbl[bbox[0]:bbox[2],bbox[1]:bbox[3]] = 1
            loss2 += criterion2(cam, cam_lbl)
        loss = loss1 + 0.01*loss2
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += 0.01*loss2.item()
    total_acc = 100 * running_acc / total
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)

def box_feat_matching(epoch, model, model2, s_loader, criterion, criterion2, optimizer, device, device2):
    print('\nEpoch: %d'%epoch)
    model.train()
    model2.eval()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_acc = 0.0
    total = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        enc_mask = F.interpolate(masks, size=(7,7), mode='bilinear').squeeze() # b,7,7
        outputs = model(inputs)
        # feat11, feat12, feat13, feat14 = outputs['l1'],outputs['l2'],outputs['l3'],outputs['l4']
        feat14 = outputs['l4']
        outputs = outputs['fc']
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)
        cropped_imgs = []
        for I, label, mask in zip(inputs, labels, masks):
            bbox = gen_pseudo_box_by_CAM(I, mask.squeeze(0), model2, device2)
            # print(I.shape)
            cropped_img = F2.resized_crop(I.unsqueeze(0), bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],size=(224,224))
            
            # cropped_img = torch.zeros_like(I.unsqueeze(0))
            # cropped_img[:,:,bbox[0]:bbox[2],bbox[1]:bbox[3]] = F2.crop(I.unsqueeze(0).unsqueeze(0), bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
            cropped_img = cropped_img.squeeze()
            cropped_imgs.append(cropped_img)
        cropped_imgs = torch.stack(cropped_imgs).to(device)
        outputs2 = model(cropped_imgs)
        # feat21, feat22, feat23, feat24 = outputs2['l1'],outputs2['l2'],outputs2['l3'],outputs2['l4']
        feat24 = outputs2['l4']
        outputs2 = outputs2['fc']

        # feat14 = torch.view(-1, feat14.shape[1], feat14.shape[2]*feat14.shape[3])
        feat14 = F.adaptive_avg_pool2d(feat14, (1,1)) # b, 2048, 1, 1
        feat14 = torch.flatten(feat14, 1) # b, 2048
        feat14 = F.normalize(feat14, p=2.0, dim=1) # b, 2048

        feat24 = F.adaptive_avg_pool2d(feat24, (1,1)) # b, 2048, 1, 1
        feat24 = torch.flatten(feat24, 1) # b, 2048
        feat24 = F.normalize(feat24, p=2.0, dim=1) 

        loss2 = criterion(outputs2, labels)
        
        # loss3 = 1-torch.mean(criterion2(outputs,outputs2)) +\
        #         1-torch.mean(criterion2(feat11, feat21)) +\
        #         1-torch.mean(criterion2(feat12, feat22)) +\
        #         1-torch.mean(criterion2(feat13, feat23)) +\
        #         1-torch.mean(criterion2(feat14, feat24))

        # print(feat14.shape, feat24.shape)
        loss3 = 1-torch.mean(criterion2(feat14,feat24))

        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        running_loss3 += loss3.item()
    total_acc = 100 * running_acc / total
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)
    print("Loss 3 : ", running_loss3 / total)

def box_masking(epoch, model, model2, s_loader, criterion, criterion2, optimizer, device, device2):
    print('\nEpoch: %d'%epoch)
    model.train()
    model2.eval()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    running_acc = 0.0
    total = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        enc_mask = F.interpolate(masks, size=(7,7), mode='bilinear').squeeze() # b,7,7
        outputs = model(inputs)
        feat11, feat12, feat13, feat14 = outputs['l1'],outputs['l2'],outputs['l3'],outputs['l4']
        # feat14 = outputs['l4']
        outputs = outputs['fc']
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)
        masked_imgs = []
        for I, label, mask in zip(inputs, labels, masks):
            bbox = gen_pseudo_box_by_CAM(I, mask.squeeze(0), model2, device2)
            # print("I shape : ",I.shape)
            masked_img = I.clone()
            masked_img[:,bbox[0]:bbox[2],bbox[1]:bbox[3]] *= 0.1 
            masked_img = masked_img.squeeze()
            masked_imgs.append(masked_img)
        masked_imgs = torch.stack(masked_imgs).to(device)
        outputs2 = model(masked_imgs)
        feat21, feat22, feat23, feat24 = outputs2['l1'],outputs2['l2'],outputs2['l3'],outputs2['l4']
        # feat24 = outputs2['l4']
        # outputs2 = outputs2['fc']

        feat11 = F.normalize(feat11, p=2.0, dim=1) # b, 2048
        feat12 = F.normalize(feat12, p=2.0, dim=1) # b, 2048
        feat13 = F.normalize(feat13, p=2.0, dim=1) # b, 2048
        feat14 = F.normalize(feat14, p=2.0, dim=1) # b, 2048

        feat21 = F.normalize(feat21, p=2.0, dim=1) 
        feat22 = F.normalize(feat22, p=2.0, dim=1) 
        feat23 = F.normalize(feat23, p=2.0, dim=1) 
        feat24 = F.normalize(feat24, p=2.0, dim=1) 

        # loss2 = criterion(outputs2, labels)
        
        loss3 = 1-torch.mean(criterion2(feat11, feat21)) +\
                1-torch.mean(criterion2(feat12, feat22)) +\
                1-torch.mean(criterion2(feat13, feat23)) +\
                1-torch.mean(criterion2(feat14, feat24))

        # print(feat14.shape, feat24.shape)
        # loss3 = 1-torch.mean(criterion2(feat14,feat24))

        # loss = loss1 + loss2 + loss3
        loss = loss1 + + loss3
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        # running_loss2 += loss2.item()
        running_loss3 += loss3.item()
    total_acc = 100 * running_acc / total
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    # print("Loss 2 : ", running_loss2 / total)
    print("Loss 3 : ", running_loss3 / total)

def point_regression(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    reg_head.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        feat = outputs['l4'] # b,2048,7,7
        outputs = outputs['fc'] 
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)
        
        gt_points = []
        for mask in masks:
            mask = mask.squeeze(0)
            m_point, _ = get_center_box(mask, mode='max')
            m_point /= torch.tensor(mask.shape)
            gt_points.append(m_point)
        gt_points = torch.stack(gt_points)
        gt_points = gt_points.to(device)
        feat = reg_head(feat) # b,2,1,1
        reg_outputs = torch.flatten(feat,1)
        reg_outputs = reg_outputs.float()
        gt_points = gt_points.float()
        loss2 = criterion2(reg_outputs, gt_points)
        
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    total_acc = 100 * running_acc / total
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)

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

def get_center_box(map, tr=0.5, mode='all'):
    map = map.clone()
    map[map>=tr]=1
    map[map<tr] = 0
    lbl_0 = skimage.measure.label(map.detach().cpu().numpy())
    props = skimage.measure.regionprops(lbl_0)
    if mode=='all':
        points = []
        boxes = []
        for prop in props:
            cy, cx = prop.centroid
            minr, minc, maxr, maxc = prop.bbox
            boxes.append([minr, minc, maxr-1, maxc-1]) # y_min, x_min, y_max, x_max
            points.append(torch.tensor([cx,cy]))
    if mode=='max':
        points = None
        boxes = None
        area = 0
        for prop in props:
            if prop.area > area:
                area = prop.area
                cy, cx = prop.centroid
                minr, minc, maxr, maxc = prop.bbox
                points = torch.tensor([cx,cy])
                boxes = [minr, minc, maxr, maxc]
    return points, boxes


def gen_cams(inputs, model, device, interpolate=False):
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    f4 = outputs['l4']
    height, width = inputs.shape[-2], inputs.shape[-1]
    feat = torch.reshape(f4, [-1,2048,7*7])
    weights = list(model.parameters())[-2]
    cams = []
    for i in range(len(weights)):
        weight = weights[i]
        weight = weight.unsqueeze(0).unsqueeze(0)
        cam = torch.bmm(weight, feat)
        cam = torch.reshape(cam, (1,7,7))
        cam = cam.unsqueeze(1)
        if interpolate==True:
            cam = F.interpolate(cam, size=(height, width), mode='bicubic')
        cam = cam-torch.min(cam)
        cam = cam/torch.max(cam)
        cams += cam
    cams = torch.stack(cams)
    cams = cams.squeeze()
    return cams

def gen_softmax_cams(inputs,model,device):
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    f4 = outputs['l4']

    height, width = inputs.shape[-2], inputs.shape[-1]
    feat = torch.reshape(f4, [-1,2048,7*7])
    weights = list(model.parameters())[-2]
    biases = list(model.parameters())[-1]
    cams = []
    for i in range(len(weights)):
        weight = weights[i]
        bias = biases[i]
        weight = weight.unsqueeze(0).unsqueeze(0)
        cam = torch.bmm(weight, feat)
        # print(cam.shape)
        cam = torch.reshape(cam, (1,7,7))
        cam = cam.unsqueeze(1)+bias
        cam = F.interpolate(cam, size=(height, width), mode='bicubic')
        cam = torch.softmax(cam, dim=-1)
        # cam = cam.squeeze()
        cams.append(cam)
    cams = torch.stack(cams)
    cams = cams.squeeze()
    return cams

def find_nearest_points(points1, points2): # points1 : Mask, points2 : CAM
    dist_list = torch.cdist(points1, points2, p=2)
    pair_list = []
    for i, dists in enumerate(dist_list):
        idx = torch.argmin(dists)
        pair_list.append(torch.tensor([i, idx])) # idx of points
    return pair_list

def gen_pseudo_box(center, bbox):
    # top_left = torch.tensor([bbox[1], bbox[0]], dtype=torch.float) # min_x, min_y
    # top_right = torch.tensor([bbox[3], bbox[0]], dtype=torch.float) # max_x, min_y
    # bottom_left = torch.tensor([bbox[1], bbox[2]], dtype=torch.float) # min_x, max_y
    # bottom_right = torch.tensor([bbox[3], bbox[2]], dtype=torch.float) # max_x, max_y

    top_left = torch.tensor([bbox[1], bbox[0]], dtype=torch.double) # min_x, min_y
    top_right = torch.tensor([bbox[3], bbox[0]], dtype=torch.double) # max_x, min_y
    bottom_left = torch.tensor([bbox[1], bbox[2]], dtype=torch.double) # min_x, max_y
    bottom_right = torch.tensor([bbox[3], bbox[2]], dtype=torch.double) # max_x, max_y
    box_points = torch.stack([top_left,top_right,bottom_left,bottom_right])

    dist_list = torch.cdist(center.unsqueeze(0), box_points)
    box_idx = torch.argmax(dist_list)
    Farthest_point = box_points[box_idx]
    dx = torch.abs(center[0]-Farthest_point[0])
    dy = torch.abs(center[1]-Farthest_point[1])
    cx, cy = center
    minr, minc, maxr, maxc = cy-dy, cx-dx, cy+dy, cx+dx 
    if minr < 0: minr = 0
    if minc < 0: minc = 0
    if maxr > 223: maxr = 223
    if maxc > 223: maxc = 223
    return torch.tensor([int(minr), int(minc), int(maxr), int(maxc)])

def gen_pseudo_box_by_CAM(inputs, mask, model, device):
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    outputs = outputs['fc']
    _, pred = torch.max(outputs, 1)
    cams = gen_cams(inputs,model,interpolate=True,device=device)
    cam = cams[pred.item()]
    # cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224,224), mode='bilinear').squeeze()
    m_point, m_box = get_center_box(mask, mode='max')
    c_points, c_boxes = get_center_box(cam, tr=0.5)
    points_pair = find_nearest_points(m_point.unsqueeze(0), torch.stack(c_points))
    s_cam_point, s_cam_box = c_points[points_pair[0][1]], c_boxes[points_pair[0][1]]
    pseudo_box = gen_pseudo_box(m_point, s_cam_box)
    return pseudo_box

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
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            total += len(images)
            val_loss_list.append(loss_value)
        return sum(val_loss_list)/total

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