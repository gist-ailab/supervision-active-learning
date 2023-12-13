import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import matplotlib.pyplot as plt
from skimage.io import imshow
import skimage
from tqdm import tqdm
import pickle
from scipy import ndimage
from torchvision.ops import masks_to_boxes
from kcenterGreedy import kCenterGreedy
import random

def collate_fn(batch):
    return tuple(zip(*batch))

def train(epoch, model, loader, criterion, optimizer, device):
    print('\nEpoch: %d'%epoch)
    if type(model)==dict:
        model = model['backbone']
    if type(optimizer)==dict:
        optimizer = optimizer['backbone']    
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for _, (inputs, labels, _, _) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    total_loss = running_loss / total
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

def test(epoch, model, loader, criterion, device, minLoss, spath):
    print('\nEpoch: %d'%epoch)
    if type(model)==dict:
        model = model['backbone']
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    with torch.no_grad():
        for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        total_loss = running_loss / total
        total_acc = 100 * running_acc / total
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
        if total_loss < minLoss and epoch!=-1:
            torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
            torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
            return total_loss
        else:
            return minLoss

def metric(model, loader, num_classes, device):
    model.eval()
    class_AP = dict()
    class_AR = dict()
    class_ARoverAP = dict()
    confusion_matrix = torch.zeros([num_classes, num_classes])
    for i, (imgs, labels, _, _) in enumerate(tqdm(loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs, _ = model(imgs)
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
        class_AP[i] = (100*TP / (TP + FP + 1e-6)).item()
        class_AR[i] = (100*TP / (TP + FN + 1e-6)).item()
        # class_ARoverAP[i] = ((TP + FN) / (TP + FP+1e-6)).item()
    print(confusion_matrix)
    # print(f'Class AP/AR: {class_ARoverAP},\n mAP/mAR : {torch.sum(torch.stack(list(class_ARoverAP.values())))/num_classes}')
    mAP = sum(class_AP.values())/num_classes+1e-6
    mAR = sum(class_AR.values())/num_classes+1e-6
    print('mAP : ',mAP)
    print('mAR : ', mAR)
    print('F1 : ', 2*(mAP * mAR)/(mAP + mAR))

def position_prediction(epoch, model, s_loader, criterion, criterion2, optimizer, device, feat_size=(7,7)):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    fh, fw = feat_size
    baseMap = torch.zeros([fh,fw])
    baseHMap = torch.zeros([fh,fw,2])
    for x in range(fw):
        for y in range(fh):
            baseHMap[x,y,:] = torch.tensor([x,y])

    for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, positions = model(inputs)
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)
        
        gt_points = []
        for mask in masks:
            mask = mask.squeeze(0)
            mask[mask>0]=1.
            m_point, _ = get_center_box(mask, mode='max')
            m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
            x, y = int(fw*m_point[0]), int(fh*m_point[1]) # 7,7

            pointMap = baseMap.clone()
            pointMap[y, x] = 1
            # pointMap = gen_gaussian_HM([y,x], size=[fh, fw], base_heatmap=baseHMap)

            gt_points.append(pointMap)
        gt_points = torch.stack(gt_points) # b, 7, 7
        gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
        gt_points = gt_points.to(device)

        positions = positions.view([gt_points.shape[0], -1])
        positions = torch.softmax(positions, -1)

        positions = positions.float()
        gt_points = gt_points.float()
        
        loss2 = 10*criterion2(positions, gt_points)
        # loss2 = criterion2(positions, gt_points)
        
        loss = loss1 + loss2
        # loss = loss1
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    total_acc = 100 * running_acc / total
    print("Total : ", total)
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)
    print("Total Loss : ", (running_loss1+running_loss2) / total)

def Position_prediction_test(epoch, model, loader, criterion, criterion2, device, minLoss, spath, feat_size=(7,7)):
    print('\nEpoch: %d'%epoch)
    model.eval()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    fh, fw = feat_size
    baseMap = torch.zeros([fh,fw])
    baseHMap = torch.zeros([fh,fw,2])
    for x in range(fw):
        for y in range(fh):
            baseHMap[x,y,:] = torch.tensor([x,y])
    with torch.no_grad():
        for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, positions = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            outputs = outputs.float()
            loss1 = criterion(outputs, labels)

            gt_points = []
            for mask in masks:
                mask = mask.squeeze(0)
                mask[mask>0]=1.
                m_point, _ = get_center_box(mask, mode='max')
                m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                x, y = int(fw*m_point[0]), int(fh*m_point[1]) # 7,7

                pointMap = baseMap.clone()
                pointMap[y, x] = 1
                # pointMap = gen_gaussian_HM([y,x], size=[fh, fw], base_heatmap=baseHMap)
                gt_points.append(pointMap)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)

            positions = positions.view([gt_points.shape[0], -1])
            positions = torch.softmax(positions, -1)

            positions = positions.float()
            gt_points = gt_points.float()
            
            loss2 = 10*criterion2(positions, gt_points)
            # loss2 = criterion2(positions, gt_points)

            loss = loss1 + loss2
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_loss = (running_loss1 + running_loss2) / total
        total_acc = 100 * running_acc / total
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Total Loss : ", (running_loss1+running_loss2) / total)
        
        if total_loss < minLoss:
            torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
            torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
            return total_loss
        else:
            return minLoss

def point4wrong(epoch, model, s_loader, u_loader, criterion, criterion2, optimizer, device, feat_size=(7,7)):
    print('For Correct Set. ')
    train(epoch, model, u_loader, criterion, optimizer, device)
    print('For Wrong Set. ')
    position_prediction(epoch, model, s_loader, criterion, criterion2, optimizer, device, feat_size=feat_size)

##########################################################################################################

def select_wrongs(model, loader, device):
    selects = []
    unselects = []
    model.eval()
    with torch.no_grad():
        for _, (inputs, labels, _, indexes) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if type(outputs)==dict:
                f1 = outputs['l1'] # b,256,56,56
                f2 = outputs['l2'] # b,512,28,28
                f3 = outputs['l3'] # b,1024,14,14
                f4 = outputs['l4'] # b,2048,7,7
                outputs = outputs['fc']
            _, preds = torch.max(outputs, 1)
            for (pred, label, index) in zip(preds, labels, indexes):
                if pred.item()!=label.item():
                    selects.append(index)
                else:
                    unselects.append(index)
        return selects, unselects

def gen_gaussian_HM(point, size=7, sigma=1., base_heatmap=None):
    if type(size)==int:
        size = [size,size]
    if base_heatmap==None:
        base_heatmap = torch.zeros((size[0],size[1],2))
        for x in range(size[0]):
            for y in range(size[1]):
                base_heatmap[x,y,:] = torch.tensor([x,y])
        heatmap = base_heatmap
    else:
        heatmap = base_heatmap.clone()
    # print(base_heatmap)
    heatmap = (heatmap - torch.tensor(point, dtype=int))/(size[0]/2)
    # heatmap = (heatmap - torch.tensor(point, dtype=int))
    heatmap = heatmap*heatmap
    heatmap = torch.sum(heatmap, dim=-1)
    heatmap = -1*heatmap/(sigma)**2
    heatmap = torch.exp(heatmap)
    # heatmap = heatmap*heatmap
    return heatmap

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