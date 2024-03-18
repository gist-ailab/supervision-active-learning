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
from sklearn import metrics
from torchvision.ops import masks_to_boxes
from kcenterGreedy import kCenterGreedy
import random
import torchvision.models as models
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import PIL.Image as Image


def init_model(device, name='resnet50', num_class=3):
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(2048, num_class)
        return_nodes = {
            'layer1':'l1',
            'layer2':'l2',
            'layer3':'l3',
            'layer4':'l4',
            'fc':'fc'
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        model = model.to(device)

    if name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, num_class)
        return_nodes = {
            'layer1':'l1',
            'layer2':'l2',
            'layer3':'l3',
            'layer4':'l4',
            'fc':'fc'
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        model = model.to(device)
    
    if name == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_class, bias=True)
        )
        return_nodes = {
            'features':'l4',
            'classifier':'fc'
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        model = model.to(device)

    if name == 'efficientnet':
        model = models.efficientnet_b1(weights='DEFAULT')
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_class, bias=True)
        )
        return_nodes = {
            'features':'l4',
            'classifier':'fc'
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        model = model.to(device)

    return model

def one_hot(labels, num_classes=3):
    one_hot_labels = np.eye(num_classes, dtype=int)[labels]
    return one_hot_labels

def roc_scores(labels, output_list, num_class=3):
    one_hot_labels = one_hot(labels, num_class)
    print(one_hot_labels.shape)
    print(output_list.shape)
    roc_score = metrics.roc_auc_score(one_hot_labels, output_list, multi_class='ovr')
    return roc_score

def collate_fn(batch):
    return tuple(zip(*batch))

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = 2.0 * (intersection+smooth) / (union+smooth)
    dice_loss = 1.0 - dice
    return dice_loss

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
        _, _, _, _, outputs = model(inputs)
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
    total_loss = running_loss / total
    total_acc = 100 * running_acc / total
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

def train_seg(epoch, model, teacher, loader, criterion1, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch) 
    model.train()
    teacher.eval()
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    total = 0
    alpha = 1.0
    for _, (inputs, _, masks, _) in enumerate(tqdm(loader)):
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        f1, f2, f3, f4, _ = teacher(inputs)
        _, _, _, _, outputs = model(inputs, f1, f2, f3, f4)
        
        total += outputs.size(0)
        loss1 = torch.mean(criterion1(outputs, masks))
        loss2 = torch.mean(criterion2(outputs, masks))
        # print(loss1, loss2)
        loss = loss1 + alpha*loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    total_loss = running_loss / total
    print('loss1 : ', running_loss1 / total)
    print('loss2 : ', running_loss2 / total)
    print(f'Train epoch : {epoch} loss : {total_loss}')

def train_csc(epoch, model, teacher1, teacher2, loader, criterion1, criterion2, optimizer, device):
    print('\nEpoch: %d'%epoch)
    model.train()
    teacher1.eval()
    teacher2.eval()
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    alpha = 0.01
    for _, (inputs, labels, masks, _) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        f1, f2, f3, f4, _ = teacher1(inputs)
        tf1, tf2, tf3, tf4, _ = teacher2(inputs, f1, f2, f3, f4)
        f1, f2, f3, f4, outputs = model(inputs)
        # print(tf1.shape)
        # print(tf2.shape)
        # print(tf3.shape)
        # print(tf4.shape)
        # loss2 = 4 - torch.mean(criterion2(f1, tf1)) \
        #             - torch.mean(criterion2(f2, tf2)) \
        #             - torch.mean(criterion2(f3, tf3)) \
        #             - torch.mean(criterion2(f4, tf4)) \
        f1 = F.normalize(f1)
        f2 = F.normalize(f2)
        f3 = F.normalize(f3)
        f4 = F.normalize(f4)
        tf1 = F.normalize(tf1)
        tf2 = F.normalize(tf2)
        tf3 = F.normalize(tf3)
        tf4 = F.normalize(tf4)
        # loss2 = criterion2(f1, tf1)+criterion2(f2, tf2)+criterion2(f3, tf3)+criterion2(f4, tf4)
        loss2 = criterion2(f4, tf4)

        _, pred = torch.max(outputs, 1)
        total += outputs.size(0)
        running_acc += (pred == labels).sum().item()
        outputs = outputs.float()
        loss1 = criterion1(outputs, labels)
        # loss = loss1 + alpha*loss2
        loss = loss1
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    total_loss = running_loss / total
    total_acc = 100 * running_acc / total
    print('loss1 : ', running_loss1 / total)
    print('loss2 : ', running_loss2 / total)
    print(f'Train epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')

def test(epoch, model, loader, criterion, device, minLoss, spath, mode, submodule=None):
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
            _, _, _, _, outputs = model(inputs)
            if type(outputs)==dict:
                outputs = outputs['fc']
            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        total_loss = running_loss / total
        total_acc = 100 * running_acc / total
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
        if total_loss > minLoss and epoch!=-1:
            if mode=='step1':
                torch.save(model.state_dict(), os.path.join(spath, 's1_model.pth'))
            if mode=='step2':
                torch.save(model.state_dict(), os.path.join(spath, 's2_model.pth'))
            if mode=='step3':
                torch.save(model.state_dict(), os.path.join(spath, 's3_model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
            return total_loss
        else:
            return minLoss

def test_seg(epoch, model, teacher, loader, criterion1, criterion2, device, minLoss, spath, mode, save=False):
    print('\nEpoch: %d'%epoch)
    model.eval()
    teacher.eval()
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for _, (inputs, _, masks, _) in enumerate(tqdm(loader)):
            inputs, masks = inputs.to(device), masks.to(device)
            f1, f2, f3, f4, _ = teacher(inputs)
            _, _, _, _, outputs = model(inputs, f1, f2, f3, f4)
            if save==True:
                for mask in outputs:
                    mask = mask.detach().cpu().numpy()      
            total += outputs.size(0)
            loss1= torch.mean(criterion1(outputs, masks))
            loss2 = torch.mean(criterion2(outputs, masks))
            loss = loss1 + loss2
            running_loss += loss.item()
        total_loss = running_loss / total
        print(f'Test epoch : {epoch} loss : {total_loss}')

        if total_loss > minLoss and epoch!=-1:
            if mode=='step1':
                torch.save(model.state_dict(), os.path.join(spath, 's1_segmodel.pth'))
            if mode=='step2':
                torch.save(model.state_dict(), os.path.join(spath, 's2_segmodel.pth'))
            if mode=='step3':
                torch.save(model.state_dict(), os.path.join(spath, 's3_segmodel.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(spath, 'segmodel.pth'))
            return total_loss
        else:
            return minLoss

def test_csc(epoch, model, teacher1, teacher2, loader, criterion1, criterion2, device, minLoss, spath, mode, submodule=None):
    print('\nEpoch: %d'%epoch)
    model.eval()
    teacher1.eval()
    teacher2.eval()
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    alpha = 0.01
    with torch.no_grad():
        for _, (inputs, labels, masks, _) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            f1, f2, f3, f4, _ = teacher1(inputs)
            tf1, tf2, tf3, tf4, _ = teacher2(inputs, f1, f2, f3, f4)
            f1, f2, f3, f4, outputs = model(inputs)

            # loss2 = 4 - torch.mean(criterion2(f1, tf1)) \
            #           - torch.mean(criterion2(f2, tf2)) \
            #           - torch.mean(criterion2(f3, tf3)) \
            #           - torch.mean(criterion2(f4, tf4)) \
            f1 = F.normalize(f1)
            f2 = F.normalize(f2)
            f3 = F.normalize(f3)
            f4 = F.normalize(f4)
            tf1 = F.normalize(tf1)
            tf2 = F.normalize(tf2)
            tf3 = F.normalize(tf3)
            tf4 = F.normalize(tf4)
            # loss2 = criterion2(f1, tf1)+criterion2(f2, tf2)+criterion2(f3, tf3)+criterion2(f4, tf4)
            loss2 = criterion2(f4, tf4)

            _, pred = torch.max(outputs, 1)
            total += outputs.size(0)
            running_acc += (pred == labels).sum().item()
            outputs = outputs.float()
            loss1 = criterion1(outputs, labels)
            # loss = loss1 + alpha*loss2
            loss = loss1
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_loss = running_loss / total
        total_loss1 = running_loss1 / total
        total_loss2 = running_loss2 / total
        total_acc = 100 * running_acc / total
        print('loss1 : ', total_loss1.item())
        print('loss2 : ', total_loss2.item())
        print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
        
        if total_loss > minLoss and epoch!=-1:
            if mode=='step1':
                torch.save(model.state_dict(), os.path.join(spath, 's1_model.pth'))
            if mode=='step2':
                torch.save(model.state_dict(), os.path.join(spath, 's2_model.pth'))
            if mode=='step3':
                torch.save(model.state_dict(), os.path.join(spath, 's3_model.pth'))
            else:
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
    labels_list = []
    outputs_list = []
    for i, (imgs, labels, _, _) in enumerate(tqdm(loader)):
        imgs, labels = imgs.to(device), labels.to(device)
        _, _, _, _, outputs = model(imgs)
        if type(outputs)==dict:
            outputs = outputs['fc']
        _, preds = torch.max(outputs, 1)
        labels_list.append(np.array(labels.detach().cpu()))
        outptus = torch.softmax(outputs, dim=-1)
        outputs_list.append(np.array(outputs.detach().cpu()))

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
    # print(labels_list)
    # labels_list = np.concatenate(labels_list, axis=0)
    # outputs_list = np.concatenate(outputs_list, axis=0)
    # auc_scores = roc_scores(labels_list, outputs_list, num_class=3)
    print(confusion_matrix)
    # print(f'Class AP/AR: {class_ARoverAP},\n mAP/mAR : {torch.sum(torch.stack(list(class_ARoverAP.values())))/num_classes}')
    mAP = sum(class_AP.values())/num_classes+1e-6
    mAR = sum(class_AR.values())/num_classes+1e-6
    print('mAP : ',mAP)
    print('mAR : ', mAR)
    print('F1 : ', 2*(mAP * mAR)/(mAP + mAR))
    # print('AUC Score : ', auc_scores)

def csc_metric(model, loader, num_classes, device):
    model.eval()
    class_AP = dict()
    class_AR = dict()
    class_ARoverAP = dict()
    confusion_matrix = torch.zeros([num_classes, num_classes])
    labels_list = []
    outputs_list = []
    for _, (inputs, labels, masks, _) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)
        _,_,_,_,outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        labels_list.append(np.array(labels.detach().cpu()))
        outptus = torch.softmax(outputs, dim=-1)
        outputs_list.append(np.array(outputs.detach().cpu()))

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
    print(confusion_matrix)

def point_regression3(epoch, model, s_loader, criterion, criterion2, optimizer, device, feat_size=(7,7)):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    fh, fw = feat_size
    # class_dict = {0:0, 1:0, 2:0}
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
        # optimizer2.zero_grad()
        outputs = model(inputs)
        f1 = outputs['l1'] # b,256,56,56
        f2 = outputs['l2'] # b,512,28,28
        f3 = outputs['l3'] # b,1024,14,14
        f4 = outputs['l4'] # b,2048,7,7
        # fh, fw = f4.shape[-2:]
        outputs = outputs['fc'] 
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)

        # for label in labels:
        #     class_dict[label.item()] += 1
        
        gt_points = []
        # print(fh, fw)
        for mask in masks:
            mask = mask.squeeze(0)
            mask[mask>0]=1.
            m_point, _ = get_center_box(mask, mode='max')
            m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
            x, y = int(fw*m_point[0]), int(fh*m_point[1]) # 7,7

            pointMap = baseMap.clone()
            pointMap[y, x] = 1
            pointMap = gen_gaussian_HM([y,x], size=[fh, fw], base_heatmap=baseHMap)

            gt_points.append(pointMap)
        gt_points = torch.stack(gt_points) # b, 7, 7
        gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
        gt_points = gt_points.to(device)

        # f1 = reg_head[0](f1) # b,1,56,56
        f1 = torch.mean(f1, 1)
        f1 = F.adaptive_avg_pool2d(f1, (fh,fw))
        # f2 = reg_head[1](f2) # b,1,28,28
        f2 = torch.mean(f2, 1)
        f2 = F.adaptive_avg_pool2d(f2, (fh,fw))
        # f3 = reg_head[2](f3) # b,1,14,14
        f3 = torch.mean(f3, 1)
        f3 = F.adaptive_avg_pool2d(f3, (fh,fw))
        # f4 = reg_head[3](f4) # b,1,7,7
        f4 = torch.mean(f4, 1)
        f4 = F.adaptive_avg_pool2d(f4, (fh,fw))
        # f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7
        f = torch.stack([f1,f2,f3,f4], 1).squeeze()
        if len(f.shape) < 4:
            f = f.unsqueeze(0)
        f = torch.mean(f, dim=1)
        # print(f.shape)

        reg_outputs = f.squeeze() # b, 7, 7
        # print('1. ', reg_outputs.shape)
        if len(reg_outputs.shape) == 2:
            reg_outputs = reg_outputs.unsqueeze(0)
        # print('2. ',reg_outputs.shape)
        reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
        # print('3. ',reg_outputs.shape)

        # reg_outputs = torch.sigmoid(reg_outputs)
        reg_outputs = torch.softmax(reg_outputs, -1)

        reg_outputs = reg_outputs.float()
        gt_points = gt_points.float()
        
        loss2 = 1*criterion2(reg_outputs, gt_points)
        # loss2 = criterion2(reg_outputs, gt_points)
        
        if epoch==-1:
            loss = loss2
        else:
            loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        # optimizer2.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
    total_acc = 100 * running_acc / total
    print("Total : ", total)
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)
    print("Total Loss : ", (running_loss1+running_loss2) / total)
    # print("Class Dict : ", class_dict[0], class_dict[1], class_dict[2])

def train_edge_similarity():
    pass

def train_mask_similarity(epoch, model, segHead, grad_cam, loader, criterion, criterion2, optimizer, optimizer2,device):
    # 1. for every batch, generate grad_cam
    # 2. train segment head with grad_cam pseudo GT
    print('\nEpoch: %d'%epoch)
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    count = 0
    for _, (inputs, labels, masks, _) in enumerate(tqdm(loader)):
        total += inputs.shape[0]
        inputs, labels = inputs.to(device), labels.to(device)
        masks = masks.squeeze().to(device)
        optimizer.zero_grad()
        optimizer2.zero_grad()

        # model.eval()
        # outputs = model(inputs)
        # outputs = outputs['fc']
        # _, preds = torch.max(outputs, 1)
        # heatmap_gt = []
        # for img, pred in zip(inputs, preds):
        #     img = img.unsqueeze(0)
        #     heatmap = grad_cam(img, pred)
        #     heatmap = np.uint8(255 * heatmap)
        #     heatmap = Image.fromarray(heatmap).resize((56,56), Image.LANCZOS)
        #     heatmap = np.array(heatmap)
        #     bn_heatmap = np.float32(heatmap) / 255
        #     bn_heatmap[bn_heatmap>0.1] = 1.
        #     bn_heatmap[bn_heatmap<0.1] = 0.
        #     if epoch<=1 and count<1:
        #         with open('/SSDg/yjh/workspace/supervision-active-learning/bn_heatmap.npy', 'wb') as f:
        #             print(np.max(bn_heatmap))
        #             print(np.min(bn_heatmap))
        #             np.save(f, bn_heatmap)

        #     bn_heatmap = torch.tensor(bn_heatmap)
        #     heatmap_gt.append(bn_heatmap)
        #     if epoch<=1 and count<1:
        #         with open('/SSDg/yjh/workspace/supervision-active-learning/heatmap.npy', 'wb') as f:
        #             print(np.max(heatmap))
        #             print(np.min(heatmap))
        #             np.save(f, heatmap)
        #         count += 1
        #         print('count',count)
                
        # heatmap_gt = torch.stack(heatmap_gt)
        # heatmap_gt = heatmap_gt.to(device)

        model.train()
        outputs = model(inputs)
        f1 = outputs['l1'] # b,256,56,56
        f2 = outputs['l2'] # b,512,28,28
        f3 = outputs['l3'] # b,1024,14,14
        f4 = outputs['l4'] # b,2048,7,7
        outputs = outputs['fc']
        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)

        seg_outputs = segHead(f1,f2,f3,f4)
        if len(seg_outputs.shape)==4:
            seg_outputs = torch.stack([seg_outputs[i,labels[i],:,:] for i in range(seg_outputs.shape[0])])
        loss2 = criterion2(seg_outputs, masks)
        # loss2 = criterion2(seg_outputs, heatmap_gt)
        loss = loss1 + loss2
        # loss = loss1

        loss.backward()
        # if epoch > 20:
        #     optimizer.step()
        optimizer.step()
        optimizer2.step()
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()

    total_acc = 100 * running_acc / total
    print("Total : ", total)
    print("Acc : ", total_acc)
    print("Loss 1 : ", running_loss1 / total)
    print("Loss 2 : ", running_loss2 / total)
    print("Total Loss : ", (running_loss1+running_loss2) / total)

def wrong_data_correction(epoch, model, s_loader, criterion, optimizer, device, feat_size=(7,7)):
    pass

def semi_point_prediction(epoch, model, s_loader, criterion, criterion2, optimizer, device, feat_size=(7,7), selected=[]):
    print('\nEpoch: %d'%epoch)
    model.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_acc = 0.0
    total = 0
    fh, fw = feat_size
    # class_dict = {0:0, 1:0, 2:0}
    baseMap = torch.zeros([fh,fw])
    baseHMap = torch.zeros([fh,fw,2])
    for x in range(fw):
        for y in range(fh):
            baseHMap[x,y,:] = torch.tensor([x,y])

    for _, (inputs, labels, masks, indexes) in enumerate(tqdm(s_loader)):
        total += inputs.shape[0]
        inputs, masks = inputs.to(device), masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if len(outputs.keys())==5:
            f1 = outputs['l1'] # b,256,56,56
            f2 = outputs['l2'] # b,512,28,28
            f3 = outputs['l3'] # b,1024,14,14
            f4 = outputs['l4'] # b,2048,7,7
            outputs = outputs['fc'] 
            
            f1 = torch.mean(f1, 1)
            f1 = F.adaptive_avg_pool2d(f1, (fh,fw))
            f2 = torch.mean(f2, 1)
            f2 = F.adaptive_avg_pool2d(f2, (fh,fw))
            f3 = torch.mean(f3, 1)
            f3 = F.adaptive_avg_pool2d(f3, (fh,fw))
            f4 = torch.mean(f4, 1)
            f4 = F.adaptive_avg_pool2d(f4, (fh,fw))
            f = torch.stack([f1,f2,f3,f4], 1).squeeze()
            if len(f.shape) < 4:
                f = f.unsqueeze(0)
            f = torch.mean(f, dim=1)

        elif len(outputs.keys())==2:
            f4 = outputs['l4']
            outputs = outputs['fc']

            f4 = torch.mean(f4, 1)
            f = F.adaptive_avg_pool2d(f4, (fh,fw))

        _, pred = torch.max(outputs, 1)
        running_acc += (pred == labels).sum().item()
        loss1 = criterion(outputs, labels)

        reg_outputs = f.squeeze() # b, 7, 7
        if len(reg_outputs.shape) == 2:
            reg_outputs = reg_outputs.unsqueeze(0)
        reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
        reg_outputs = torch.softmax(reg_outputs, -1)

        gt_points = []
        for mask, idx, reg_output in zip(masks, indexes, reg_outputs):
            if idx in selected and torch.max(mask)!=0:
                mask = mask.squeeze(0)
                mask[mask>0]=1.
                m_point, _ = get_center_box(mask, mode='max')
                m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                x, y = int(fw*m_point[0]), int(fh*m_point[1]) # 7,7
                # pointMap = baseMap.clone()
                # pointMap[y, x] = 1
                pointMap = gen_gaussian_HM([y,x], size=[fh, fw], base_heatmap=baseHMap)
            else:
                idx = torch.argmax(reg_output)
                y = int(idx//fh)
                x = int(idx - fh*y)
                # print(y, x)
                pointMap = gen_gaussian_HM([y,x], size=[fh, fw], base_heatmap=baseHMap)
                # pointMap = pointMap.view([fh, fw])
                # print(pointMap.shape)
            gt_points.append(pointMap)
        gt_points = torch.stack(gt_points) # b, 7, 7
        gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
        gt_points = gt_points.to(device)

        reg_outputs = reg_outputs.float()
        gt_points = gt_points.float()
        
        loss2 = 10*criterion2(reg_outputs, gt_points)
        loss = loss1 + loss2
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

def regression_test3(epoch, model, loader, criterion, criterion2, device, minLoss, spath, feat_size=(7,7)):
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
            outputs = model(inputs)
            if type(outputs)==dict:
                f1 = outputs['l1'] # b,256,56,56
                f2 = outputs['l2'] # b,512,28,28
                f3 = outputs['l3'] # b,1024,14,14
                f4 = outputs['l4'] # b,2048,7,7
                outputs = outputs['fc']
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

            # f1 = reg_head[0](f1) # b,1,56,56
            f1 = torch.mean(f1, 1)
            f1 = F.adaptive_avg_pool2d(f1, (fh,fw))
            # f2 = reg_head[1](f2) # b,1,28,28
            f2 = torch.mean(f2, 1)
            f2 = F.adaptive_avg_pool2d(f2, (fh,fw))
            # f3 = reg_head[2](f3) # b,1,14,14
            f3 = torch.mean(f3, 1)
            f3 = F.adaptive_avg_pool2d(f3, (fh,fw))
            # f4 = reg_head[3](f4) # b,1,7,7
            f4 = torch.mean(f4, 1)
            f4 = F.adaptive_avg_pool2d(f4, (fh,fw))
            # f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7
            f = torch.stack([f1,f2,f3,f4], 1).squeeze()
            if len(f.shape) < 4:
                f = f.unsqueeze(0)
            f = torch.mean(f, dim=1)

            reg_outputs = f.squeeze() # b, 7, 7
            if len(reg_outputs.shape) == 2:
                reg_outputs = reg_outputs.unsqueeze(0)
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            
            # reg_outputs = torch.sigmoid(reg_outputs)
            reg_outputs = torch.softmax(reg_outputs, -1)

            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            
            loss2 = 1*criterion2(reg_outputs, gt_points)
            # loss2 = criterion2(reg_outputs, gt_points)

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
    print('For Wrong Set. ')
    point_regression3(epoch, model, s_loader, criterion, criterion2, optimizer, device, feat_size=feat_size)
    print('For Correct Set. ')
    train(epoch, model, u_loader, criterion, optimizer, device)
    

#---------------------------------------------------------------------
def select_wrongs(model, loader, device):
    selects = []
    unselects = []
    model.eval()
    with torch.no_grad():
        for _, (inputs, labels, masks, indexes) in enumerate(tqdm(loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, _, _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for (pred, label, mask, index) in zip(preds, labels, masks, indexes):
                if pred.item()!=label.item() and torch.max(mask)==1.:
                    selects.append(index)
                else:
                    unselects.append(index)
        return selects, unselects

def gen_checklist(selects, unselects):
    
    pass

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
    return masked_input

toggle = True
if toggle:
    def data_selection(model, loader, criterion, device, ratio=0.1, preselected=[], mode='random'):
        selected = []
        unselected = []
        total = 0
        model.eval()
        # print(total)
        if mode=='random': # divergence
            for i, (inputs, _, masks, index) in enumerate(tqdm(loader)):
                unselected.append(index)
            random.shuffle(unselected)
            data_len = len(unselected)
            selected = unselected[:int(ratio*data_len)]
            unselected = unselected[int(ratio*data_len):]
        if 'entropy' in mode: # uncertainty
            entropy_list = []
            with torch.no_grad():
                for i, (inputs, _, masks, index) in enumerate(tqdm(loader)):
                    outputs = model(inputs)
                    outputs = outputs['fc']
                    _, preds = torch.max(outputs, 1)
                    entropy = criterion(outputs, preds)
                    entropy_list.append((entropy, index))
            entropy_list.sort(key=lambda x : x[0])
            if 'high' in mode:
                entropy_list.reverse()
            for sample in entropy_list:
                if len(selected) < int(len(entropy_list)*ratio):
                    selected.append(sample[1])
                else:
                    unselected.append(sample[1])
        if 'confi' in mode: # uncertainty
            confi_list = []
            with torch.no_grad():
                for i, (inputs, _, masks, index) in enumerate(tqdm(loader)):
                    outputs = model(inputs)
                    outputs = outputs['fc']
                    conf = torch.max(outputs)
                    confi_list.append((conf, index))
            confi_list.sort(key=lambda x : x[0])
            if 'high' in mode:
                confi_list.reverse()
            for sample in confi_list:
                if len(selected) < int(len(confi_list)*ratio):
                    selected.append(sample[1])
                else:
                    unselected.append(sample[1])
        if 'lloss' in mode:
            uncertainty = get_uncertainty(model, loader)
            arg = np.argsort(uncertainty)
            selected = arg[:int(ratio*data_len)]
            unselected = arg[int(ratio*data_len):]

        if 'gcn' in mode:
            binary_labels = torch.ones([len(testset), 1])
            for data in testloader:
                img, label, mask, idx = data
                if idx in preselected:
                    binary_labels[idx,:] = 0

            features = get_features(model, unlabeled_loader)
            features = nn.functional.normalize(features)
            adj = aff_to_adj(features)

            gcn_module = GCN(nfeat=features.shape[1],
                            nhid=128,
                            nclass=1,
                            dropout=0.3).cuda()
                                    
            models      = {'gcn_module': gcn_module}
            optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=1e-3, weight_decay=5e-4)
            optimizers = {'gcn_module': optim_backbone}

            lbl = np.arange(SUBSET, SUBSET+(cycle+1)*ADDENDUM, 1)
            nlbl = np.arange(0, SUBSET, 1)
            
            ############
            for _ in range(200):

                optimizers['gcn_module'].zero_grad()
                outputs, _, _ = models['gcn_module'](features, adj)
                lamda = args.lambda_loss 
                loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
                loss.backward()
                optimizers['gcn_module'].step()

            models['gcn_module'].eval()
            with torch.no_grad():
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = features.cuda()
                    labels = binary_labels.cuda()
                scores, _, feat = models['gcn_module'](inputs, adj)
                
                feat = feat.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET,(SUBSET + (cycle+1)*ADDENDUM))
                sampling2 = kCenterGreedy(feat)  
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2        
        return selected, unselected

    def get_uncertainty(models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            uncertainty = torch.tensor([]).cuda()
        with torch.no_grad():
            for inputs, _, _, _ in unlabeled_loader:
                with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                    inputs = inputs.cuda()
                features = models['backbone'](inputs)['l4']
                pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))
                uncertainty = torch.cat((uncertainty, pred_loss), 0)
        return uncertainty.cpu()

    def get_features(models, unlabeled_loader):
        models['backbone'].eval()
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            features = torch.tensor([]).cuda()    
        with torch.no_grad():
                for inputs, _, _, _ in unlabeled_loader:
                    with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                        inputs = inputs.cuda()
                        features_batch = models['backbone'](inputs)['l4']
                    features = torch.cat((features, features_batch), 0)
                feat = features #.detach().cpu().numpy()
        return feat

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

    def point_regression(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, optimizer2, device):
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
            optimizer2.zero_grad()
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
            
            if epoch==-1:
                loss = loss2
            else:
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)

    def point_regression2(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, optimizer2, device):
        print('\nEpoch: %d'%epoch)
        model.train()
        reg_head.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
            total += inputs.shape[0]
            inputs, masks = inputs.to(device), masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
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
                m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                pointMap = baseMap.clone()
                x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                pointMap[y, x] = 1
                gt_points.append(pointMap)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)
            feat = reg_head(feat) # b,1,7,7
            reg_outputs = feat.squeeze() # b, 7, 7
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            reg_outputs = torch.softmax(reg_outputs, -1)
            # reg_outputs = reg_outputs.view([reg_outputs.shape[0], 7, 7]) # b,7,7
            # print(reg_outputs.shape)
            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            loss2 = 10*criterion2(reg_outputs, gt_points)
            
            if epoch==-1:
                loss = loss2
            else:
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Total Loss : ", (running_loss1+running_loss2) / total)

    def point_regression4(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, optimizer2, device):
        print('\nEpoch: %d'%epoch)
        model.train()
        for head in reg_head:
            head.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
            total += inputs.shape[0]
            inputs, masks = inputs.to(device), masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = model(inputs)
            f1 = outputs['l1'] # b,256,56,56
            f2 = outputs['l2'] # b,512,28,28
            f3 = outputs['l3'] # b,1024,14,14
            f4 = outputs['l4'] # b,2048,7,7
            outputs = outputs['fc'] 
            _, pred = torch.max(outputs, 1)
            running_acc += (pred == labels).sum().item()
            loss1 = criterion(outputs, labels)
            
            gt_points = []
            for mask in masks:
                mask = mask.squeeze(0)
                m_point, _ = get_center_box(mask, mode='max')
                m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                pointMap = baseMap.clone()
                x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                pointMap[y, x] = 1
                gt_points.append(pointMap)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)

            weight = list(model.parameters())[-2]
            bias = list(model.parameters())[-1]
            cams = F.conv2d(f4, weight.view(weight.shape[0], weight.shape[1], 1, 1),
                        bias, stride=1, padding=0, dilation=1, groups=1) # b, n_cls, 7, 7
            labeled_cams = []
            for cam, label in zip(cams, labels):
                labeled_cam = cam[int(label)]
                labeled_cam = labeled_cam.unsqueeze(0)
                labeled_cams.append(labeled_cam)
            labeled_cams = torch.stack(labeled_cams)
            # print(labeled_cams.shape) # b,1,7,7

            f1 = reg_head[0](f1) # b,1,56,56
            f1 = F.adaptive_avg_pool2d(f1, (7,7))
            f2 = reg_head[1](f2) # b,1,28,28
            f2 = F.adaptive_avg_pool2d(f2, (7,7))
            f3 = reg_head[2](f3) # b,1,14,14
            f3 = F.adaptive_avg_pool2d(f3, (7,7))
            f4 = reg_head[3](f4) # b,1,7,7

            f = reg_head[4](torch.stack([f1,f2,f3,f4,labeled_cams], 1).squeeze()) # b,1,7,7

            reg_outputs = f.squeeze() # b, 7, 7
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            reg_outputs = torch.softmax(reg_outputs, -1)
            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            loss2 = 10*criterion2(reg_outputs, gt_points)
            
            if epoch==-1:
                loss = loss2
            else:
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Total Loss : ", (running_loss1+running_loss2) / total)

    def reg_feat_distil(epoch, model, reg_head, s_loader, criterion, criterion2, criterion3, optimizer, optimizer2, device):
        print('\nEpoch: %d'%epoch)
        if type(model)==dict and reg_head==None:
            model_dict = model
            model = model_dict['backbone']
            reg_head = model_dict['reg_module']
        if type(optimizer)==dict and optimizer2==None:
            opti_dict = optimizer
            optimizer2 = opti_dict['reg_head']
            optimizer = opti_dict['backbone']
        model.train()
        for head in reg_head:
            head.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
            total += inputs.shape[0]
            inputs, masks = inputs.to(device), masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = model(inputs)
            f1 = outputs['l1'] # b,256,56,56
            f2 = outputs['l2'] # b,512,28,28
            f3 = outputs['l3'] # b,1024,14,14
            f4 = outputs['l4'] # b,2048,7,7
            outputs = outputs['fc'] 
            _, pred = torch.max(outputs, 1)
            running_acc += (pred == labels).sum().item()
            loss1 = criterion(outputs, labels)
            
            gt_points = []
            for mask in masks:
                mask = mask.squeeze(0)
                m_point, _ = get_center_box(mask, mode='max')
                m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                pointMap = baseMap.clone()
                x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                pointMap[y, x] = 1
                gt_points.append(pointMap)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)

            reg_f1 = reg_head[0](f1) # b,1,56,56
            mean_f1 = torch.mean(f1, dim=1)
            f1 = F.adaptive_avg_pool2d(reg_f1, (7,7)) # b,1,7,7
            reg_f2 = reg_head[1](f2) # b,1,28,28
            mean_f2 = torch.mean(f2, dim=1)
            f2 = F.adaptive_avg_pool2d(reg_f2, (7,7)) # b,1,7,7
            reg_f3 = reg_head[2](f3) # b,1,14,14
            mean_f3 = torch.mean(f3, dim=1)
            f3 = F.adaptive_avg_pool2d(reg_f3, (7,7)) # b,1,7,7
            reg_f4 = reg_head[3](f4) # b,1,7,7
            mean_f4 = torch.mean(f4, dim=1)
            f4 = F.adaptive_avg_pool2d(reg_f4, (7,7)) # b,1,7,7
            f = torch.stack([f1,f2,f3,f4], 1).squeeze()
            if len(f.shape) < 4:
                f = f.unsqueeze(0)
            f = torch.mean(f, dim=1)
            # print(f.shape)

            if f.shape[0] == 1:
                reg_outputs = f.squeeze() # 7, 7
                reg_outputs = reg_outputs.unsqueeze(0) # 1, 7, 7
            else:
                reg_outputs = f.squeeze() # b, 7, 7
            if len(reg_outputs.shape)==2:
                reg_outputs = reg_outputs.unsqueeze(0)
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            reg_outputs = torch.softmax(reg_outputs, -1)
            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            loss2 = 10*criterion2(reg_outputs, gt_points)
            
            reg_f1 = reg_f1.reshape(reg_f1.size(0), -1)
            reg_f2 = reg_f2.reshape(reg_f2.size(0), -1)
            reg_f3 = reg_f3.reshape(reg_f3.size(0), -1)
            reg_f4 = reg_f4.reshape(reg_f4.size(0), -1)

            mean_f1 = mean_f1.reshape(mean_f1.size(0), -1)
            mean_f2 = mean_f2.reshape(mean_f2.size(0), -1)
            mean_f3 = mean_f3.reshape(mean_f3.size(0), -1)
            mean_f4 = mean_f4.reshape(mean_f4.size(0), -1)

            dist1 = (1-torch.mean(criterion3(mean_f1, reg_f1)))/2
            dist2 = (1-torch.mean(criterion3(mean_f2, reg_f2)))/2
            dist3 = (1-torch.mean(criterion3(mean_f3, reg_f3)))/2
            dist4 = (1-torch.mean(criterion3(mean_f4, reg_f4)))/2
            loss3 = 10*(dist1+dist2+dist3+dist4)/4

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Loss 3 : ", running_loss3 / total)
        print("Total Loss : ", (running_loss1+running_loss2+running_loss3) / total)

    def box_regression(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, optimizer2, device):
        print('\nEpoch: %d'%epoch)
        model.train()
        for head in reg_head:
            head.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
            total += inputs.shape[0]
            inputs, masks = inputs.to(device), masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = model(inputs)
            f1 = outputs['l1'] # b,256,56,56
            f2 = outputs['l2'] # b,512,28,28
            f3 = outputs['l3'] # b,1024,14,14
            f4 = outputs['l4'] # b,2048,7,7
            outputs = outputs['fc'] 
            _, pred = torch.max(outputs, 1)
            running_acc += (pred == labels).sum().item()
            loss1 = criterion(outputs, labels)
            
            gt_points = []
            for mask in masks:
                mask = mask.unsqueeze(0) # 1,1,224,224
                # print("max1 : ",torch.max(mask))
                # mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (7,7), mode='nearest').squeeze()
                mask = F.adaptive_avg_pool2d(mask,(7,7))
                # print("max2 : ",torch.max(mask))
                # print(mask.shape)
                mask = mask.squeeze().unsqueeze(0)
                bbox = torchvision.ops.masks_to_boxes(mask)[0].int()
                bbox_mask = torch.zeros_like(mask) # 7,7
                bbox_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1.0
                gt_points.append(bbox_mask)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)

            f1 = reg_head[0](f1) # b,1,56,56
            f1 = F.adaptive_avg_pool2d(f1, (7,7)) # b,1,7,7
            f2 = reg_head[1](f2) # b,1,28,28
            f2 = F.adaptive_avg_pool2d(f2, (7,7)) # b,1,7,7
            f3 = reg_head[2](f3) # b,1,14,14
            f3 = F.adaptive_avg_pool2d(f3, (7,7)) # b,1,7,7
            f4 = reg_head[3](f4) # b,1,7,7
            # f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7
            f = torch.mean(torch.stack([f1,f2,f3,f4], 1).squeeze(), dim=1)
            # print(f.shape)

            reg_outputs = f.squeeze() # b, 7, 7
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            reg_outputs = torch.softmax(reg_outputs, -1)
            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            loss2 = 0.1*criterion2(reg_outputs, gt_points)
            
            if epoch==-1:
                loss = loss2
            else:
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Total Loss : ", (running_loss1+running_loss2) / total)

    def mask_regression(epoch, model, reg_head, s_loader, criterion, criterion2, optimizer, optimizer2, device):
        print('\nEpoch: %d'%epoch)
        model.train()
        for head in reg_head:
            head.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        for _, (inputs, labels, masks, _) in enumerate(tqdm(s_loader)):
            total += inputs.shape[0]
            inputs, masks = inputs.to(device), masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs = model(inputs)
            f1 = outputs['l1'] # b,256,56,56
            f2 = outputs['l2'] # b,512,28,28
            f3 = outputs['l3'] # b,1024,14,14
            f4 = outputs['l4'] # b,2048,7,7
            outputs = outputs['fc'] 
            _, pred = torch.max(outputs, 1)
            running_acc += (pred == labels).sum().item()
            loss1 = criterion(outputs, labels)
            
            gt_points = []
            for mask in masks:
                mask = mask.unsqueeze(0) # 1,1,224,224
                mask = F.adaptive_avg_pool2d(mask,(7,7))
                mask = mask.squeeze()
                gt_points.append(mask)
            gt_points = torch.stack(gt_points) # b, 7, 7
            gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
            gt_points = gt_points.to(device)

            f1 = reg_head[0](f1) # b,1,56,56
            f1 = F.adaptive_avg_pool2d(f1, (7,7)) # b,1,7,7
            f2 = reg_head[1](f2) # b,1,28,28
            f2 = F.adaptive_avg_pool2d(f2, (7,7)) # b,1,7,7
            f3 = reg_head[2](f3) # b,1,14,14
            f3 = F.adaptive_avg_pool2d(f3, (7,7)) # b,1,7,7
            f4 = reg_head[3](f4) # b,1,7,7
            # f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7
            f = torch.mean(torch.stack([f1,f2,f3,f4], 1).squeeze(), dim=1)
            # print(f.shape)

            reg_outputs = f.squeeze() # b, 7, 7
            reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
            reg_outputs = torch.softmax(reg_outputs, -1)
            reg_outputs = reg_outputs.float()
            gt_points = gt_points.float()
            loss2 = 0.01*criterion2(reg_outputs, gt_points)
            
            if epoch==-1:
                loss = loss2
            else:
                loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        total_acc = 100 * running_acc / total
        print("Acc : ", total_acc)
        print("Loss 1 : ", running_loss1 / total)
        print("Loss 2 : ", running_loss2 / total)
        print("Total Loss : ", (running_loss1+running_loss2) / total)

    def regression_test(epoch, model, loader, criterion, criterion2, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        model.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    feat = outputs['l4']
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
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
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
            total_loss = (running_loss1 + running_loss2) / total
            total_acc = 100 * running_acc / total
            print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
            print("Loss 1 : ", running_loss1 / total)
            print("Loss 2 : ", running_loss2 / total)
            
            if total_loss < minLoss:
                torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
                if reg_head is not None:
                    torch.save(reg_head.state_dict(), os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head.state_dict(), os.path.join(spath, 'reg_head.pth'))
                return total_loss
            else:
                return minLoss

    def regression_test2(epoch, model, loader, criterion, criterion2, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        model.eval()
        reg_head.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    feat = outputs['l4']
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
                loss1 = criterion(outputs, labels)

                gt_points = []
                for mask in masks:
                    mask = mask.squeeze(0)
                    m_point, _ = get_center_box(mask, mode='max')
                    m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                    pointMap = baseMap.clone()
                    x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                    pointMap[y, x] = 1
                    gt_points.append(pointMap)
                gt_points = torch.stack(gt_points) # b, 7, 7
                gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
                gt_points = gt_points.to(device)
                feat = reg_head(feat) # b,1,7,7
                reg_outputs = feat.squeeze() # b, 7, 7
                reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
                reg_outputs = torch.softmax(reg_outputs, -1)
                # reg_outputs = reg_outputs.view([reg_outputs.shape[0], 7, 7]) # b,7,7
                # print(reg_outputs.shape)
                reg_outputs = reg_outputs.float()
                gt_points = gt_points.float()
                loss2 = 10*criterion2(reg_outputs, gt_points)

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
                if reg_head is not None:
                    torch.save(reg_head.state_dict(), os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head.state_dict(), os.path.join(spath, 'reg_head.pth'))
                return total_loss
            else:
                return minLoss

    def reg_distil_test(epoch, model, loader, criterion, criterion2, criterion3, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        if type(model)==dict:
            model_dict = model
            model = model_dict['backbone']
            reg_head = model_dict['reg_module']
        model.eval()
        for head in reg_head:
            head.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    f1 = outputs['l1'] # b,256,56,56
                    f2 = outputs['l2'] # b,512,28,28
                    f3 = outputs['l3'] # b,1024,14,14
                    f4 = outputs['l4'] # b,2048,7,7
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
                loss1 = criterion(outputs, labels)

                gt_points = []
                for mask in masks:
                    mask = mask.squeeze(0)
                    m_point, _ = get_center_box(mask, mode='max')
                    m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                    pointMap = baseMap.clone()
                    x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                    pointMap[y, x] = 1
                    gt_points.append(pointMap)
                gt_points = torch.stack(gt_points) # b, 7, 7
                gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
                gt_points = gt_points.to(device)

                reg_f1 = reg_head[0](f1) # b,1,56,56
                mean_f1 = torch.mean(f1, dim=1)
                f1 = F.adaptive_avg_pool2d(reg_f1, (7,7)) # b,1,7,7
                reg_f2 = reg_head[1](f2) # b,1,28,28
                mean_f2 = torch.mean(f2, dim=1)
                f2 = F.adaptive_avg_pool2d(reg_f2, (7,7)) # b,1,7,7
                reg_f3 = reg_head[2](f3) # b,1,14,14
                mean_f3 = torch.mean(f3, dim=1)
                f3 = F.adaptive_avg_pool2d(reg_f3, (7,7)) # b,1,7,7
                reg_f4 = reg_head[3](f4) # b,1,7,7
                mean_f4 = torch.mean(f4, dim=1)
                f4 = F.adaptive_avg_pool2d(reg_f4, (7,7)) # b,1,7,7
                f = torch.stack([f1,f2,f3,f4], 1).squeeze()
                if len(f.shape) < 4:
                    f = f.unsqueeze(0)
                f = torch.mean(f, dim=1)

                reg_outputs = f.squeeze() # b, 7, 7
                if len(reg_outputs.shape) == 2:
                    reg_outputs = reg_outputs.unsqueeze(0)
                reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
                reg_outputs = torch.softmax(reg_outputs, -1)
                reg_outputs = reg_outputs.float()
                gt_points = gt_points.float()
                loss2 = 10*criterion2(reg_outputs, gt_points)

                reg_f1 = reg_f1.reshape(reg_f1.size(0), -1)
                reg_f2 = reg_f2.reshape(reg_f2.size(0), -1)
                reg_f3 = reg_f3.reshape(reg_f3.size(0), -1)
                reg_f4 = reg_f4.reshape(reg_f4.size(0), -1)

                mean_f1 = mean_f1.reshape(mean_f1.size(0), -1)
                mean_f2 = mean_f2.reshape(mean_f2.size(0), -1)
                mean_f3 = mean_f3.reshape(mean_f3.size(0), -1)
                mean_f4 = mean_f4.reshape(mean_f4.size(0), -1)

                dist1 = (1-torch.mean(criterion3(mean_f1, reg_f1)))/2
                dist2 = (1-torch.mean(criterion3(mean_f2, reg_f2)))/2
                dist3 = (1-torch.mean(criterion3(mean_f3, reg_f3)))/2
                dist4 = (1-torch.mean(criterion3(mean_f4, reg_f4)))/2
                loss3 = 10*(dist1+dist2+dist3+dist4)/4

                loss = loss1 + loss2 + loss3

                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
            total_loss = (running_loss1 + running_loss2 + running_loss3) / total
            total_acc = 100 * running_acc / total
            print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
            print("Loss 1 : ", running_loss1 / total)
            print("Loss 2 : ", running_loss2 / total)
            print("Loss 3 : ", running_loss3 / total)
            print("Total Loss : ", (running_loss1+running_loss2+running_loss3) / total)
            
            if total_loss < minLoss:
                torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
                minLoss = total_loss
                if reg_head is not None:
                    reg_head_state_dict = dict()
                    reg_head_state_dict['l1'] = reg_head[0].state_dict()
                    reg_head_state_dict['l2'] = reg_head[1].state_dict()
                    reg_head_state_dict['l3'] = reg_head[2].state_dict()
                    reg_head_state_dict['l4'] = reg_head[3].state_dict()
                    torch.save(reg_head_state_dict, os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head_state_dict, os.path.join(spath, 'reg_head.pth'))
                return total_loss
            else:
                return minLoss

    def regression_test4(epoch, model, loader, criterion, criterion2, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        model.eval()
        for head in reg_head:
            head.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    f1 = outputs['l1'] # b,256,56,56
                    f2 = outputs['l2'] # b,512,28,28
                    f3 = outputs['l3'] # b,1024,14,14
                    f4 = outputs['l4'] # b,2048,7,7
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
                loss1 = criterion(outputs, labels)

                gt_points = []
                for mask in masks:
                    mask = mask.squeeze(0)
                    m_point, _ = get_center_box(mask, mode='max')
                    m_point /= torch.tensor(mask.shape) # range : 0 ~ 1
                    pointMap = baseMap.clone()
                    x, y = int(7*m_point[0]), int(7*m_point[1]) # 7,7
                    pointMap[y, x] = 1
                    gt_points.append(pointMap)
                gt_points = torch.stack(gt_points) # b, 7, 7
                gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
                gt_points = gt_points.to(device)

                weight = list(model.parameters())[-2]
                bias = list(model.parameters())[-1]
                cams = F.conv2d(f4, weight.view(weight.shape[0], weight.shape[1], 1, 1),
                            bias, stride=1, padding=0, dilation=1, groups=1) # b, n_cls, 7, 7
                labeled_cams = []
                for cam, label in zip(cams, labels):
                    labeled_cam = cam[int(label)]
                    labeled_cam = labeled_cam.unsqueeze(0)
                    labeled_cams.append(labeled_cam)
                labeled_cams = torch.stack(labeled_cams)

                f1 = reg_head[0](f1) # b,1,56,56
                f1 = F.adaptive_avg_pool2d(f1, (7,7))
                f2 = reg_head[1](f2) # b,1,28,28
                f2 = F.adaptive_avg_pool2d(f2, (7,7))
                f3 = reg_head[2](f3) # b,1,14,14
                f3 = F.adaptive_avg_pool2d(f3, (7,7))
                f4 = reg_head[3](f4) # b,1,7,7
                f = reg_head[4](torch.stack([f1,f2,f3,f4, labeled_cams], 1).squeeze()) # b,1,7,7

                reg_outputs = f.squeeze() # b, 7, 7
                reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
                reg_outputs = torch.softmax(reg_outputs, -1)
                reg_outputs = reg_outputs.float()
                gt_points = gt_points.float()
                loss2 = 10*criterion2(reg_outputs, gt_points)

                loss = loss1 + loss2
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
            total_loss = (running_loss1 + running_loss2) / total
            total_acc = 100 * running_acc / total
            print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
            print("Loss 1 : ", running_loss1 / total)
            print("Loss 2 : ", running_loss2 / total)
            print("Total Loss : ", (running_loss1+running_loss2) / total)
            
            # if total_loss < minLoss:
            if total_acc > minLoss:
                torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
                if reg_head is not None:
                    reg_head_state_dict = dict()
                    reg_head_state_dict['l1'] = reg_head[0].state_dict()
                    reg_head_state_dict['l2'] = reg_head[1].state_dict()
                    reg_head_state_dict['l3'] = reg_head[2].state_dict()
                    reg_head_state_dict['l4'] = reg_head[3].state_dict()
                    torch.save(reg_head_state_dict, os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head_state_dict, os.path.join(spath, 'reg_head.pth'))
                return total_acc
            else:
                return minLoss

    def box_regression_test(epoch, model, loader, criterion, criterion2, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        model.eval()
        for head in reg_head:
            head.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    f1 = outputs['l1'] # b,256,56,56
                    f2 = outputs['l2'] # b,512,28,28
                    f3 = outputs['l3'] # b,1024,14,14
                    f4 = outputs['l4'] # b,2048,7,7
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
                loss1 = criterion(outputs, labels)

                gt_points = []
                for mask in masks:
                    mask = mask.unsqueeze(0) # 1,1,224,224
                    # mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (7,7), mode='nearest').squeeze()
                    mask = F.adaptive_avg_pool2d(mask,(7,7))
                    mask = mask.squeeze().unsqueeze(0)
                    bbox = torchvision.ops.masks_to_boxes(mask)[0].int()
                    bbox_mask = torch.zeros_like(mask) # 7,7
                    bbox_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1.0
                    gt_points.append(bbox_mask)
                gt_points = torch.stack(gt_points) # b, 7, 7
                gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
                gt_points = gt_points.to(device)

                f1 = reg_head[0](f1) # b,1,56,56
                f1 = F.adaptive_avg_pool2d(f1, (7,7))
                f2 = reg_head[1](f2) # b,1,28,28
                f2 = F.adaptive_avg_pool2d(f2, (7,7))
                f3 = reg_head[2](f3) # b,1,14,14
                f3 = F.adaptive_avg_pool2d(f3, (7,7))
                f4 = reg_head[3](f4) # b,1,7,7
                f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7

                reg_outputs = f.squeeze() # b, 7, 7
                reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
                reg_outputs = torch.softmax(reg_outputs, -1)
                reg_outputs = reg_outputs.float()
                gt_points = gt_points.float()
                loss2 = 0.1*criterion2(reg_outputs, gt_points)

                loss = loss1 + loss2
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
            total_loss = (running_loss1 + running_loss2) / total
            total_acc = 100 * running_acc / total
            print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
            print("Loss 1 : ", running_loss1 / total)
            print("Loss 2 : ", running_loss2 / total)
            print("Total Loss : ", (running_loss1+running_loss2) / total)
            
            # if total_loss < minLoss:
            if total_acc > minLoss:
                torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
                if reg_head is not None:
                    reg_head_state_dict = dict()
                    reg_head_state_dict['l1'] = reg_head[0].state_dict()
                    reg_head_state_dict['l2'] = reg_head[1].state_dict()
                    reg_head_state_dict['l3'] = reg_head[2].state_dict()
                    reg_head_state_dict['l4'] = reg_head[3].state_dict()
                    torch.save(reg_head_state_dict, os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head_state_dict, os.path.join(spath, 'reg_head.pth'))
                return total_acc
            else:
                return minLoss

    def mask_regression_test(epoch, model, loader, criterion, criterion2, device, minLoss, spath, reg_head=None):
        print('\nEpoch: %d'%epoch)
        model.eval()
        for head in reg_head:
            head.eval()
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_acc = 0.0
        total = 0
        baseMap = torch.zeros([7,7])
        with torch.no_grad():
            for _, (inputs, labels, masks, index) in enumerate(tqdm(loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if type(outputs)==dict:
                    f1 = outputs['l1'] # b,256,56,56
                    f2 = outputs['l2'] # b,512,28,28
                    f3 = outputs['l3'] # b,1024,14,14
                    f4 = outputs['l4'] # b,2048,7,7
                    outputs = outputs['fc']
                _, pred = torch.max(outputs, 1)
                total += outputs.size(0)
                running_acc += (pred == labels).sum().item()
                outputs = outputs.float()
                loss1 = criterion(outputs, labels)

                gt_points = []
                for mask in masks:
                    mask = mask.unsqueeze(0) # 1,1,224,224
                    mask = F.adaptive_avg_pool2d(mask,(7,7))
                    mask = mask.squeeze()
                    gt_points.append(mask)
                gt_points = torch.stack(gt_points) # b, 7, 7
                gt_points = gt_points.view([gt_points.shape[0], -1]) # b, 49
                gt_points = gt_points.to(device)

                f1 = reg_head[0](f1) # b,1,56,56
                f1 = F.adaptive_avg_pool2d(f1, (7,7))
                f2 = reg_head[1](f2) # b,1,28,28
                f2 = F.adaptive_avg_pool2d(f2, (7,7))
                f3 = reg_head[2](f3) # b,1,14,14
                f3 = F.adaptive_avg_pool2d(f3, (7,7))
                f4 = reg_head[3](f4) # b,1,7,7
                f = reg_head[4](torch.stack([f1,f2,f3,f4], 1).squeeze()) # b,1,7,7

                reg_outputs = f.squeeze() # b, 7, 7
                reg_outputs = reg_outputs.view([reg_outputs.shape[0], -1]) # b, 49
                reg_outputs = torch.softmax(reg_outputs, -1)
                reg_outputs = reg_outputs.float()
                gt_points = gt_points.float()
                loss2 = 0.01*criterion2(reg_outputs, gt_points)

                loss = loss1 + loss2
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
            total_loss = (running_loss1 + running_loss2) / total
            total_acc = 100 * running_acc / total
            print(f'Test epoch : {epoch} loss : {total_loss} Acc : {total_acc}%')
            print("Loss 1 : ", running_loss1 / total)
            print("Loss 2 : ", running_loss2 / total)
            print("Total Loss : ", (running_loss1+running_loss2) / total)
            
            # if total_loss < minLoss:
            if total_acc > minLoss and epoch!=-1:
                torch.save(model.state_dict(), os.path.join(spath, f'ACC_{total_acc:.2f}.pth'))
                torch.save(model.state_dict(), os.path.join(spath, 'model.pth'))
                if reg_head is not None:
                    reg_head_state_dict = dict()
                    reg_head_state_dict['l1'] = reg_head[0].state_dict()
                    reg_head_state_dict['l2'] = reg_head[1].state_dict()
                    reg_head_state_dict['l3'] = reg_head[2].state_dict()
                    reg_head_state_dict['l4'] = reg_head[3].state_dict()
                    torch.save(reg_head_state_dict, os.path.join(spath, f'reg_{total_acc:.2f}.pth'))
                    torch.save(reg_head_state_dict, os.path.join(spath, 'reg_head.pth'))
                return total_acc
            else:
                return minLoss
