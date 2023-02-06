import os,sys
import cv2
from dataset import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from resnet import *
from PIL import Image

torch.random.manual_seed(20230160)

data_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
# selected = [i for i in range(0,15849)]
# trainset = ilsvrc30(data_path, 'train', selected)
# test_loader = DataLoader(trainset, 1, drop_last=True, shuffle=True, num_workers=4)

selected = [i for i in range(0,1500)]
testset = ilsvrc30(data_path, 'val', selected)
test_loader = DataLoader(testset, 1, drop_last=True, shuffle=True, num_workers=4)

# selected = [i for i in range(0,3001)]
# trainset = chestX(data_path, 'train', selected)
# train_loader = DataLoader(trainset, 32, drop_last=True, shuffle=True, num_workers=4)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = '/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/all/seed10/loss4/0_20.614_model.pt'
model_path = '/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/loss_1500/seed20/loss4/test_-1_40.333_model.pt'
# model_path = '/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/imagenet30/box_loss/zero/seed5/loss/32_74.183_model.pt'
model = ResNet18(num_classes=30)
model = model.to(device)

model_para = torch.load(model_path)
# model_para['model'].update(model_para['linear'])
model.load_state_dict(model_para['model'])
classif_loss = nn.CrossEntropyLoss()

pbar = tqdm(test_loader)
model.eval()
for idx, (images, labels, heatmaps, img_id) in enumerate(pbar):
    images2 = torch.cat((images[:,2,:], images[:,1,:,:], images[:,0,:,:]), dim=0)
    labels = labels.to(device)
    # print(images.shape)
    images = images.to(device)
    outputs, acts = model(images)
    
    outputs = torch.softmax(outputs, dim=-1)
    conf, predicted = outputs.max(1)
    print(conf)
    print(labels, predicted)
    # if predicted.eq(labels).sum().item():
        # print(labels, predicted)
        # print(predicted.eq(labels).sum().item())
        
    save_image(heatmaps, './temp_heatmap.png')
    save_image(images2, './temp_img.png')
    
    b,c,h,w = acts.shape
    weight = list(model.parameters())[-2].data
    beforDot = torch.reshape(acts, (b,c,h*w))
    weights = torch.stack([weight[i].unsqueeze(0) for i in predicted], dim=0)

    cam = torch.bmm(weights, beforDot)
    cam = torch.reshape(cam, (b, h, w))
    # print("1",cam.shape)
    cam = torch.stack([cam[i]-torch.min(cam[i]) for i in range(b)], dim=0)
    cam = torch.stack([cam[i]/torch.max(cam[i]) for i in range(b)], dim=0)
    # print("2",cam.shape)
    cam = cam.unsqueeze(dim=0)
    # cam = cam.unsqueeze(dim=0)
    pred_hmap = F.interpolate(cam, size=(256,256))
    # save_image(heatmap_output, './temp_output_heatmap.png')
    save_image(pred_hmap, './temp_output_heatmap.png')
    break


# def test(epoch, best_acc):
#         model.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         max_loss = 0
#         min_loss = 999
#         with torch.no_grad():
#             pbar = tqdm(test_loader)
#             for idx, (images, labels, _, img_id) in enumerate(pbar):
#                 images, labels = images.to(device), labels.to(device)
#                 outputs, _ = model(images)
                
#                 loss = classif_loss(outputs, labels)
#                 if max_loss < loss: max_loss = loss
#                 if min_loss > loss: min_loss = loss
                
#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 # print(predicted, labels)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
#                 pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
#             acc = 100*correct/total
#             print(max_loss, min_loss)
#             return best_acc 
        
# best_acc = 0
# best_acc = test(0, best_acc)
# print(best_acc)