import os,sys
import torch
import torchvision
import argparse
import tqdm
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from datasets import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dpath', type=str, default='/ailab_mat/dataset/bdd100k')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/bdd100k')
parser.add_argument('--epoch1', type=int, default=100)
parser.add_argument('--epoch2', type=int, default=50)
parser.add_argument('--dataset', type=str, default='BDD100k')
parser.add_argument('--query', type=str, default='')
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--note', type=str, default='baseline')
args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.isdir(args.dpath):
    os.mkdir(args.dpath)
if not os.path.isdir(args.spath):
    os.mkdir(args.spath)

if not args.seed==None:
    save_path = os.path.join(args.spath, f'seed{args.seed}')
else:
    save_path = os.path.join(args.spath, 'current')
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
save_path = os.path.join(save_path, args.note)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
#---------------------------------------------------------------------------
trainset = BDD100K(args.dpath, 'train')
valset = BDD100K(args.dpath, 'val')

trainSubset = Subset(trainset, [i for i in range(0, len(trainset), 10)])
valSubset = Subset(valset, [i for i in range(0, len(valset), 10)])

trainloader = DataLoader(trainSubset, args.batch, shuffle=True, num_workers=4, collate_fn=collate_fn)
valloader = DataLoader(valSubset, args.batch, shuffle=False, num_workers=4, collate_fn=collate_fn)

model = models.detection.fasterrcnn_resnet50_fpn_v2(
    progress = True,
    weights_backbone = models.ResNet50_Weights,
    weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights
)
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 10)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 10*4)
# print(torch.cuda.device_count())
# model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.3)

bestAcc = 0
bestLoss = 100.0
for i in range(args.epoch1):
    train_loss = train_detector(i, model, trainloader, optimizer, device)
    print(train_loss)
    val_loss = val_detector(i, model, valloader, device, bestAcc, args.spath)
    print(val_loss)
    if np.isnan(val_loss):
        print(i)
        break
    if val_loss <= bestLoss:
        torch.save(model.state_dict(), os.path.join(args.spath, 'model_para10.pth'))
        bestLoss = val_loss