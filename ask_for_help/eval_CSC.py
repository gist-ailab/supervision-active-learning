import os,sys
import torch
import torchvision
import argparse
import tqdm
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms.functional as F2
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, WeightedRandomSampler
from model import *
from datasets import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dpath1', type=str, default='/ailab_mat/dataset/HAM10000/')
parser.add_argument('--dpath2', type=str, default='/SSDg/yjh/datas/isic2017/imageFolder')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/CSCModel')
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--epoch1', type=int, default=20)
parser.add_argument('--epoch2', type=int, default=30)
parser.add_argument('--epoch3', type=int, default=30)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr1', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-4)
parser.add_argument('--lr3', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--note', type=str, default='')
parser.add_argument('--num_trial', type=int, default=1)
args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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

testset1 = ISIC2017_2(args.dpath2, mode='train') # 750
print('Num testset1 : ', len(testset1))
testset2 = ISIC2017_2(args.dpath2, mode='test') # 600
print('Num testset2 : ', len(testset2))
testset3 = ISIC2017_2(args.dpath2, mode='val') # 150
print('Num testset3 : ', len(testset3))

testloader1 = DataLoader(testset1, args.batch, shuffle=True, num_workers=4)
testloader2 = DataLoader(testset2, args.batch, shuffle=False, num_workers=4)
testloader3 = DataLoader(testset3, args.batch, shuffle=False, num_workers=4)

cls1_model = CSC_cls1(num_class=2, model_name='resnet50').cuda()
cls1_model = nn.DataParallel(cls1_model, device_ids=[0,1,2,3])
cls1_model.load_state_dict(torch.load(os.path.join(save_path, 's1_model.pth')))

num_classes=2
metric(cls1_model, testloader2, num_classes=num_classes, device=device1)

seg_model = CSC_Seg(num_class=1, model_name='resnet50').cuda()
seg_model = nn.DataParallel(seg_model, device_ids=[0,1,2,3])
seg_model.load_state_dict(torch.load(os.path.join(save_path, 's2_segmodel.pth')))
test_seg(-1, seg_model, cls1_model, testloader3, criterion2, criterion3, device1, s2_minloss, save_path, mode='step2')

cls2_model = CSC_cls2(num_class=2, model_name='resnet50').cuda()
cls2_model = nn.DataParallel(cls2_model, device_ids=[0,1,2,3])
cls2_model.load_state_dict(torch.load(os.path.join(save_path, 's3_model.pth')))

num_classes=2
csc_metric(cls2_model, cls1_model, seg_model, testloader2, num_classes=num_classes, device=device1)

