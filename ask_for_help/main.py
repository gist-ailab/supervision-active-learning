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
parser.add_argument('--dpath', type=str, default='/ailab_mat/dataset/ISIC_skin_disease')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/ilsvrc')
parser.add_argument('--epoch1', type=int, default=30)
parser.add_argument('--epoch2', type=int, default=30)
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--query', type=str, default='')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)
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

#------------------------------------------------------------
if args.dataset == 'ISIC2017':
    # trainset = ISIC2017(args.dpath, 'Training', None)
    trainset = ilsvrc30(args.dpath, 'train')
    trainset1 = Subset(trainset, [2*i for i in range(len(trainset)//2)])
    trainset2 = Subset(trainset, [2*i+1 for i in range(len(trainset)//2)])
    
    # valset = ISIC2017(args.dpath, 'Validation')
    valset = ilsvrc30(args.dpath, 'val')
    # testset = ISIC2017(args.dpath, 'Test')
    testset = ilsvrc30(args.dpath, 'test')
    
    trainloader = DataLoader(trainset, args.batch, shuffle=True, num_workers=4)
    train1loader = DataLoader(trainset1, args.batch, shuffle=True, num_workers=4)
    train2loader = DataLoader(trainset2, args.batch, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, args.batch, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, args.batch, shuffle=False, num_workers=4)

    model = models.resnet18()
    model.fc = nn.Linear(512, 30)
    if args.mode == 'base':
        model = model.to(device)
    else:
        return_nodes = {
            'layer1':'l1',
            'layer2':'l2',
            'layer3':'l3',
            'layer4':'l4',
            'fc':'fc'
        }
        model = create_feature_extractor(model, return_nodes=return_nodes)
        model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=1e-3)

if args.mode=='base':
    # bestAcc = 100
    # for i in range(args.epoch1):
    #     train(i, model, train1loader, criterion, optimizer, device)
    #     test(i, model, valloader, criterion, device, bestAcc, save_path)
    
    model.load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/ilsvrc/seed0/baseline/ACC_65.93.pth'))
    bestAcc = 100
    optimizer = optim.SGD(model.parameters(), args.lr*0.1, weight_decay=1e-3)
    for i in range(args.epoch1, args.epoch1+args.epoch2):
        train(i, model, train2loader, criterion, optimizer, device)
        bestAcc = test(i, model, valloader, criterion, device, bestAcc, save_path)
    print(bestAcc)
    # model.load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/seed0/baseline/ACC_84.67.pth'))
    # test(-1, model, testloader, criterion, device, 100, save_path)

if args.mode=='point':
    base_heatmap = torch.zeros((224,224,2))
    for x in range(224):
            for y in range(224):
                base_heatmap[x,y,:] = torch.tensor([y,x])
    # bestAcc = 100
    # for i in range(args.epoch1):
    #     train(i, model, train1loader, criterion, optimizer, device)
    #     test(i, model, valloader, criterion, device, bestAcc, save_path)
    bestAcc = 0
    optimizer = optim.SGD(model.parameters(), args.lr*0.1, weight_decay=1e-3)
    criterion2 = nn.CosineSimilarity(dim=-1)
    selection_loader = DataLoader(trainset2, batch_size=1, shuffle=False)
    s_idx, u_idx = data_selection(model, selection_loader, criterion, ratio=args.ratio, mode='random')
    s_subsetRandomSampler = SubsetRandomSampler(s_idx)
    u_subsetRandomSampler = SubsetRandomSampler(u_idx)
    s_loader = DataLoader(trainset, batch_size=args.batch, shuffle=False, sampler=s_subsetRandomSampler)
    u_loader = DataLoader(trainset, batch_size=args.batch, shuffle=False, sampler=u_subsetRandomSampler)
    model.load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/ilsvrc/seed0/baseline/ACC_60.87.pth'))
    
    for i in range(args.epoch1, args.epoch1+args.epoch2):
        # supervision_train1(i, model, s_loader, u_loader, criterion, criterion2, optimizer, device, base_heatmap, ratio=1.0)
        supervision_train2(i, model, train2loader, s_loader, criterion, criterion2, optimizer, device, base_heatmap)
        bestAcc = test(i, model, valloader, criterion, device, bestAcc, save_path)
    print(bestAcc)