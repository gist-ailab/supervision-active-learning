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
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, WeightedRandomSampler
from datasets import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dpath1', type=str, default='/ailab_mat/dataset/HAM10000/balanced')
parser.add_argument('--dpath2', type=str, default='/ailab_mat/dataset/ISIC_skin_disease/balanced')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000')
parser.add_argument('--epoch1', type=int, default=60)
parser.add_argument('--epoch2', type=int, default=20)
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--query', type=str, default='')
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
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
    trainset = HAM10000(args.dpath1, mode='train') # 3000
    print('Num trainset : ', len(trainset))
    validset = HAM10000(args.dpath1, mode='test') # 450
    print('Num validset : ', len(validset))
    testset1 = ISIC2017(args.dpath2, mode='train') # 750
    print('Num testset1 : ', len(testset1))
    testset2 = ISIC2017(args.dpath2, mode='test') # 270
    print('Num testset2 : ', len(testset2))
    
    trainloader = DataLoader(trainset, args.batch, shuffle=True, num_workers=4)
    valloader = DataLoader(validset, args.batch, shuffle=False, num_workers=4)
    testloader1 = DataLoader(testset1, args.batch, shuffle=False, num_workers=4)
    testloader2 = DataLoader(testset2, args.batch, shuffle=False, num_workers=4)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 3)
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
optimizer = optim.Adam(model.parameters(), args.lr)
criterion = nn.CrossEntropyLoss()


if args.mode=='base':
    bestAcc = 0.0
    for i in range(args.epoch1):
        train(i, model, trainloader, criterion, optimizer, device)
        bestAcc = test(i, model, valloader, criterion, device, bestAcc, save_path)
    print(bestAcc)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
    test(-1, model, testloader2, criterion, device, bestAcc, save_path)
    # metric(model, testloader1, num_classes=3, device=device)

if args.mode=='point':
    model.load_state_dict(torch.load(os.path.join(save_path, '..', 'baseline','model.pth')))
    
    optimizer = optim.Adam(model.parameters(), args.lr, betas=[args.beta1, args.beta2], eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.L1Loss()
    # criterion2 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    selection_loader = DataLoader(testset1, batch_size=1, shuffle=False)
    s_idx, entropy_list = data_selection(model, selection_loader, criterion, device, ratio=args.ratio, mode='low_entropy')
    
    s_subsetRandomSampler = SubsetRandomSampler(s_idx)
    s_loader = DataLoader(testset1, batch_size=args.batch, shuffle=False, sampler=s_subsetRandomSampler)
    
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    
    # for name, param in model.named_parameters():
    #     if 'fc' in name or 'layer4' in name:
    #         param.requires_grad = True
            
    # for name, param in model.named_parameters():
    #     if 'layer4' in name:
    #         param.requires_grad = True
    
    bestAcc = 100.0
    # test(-1, model, testloader2, criterion, device, bestAcc, save_path)
    # metric(model, testloader2, num_classes=3, device=device)
    bestAcc = 0.0
    for i in range(0, args.epoch2):
        # activation_map_matching(i, model, s_loader, criterion, criterion2, optimizer, device)
        train(i, model, s_loader, criterion, optimizer, device)
        bestAcc = test(i, model, testloader2, criterion, device, bestAcc, save_path)
        metric(model, testloader2, num_classes=3, device=device)
    torch.save(model.state_dict, os.path.join(save_path, 'model.pth'))
    print(bestAcc)