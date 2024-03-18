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
from model import PPM

parser = argparse.ArgumentParser()
parser.add_argument('--dpath1', type=str, default='/ailab_mat/dataset/HAM10000/')
# parser.add_argument('--dpath2', type=str, default='/home/yunjae_heo/datas/CUB_dataset/datas')
# parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/CUB200')
# parser.add_argument('--dataset', type=str, default='CUB200')
parser.add_argument('--dpath2', type=str, default='/SSDg/yjh/datas/isic2017/augx60_dataset')
# parser.add_argument('--dpath2', type=str, default='/SSDg/yjh/datas/isic2017/imageFolder')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/HAM10000')
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--pretrained', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/HAM10000/seed0/ham10000_origin/model.pth')
parser.add_argument('--epoch1', type=int, default=20)
parser.add_argument('--epoch2', type=int, default=20)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--mode', type=str, default='point')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--num_train', type=int, default=20)
parser.add_argument('--num_trial', type=int, default=10)
parser.add_argument('--f_size', type=int, default=7)
args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
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
# save_path = os.path.join(save_path, 'melanoma_classification')
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)
save_path = os.path.join(save_path, args.note)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#------------------------------------------------------------
if args.dataset == 'ISIC2017':
    trainset = HAM10000_origin(args.dpath1, mode='train') # 10015
    print('Num trainset : ', len(trainset))
    validset = HAM10000_origin(args.dpath1, mode='test') # 1511
    print('Num validset : ', len(validset))
    testset1 = ISIC2017_2(args.dpath2, mode='train') # 750
    print('Num testset1 : ', len(testset1))
    testset2 = ISIC2017_2(args.dpath2, mode='test') # 600
    print('Num testset2 : ', len(testset2))
    testset3 = ISIC2017_2(args.dpath2, mode='val') # 150
    print('Num testset3 : ', len(testset3))
    
    trainloader = DataLoader(trainset, args.batch, shuffle=True, num_workers=4)
    valloader = DataLoader(validset, args.batch, shuffle=False, num_workers=4)
    testloader1 = DataLoader(testset1, args.batch, shuffle=True, num_workers=4)
    testloader2 = DataLoader(testset2, args.batch, shuffle=False, num_workers=4)
    testloader3 = DataLoader(testset3, args.batch, shuffle=False, num_workers=4)

    model = init_model(device=device1)

if args.dataset == 'CUB200':
    testset1 = CUB200(args.dpath2, mode='train', num_train=args.num_train)
    testset2 = CUB200(args.dpath2, mode='test')
    testset3 = CUB200(args.dpath2, mode='val')
    print('Num testset1 : ', len(testset1))
    print('Num testset2 : ', len(testset2))
    print('Num testset3 : ', len(testset3))

    testloader1 = DataLoader(testset1, args.batch, shuffle=True, num_workers=4)
    testloader2 = DataLoader(testset2, args.batch, shuffle=False, num_workers=4)
    testloader3 = DataLoader(testset3, args.batch, shuffle=False, num_workers=4)

    model = PPM(num_class=200)
    model = model.to(device1)

torch.backends.cudnn.benchmark = True
optimizer = optim.Adam(model.parameters(), args.lr)
criterion = nn.CrossEntropyLoss()

if args.mode=='base':
    MinLoss = 0.0
    for i in range(args.epoch1):
        train(i, model, trainloader, criterion, optimizer, device1)
        MinLoss = test(i, model, valloader, criterion, device1, MinLoss, save_path)
    print(MinLoss)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
    test(-1, model, testloader2, criterion, device1, MinLoss, save_path)
    # metric(model, testloader1, num_classes=3, device=device1)

if args.mode=='point':
    for trial in range(args.num_trial):
        print('Trial : ', trial)
        if args.dataset == 'ISIC2017':
            model = init_model(device=device1, name='resnet50')
        if args.dataset == 'CUB200':
            model = PPM(num_class=200)
            model = model.to(device1)

        optimizer = optim.Adam(model.parameters(), args.lr, betas=[args.beta1, args.beta2], eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        # criterion2 = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()

        MinLoss = 999
        for i in range(0, args.epoch1):
            train(i, model, testloader1, criterion, optimizer, device1)
            MinLoss = test(i, model, testloader3, criterion, device1, MinLoss, save_path)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
        # test(-1, model, testloader2, criterion, device1, MinLoss, save_path)
        if args.dataset == 'CUB200': num_classes=200
        if args.dataset == 'ISIC2017': num_classes=3
        metric(model, testloader2, num_classes=num_classes, device=device1)

        selects, unselects = select_wrongs(model, testloader1, device1)

        model = init_model(device=device1, name='resnet18')
        optimizer = optim.SGD(model.parameters(), args.lr)
        # optimizer = optim.Adam(model.parameters(), args.lr, betas=[args.beta1, args.beta2], eps=1e-8)
        criterion2 = nn.BCELoss()

        MinLoss = 999
        for i in range(0, args.epoch2):
            semi_point_prediction(i, model, testloader1, criterion, criterion2, optimizer, device1, selected=selects)
            MinLoss = test(i, model, testloader3, criterion, device1, MinLoss, save_path)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
        # test(-1, model, testloader2, criterion, device1, MinLoss, save_path)
        if args.dataset == 'CUB200': num_classes=200
        if args.dataset == 'ISIC2017': num_classes=3
        metric(model, testloader2, num_classes=num_classes, device=device1)