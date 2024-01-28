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
from model import *
from datasets import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dpath1', type=str, default='/ailab_mat/dataset/HAM10000/')
# parser.add_argument('--dpath2', type=str, default='/home/yunjae_heo/datas/CUB_dataset/datas')
# parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/CUB200')
# parser.add_argument('--dataset', type=str, default='CUB200')
parser.add_argument('--dpath2', type=str, default='/SSDg/yjh/datas/isic2017/augx60_dataset')
# parser.add_argument('--dpath2', type=str, default='/SSDg/yjh/datas/isic2017/imageFolder')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/HAM10000')
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--epoch1', type=int, default=10)
parser.add_argument('--epoch2', type=int, default=10)
parser.add_argument('--epoch3', type=int, default=10)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr1', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-5)
parser.add_argument('--lr3', type=float, default=1e-5)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--num_trial', type=int, default=10)
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
save_path = os.path.join(save_path, args.note)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

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

    for trial in range(args.num_trial):
        model = init_model(device=device1, name='resnet50')
        optimizer = optim.Adam(model.parameters(), args.lr1, betas=[args.beta1, args.beta2], eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        
        # Train Backbone
        print("STEP 1 ---------------------------------------------------------------------")
        s1_minloss = 999
        for i in range(0, args.epoch1):
            train(i, model, testloader1, criterion, optimizer, device1)
            s1_minloss = test(i, model, testloader3, criterion, device1, s1_minloss, save_path, mode='step1')
        model.load_state_dict(torch.load(os.path.join(save_path, 's1_model.pth')))
        num_classes=3
        metric(model, testloader2, num_classes=num_classes, device=device1)

        # Mask Similarity
        print("STEP 2 ---------------------------------------------------------------------")
        for name, para in model.named_paramters():
            para.requires_grad = False
            if 'fc' in name:
                para.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), args.lr2, betas=[args.beta1, args.beta2], eps=1e-8)

        segHead = seghead(num_classes=1, model_name='resnet50')
        segHead = segHead.to(device1)
        criterion2 = nn.BCELoss()
        optimizer2 = optim.Adam(segHead.parameters(), args.lr2, betas=[args.beta1, args.beta2], eps=1e-8)

        model.eval()
        target_layer = model.layer4.get_submodule('2').conv3.eval()
        grad_cam = GradCAM(model, target_layer)

        s2_minloss = 999
        for i in range(0, args.epoch2):
            # train_edge_similarity()
            train_mask_similarity(i, model, segHead, grad_cam, testloader1, criterion, criterion2, optimizer, optimizer2, device1)
            s2_minloss = test(i, model, testloader3, criterion, device1, s2_minloss, save_path, mode='step2')
        model.load_state_dict(torch.load(os.path.join(save_path, 's2_model.pth')))
        num_classes=3
        metric(model, testloader2, num_classes=num_classes, device=device1)

        # Wrong Data Correction
        print("STEP 3 ---------------------------------------------------------------------")
        # for para in model.parameters():
        #     para.requires_grad = True
        # optimizer = optim.Adam(model.parameters(), args.lr3, betas=[args.beta1, args.beta2], eps=1e-8)
        # selects, unselects = select_wrongs(model, testloader1, device1)
        
        # s3_minloss = 999
        # for i in range(0, args.epoch2):
        #     wrong_data_correction()
        #     s3_minloss = test(i, model, testloader3, criterion, device1, s3_minloss, save_path, mode='step3')
        # model.load_state_dict(torch.load(os.path.join(save_path, 's3_model.pth')))
        # num_classes=3
        # metric(model, testloader2, num_classes=num_classes, device=device1)