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
parser.add_argument('--dpath2', type=str, default='/ailab_mat/dataset/ISIC_skin_disease/balanced_train')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000')
parser.add_argument('--epoch1', type=int, default=60)
parser.add_argument('--epoch2', type=int, default=10)
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--query', type=str, default='')
parser.add_argument('--batch', type=int, default=28)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr2', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--mode', type=str, default='point')
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--note', type=str, default='')
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
save_path = os.path.join(save_path, 'Eval_with_Val_origin_val_test')
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
    testset1_idx = [i for i in range(750)]
    random.shuffle(testset1_idx)
    selected = testset1_idx[:int(len(testset1_idx)*args.ratio)]
    testSubset1 = Subset(testset1, selected)
    print('Num testset1 : ', len(testset1))
    print('Num Selected : ', len(testSubset1))
    testset2 = ISIC2017(args.dpath2, mode='test') # 600
    print('Num testset2 : ', len(testset2))
    testset3 = ISIC2017(args.dpath2, mode='val') # 150
    print('Num testset3 : ', len(testset3))
    
    trainloader = DataLoader(trainset, args.batch, shuffle=True, num_workers=4)
    valloader = DataLoader(validset, args.batch, shuffle=False, num_workers=4)
    testloader1 = DataLoader(testSubset1, args.batch, shuffle=True, num_workers=4)
    testloader2 = DataLoader(testset2, args.batch, shuffle=False, num_workers=4)
    testloader3 = DataLoader(testset3, args.batch, shuffle=False, num_workers=4)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(2048, 3)
if args.mode == 'base':
    model = model.to(device1)
else:
    return_nodes = {
        'layer1':'l1',
        'layer2':'l2',
        'layer3':'l3',
        'layer4':'l4',
        'fc':'fc'
    }
    model = create_feature_extractor(model, return_nodes=return_nodes)
    model = model.to(device1)
# reg_head = nn.Conv2d(2048, 1, (1,1))
reg_head1 = nn.Conv2d(256, 1, (1,1))
reg_head2 = nn.Conv2d(512, 1, (1,1))
reg_head3 = nn.Conv2d(1024, 1, (1,1))
reg_head4 = nn.Conv2d(2048, 1, (1,1))

reg_head1 = reg_head1.to(device1)
reg_head2 = reg_head2.to(device1)
reg_head3 = reg_head3.to(device1)
reg_head4 = reg_head4.to(device1)
reg_head = [reg_head1,reg_head2,reg_head3,reg_head4]

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
    model.load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000/seed0/Eval_with_Val/HAM_baseline/model_61.48.pth'))
    
    optimizer = optim.Adam(model.parameters(), args.lr, betas=[args.beta1, args.beta2], eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    # criterion2 = nn.CrossEntropyLoss()
    criterion2 = nn.BCELoss()
    # selection_loader = DataLoader(testset1, batch_size=1, shuffle=False)
    # s_idx, entropy_list = data_selection(model, selection_loader, criterion, device1, ratio=args.ratio, mode='low_entropy')
    
    # s_subsetRandomSampler = SubsetRandomSampler(s_idx)
    # s_loader = DataLoader(testset1, batch_size=args.batch, shuffle=False, sampler=s_subsetRandomSampler)
            
    # for name, param in model.named_parameters():
    #     if 'layer4' in name:
    #         param.requires_grad = True
    
    MinLoss = 100.0
    # test(-1, model, testloader2, criterion, device1, MinLoss, save_path)
    # metric(model, testloader2, num_classes=3, device=device1)
    
    # MinLoss = 999
    MaxAcc = 999

    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # optimizer2 = optim.Adam(reg_head.parameters(), 0.001)
    # for i in range(10):
    #     point_regression(-1, model, reg_head, testloader1, criterion, criterion2, optimizer, optimizer2, device1)
    # for name, param in model.named_parameters():
    #     param.requires_grad = True
    reg_head_parameters = list(reg_head[0].parameters()) +\
                          list(reg_head[1].parameters()) +\
                          list(reg_head[2].parameters()) +\
                          list(reg_head[3].parameters())
    optimizer2 = optim.Adam(reg_head_parameters, args.lr2)
    for i in range(0, args.epoch2):
        point_regression3(i, model, reg_head, testloader1, criterion, criterion2, optimizer, device1)
        MaxAcc = regression_test3(i, model, testloader3, criterion, criterion2, device1, MaxAcc, save_path, reg_head)
        # MaxAcc = test(i, model, testloader3, criterion, device1, MaxAcc, save_path, reg_head)
        # metric(model, testloader2, num_classes=3, device=device1)
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
    # MinLoss = 0
    # MaxAcc = 100
    test(-1, model, testloader2, criterion, device1, MaxAcc, save_path)
    # torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    # print(MaxAcc)