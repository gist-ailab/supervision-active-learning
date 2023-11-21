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
from models.query_models import LossNet
from datasets import *
from utils import *
from config import *
from selection_methods import query_samples

parser = argparse.ArgumentParser()
parser.add_argument('--dpath1', type=str, default='/ailab_mat/dataset/HAM10000/balanced')
parser.add_argument('--dpath2', type=str, default='/ailab_mat/dataset/ISIC_skin_disease/balanced_train')
parser.add_argument('--spath', type=str, default='/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000')
parser.add_argument('--epoch1', type=int, default=60)
parser.add_argument('--epoch2', type=int, default=10)
parser.add_argument('--dataset', type=str, default='ISIC2017')
parser.add_argument('--batch', type=int, default=28)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--mode', type=str, default='point')
parser.add_argument('--note', type=str, default='baseline')

parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-m","--method_type", type=str, default="lloss",
                    help="")

args = parser.parse_args()

random.seed(SEED)
torch.random.manual_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if not os.path.isdir(args.spath):
    os.mkdir(args.spath)

save_path = os.path.join(args.spath, f'seed{SEED}')
if not os.path.isdir(save_path):
    os.mkdir(save_path)
# save_path = os.path.join(save_path, 'Eval_with_Val')
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)
save_path = os.path.join(save_path, args.note)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

if args.dataset == 'ISIC2017':
    # trainset = HAM10000(args.dpath1, mode='train') # 3000
    # print('Num trainset : ', len(trainset))
    # validset = HAM10000(args.dpath1, mode='test') # 450
    # print('Num validset : ', len(validset))
    testset1 = ISIC2017(args.dpath2, mode='train') # 750
    print('Num testset1 : ', len(testset1))
    testset2 = ISIC2017(args.dpath2, mode='test') # 270
    print('Num testset2 : ', len(testset2))
    testset3 = ISIC2017(args.dpath2, mode='val') # 90
    print('Num testset3 : ', len(testset3))

method = args.method_type
methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL', 'HighEntropy', 'LowEntropy']
assert method in methods, 'No method %s! Try options %s'%(method, methods)

for trial in range(TRIALS):
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
    model.load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000/seed0/Eval_with_Val/HAM_baseline/model_61.48.pth'))

    reg_head1 = nn.Sequential(nn.Conv2d(256, 16, (1,1)),
                        nn.Conv2d(16, 1, (1,1)))
    reg_head2 = nn.Sequential(nn.Conv2d(512, 64, (1,1)),
                            nn.Conv2d(64, 1, (1,1)))
    reg_head3 = nn.Sequential(nn.Conv2d(1024, 128, (1,1)),
                            nn.Conv2d(128, 1, (1,1)))
    reg_head4 = nn.Sequential(nn.Conv2d(2048, 512, (1,1)),
                            nn.Conv2d(512, 1, (1,1)))
    reg_head5 = nn.Conv2d(4,1,(1,1))

    reg_head1 = reg_head1.to(device1)
    reg_head2 = reg_head2.to(device1)
    reg_head3 = reg_head3.to(device1)
    reg_head4 = reg_head4.to(device1)
    reg_head5 = reg_head5.to(device1)
    reg_head = [reg_head1,reg_head2,reg_head3,reg_head4, reg_head5]

    reg_head_parameters = list(reg_head[0].parameters()) +\
                          list(reg_head[1].parameters()) +\
                          list(reg_head[2].parameters()) +\
                          list(reg_head[3].parameters()) +\
                          list(reg_head[4].parameters())

    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)

    labeled_set = indices[:ADDENDUM]
    unlabeled_set = [x for x in indices if x not in labeled_set]

    data_unlabeled = ISIC2017(args.dpath2, mode='train')
    train_loader = DataLoader(testset1, batch_size=BATCH, 
                              sampler=SubsetRandomSampler(labeled_set), 
                              pin_memory=True, drop_last=True, num_workers=4)
    val_loader  = DataLoader(testset3, batch_size=BATCH, num_workers=4)
    test_loader  = DataLoader(testset2, batch_size=BATCH, num_workers=4)
    dataloaders  = {'train': train_loader, 'val' : val_loader, 'test': test_loader}
    
    for cycle in range(CYCLES):
        print(f"Trial : {trial}, Cycle : {cycle}")
        subset = unlabeled_set[:]
        SUBSET = len(subset)
        Models = {'backbone' : model}
        if method =='lloss':
            loss_module = LossNet().to(device1)
            Models = {'backbone': model, 'module': loss_module}
        
        if INIT_PARA:
            Models['backbone'].load_state_dict(torch.load('/ailab_mat/personal/heo_yunjae/supervision_active_learning/ask_for_help/parameters/balanced_HAM10000/seed0/Eval_with_Val/HAM_baseline/model_61.48.pth'))
        torch.backends.cudnn.benchmark = True
        
        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()
        criterion3 = nn.CosineSimilarity()

        optim_backbone = optim.Adam(Models['backbone'].parameters(), args.lr, betas=[args.beta1, args.beta2], eps=1e-8)
        optimizer2 = optim.Adam(reg_head_parameters, args.lr2)
        optimizers = {'backbone': optim_backbone, 'reg_head': optimizer2}

        if method == 'lloss':
            optim_module = optim.SGD(Models['module'].parameters(), lr=LR, 
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone, 'reg_head': optimizer2, 'module': optim_module}
        
        minLoss = 999
        for i in range(args.epoch2):
            train(i, Models, dataloaders['train'], criterion, optimizers, device1)
            reg_feat_distil(i, Models, reg_head, dataloaders['train'], criterion, criterion2, criterion3, optimizers, None, device1)
            minLoss = reg_distil_test(i, Models, dataloaders['val'], criterion, criterion2, criterion3, device1, MinLoss, save_path, reg_head)
        print('Min Loss : ', minLoss)
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.pth')))
        Models = {'backbone' : model, 'reg_module' : reg_head}
        if method =='lloss':
            Models = {'backbone': model, 'reg_module' : reg_head, 'module': loss_module}
        test(-1, Models, dataloaders['test'], criterion, device1, minLoss, save_path)
        
        if cycle==CYCLES-1:
            print(f'Trial {trial} Finished')
            break

        arg = query_samples(Models, method, data_unlabeled, subset, labeled_set, cycle, args)
        print(len(arg))
        print(arg[:5])

        labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
        listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
        unlabeled_set = listd + unlabeled_set[SUBSET:]
        dataloaders['train'] = DataLoader(testset1, batch_size=BATCH, 
                                          sampler=SubsetRandomSampler(labeled_set), 
                                          pin_memory=True, num_workers=4)
