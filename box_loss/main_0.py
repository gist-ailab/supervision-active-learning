import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/yunjae_heo/SSD/yunjae.heo/chestx-det')
parser.add_argument('--save_path', type=str, default='/home/yunjae_heo/workspace/ailab_mat/Parameters/supervision/box_loss')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['loss'], default='loss')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = args.episode
if not os.path.isdir(args.data_path):
    os.mkdir(args.data_path)
if not args.seed==None:
    save_path = os.path.join(args.save_path, f'seed{args.seed}',args.query_algorithm)
else:
    save_path = os.path.join(args.save_path, 'current', args.query_algorithm)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
if __name__ == "__main__":
    selected = []
    trainset = chestX(args.data_path, 'train', selected)
    testset = chestX(args.data_path, 'test', [])
    trainloader = DataLoader(trainset, args.batch_size, drop_last=True, shuffle=True)
    testloader = DataLoader(testset, args.batch_size, drop_last=False, shuffle=False)
    
    model = ResNet18()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    classif_loss = nn.CrossEntropyLoss()
    