import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet import *

def get_rand_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})

def test(epoch, model, test_loader, criterion, save_path, sign, device, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
        if not os.path.isdir(os.path.join(save_path,sign)):
            os.mkdir(os.path.join(save_path,sign))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_path,sign,'model.pt'))
    return acc

def query_test(epoch, model, test_loader, criterion, save_path, sign, device, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
        if not os.path.isdir(os.path.join(save_path,sign)):
            os.mkdir(os.path.join(save_path,sign))
        if acc > best_acc:
            torch.save(model.state_dict(), os.path.join(save_path,sign,'query_model.pt'))
    return acc

def binary_train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
        
    print(f'epoch : {epoch} _________________________________________________')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.view(inputs.shape[0], -1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})

def domain_gap_prediction(model, criterion, ulbl_loader, ulbl_idx, sign, device, K):
    model.eval()
    if sign=='low_conf':
        conf_list = torch.tensor([]).to(device)
        with torch.no_grad():
            pbar = tqdm(ulbl_loader)
            for i, (inputs, _) in enumerate(pbar):
                inputs = inputs.to(device)
                outputs = model(inputs)
                confidence = torch.max(F.softmax(outputs, dim=1),dim=1)
                conf_list = torch.cat((conf_list,confidence.values),0)
            arg = conf_list.argsort().cpu().numpy()
        return list(arg[:K])
    
    if sign=='high_entropy':
        entr_list = torch.tensor([]).to(device)
        with torch.no_grad():
            pbar = tqdm(ulbl_loader)
            for i, (inputs, _) in enumerate(pbar):
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
                entropy = -outputs*outputs.log()
                entropy = entropy.sum(dim=1)
                entr_list = torch.cat((entr_list,entropy),0)
            arg = entr_list.argsort().cpu().numpy()
        return list(arg[-K:])
    
    if sign=='jensen':
        div_list = torch.tensor([]).to(device)
        with torch.no_grad():
            pbar = tqdm(ulbl_loader)
            for i, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.shape[0], -1)
                targets = targets.to(device)
                outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(targets, outputs)
                js_div = loss1[:,0]+loss2[:,0]
                div_list = torch.cat((div_list,js_div),0)
            arg = div_list.argsort().cpu().numpy()
        return list(arg[:K])
    
    if sign=='kld':
        div_list = torch.tensor([]).to(device)
        with torch.no_grad():
            pbar = tqdm(ulbl_loader)
            for i, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.shape[0], -1)
                targets = targets.to(device)
                outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                kld_div = loss1[:,0]
                div_list = torch.cat((div_list,kld_div),0)
            arg = div_list.argsort().cpu().numpy()
        return list(arg[:K])
    
    if sign=='pad':
        div_list = torch.tensor([]).to(device)
        with torch.no_grad():
            pbar = tqdm(ulbl_loader)
            for i, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(device)
                inputs = inputs.view(inputs.shape[0], -1)
                targets = targets.to(device)
                outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                pad_div = loss1
                div_list = torch.cat((div_list,pad_div),0)
            arg = div_list.argsort().cpu().numpy()
        return list(arg[-K:])
    
    if sign=='random':
        return list(np.random.randint(0, len(ulbl_idx), size=K))
    
def query_algorithm(model, criterion, ulbl_loader, ulbl_idx, device, model_paths, K):
    model_dict = dict()
    for i in range(len(model_paths)):
        model_dict[i] = ResNet18().to(device)
        model_dict[i].load_state_dict(torch.load(model_paths[i]))
    
    conf_list = torch.tensor([]).to(device)
    with torch.no_grad():
        pbar = tqdm(ulbl_loader)
        for i, (inputs, _) in enumerate(pbar):
            inputs = inputs.to(device)
            temp_tensor = torch.tensor([]).to(device)
            for j in range(len(model_dict)):
                outputs = model_dict[j](inputs)
                confidence = torch.max(F.softmax(outputs, dim=1),dim=1)
                if len(temp_tensor)==0:
                    temp_tensor = torch.cat((temp_tensor, confidence.values), 0)
                else:
                    temp_tensor = temp_tensor + confidence.values
            conf_list = torch.cat((conf_list,temp_tensor),0)
        arg = conf_list.argsort().cpu().numpy()
    return list(arg[:K])

def model_freeze(model):
    for _,child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False

def model_unfreeze(model):
    for _,child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True