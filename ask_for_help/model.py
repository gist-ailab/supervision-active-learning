import os,sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

class PPM(torch.nn.Module):
    def __init__(self, num_class=200, model_name='resnet50'):
        super(PPM, self).__init__()
        if model_name=='resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            module_name = list()
            for name, module in self.backbone.named_modules():
                name = name.split('.')[0]
                if not name=='' and name not in module_name:
                    module_name.append(name)
            self.module_dict = nn.ModuleDict()
            for name in module_name:
                self.module_dict[name] = getattr(self.backbone, name)
            self.module_dict['fc'] = nn.Linear(2048, num_class)

            self.module_dict['pos_head1'] = nn.Conv2d(256, 1, 1, 1)
            self.module_dict['pos_head2'] = nn.Conv2d(512, 1, 1, 1)
            self.module_dict['pos_head3'] = nn.Conv2d(1024, 1, 1, 1)
            self.module_dict['pos_head4'] = nn.Conv2d(2048, 1, 1, 1)
            self.module_dict['adaptive_pool'] = nn.AdaptiveAvgPool2d((7,7))

            self.module_dict['pos_bn1'] = nn.BatchNorm2d(256)
            self.module_dict['pos_bn2'] = nn.BatchNorm2d(512)
            self.module_dict['pos_bn3'] = nn.BatchNorm2d(1024)
            self.module_dict['pos_bn4'] = nn.BatchNorm2d(2048)

    def forward(self, x):
        outputs = self.module_dict['conv1'](x)
        outputs = self.module_dict['bn1'](outputs)
        outputs = self.module_dict['relu'](outputs)
        outputs = self.module_dict['maxpool'](outputs)
        
        outputs = self.module_dict['layer1'](outputs)
        feat1 = outputs
        pos1 = self.module_dict['pos_head1'](outputs)
        pos1 = self.module_dict['adaptive_pool'](pos1)
        weight = self.module_dict['pos_head1'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn1'](outputs)
        outputs = feat1 + outputs
        
        outputs = self.module_dict['layer2'](outputs)
        feat2 = outputs
        pos2 = self.module_dict['pos_head2'](outputs)
        pos2 = self.module_dict['adaptive_pool'](pos2)
        weight = self.module_dict['pos_head2'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn2'](outputs)
        outputs = feat2 + outputs
        
        outputs = self.module_dict['layer3'](outputs)
        feat3 = outputs
        pos3 = self.module_dict['pos_head3'](outputs)
        pos3 = self.module_dict['adaptive_pool'](pos3)
        weight = self.module_dict['pos_head3'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn3'](outputs)
        outputs = feat3 + outputs
        
        outputs = self.module_dict['layer4'](outputs)
        feat4 = outputs
        pos4 = self.module_dict['pos_head4'](outputs)
        pos4 = self.module_dict['adaptive_pool'](pos4)
        weight = self.module_dict['pos_head4'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn4'](outputs)
        outputs = feat4 + outputs
        
        # print('1 ', outputs.shape)
        outputs = self.module_dict['avgpool'](outputs)
        # print('2 ', outputs.shape)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.module_dict['fc'](outputs)
        # print('3 ', outputs.shape)
        
        pos_outputs = torch.mean(torch.stack([pos1, pos2, pos3, pos4], 1), 1)
        pos_outputs = pos_outputs.squeeze()

        return outputs, pos_outputs

class CHConcat(torch.nn.Module):
    def __init__(self, num_class):
        super(CHConcat, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        module_name = list()
        for name, module in self.backbone.named_modules():
            name = name.split('.')[0]
            if not name=='' and name not in module_name:
                module_name.append(name)
        self.module_dict = nn.ModuleDict()
        for name in module_name:
            self.module_dict[name] = getattr(self.backbone, name)
        self.module_dict['conv1'] = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        self.module_dict['fc'] = nn.Linear(2048, num_class)

    def forward(self, x):
        outputs = self.module_dict['conv1'](x)
        outputs = self.module_dict['bn1'](outputs)
        outputs = self.module_dict['relu'](outputs)
        outputs = self.module_dict['maxpool'](outputs)
        
        outputs = self.module_dict['layer1'](outputs)
        outputs = self.module_dict['layer2'](outputs)
        outputs = self.module_dict['layer3'](outputs)      
        outputs = self.module_dict['layer4'](outputs)     
        # print('1 ', outputs.shape)
        outputs = self.module_dict['avgpool'](outputs)
        # print('2 ', outputs.shape)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.module_dict['fc'](outputs)
        # print('3 ', outputs.shape)
        return outputs

class ICICNet(torch.nn.Module): # intra class consistency and inter class discrimination
    def __init__(self, num_class, model_name):
        super(ICICNet, self).__init__()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class seghead(torch.nn.Module):
    def __init__(self, num_classes, model_name):
        super(seghead, self).__init__()
        if model_name == 'resnet50':
            self.up1 = UpConv(2048, 1024)
            self.conv1 = ConvBlock(2048, 1024)
            self.up2 = UpConv(1024, 512)
            self.conv2 = ConvBlock(1024, 512)
            self.up3 = UpConv(512, 256)
            self.conv3 = ConvBlock(512, 256)
            self.up4 = UpConv(256, 128)
            self.conv4 = ConvBlock(128, 128)
            self.up5 = UpConv(128, 64)
            self.conv5 = ConvBlock(64, 64)

            self.final = nn.Conv2d(64, num_classes, kernel_size=1)
            if num_classes == 1: self.normalize = nn.Sigmoid()
            else: self.normalize = nn.Softmax(dim=1)

    def forward(self, d1, d2, d3, d4):
        # x : input batch
        # d1 : outputs of backbone layer1
        # d2 : outputs of backbone layer2
        # d3 : outputs of backbone layer3
        # d4 : outputs of backbone layer4

        # 디코더 + 스킵 연결
        u1 = self.up1(d4)
        u1 = torch.cat((u1, d3), dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat((u2, d2), dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat((u3, d1), dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = self.conv4(u4)

        u5 = self.up5(u4)
        u5 = self.conv5(u5)

        out = self.final(u5)
        out = self.normalize(out)
        out = out.squeeze()
        # 최종 출력
        return out

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradients(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]

    def __call__(self, x, class_idx):
        output = self.model(x)
        if type(output)==dict:
            output = output['fc']
        self.model.zero_grad()
        class_loss = output[0, class_idx]
        class_loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activation[0]
        for i in range(activation.shape[0]):
            activation[i] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).cpu()
        try:
            heatmap = np.maximum(heatmap, 0)
        except:
            heatmap = np.maximum(heatmap.detach().cpu(), 0)
        heatmap /= torch.max(heatmap)

        return heatmap

