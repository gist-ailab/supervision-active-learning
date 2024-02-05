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
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class scSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(scSE, self).__init__()
        self.avgpool = nn.adaptive_avg_pool2d((1,1))
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, (1,1))
        self.conv2 = nn.Conv2d(in_channels, in_channels//16, (1,1))
        self.conv3 = nn.Conv2d(in_channels//16,out_channels//2, (1,1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        branch1 = self.sigmoid(self.conv3(self.relu(self.conv2(self.avgpool(x)))))
        branch2 = self.sigmoid(self.conv1(x))
        att1 = branch1*x
        att2 = branch2*x
        out = torch.concat((att1, att2), 1)
        return out

class SCAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCAttention, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.scSE1 = scSE(in_channels, in_channels)
        self.scSE2 = ScSE(in_channels, out_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        out = self.up(x)
        out = self.scSE1(out)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.scSE2(out)
        out = self.dropout(out)
        return out

class SCAttentionOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCAttention, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.scSE1 = scSE(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        out = self.up(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.scSE1(out)
        out = self.dropout(out)
        return out


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

            self.final = nn.Conv2d(256, num_classes, kernel_size=1)
            if num_classes == 1: self.normalize = nn.Sigmoid()
            else: self.normalize = nn.Softmax(dim=1)

    def forward(self, d1, d2, d3, d4):
        # x : input batch
        # d1 : outputs of backbone layer1 : 256, 56, 56
        # d2 : outputs of backbone layer2 : 512, 28, 28
        # d3 : outputs of backbone layer3 : 1024, 14, 14
        # d4 : outputs of backbone layer4 : 2048, 7, 7

        # 디코더 + 스킵 연결
        u1 = self.up1(d4) # 1024, 14, 14
        u11 = torch.cat((u1, d3), dim=1) # 2048, 14, 14
        u1 = self.conv1(u11) + u1 # 1024, 14, 14 

        u2 = self.up2(u1)
        u22 = torch.cat((u2, d2), dim=1)
        u2 = self.conv2(u22) + u2

        u3 = self.up3(u2)
        u33 = torch.cat((u3, d1), dim=1)
        u3 = self.conv3(u33) + u3

        out = self.final(u3)
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

class CSC_cls1(torch.nn.Module):
    def __init__(self, num_clss, model_name='resnet50'):
        super(CSC_cls1, self).__init__()
        if model_name=='resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone.fc = nn.Linear(2048, num_class)
            backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.modlist = nn.ModuleDict()
            for name, para in backbone.named_modules():
                name = name.split('.')[0]
                if not name=='':
                    self.modlist[name] = para
        
    def forward(self, x):
        conv1_out = self.modlist['conv1'](x)
        bn1_out = self.modlist['bn1'](conv1_out)
        relu_out = self.modlist['relu'](bn1_out)
        maxpool_out = self.modlist['maxpool'](relu_out)
        l1_out = self.modlist['layer1'](maxpool_out)
        l2_out = self.modlist['layer2'](l1_out)
        l3_out = self.modlist['layer3'](l2_out)
        l4_out = self.modlist['layer4'](l3_out)

        #attn
        b, c, h, w = l4_out.shape
        att1 = l4_out.reshape([b,c,h*w])
        att2 = torch.permute(att1, (0,2,1))
        att = torch.bmm(att1, att2) # b, c, c
        att_out = torch.bmm(att, l4_out)
        avgpool_out = self.modlist['avgpool']()
        avgpool_out = avgpool_out.squeeze()
        fc_out = self.modlist['fc'](avgpool_out)
        return l1_out, l2_out, l3_out, l4_out, fc_out

class CSC_cls2(torch.nn.Module):
    def __init__(self, num_clss, model_name='resnet50'):
        super(CSC_cls2, self).__init__()
        if model_name=='resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
            backbone.fc = nn.Linear(2048, num_class)
            backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.modlist = nn.ModuleDict()
            for name, para in backbone.named_modules():
                name = name.split('.')[0]
                if not name=='':
                    self.modlist[name] = para
        
    def forward(self, x, f1, f2, f3,f4):
        conv1_out = self.modlist['conv1'](x)
        bn1_out = self.modlist['bn1'](conv1_out)
        relu_out = self.modlist['relu'](bn1_out)
        maxpool_out = self.modlist['maxpool'](relu_out)
        l1_out = self.modlist['layer1'](maxpool_out)
        e1 = torch.sigmoid(f1) * l1_out
        e1_out = l1_out + e1

        l2_out = self.modlist['layer2'](e1_out)
        e2 = torch.sigmoid(f2) * l2_out
        e2_out = l2_out + e2

        l3_out = self.modlist['layer3'](e2_out)
        e3 = torch.sigmoid(f3) * l3_out
        e3_out = l3_out + e3

        l4_out = self.modlist['layer4'](e3_out)
        e4 = torch.sigmoid(f4) * l4_out
        e4_out = l4_out + e4

        #attn
        b, c, h, w = e4_out.shape
        att1 = e4_out.reshape([b,c,h*w])
        att2 = torch.permute(att1, (0,2,1))
        att = torch.bmm(att1, att2) # b, c, c
        att_out = torch.bmm(att, l4_out)

        avgpool_out = self.modlist['avgpool']()
        avgpool_out = avgpool_out.squeeze()
        fc_out = self.modlist['fc'](avgpool_out)
        
        return fc_out

class CSC_Seg(torch.nn.Module):
    def __init__(self, num_class=1, model_name='resnet50'):
        super(CSC_Seg, self).__init__()
        if model_name=='resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.modlist = nn.ModuleDict()
            for name, para in backbone.named_modules():
                name = name.split('.')[0]
                if not name=='' and not name=='fc':
                    self.modlist[name] = para
            self.modlist['up1'] = SCAttention(2048, 1024)
            self.modlist['up2'] = SCAttention(1024, 512)
            self.modlist['up3'] = SCAttention(512, 256)
            self.modlist['up4'] = SCAttention(256, 64)
            self.modlist['up5'] = SCAttentionOut(64, 32)
            self.fc = nn.Conv2d(32, num_class, (1,1))
    
    def forward(self, x, f1, f2, f3, f4):
        #Encoder
        conv1_out = self.modlist['conv1'](x)
        bn1_out = self.modlist['bn1'](conv1_out)
        relu_out = self.modlist['relu'](bn1_out)
        maxpool_out = self.modlist['maxpool'](relu_out)
        l1_out = self.modlist['layer1'](maxpool_out)
        e1 = torch.sigmoid(f1) * l1_out
        e1_out = l1_out + e1

        l2_out = self.modlist['layer2'](e1_out)
        e2 = torch.sigmoid(f2) * l2_out
        e2_out = l2_out + e2

        l3_out = self.modlist['layer3'](e2_out)
        e3 = torch.sigmoid(f3) * l3_out
        e3_out = l3_out + e3

        l4_out = self.modlist['layer4'](e3_out)
        e4 = torch.sigmoid(f4) * l4_out
        e4_out = l4_out + e4

        #Decoder
        up1_out = self.modlist['up1'](e4_out) + e3_out
        up2_out = self.modlist['up2'](up1_out) + e2_out
        up3_out = self.modlist['up3'](up2_out) + e1_out
        up4_out = self.modlist['up4'](up3_out) + e3_out
        up5_out = self.modlist['up5'](up4_out) + conv1_out
        out = self.fc(up5_out)
        return l1_out, l2_out, l3_out, l4_out, out
