import os,sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

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
        pos1 = self.module_dict['pos_head1'](outputs)
        pos1 = self.module_dict['adaptive_pool'](pos1)
        weight = self.module_dict['pos_head1'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn1'](outputs)
        
        outputs = self.module_dict['layer2'](outputs)
        pos2 = self.module_dict['pos_head2'](outputs)
        pos2 = self.module_dict['adaptive_pool'](pos2)
        weight = self.module_dict['pos_head2'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn2'](outputs)
        
        outputs = self.module_dict['layer3'](outputs)
        pos3 = self.module_dict['pos_head3'](outputs)
        pos3 = self.module_dict['adaptive_pool'](pos3)
        weight = self.module_dict['pos_head3'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn3'](outputs)
        
        outputs = self.module_dict['layer4'](outputs)
        pos4 = self.module_dict['pos_head4'](outputs)
        pos4 = self.module_dict['adaptive_pool'](pos4)
        weight = self.module_dict['pos_head4'].weight
        outputs = weight*outputs
        outputs = self.module_dict['pos_bn4'](outputs)
        
        # print('1 ', outputs.shape)
        outputs = self.module_dict['avgpool'](outputs)
        # print('2 ', outputs.shape)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.module_dict['fc'](outputs)
        # print('3 ', outputs.shape)
        
        pos_outputs = torch.mean(torch.stack([pos1, pos2, pos3, pos4], 1), 1)
        pos_outputs = pos_outputs.squeeze()

        return outputs, pos_outputs