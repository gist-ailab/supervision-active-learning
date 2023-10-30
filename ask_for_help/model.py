import os,sys
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class res_size5(torch.nn.Module):
    def __init__(self, num_class=3, model_name='resnet50'):
        super(res50_size5, self).__init__()
        if model_name=='resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(2048, num_class)
            return_nodes = {
            'layer1':'l1',
            'layer2':'l2',
            'layer3':'l3',
            'layer4':'l4',
            'fc':'fc'
            }
            self.model = create_feature_extractor(self.backbone, return_nodes)
            self.feat_layer = nn.Conv2d(2048, 512, (3,3))
            self.GAP = nn.Conv2d(512, 1, (1,1))
    
    def forward(self, x):
        outputs_dict = self.model(x)
        outputs = outputs_dict['fc']
        feats = outputs['l4'] # b, 2048, 7, 7
        feats = self.feat_layer(feats) # b, 512, 5, 5
        feats = self.GAP(feats) # b, 1, 5, 5
        feats = feats.squeeze() # b, 5, 5
        return outputs, feats
    
class res_size7(torch.nn.Module):
    def __init__(self, num_class=3, model_name='resnet50'):
        super(res50_size7, self).__init__()
        if model_name=='resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(2048, num_class)
            return_nodes = {
            'layer1':'l1',
            'layer2':'l2',
            'layer3':'l3',
            'layer4':'l4',
            'fc':'fc'
            }
            self.model = create_feature_extractor(self.backbone, return_nodes)
            self.feat_layer = nn.Conv2d(2048, 512, (1,1))
            self.GAP = nn.Conv2d(512, 1, (1,1))
    
    def forward(self, x):
        outputs_dict = self.model(x)
        outputs = outputs_dict['fc']
        feats = outputs['l4'] # b, 2048, 7, 7
        feats = self.feat_layer(feats) # b, 512, 7, 7
        feats = self.GAP(feats) # b, 1, 7, 7
        feats = feats.squeeze() # b, 7, 7
        return outputs, feats