import torch
import torch.nn as nn
import torch.nn.functional as F

class branch(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass
    
class Linear(nn.Module):
    def __init__(self, num_classes=10):
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        size = x.shape[-1]
        out = F.avg_pool2d(x, size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out