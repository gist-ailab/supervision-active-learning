'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out3 = out2+self.shortcut(x)
        out4 = F.relu(out3)
        return out4


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.autograd.set_detect_anomaly(True):
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            
            size = out4.shape[-1]
            out5 = F.avg_pool2d(out4, size)
            out6 = out5.view(out5.size(0), -1)
            out7 = self.linear(out6)
            return out7, out4

class GradCamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.model = model
        # self.layerhook.append(self.model.layer4.register_forward_hook(self.forward_hook()))
        # self.attach_hook()
        
        for p in self.model.parameters():
            p.requires_grad = True
    def attach_hook(self):
        self.layerhook.append(self.model.layer4.register_forward_hook(self.forward_hook()))
        
    def detach_hook(self):
        for layerhook in self.layerhook:
            layerhook.remove()
        for tensorhook in self.tensorhook:
            tensorhook.remove()
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        if self.model.training:
            if len(self.layerhook)==0:
                self.attach_hook()
        else:
            self.detach_hook()
        out = self.model(x)
        return out, self.selected_out

class BoxProposal(nn.Module):
    def __init__(self, output_size=256):
        super(BoxProposal, self).__init__()
        pass
    
class heatmap_model(nn.Module):
    def __init__(self, input_size=32, output_size=256):
        super(heatmap_model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.channel_layer = nn.Conv2d(256,32,kernel_size=1,stride=1)
        
        self.layer1 = nn.Linear(32*self.input_size*self.input_size, 64*64)
        self.layer2 = nn.Linear(64*64, 32*32)
        self.layer3 = nn.Linear(32*32, 16*16)
        self.layer4 = nn.Linear(16*16, 32*32)
        self.layer5 = nn.Linear(32*32, 64*64)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.channel_layer(x)
        out = out.view(out.size(0), -1)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        out = self.relu(out)
        out = self.layer5(out)
        out = out.view(out.size(0), 1, 64, 64)
        out = F.interpolate(out, size=(256,256))
        # print(out.shape)
        return out.squeeze()

class Linear(nn.Module):
    def __init__(self, num_classes=10):
        super(Linear, self).__init__()
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        size = x.shape[-1]
        out = F.avg_pool2d(x, size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


