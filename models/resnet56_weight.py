import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


class WeightNet(nn.Module):

    def __init__(self, inp, oup, ksize, stride):
        super().__init__()

        self.M = 16
        self.G = 4

        inp_gap = max(16, inp//16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = nn.Conv2d(inp_gap, self.M*oup, 1, 1, 0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M*oup, oup*inp*ksize*ksize, 1, 1, 0, groups=self.G*oup, bias=False)
        


    def forward(self, x, x_gap):
        batch_size=x_gap.shape[0]
        x_w = self.wn_fc1(x_gap) # Bx16x1x1 -> Bx(MxCout)x1x1
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w) # Bx(MxCout)x1x1 -> Bx(CinxCoutxKxK)x1x1

       

        if x.shape[0] == 1: # case of batch size = 1
            x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize) # CoutxCinx3x3
            x = F.conv2d(x, weight=x_w, stride=self.stride, padding=1)
            return x
        
        x = x.reshape(1, -1, x.shape[2], x.shape[3])# BxCinxHxW -> 1x(BxCin)xHxW 
        x_w = x_w.reshape(x_w.shape[0]*self.oup, self.inp, self.ksize, self.ksize) # Bx(CinxCoutxKxK)x1x1->(BxCout)xCinxKxK
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=1, groups=batch_size)
        x = x.reshape(-1, self.oup, x.shape[2], x.shape[3])# BxCxHxW
        return x



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = WeightNet(in_planes, planes, ksize=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = WeightNet(planes, planes, ksize=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.reduce = nn.Conv2d(in_planes, max(16, in_planes//16), 1, 1, 0, bias=True)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
        
            if option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        x_gap = x.mean(axis=2,keepdims=True).mean(axis=3,keepdims=True)
        x_gap = self.reduce(x_gap)

        out = F.relu(self.bn1(self.conv1(x, x_gap)))
        out = self.bn2(self.conv2(out, x_gap))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.dropout=nn.Dropout(0.2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out



def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def resnet_weight():
    return ResNet(BasicBlock, [9, 9, 9])    




# if __name__ == '__main__':
    # resnet56 = resnet_cond(2,2)

    # x = torch.randn(32, 3, 32, 32)

    # res=resnet56(x)
