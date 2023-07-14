import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def set_grad(model, grad):
    for param in model.parameters():
        param.requires_grad = grad



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Parameter):
        init.kaiming_normal_(m.weight)


class BaseNet(nn.Module):
    def __init__(self, inplanes=3, stride=2, w_base=16, num_classes=10, embedding_size=64):
        super(BaseNet, self).__init__() 
      
        self.conv1 = nn.Conv2d(inplanes, w_base, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_base)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(w_base, w_base*2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(w_base*2)

        self.conv3 = nn.Conv2d(w_base*2, w_base*4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_base*4)

        self.conv4 = nn.Conv2d(w_base*4, w_base*4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(w_base*4)

        self.fc=nn.Linear(w_base*4*2*2, embedding_size)

        self.softmax=nn.Softmax(dim=1)

        self.classifer=nn.Linear(embedding_size, num_classes)

        self.apply(_weights_init)

        
    def forward(self, x):
        out= self.conv1(x)
        out= self.bn1(out)
        out= self.relu(out)
        
        out= self.conv2(out)
        out= self.bn2(out)
        out= self.relu(out)

        out= self.conv3(out)
        out= self.bn3(out)
        out= self.relu(out)
        
        out= self.conv4(out)
        out= self.bn4(out)
        out= self.relu(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        emb = self.softmax(out)

        out=self.classifer(out)
       
        return out, emb


class MoEBlock(nn.Module):
    def __init__(self, embedding_size, output):
        super(MoEBlock, self).__init__()
        self.moe1 = nn.Parameter(torch.zeros(embedding_size, output), requires_grad=True)
        self.moe2 = nn.Parameter(torch.zeros(embedding_size, output), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

        self.apply(_weights_init)
        
       
    def forward(self, x):
        x, gx =x
        o1 = self.relu(x @ self.moe1)
        o2 = self.relu(x @ self.moe2)
        gx.append(o1)
        gx.append(o2)
        x = x, gx
        return x



class MoE(nn.Module):
    def __init__(self, embedding_size=64, layer1_out=16, layer2_out=32, layer3_out=64):
        super(MoE, self).__init__()
        self.num=9
        self.layer1 =self._make_layer(self.num, embedding_size, layer1_out)
        self.layer2 =self._make_layer(self.num, embedding_size, layer2_out)
        self.layer3 =self._make_layer(self.num, embedding_size, layer3_out)
      
    def _make_layer(self, num, emb_size, layer_out):
        layers=[]
        for i in range(num):
            layers.append(MoEBlock(emb_size, layer_out))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = [x, []]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        _, gx=x
      
        return gx


# class MoEBlock(nn.Module):
#     def __init__(self, embedding_size, output):
#         super(MoEBlock, self).__init__()
#         self.moe1 = nn.Linear(embedding_size, output, bias=False)
#         self.moe2 = nn.Linear(embedding_size, output, bias=False)
#         self.relu = nn.ReLU(inplace=True)

#         self.apply(_weights_init)
    
       
#     def forward(self, x):
#         x, gx =x
#         o1 = self.relu(self.moe1(x))
#         o2 = self.relu(self.moe2(x))
#         gx.append(o1)
#         gx.append(o2)
#         x = x, gx
#         return x



# class MoE(nn.Module):
#     def __init__(self, embedding_size=64, layer1_out=16, layer2_out=32, layer3_out=64):
#         super(MoE, self).__init__()
#         self.num=9
#         self.layer1 =self._make_layer(self.num, embedding_size, layer1_out)
#         self.layer2 =self._make_layer(self.num, embedding_size, layer2_out)
#         self.layer3 =self._make_layer(self.num, embedding_size, layer3_out)
      
#     def _make_layer(self, num, emb_size, layer_out):
#         layers=[]
#         for i in range(num):
#             layers.append(MoEBlock(emb_size, layer_out))

#         return nn.Sequential(*layers)
        
#     def forward(self, x):
#         x = [x, []]
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         _, gx=x
      
#         return gx


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, moe=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.moe = moe
        if stride != 1 or in_planes != planes:
            
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.moe:
            x, drop =x

        out = self.relu(self.bn1(self.conv1(x)))
        if self.moe:
            out = out* (drop[0].unsqueeze(-1).unsqueeze(-1))
            
        out = self.bn2(self.conv2(out))
        if self.moe:
            out = out* (drop[1].unsqueeze(-1).unsqueeze(-1))
        
        out += self.shortcut(x)
        out = self.relu(out)

        if self.moe: 
            drop = drop[2:]
            out =[out, drop]
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, w_base=16, embedding_size=64):
        super(ResNet, self).__init__()
        self.basenet= BaseNet(num_classes=num_classes,  w_base=w_base, embedding_size=embedding_size)
        self.moe=MoE(embedding_size=embedding_size, layer1_out=w_base, layer2_out=w_base*2, layer3_out=w_base*4)


        self.in_planes = w_base
        self.conv1 = nn.Conv2d(3, w_base, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(w_base)
        self.layer1 = self._make_layer(block, w_base, num_blocks[0], stride=1, moe=True)
        self.layer2 = self._make_layer(block, w_base*2, num_blocks[1], stride=2, moe=True)
        self.layer3 = self._make_layer(block, w_base*4, num_blocks[2], stride=2, moe=True)
        self.linear = nn.Linear(w_base*4, num_classes)
        
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, moe=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, moe))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    
    def set_freeze(self):
        set_grad(self.basenet, False)
        set_grad(self.moe, False)


    def forward(self, x):

        basenet_out, emb = self.basenet(x) 
        e =self.moe(emb)
        # e =[torch.zeros(1, 64).to("cuda:0") for i in range(54)]
        
        out = self.relu(self.bn1(self.conv1(x)))

        out = [out, e]
        out  = self.layer1(out)

        out  = self.layer2(out)
        out  = self.layer3(out)
        out, _ = out
        
       
        # out = F.avg_pool2d(out, out.size()[3])
        out = F.avg_pool2d(out, 8)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, basenet_out, e


class L1_loss(nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
    def forward(self, x):
        loss = 0.0
        for i in range(len(x)):
            loss = loss + x[i].mean()
        return loss 




def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56(num_classes, w_base, embedding_size):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, w_base, embedding_size)



def resnet110(num_classes, w_base, embedding_size):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, w_base, embedding_size)


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


if __name__ == '__main__':
    resne110 = resnet110(100, 32, 128)

    x = torch.randn(32, 3, 32, 32)
    
    emb1,emb2, emb3 = resne110(x)
    print(emb1.size())
    print(emb2.size())
    print(emb3.size())
    # e = moe(emb)
    # res = resnet56(x, e)
    # n =torch.ones(32, 64)
    # print(torch.sum(n))



