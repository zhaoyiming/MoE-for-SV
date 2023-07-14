
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter



class _routing(nn.Module):
    def __init__(self, c_in, num_experts):
        super(_routing, self).__init__()
      
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
     
        x = x.mean(axis=2,keepdims=True).mean(axis=3,keepdims=True)
        x = torch.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CondConv2D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=32):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

      
        self._routing_fn = _routing(in_channels, num_experts)
        
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs):
  
        res = []
        for input in inputs:
          
            input = input.unsqueeze(0)
            routing_weights = self._routing_fn(input)
            kernels = torch.sum(routing_weights[: ,None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)

if __name__ == '__main__':
    a=torch.rand(32, 3, 32, 32)

    conv1= CondConv2D(3, 32, 3)
    b=conv1(a)
    print(b.size())