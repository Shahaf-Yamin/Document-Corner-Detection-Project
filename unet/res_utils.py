import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple, List, Callable, Any
import torch.nn.functional as F

class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicDeConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__( self, in_channels: int,) -> None:
        super().__init__()
        self.branch1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch2 = BasicDeConv2D(in_channels, 128, kernel_size=3)
        self.branch3 = BasicDeConv2D(in_channels, 128, kernel_size=5)
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
                                      BasicDeConv2D(in_channels, 128, kernel_size=3))

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

# class Mlif(nn.Module):
#     def __init__(self):