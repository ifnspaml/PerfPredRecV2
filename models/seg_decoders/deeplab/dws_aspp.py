import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util import upsample

class _ConvModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ConvModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class _DWSConvModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_DWSConvModule, self).__init__()
        self.dw_atrous_conv = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, groups=inplanes,
                                        stride=1, padding=padding, dilation=dilation, bias=False)
        self.pw_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.dw_atrous_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DWSASPP(nn.Module):
    def __init__(self, backbone="resnet18", BatchNorm=nn.BatchNorm2d):
        super(DWSASPP, self).__init__()

        if "resnet18" in backbone:
            inplanes = 512
            outplanes = 128
        elif "resnet50" in backbone:
            inplanes = 2048
            outplanes = 256
        elif "convnext_tiny" in backbone:
            inplanes = 768
            outplanes = 192
        elif "swin_t" in backbone:
            inplanes = 768
            outplanes = 192
        else:
            raise NotImplementedError(f"{backbone} is not supported.")

        # if "swin_t" in backbone:
        dilations = [1, 6, 12, 18]
        # else:
        #     dilations = [1, 3, 6, 9]

        self.aspp1 = _ConvModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _DWSConvModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _DWSConvModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _DWSConvModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                             BatchNorm(outplanes),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(outplanes * 5, outplanes, 1, bias=False)
        self.bn1 = BatchNorm(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = upsample(self.training, x5, x4.size()[2:])
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_dwsaspp(backbone, BatchNorm):
    return DWSASPP(backbone, BatchNorm)