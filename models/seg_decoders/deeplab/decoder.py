import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dws_aspp import _DWSConvModule
from ...util import upsample

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        self.num_classes = num_classes

        if 'resnet18' in backbone:
            low_level_inplanes = 64
            low_level_outplanes = 12
            aspp_outplanes = 128
        elif 'resnet50' in backbone:
            low_level_inplanes = 256
            low_level_outplanes = 48
            aspp_outplanes = 256
        elif 'convnext_tiny' in backbone:
            low_level_inplanes = 96
            low_level_outplanes = 18
            aspp_outplanes = 192
        elif 'swin_t' in backbone:
            low_level_inplanes = 96
            low_level_outplanes = 18
            aspp_outplanes = 192
        else:
            raise NotImplementedError(f"{backbone} is not supported.")

        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_outplanes, 1, bias=False)
        self.bn1 = BatchNorm(low_level_outplanes)

        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(_DWSConvModule(aspp_outplanes + low_level_outplanes,
                                                      aspp_outplanes,
                                                      3,
                                                      padding=1,
                                                      dilation=1,
                                                      BatchNorm=BatchNorm),
                                       _DWSConvModule(aspp_outplanes,
                                                      aspp_outplanes,
                                                      3,
                                                      padding=1,
                                                      dilation=1,
                                                      BatchNorm=BatchNorm),
                                       nn.Conv2d(aspp_outplanes, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = upsample(self.training, x, low_level_feat.size()[2:])
        x = torch.cat((x, low_level_feat), dim=1)
        idlc = []
        for last_conv in self.last_conv:
            x = last_conv(x)
            idlc.append(x)
        return x, idlc 

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)