import math
from itertools import chain

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ... util import _Upsample, SpatialPyramidPooling, BasicBlock, Bottleneck

__all__ = ['ResNet', 'resnet18', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True, spp_grids=(8, 4, 2, 1),
                 spp_square_grid=False, strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1], atrous=False, kaiming=True):
        super(ResNet, self).__init__()
        self.dims = []
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block = block
        upsamples = []

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[0])
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        # If inherited from DeepLabv3+, the decoder is ignored, and thus only the backbone layers are initialized.
        if atrous:
            self.layer4 = self._make_MG_unit(block, 512, blocks=[1, 2, 4], stride=strides[3], dilation=dilations[3], BatchNorm=True)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[0])

            num_levels = 3
            self.spp_size = num_features
            bt_size = self.spp_size

            level_size = self.spp_size // num_levels

            self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                            out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                            bn_momentum=0.01 / 2, use_bn=self.use_bn)
            self.upsample = nn.ModuleList(list(reversed(upsamples)))

            self.random_init = [self.spp, self.upsample]
            
        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        self.num_features = num_features

        init_method = "Kaiming" if kaiming is True else "custom"
        print(f"\nInitialize backbone network using {init_method} initialization.\n")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if kaiming:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self._load_pretrained(layers)

    def _init_layer(self, layer, kaiming=True):
        if isinstance(layer, nn.Conv2d):
            if kaiming:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            else:
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()

    def _load_pretrained(self, layers):
        if layers == [2, 2, 2, 2]:
            url = model_urls['resnet18']
        elif layers == [3, 4, 6, 3]:
            url = model_urls['resnet50']

        model_dict = {}
        pretrain_dict = model_zoo.load_url(url)
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient,
                        use_bn=self.use_bn, dilation=dilation)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn, dilation=dilation)]
        self.dims.append(self.inplanes)
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, use_bn=BatchNorm, efficient=self.efficient))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, use_bn=BatchNorm, efficient=self.efficient))

        return nn.Sequential(*layers)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [self.spp.forward(skip)]
        return features

    def forward_down_only(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [x]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [x]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [x]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [x]
        return features, x

    def forward_up(self, features):
        features = features[::-1]
        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}

    def forward(self, image):
        return self.forward_up(self.forward_down(image))


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model