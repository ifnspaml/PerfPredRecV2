import math
from itertools import chain

import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .dws_aspp import build_dwsaspp
from .decoder import build_decoder
from ..swiftnet.resnet.full_network import ResNet
from ..convnext.full_network import ConvNeXt
from ..util import BasicBlock, Bottleneck, upsample

CONFIG = {
    'resnet18'     : {'layers': [2, 2, 2, 2], 'url': "https://download.pytorch.org/models/resnet18-5c106cde.pth"},
    'resnet50'     : {'layers': [3, 4, 6, 3], 'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'},
    "convnext_tiny": {'layers': None,         'url': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"}
}

# --------------------------------------------------------------------------------
#  Custom function for the specific model architecture to load/update state_dict
# --------------------------------------------------------------------------------
def load_state_dict_into_model(model, pretrained_dict):
    model_dict = model.state_dict()
    if list(pretrained_dict.keys())[0] == 'backbone.conv1.weight':
        pretrained_dict = {k.replace('backbone', 'loaded_model.backbone'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('logits', 'loaded_model.logits'): v for k, v in pretrained_dict.items()}
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            print("State_dict mismatch!", flush=True)
            continue
        model_dict[name].copy_(param)
    model.load_state_dict(pretrained_dict, strict=True)

# TODO: Extend inheritance to ConvNeXt and implement call function to call only either one of the parent class depending on a condition
class DeepLab(ResNet, ConvNeXt):
    def __init__(self, num_classes, backbone='resnet18', pretrained=True,
                 freeze_bn=False, atrous=False, strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1]):
        if atrous:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]

        # Call ResNet class to get the resnet-based backbone
        # TODO: Refractor the constructor
        if 'resnet' in backbone:
            self.block = BasicBlock if backbone=='resnet18' else Bottleneck
            super().__init__(self.block, CONFIG[backbone]['layers'], efficient=False,
                             strides=strides, dilations=dilations, kaiming=False,
                             atrous=atrous)
        elif backbone == 'convnext_tiny':
            super(ResNet, self).__init__(strides=[4, 2, 2, 1], dilations=[1, 1, 1, 2])
        else:
            ValueError(f"Backbone network {backbone} is not supported.")

        self.backbone = backbone

        # Override layer_4 attribute if applying atrous convolution
        # if atrous: 
        #     print("\nAtrous convolution enabled.")
        #     blocks = [1, 2, 4]
        #     self.inplanes = 256 if backbone=='resnet18' else 1024
        #     self.layer4 = self._make_MG_unit(self.block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=True)
        if 'resnet' in backbone:
            self.fine_tune = [self.conv1, self.bn1, self.act1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]

        # debugger for now
        # if self.upsample is not None:
        #     del self.upsample
        # if self.spp is not None:
        #     del self.spp

        # This initializes the backbone network. Skipping this drops the scores somehow, although the pretrained weigths loaded below.
        # self._init_weight()

        # if pretrained:
            # self._load_pretrained_model()

        BatchNorm = nn.BatchNorm2d
        self.aspp = build_dwsaspp(backbone, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.random_init = [self.aspp, self.decoder]

        self.freeze_bn = freeze_bn

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        print("\nGetting the pretrained weights from ", CONFIG[self.backbone]['url'])
        state_dict = load_state_dict_from_url(CONFIG[self.backbone]['url'], progress=True)
        print("Load pretrained weights")
        for param_tensor in state_dict:
            try:
                self.state_dict()[param_tensor].size()
            except:
                print(f"Could not find {param_tensor} in model state dict")
        print()
        self.load_state_dict(state_dict, strict=False)


    def forward(self, input_):
        features = self.forward_down_only(input_)
        x, low_level_feat = features[-1], features[0]
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = upsample(self.training, x, input_.shape[2:4])
        return x

    def forward_down(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x, skip = self.forward_resblock(x, self.layer1)
        low_level_feat = x
        x, skip = self.forward_resblock(x, self.layer2)
        x, skip = self.forward_resblock(x, self.layer3)
        x, skip = self.forward_resblock(x, self.layer4)
        return x, low_level_feat

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def random_init_params(self):
        return chain(*[layer.parameters() for layer in self.random_init])

    def fine_tune_params(self):
        return chain(*[layer.parameters() for layer in self.fine_tune])
