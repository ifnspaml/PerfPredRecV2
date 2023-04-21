import torch.nn as nn
from ...encoders.resnet.full_network import ResNet
from ...encoders.convnext.full_network import ConvNeXt
from ...seg_decoders.deeplab.fw_adaptor import DeepLab
from ...util import BasicBlock, Bottleneck

CONFIG = {
    'resnet18'     : {'layers': [2, 2, 2, 2], 'url': "https://download.pytorch.org/models/resnet18-5c106cde.pth", 'block': BasicBlock},
    'resnet50'     : {'layers': [3, 4, 6, 3], 'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'block': Bottleneck},
    "convnext_tiny": {'layers': None,         'url': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"}
}

class DeepLab(DeepLab):
    def __init__(self, encoder, encoder_name, num_classes):
        super().__init__(encoder, encoder_name, num_classes)

    def forward(self, batch):
        features, _ = self.encoder.forward_down_only(batch)
        x, low_level_feat = features[-1], features[0]
        backbone_output = x
        x = self.aspp(x)
        x, idlc = self.decoder(x, low_level_feat)
        return {
            'spp_input': backbone_output,
            'logits': x,
            'low_level_feat': low_level_feat,
            'idlc': idlc,
        }
    
    def fine_tune_params(self):
        return super().fine_tune_params()

def build_deeplabrec(num_classes, encoder='resnet18', atrous=1):
    if 'resnet' in encoder:
        block = CONFIG[encoder]['block']
        layers = CONFIG[encoder]['layers']
        if atrous:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        model = DeepLab(ResNet(block, layers, efficient=False,
            strides=strides, dilations=dilations, kaiming=False, atrous=atrous),
            encoder_name=encoder, num_classes=num_classes,
        )
    elif encoder == 'convnext_tiny':
        model = DeepLab(
            ConvNeXt(strides=[4, 2, 2, 1], dilations=[1, 1, 1, 2]),
            encoder_name=encoder, num_classes=num_classes,
        ) 
    elif encoder == 'swin_t':
        from ...encoders.swin_transformer import swin
        net = swin.build_backbone(pretrained=True)
        model = DeepLab(net, encoder_name=encoder, num_classes=num_classes)
    else:
        ValueError(f"Backbone network {encoder} is not supported.")
    return model