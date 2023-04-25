from itertools import chain
import torch.nn as nn
import torch.nn.functional as F

from .dws_aspp import build_dwsaspp
from .decoder import build_decoder
from ...encoders.resnet.full_network import ResNet
from ...encoders.convnext.full_network import ConvNeXt
# from ..swin_transformer import swin
from ...util import BasicBlock, Bottleneck, upsample
CONFIG = {
    'resnet18'     : {'layers': [2, 2, 2, 2], 'url': "https://download.pytorch.org/models/resnet18-5c106cde.pth", 'block': BasicBlock},
    'resnet50'     : {'layers': [3, 4, 6, 3], 'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'block': Bottleneck},
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

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class Wrapper(nn.Module):
    def __init__(self, net):
        super(Wrapper, self).__init__()
        self.net = net
    
    def forward_down_only(self, batch):
        features = self.net(batch).reshaped_hidden_states
        features = list(features)#[:-1]
        return features, None

class DeepLab(nn.Module):
    def __init__(self, encoder, encoder_name, num_classes):
        super(DeepLab, self).__init__()
        self.encoder_name = encoder_name
        self.encoder = encoder
        if encoder_name == 'swin_t': 
            self.encoder = Wrapper(self.encoder)
            self.fine_tune = [self.encoder]
        else:
            self.fine_tune = self.encoder.fine_tune
        self.aspp = build_dwsaspp(self.encoder_name, nn.BatchNorm2d)
        self.decoder = build_decoder(num_classes, self.encoder_name, nn.BatchNorm2d)
        self.random_init = [self.aspp, self.decoder]

    def forward(self, batch):
        features, _ = self.encoder.forward_down_only(batch)
        x, low_level_feat = features[-1], features[0]
        x = self.aspp(x)
        x, _ = self.decoder(x, low_level_feat)
        x = upsample(self.training, x, batch.shape[2:4])
        return x
    
    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])
    
    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    #FIXME: Make this a global function    
    def configure_params(self, wd=0.25e-4, lr=1e-4):
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name and "encoder" in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": wd,
                "learning_rate": lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "learning_rate": lr,
            },
        ]
        return optimizer_grouped_parameters

def build_deeplab(num_classes, encoder='resnet18', atrous=1, pretrained=True):
    if 'resnet' in encoder:
        block, layers = CONFIG[encoder]['block'], CONFIG[encoder]['layers']
        if atrous:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]

        net = DeepLab(ResNet(block, layers, efficient=False,
            strides=strides, dilations=dilations, kaiming=False, atrous=atrous),
            encoder_name=encoder, num_classes=num_classes,
        )
    elif encoder == 'convnext_tiny':
        net = DeepLab(
            ConvNeXt(strides=[4, 2, 2, 1], dilations=[1, 1, 1, 2]),
            encoder_name=encoder, num_classes=num_classes,
        ) 
    elif encoder == 'swin_t':
        backbone = swin.build_backbone(pretrained=True)
        net = DeepLab(backbone, encoder_name=encoder, num_classes=num_classes)
    else:
        ValueError(f"Backbone network {encoder} is not supported.")

    return net