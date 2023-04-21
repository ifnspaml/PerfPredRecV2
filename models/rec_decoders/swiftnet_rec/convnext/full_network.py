from itertools import chain
import torch.nn as nn
import torch

from timm.models.registry import register_model

from .... encoders.convnext.full_network import ConvNeXt

__all__ = ['ConvNeXt', 'convnext_tiny']

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}

class ConvNeXtRec(ConvNeXt):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, *, num_features=128, k_up=3, spp_grids=(8, 4, 2, 1),
                 spp_square_grid=False, norm=True, strides=[4, 2, 2, 2], dilations=[1, 1, 1, 1],
                 pretrained=True):
        super().__init__(depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                 layer_scale_init_value=layer_scale_init_value, pretrained=True)
    
    def forward_upsampling(self, features):
        features = features[::-1]
        x = features[0]
        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples.append(x)
        return x, upsamples

    def forward(self, batch):
        features, backbone_output = self.forward_down_only(batch)
        del features[-1]
        spp_out = self.spp.forward(backbone_output)
        features.append(spp_out)
        prelogits, upsamples = self.forward_upsampling(features)

        semseg_dict = {'prelogits': prelogits,
                       'features': features,
                       'upsamples': upsamples}

        # Note that this dictionary only contains SwiftNet-specific outputs for now
        dict_ = {'semseg': semseg_dict,
                 'backbone_output': backbone_output,
                 'spp_input': backbone_output,
                 'skips_and_spp': features}

        return dict_

@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtRec(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model