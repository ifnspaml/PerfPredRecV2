from itertools import chain

import torch.nn as nn
import torch

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from ... util import _UpsampleGELU, SpatialPyramidPoolingGELU, LayerNorm

__all__ = ['ConvNeXt', 'convnext_tiny']

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
}

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    expansion=1

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
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
        super().__init__()
        self.num_features = num_features
        self.dims = dims
        self.block = Block

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=strides[i+1],
                          padding='same' if strides[i+1] == 1 else 0)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.upsamples = []  # 3 upsampling stages

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[self.block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

            # Create lateral layers
            if i < 3:
                lat = _UpsampleGELU(self.num_features, dims[i], self.num_features, norm=norm, k=k_up)
                self.upsamples.append(lat)

        if strides == [4, 2, 2, 2]:
            self.upsample = nn.ModuleList(list(reversed(self.upsamples)))
            num_levels = 3
            self.spp_size = num_features
            bt_size = self.spp_size

            level_size = self.spp_size // num_levels

            self.spp = SpatialPyramidPoolingGELU(dims[-1], num_levels, bt_size=bt_size, level_size=level_size,
                                                    out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                                    norm=norm)

            self.random_init = [self.spp, self.upsample]
        self.fine_tune = [self.downsample_layers, self.stages]
        self.apply(self._init_weights)

        # TODO: Load pretrained network
        if pretrained:
            url = model_urls['convnext_tiny_1k']
            checkpoint = torch.hub.load_state_dict_from_url(url=url, check_hash=True)
            self.load_state_dict(checkpoint["model"], strict=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_down_only(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features, x

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features += [x]
        del features[-1]
        features += [self.spp.forward(x)]
        return features

    def forward_upsampling(self, features):
        features = features[::-1]
        x = features[0]
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_upsampling(x)
        return x, None

    # def forward(self, x):
    #     # This part follows the logic of renet\full_network

    #     image_size = x.shape[2:]
    #     dict_ = {}
    #     dict_['semseg'] = {}

    #     # Note that here spp_input and backbone output are equal
    #     skips_and_spp, spp_input = self.forward_features_only(x)
    #     return logits

@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model