#!/usr/bin/env python

import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel

from itertools import chain
from ..util import SpatialPyramidPoolingGELU, _UpsampleGELU

class SwinSeg(nn.Module):
    def __init__(self, atrous=False, 
        num_features=128, 
        spp_grids=(8, 4, 2, 1),
        spp_square_grid=False, 
        dims=[96, 192, 384, 768],
        pretrained=True,
    ):
        super(SwinSeg, self).__init__()
        self.backbone = build_backbone(pretrained=pretrained)
        self.num_features = num_features
        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size
        self.dims = dims

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPoolingGELU(dims[-1],
            num_levels, 
            bt_size=bt_size, 
            level_size=level_size,
            out_size=self.spp_size,
            grids=spp_grids, 
            square_grid=spp_square_grid,
        )
        self.upsamples = []

        for i in range(3):
            lat = _UpsampleGELU(self.num_features, dims[i], self.num_features, norm=True, k=3)
            self.upsamples.append(lat)

        self.upsamples = nn.ModuleList(self.upsamples[::-1])

        self.random_init = [self.spp, self.upsamples]

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return self.backbone.parameters()
    
    def forward_up(self, features):
        features = features[::-1]
        x = features[0]
        upsamples = []
        for skip, up in zip(features[1:], self.upsamples):
            x = up(x, skip)
            upsamples.append(x)
        return x, upsamples
    
    def forward(self, batch):
        x = self.backbone(batch).reshaped_hidden_states
        spp_input, features = x[-1], list(x[:-2])

        spp_output = self.spp(spp_input)
        features.append(spp_output)
        prelogits, upsamples = self.forward_up(features)
        return prelogits, upsamples


def build_backbone(height=768, width=768, window_size=8, pretrained=True):
    if pretrained:
        model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    else:
        model = SwinModel()
    print(model.config)
    exit(0)
    model.config.window_size = window_size
    model.config.image_size = height
    model.config.image_size = width
    model.config.output_hidden_states = True
    return model

def build_seg(pretrained=True):
    return SwinSeg(pretrained=pretrained)