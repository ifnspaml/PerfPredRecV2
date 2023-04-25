# This is the standard SwiftNet decoder modified as an image reconstruction decoder

from itertools import chain

import torch
import torch.nn as nn

from  . config import decoder_layers
from models.util import _UpsampleBlend, _Upsample, upsample, _BNActConv, BasicBlock, SpatialPyramidPooling

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')

class SwiftNetDecoder(nn.Module):
    def __init__(self, block=BasicBlock, delta_d=0, idlc=0, route='alt_fw', *, num_features=128, k_up=3, use_bn=True, use_skips=True, use_spp=False, inplanes=64, dims=[96, 192, 384, 768]):

        super().__init__()  # inherit from higher parents class
        # TODO: Adjust the inplanes depending on the backbone network. inplanes(rn50) = 256, inplace(cn-t) = 96
        # self.inplanes = 64 * block.expansion
        self.inplanes = inplanes[0]
        self.use_bn = use_bn  # use Batch Normalization
        self.use_spp = use_spp
        upsamples = []  # create an array for the different upsampling layers
        self.decoder_layers = decoder_layers[route][delta_d]
        print(f"Following decoder layers are frozen: {self.decoder_layers}")
        if self.decoder_layers is not None:
            print(decoder_layers[route]["propagation"])
        self.logits = _BNActConv(num_maps_in=128, num_maps_out=3, batch_norm=use_bn)
        self.idlc = idlc
        print(f"\nApplying IDLC in upsample layer index: {self.idlc}.")

        if use_skips:
            upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
            self._make_layer(block, inplanes[1])
            upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
            self._make_layer(block, inplanes[2])
            if self.use_spp:
                upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
                self._make_layer(block, inplanes[3])
            else:
                # As the SPP is skipped we directly feed the 512 output feature maps to the upsample module
                upsamples += [_Upsample(self.inplanes * 2, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]
                self._make_layer(block, inplanes[3])
        else:
            upsamples += [_UpsampleBlend(num_features, num_features, use_bn=self.use_bn)]
            self._make_layer(block, 128)
            upsamples += [_UpsampleBlend(num_features, num_features, use_bn=self.use_bn)]
            self._make_layer(block, 256)
            if self.use_spp:
                upsamples += [_UpsampleBlend(num_features, num_features, use_bn=self.use_bn)]
                self._make_layer(block, 512)
            else:
                # As the SPP is skipped we directly feed the 512 output feature maps to the upsample module
                upsamples += [_UpsampleBlend(self.inplanes * 2, num_features, use_bn=self.use_bn)]
                self._make_layer(block, 512)

        if self.use_spp:
            spp_grids = (8, 4, 2, 1)
            spp_square_grid = False
            num_levels = 3
            self.spp_size = num_features
            bt_size = self.spp_size
            level_size = self.spp_size // num_levels
            self.spp_rec = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                                 out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                                 bn_momentum=0.01 / 2, use_bn=self.use_bn)

        self.upsample = nn.ModuleList(list(reversed(upsamples)))
        # FIXME: Here the spp module seems not to be included. A potential bug?
        self.random_init = self.upsample
        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # TODO?
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes):
        self.inplanes = planes * block.expansion

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def freeze_decoders(self, mode=True):
        """
        Freeze selected last reconstruction decoder layers. 
        """
        super().train(mode)
        for name, param in self.named_parameters():
            if any(x in name for x in self.decoder_layers):
                param.requires_grad = False

        for name, module in self.named_modules():
            if any(x in name for x in self.decoder_layers):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
        
    def forward(self, features, image_size, segmentation):
        if self.use_spp:
            x = features[0]  # Get the input to SPP, i.e., "spp_input"
            x = self.spp_rec.forward(x)  # features resembles the last lateral output just before spp. This is eqaul to x = features[0] in the else branch
            features = features[1][::-1]  # Get the skips and SPP, i.e., "skips_and_spp".
        else:
            x = features[0]  # Get the input to SPP, i.e., "spp_input"
            features = features[1][::-1]  # features resembles a list of skips and spp output. Reverse the list.
            # take the first element after the reverse operation
        
        for skip, up, seg, merge in zip(features[1:], self.upsample, segmentation, self.idlc):
            x = up(x, skip)
            if merge:
                x = x + seg

        x = self.logits.forward(x)
        x = upsample(self.training, x, image_size)

        # the Sigmoid-function transforms the input to [0, 1]-space
        x = torch.sigmoid(x)

        # Perform zero-mean normalization to have the space as the input
        x = x - MEAN
        x = x / STD

        return x
