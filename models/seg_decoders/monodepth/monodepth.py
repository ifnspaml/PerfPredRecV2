#!/usr/bin/env python
import numpy as np
import torch.nn as nn

from . import networks

class Backbone(nn.Module):
    def __init__(self, num_layers, split_pos, pretrained=False):
        super().__init__()

        self.encoder = networks.ResnetEncoder(num_layers, pretrained)
        self.num_layers = num_layers  # This information is needed in the train loop for the sequential training

        # Number of channels for the skip connections and internal connections
        # of the decoder network, ordered from input to output
        self.shape_enc = tuple(reversed(self.encoder.num_ch_enc))
        self.shape_dec = (256, 128, 64, 32, 16)

        # self.decoder = networks.PartialDecoder.gen_head(self.shape_dec, self.shape_enc, split_pos)

    def forward(self, x):
        # The encoder produces outputs in the order
        # (highest res, second highest res, …, lowest res)
        x = self.encoder(x)
        # The decoder expects it's inputs in the order they are
        # used. E.g. (lowest res, second lowest res, …, highest res)
        x = tuple(reversed(x))
        # Replace some elements in the x tuple by decoded
        # tensors and leave others as-is
        return x

class MonoSeg(nn.Module):
    def __init__(self, common, num_classes):
        super(MonoSeg, self).__init__()

        chs_dec = np.array([256, 128, 64, 32, 16])
        chs_enc = common.encoder.num_ch_enc[::-1]
        self.decoder = networks.PartialDecoder(chs_dec, chs_enc, start=0, end=10)
        self.multires = networks.MultiResSegmentation(self.decoder.chs_x()[-1:], num_classes)

    def forward(self, image_size, *x):
        upsamples, x = self.decoder(image_size, *x)
        x = self.multires(*x[-1:])
        x_lin = x[-1]

        return upsamples, x_lin

class MonoDepth(nn.Module):
    def __init__(self, split_pos=1, num_layers=18, weights_init='pretrained', num_classes=19):
        super().__init__()
        split_pos = max((2 * split_pos) - 1, 0)
        self.common = Backbone(
            num_layers, split_pos,
            weights_init == 'pretrained'
        )
        self.seg = MonoSeg(self.common, num_classes)

    def forward(self, batch):
        image_size = batch.shape[2:4]
        features = self.common(batch)
        # logits = self.seg(image_size, *features)
        # return logits, features
        upsamples, logits = self.seg(image_size, *features)
        return logits, upsamples, features
    # 
    def fine_tune_params(self):
        return self.common.encoder.parameters()
    
    def random_init_params(self):
        return self.seg.parameters()

def build(num_layers=18, num_classes=19):
    return MonoDepth(num_layers=num_layers, num_classes=num_classes)

if __name__ == '__main__':
    model = build()
    sample = torch.rand(2, 3, 768, 768)
    print(model)
