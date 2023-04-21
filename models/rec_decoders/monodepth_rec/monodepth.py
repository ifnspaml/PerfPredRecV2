#!/usr/bin/env python
from ...seg_decoders.monodepth.monodepth import MonoDepth

class MonoDepth(MonoDepth):
    def __init__(self, split_pos=1, num_layers=18, weights_init='pretrained', num_classes=19):
        super().__init__(split_pos=split_pos, num_layers=num_layers, weights_init=weights_init, num_classes=num_classes)
    
def build(num_layers=18, num_classes=19):
    return MonoDepth(num_layers=num_layers, num_classes=num_classes)