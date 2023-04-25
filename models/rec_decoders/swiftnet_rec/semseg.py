from itertools import chain

import torch.nn as nn

from models.rec_decoders.swiftnet_rec.rec_decoders.swiftnet import SwiftNetDecoder  # New ae-decoder-object from class Basis_Dec (SwiftNet)
from .rec_decoders import SUPPORTED_REC_DECODER
from ...util import upsample, _BNActConv, BasicBlock, Bottleneck


class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, rec_decoder=None, use_bn=True,
                 efficient=True, lateral=False, delta_d=0, idlc=0, route='lat_bw', **kwargs):
        super().__init__()
        self.rec_decoder_name = rec_decoder
        self.backbone = backbone
        self.num_classes = num_classes
        self.lateral = lateral

        # use the new universal BNActConv for different Activation-Functions
        self.logits = _BNActConv(self.backbone.num_features, num_classes, batch_norm=use_bn)

        # last shared decoder layer, default: 0
        self.delta_d = delta_d

        print('\nFreezing the complete segmentation network')
        print('Decoder architecture for the reconstruction decoder: ', self.rec_decoder_name)

        # The reconstruction decoder is constructed here
        assert self.rec_decoder_name in SUPPORTED_REC_DECODER, \
            f"The decoder type {self.rec_decoder_name} is not supported."

        self.rec_decoder = SwiftNetDecoder(use_skips=False if 'noskip' in self.rec_decoder_name else True,
                                            use_spp=False if 'nospp' in self.rec_decoder_name else True,
                                            delta_d=delta_d, idlc=idlc, route=route, inplanes=self.backbone.dims)

    # Upsamling with the new AE-Decoder
    def forward_up_rec_decoder(self, features, image_size,
                               upsamples, features_resnet=None
                               ):
        if 'swiftnet' in self.rec_decoder_name:
            return self.rec_decoder(features, image_size, upsamples)
        elif 'resnet' in self.rec_decoder_name:
            return self.rec_decoder(features, features_resnet)
        else:
            ValueError(f"The decoder type {self.rec_decoder_name} is not supported.")

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN and/or entire backbone parameters
        """
        # First set all modules to the train mode
        super().train(mode)

        # Reset the models that are frozen, i.e. that are not trained, back to the eval() mode.
        self.backbone.eval()
        self.logits.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.logits.parameters():
            param.requires_grad = False

        # Reset the BN modules that are frozen, i.e. that are not trained, back to the eval() mode.
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        if self.delta_d > 0:
            print(f"\nFreezeing selected last shared decoder layers")
            self.rec_decoder.freeze_decoders(mode=mode)

    def eval(self):
        """
        Override the default eval() to freeze the BN and/or entire backbone parameters
        """
        super().eval()

    def forward(self, batch):
        seg_decoder_output = self.backbone(batch)
        # Output of semseg just before last _BNActConv block
        semseg_prelogits = seg_decoder_output['semseg']['prelogits']
        upsamples = seg_decoder_output['semseg']['upsamples']

        # Compute Mask for the Semantic Segmentation
        logits = self.logits.forward(semseg_prelogits)
        upsampled = upsample(self.training, logits, batch.shape[2:4])

        if 'swiftnet' in self.rec_decoder_name:
            features = [seg_decoder_output['spp_input'], seg_decoder_output['skips_and_spp']]
            rec_decoder_output = self.forward_up_rec_decoder(features, batch.shape[2:4], upsamples)
        elif 'resnet' in self.rec_decoder_name:
            features = rec_decoder_output['backbone_output']
            rec_decoder_output = self.forward_up_rec_decoder(features, batch.shape[2:4],
                                                             rec_decoder_output['skips_and_spp'])
        else:
            ValueError(f"The decoder type {self.rec_decoder_name} is not supported.")

        return {'logits': upsampled, 'image_reconstruction': rec_decoder_output}

    def random_init_params(self):
        return chain(
            *([self.rec_decoder.parameters(), self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
