from itertools import chain

import torch.nn as nn

from .decoder import build_rec_decoder

from ...seg_decoders.deeplab.dws_aspp import build_dwsaspp
from ...util import upsample

from .config import decoder_layers

class SemsegModel(nn.Module):
    def __init__(self, backbone, encoder='resnet18', delta_d=0, route='bw', idlc=[0,0]):
        super().__init__()
        self.backbone = backbone
        num_layers = 3 # (R, G, B) channel
        self.delta_d = delta_d
        self.idlc = idlc
        print(f"\nApplying IDLC in upsample layer index: {self.idlc}.")
        self.decoder_layers = decoder_layers[route][self.delta_d]
        if self.decoder_layers is not None:
            print(decoder_layers[route]["propagation"])
        print(f"Following decoder layers are frozen: {self.decoder_layers}")

        print('\nFreezing the complete segmentation network')
        print('Decoder architecture for the reconstruction decoder: deeplab')

        self.aspp = build_dwsaspp(encoder, nn.BatchNorm2d) # (backbone, output_stride, nn.BatchNorm2d)
        self.rec_decoder = build_rec_decoder(num_layers, encoder, nn.BatchNorm2d, idlc=self.idlc)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN and/or entire backbone parameters
        """
        # First set all modules to the train mode
        super().train(mode)

        # Reset the models that are frozen, i.e. that are not trained, back to the eval() mode.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Reset the BN modules that are frozen, i.e. that are not trained, back to the eval() mode.
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        if self.delta_d > 0:
            print("Freezing parts of reconstruction decoder.")
            self.freeze_decoders(mode=mode)

    def freeze_decoders(self, mode=True):
        """
        Freeze selected last reconstruction decoder layers. 
        """
        for name, param in self.aspp.named_parameters():
            if any(x in name for x in self.decoder_layers):
                param.requires_grad = False

        for name, param in self.rec_decoder.named_parameters():
            if any(x in name for x in self.decoder_layers):
                param.requires_grad = False

        for name, module in self.aspp.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

        for name, module in self.rec_decoder.named_modules():
            if any(x in name for x in self.decoder_layers):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

    def eval(self):
        """
        Override the default eval() to freeze the BN and/or entire backbone parameters
        """
        super().eval()

    # Upsamling with the new AE-Decoder
    def forward_up_rec_decoder(self, features, image_size, idlc):
        x, low_level_feat = features
        x = self.aspp(x)
        return self.rec_decoder(x, low_level_feat, image_size, idlc)

    def forward(self, batch):
        seg_decoder_output = self.backbone(batch)
        # Get the segmentation logits
        logits = seg_decoder_output['logits']

        # Get the delta_d feature
        idlc = seg_decoder_output['idlc']
        upsampled = upsample(self.training, logits, batch.shape[2:4])
        features = [seg_decoder_output['spp_input'], seg_decoder_output['low_level_feat']]
        rec_decoder_output = self.forward_up_rec_decoder(features, batch.shape[2:4], idlc)

        return {'logits': upsampled, 'image_reconstruction': rec_decoder_output}

    def random_init_params(self):
        return chain(
            *([self.aspp.parameters(), self.rec_decoder.parameters()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
