import torch.nn as nn

from .config import layer_map
from .multihead import SemsegModel
from .full_network import build_deeplabrec

# --------------------------------------------------------------------------------
#  Custom function for the specific model architecture to load/update state_dict
# --------------------------------------------------------------------------------
# TODO: Adjust this function to make it robust, since this is purely adopted from SwiftNet
def load_state_dict_into_model(model, pretrained_dict):
    model_dict = model.state_dict()
    for name, param in pretrained_dict.items():
        if name not in model_dict:
            print(f"{name}, State_dict mismatch!", flush=True)
            continue
        model_dict[name].copy_(param)
    model.load_state_dict(pretrained_dict)


# Class DeepLabRec based on DeepLabV3+ and an separate Autoencoder-Decoder for Image-Upsamplingl
class DeepLabRec(nn.Module):
    def __init__(self, num_classes_wo_bg, encoder='resnet18', atrous=1, delta_d=0, idlc=[0,0], route='bw'):
        super().__init__()
        # Create the DeepLabV3+ segmentation
        self.encoder = encoder
        model = build_deeplabrec(num_classes_wo_bg, encoder=encoder, atrous=atrous)

        # Pass the DeepLabV3+ segmentation and create a image reconstruction on top
        self.loaded_model = SemsegModel(model, encoder=encoder, delta_d=delta_d, idlc=idlc, route=route)
        self.loaded_model.eval()

    def forward(self, batch):
        output_dict = self.loaded_model.forward(batch)
        return output_dict

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()

    def update_params(self, pretrained_dict, model_dict, init_mode='random'):
        """
        Copy and transfer the segmentation to reconstruction weights.
        """
        if self.encoder != 'swin_t':
            encoder_old = 'encoder'
            encoder_new = 'loaded_model.backbone.encoder'
        else:
            encoder_old = 'encoder.net'
            encoder_new = 'loaded_model.backbone.encoder.net'
        pretrained_dict = {k.replace(encoder_old, encoder_new): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('aspp.', 'loaded_model.backbone.aspp.'): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('decoder', 'loaded_model.backbone.decoder'): v for k, v in pretrained_dict.items()}
        if init_mode == 'random':
            print("Reconstruction decoder is randomly initialized.")
            return {k: v for k, v in pretrained_dict.items() if k in model_dict}

        print("Reconstruction decoder is initialized with segmentation weights.")
        seg_decoder, seg_spp = layer_map["seg_decoder"], layer_map["seg_spp"]
        rec_decoder, rec_spp = layer_map["rec_decoder"], layer_map["rec_spp"]
        for k, _ in model_dict.items():
            if all(x in k for x in ['backbone', seg_decoder]) and 'last_conv.2' not in k:
                old_key = k
                new_key = k.replace(seg_decoder, rec_decoder)
                pretrained_dict[new_key] = pretrained_dict[old_key]
            elif all(x in k for x in ['backbone', seg_spp]):
                old_key = k
                new_key = k.replace(seg_spp, rec_spp)
                pretrained_dict[new_key] = pretrained_dict[old_key]
        return pretrained_dict