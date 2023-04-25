import torch.nn as nn

from .semseg import SemsegModel
from .rec_decoders.config import layer_map
from .resnet.full_network import resnet18, resnet50
from .convnext.full_network import convnext_tiny

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
    model.load_state_dict(pretrained_dict, strict=False)


# Class SwiftNetRec based on SwiftNet and an separate Autoencoder-Decoder for Image-Upsampling
class SwiftNetRec(nn.Module):
    def __init__(self, num_classes_wo_bg, encoder='resnet18', delta_d=0, idlc=[0,0,0], route='alt_fw', **kwargs):
        super().__init__()
        # Create the SwiftNet model
        self.encoder = encoder
        if encoder == 'resnet18':
            model = resnet18(pretrained=True, efficient=False, use_bn=True, **kwargs)
        elif encoder == 'resnet50':
            model = resnet50(pretrained=True, efficient=False, use_bn=True, **kwargs)
        elif encoder == 'convnext_tiny':
            model = convnext_tiny(pretrained=True, in_22k=False)
        elif encoder == 'swin_t':
            from .swin_rec import swin_rec
            model = swin_rec.build_seg(pretrained=True)
        else:
            ValueError(f"Encoder/backbone {encoder} is not recognized.")

        # Pass the SwiftNet model and create a image reconstruction on top
        self.loaded_model = SemsegModel(model, num_classes_wo_bg,
                                        use_bn=True,
                                        efficient=False,
                                        delta_d=delta_d,
                                        idlc=idlc,
                                        route=route,
                                        **kwargs)
        self.loaded_model.eval()

    def forward(self, batch):
        output_dict = self.loaded_model.forward(batch)
        return output_dict

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()

    # TODO: Might refractor this
    def update_params(self, pretrained_dict, model_dict, init_mode='random'):
        """
        Copy and transfer the segmentation to reconstruction weights.
        """
        if init_mode == 'random':
            print("Reconstruction decoder is randomly initialized.")
            if self.encoder == 'resnet50':
                pretrained_dict = self.compare(pretrained_dict)
            return {k: v for k, v in pretrained_dict.items() if k in model_dict}

        print("Reconstruction decoder is initialized with segmentation weights.")
        seg_decoder, seg_spp = layer_map["seg_decoder"], layer_map["seg_spp"]
        rec_decoder, rec_spp = layer_map["rec_decoder"], layer_map["rec_spp"]

        # FIXME: 
        if self.encoder == 'swin_t': seg_decoder = seg_decoder.replace('upsample', 'upsamples')

        def replace(seg, rec):
            old_key = k
            new_key = k.replace(seg, rec)
            pretrained_dict[new_key] = pretrained_dict[old_key]

        # TODO: Might refractor this
        for k, _ in model_dict.items():
            if all(x in k for x in ['backbone', seg_decoder]) and 'last_conv.2' not in k:
                replace(seg_decoder, rec_decoder)
            elif all(x in k for x in ['backbone', seg_spp]):
                replace(seg_spp, rec_spp)
        return pretrained_dict
    
    def compare(self, pretrained_dict):
        # pretrained_dict = {k.replace('backbone.seg_decoder.logits', 'logits'): v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k.replace('seg_decoder.spp_ae', 'spp'): v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k.replace('seg_decoder.upsample', 'upsample'): v for k, v in pretrained_dict.items()}
        # return pretrained_dict
        new_dict = {}
        for k, v in pretrained_dict.items():
            if 'backbone.seg_decoder.logits' in k:
                old_key = k
                new_key = k.replace('backbone.seg_decoder.logits', 'logits')
                new_dict[new_key] = pretrained_dict[old_key]
            elif 'seg_decoder.spp_ae' in k:
                old_key = k
                new_key = k.replace('seg_decoder.spp_ae', 'spp')
                new_dict[new_key] = pretrained_dict[old_key]
            elif 'seg_decoder.upsample' in k:
                old_key = k
                new_key = k.replace('seg_decoder.upsample', 'upsample')
                new_dict[new_key] = pretrained_dict[old_key]
            else:
                new_dict[k] = v

        return new_dict

# if __name__ == "__main__":
#     from .. swin_transformer import SwinSeg
#     net = SwinSeg(19)
#     loaded_model = SemsegModel(net, 19, use_bn=True,efficient=False)