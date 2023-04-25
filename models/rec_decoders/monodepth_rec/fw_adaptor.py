import torch.nn as nn
from . import monodepth
from .multihead import SemsegModel
from .config import layer_map

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
    model.load_state_dict(pretrained_dict, strict=True)

class MonoDepthRec(nn.Module):
    def __init__(self, num_classes_wo_bg, encoder='resnet18', idlc=[0,0,0,0], delta_d=0, route='fw', rec_decoder='md'):
        super().__init__()
        # TODO: Build support for CN-T backbone
        self.num_layers = {'resnet18': 18, 'resnet50': 50}[encoder]
        self.rec_decoder = rec_decoder
        self.backbone = monodepth.build(num_layers=self.num_layers, num_classes=num_classes_wo_bg)
        self.loaded_model = SemsegModel(self.backbone, delta_d=delta_d, route=route, idlc=idlc, rec_decoder=rec_decoder)
        del self.backbone
        self.loaded_model.eval()

    def forward(self, batch):
        output = self.loaded_model(batch)
        return output

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()

    def update_params(self, pretrained_dict, model_dict, init_mode='random'):
        pretrained_dict = {k.replace('backbone', 'loaded_model.backbone'): v for k, v in pretrained_dict.items()}
        if init_mode == 'random':
            return {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("Reconstruction decoder is initialized with segmentation weights.")
        seg_decoder, rec_decoder = layer_map["seg_decoder"], layer_map["rec_decoder"]
        for k, _ in model_dict.items():
            if seg_decoder in k:
                old_key = k
                if 'rm_early_skip' in self.rec_decoder and 'step_7' in old_key:
                    continue
                new_key = k.replace(seg_decoder, rec_decoder)
                pretrained_dict[new_key] = pretrained_dict[old_key]
        return pretrained_dict