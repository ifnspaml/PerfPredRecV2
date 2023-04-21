import torch.nn as nn
from . import monodepth

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


class MonoDepth(nn.Module):
    """
    A wrapper for SGDepth segmentation based on Marvin's codebase.

    Parameters
    ----------
    num_classes_wo_bg : int
        Number of categories
    backbone : nn.Module
        The full SGDepth segmentation network
    
    Returns
    -------
    Tensor
        Segmentation maps if the same dimension
    """
    def __init__(self, num_classes_wo_bg, encoder='resnet18'):
        super().__init__()
        self.num_layers = {'resnet18': 18, 'resnet50': 50}[encoder]
        self.backbone = monodepth.build(num_layers=self.num_layers, num_classes=num_classes_wo_bg)
        self.backbone.eval()

    def forward(self, batch):
        output = self.backbone.forward(batch)
        return output[0]

    def random_init_params(self):
        return self.backbone.random_init_params()

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()