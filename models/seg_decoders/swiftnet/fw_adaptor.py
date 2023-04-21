import torch.nn as nn

from .semseg import SemsegModel
from ... encoders.resnet.full_network import resnet18, resnet50
from ... encoders.convnext.full_network import convnext_tiny


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

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class SwiftNet(nn.Module):
    def __init__(self, num_classes_wo_bg, encoder='resnet18', **kwargs):
        super().__init__()
        use_bn = True

        # TODO: Modify this: Call function by applying register.
        self.encoder = encoder
        if encoder == 'resnet18':
            resnet = resnet18(pretrained=True, efficient=False, use_bn=use_bn, **kwargs)
        elif encoder == 'resnet50':
            resnet = resnet50(pretrained=True, efficient=False, use_bn=use_bn, **kwargs)
        elif encoder == 'convnext_tiny':
            resnet = convnext_tiny(pretrained=True, in_22k=False)
        elif encoder == 'swin_t':
            resnet = SwinSeg(pretrained=True)
        else:
            ValueError(f"Encoder/backbone {encoder} is not recognized.")
        self.loaded_model = SemsegModel(resnet, num_classes_wo_bg, use_bn=use_bn, freeze_complete_backbone=False,
                                        freeze_backbone_bn=False, freeze_backbone_bn_affine=False)
        self.loaded_model.eval()

    def forward(self, batch):
        output = self.loaded_model.forward(batch)
        return output

    def random_init_params(self):
        return self.loaded_model.random_init_params()

    def fine_tune_params(self):
        return self.loaded_model.fine_tune_params()

    #FIXME: Make this a global function    
    def configure_params(self, wd=0.25e-4, lr=1e-4):
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name and "encoder" in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if n in decay_parameters],
                "weight_decay": wd,
                "learning_rate": lr,
            },
            {
                "params": [p for n, p in self.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
                "learning_rate": lr,
            },
        ]
        return optimizer_grouped_parameters
