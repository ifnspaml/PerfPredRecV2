from ....encoders.swin_transformer import SwinSeg

class SwinRec(SwinSeg):
    def __init__ (self, pretrained=True):
        super().__init__(pretrained=pretrained)
    
    def forward(self, batch):
        x = self.backbone(batch).reshaped_hidden_states
        spp_input, features = x[-1], list(x[:-2])

        spp_output = self.spp(spp_input)
        features.append(spp_output)
        prelogits, upsamples = self.forward_up(features)

        semseg_dict = {'prelogits': prelogits,
                       'features' : features,
                       'upsamples': upsamples}

        # Note that this dictionary only contains SwiftNet-specific outputs for now
        dict_ = {'semseg': semseg_dict,
                 'backbone_output': spp_input,
                 'spp_input': spp_input,
                 'skips_and_spp': features}

        return dict_


def build_seg(pretrained=True):
    return SwinRec(pretrained=pretrained)