import torch
import torch.nn as nn
from .config import decoder_layers
from ...seg_decoders.monodepth.monodepth import MonoSeg
from ...seg_decoders.monodepth import UpSkipBlock

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')

def unsqueeze_merge_squeeze(x, seg_feat, idx):
    x_pre = x[:idx]
    x_cur = x[idx - 1]
    x_pst = x[idx + 1:]
    # print(x_cur.shape, seg_feat.shape)
    # exit(0)
    x_cur += seg_feat
    return x_pre + (x_cur, ) + x_pst
    

class MonoDepthRec(MonoSeg):
    def __init__(self, common, num_outputs, route='fw', delta_d=0, idlc=[0, 0, 0, 0], rm_early_skip=False):
        super().__init__(common, num_outputs)
        self.decoder_layers = decoder_layers[route][delta_d]
        print(f'Freezing following reconstruction decoder layers: {self.decoder_layers} during training.')
        self.length = len(self.decoder.blocks)
        self.up_stage_indices = [1, 3, 5, 7]
        self.merge_dec = {idx: i for idx, i in zip(self.up_stage_indices, idlc)}
        print(f"Decoder connection gates: {self.merge_dec}.")

        if rm_early_skip is True:
            step_7_in_ch, step_7_out_ch, pos = self.decoder.blocks[f'step_7'].ch_in, self.decoder.blocks[f'step_7'].ch_out, self.decoder.blocks[f'step_7'].pos
            self.decoder.blocks[f'step_7'] = UpSkipBlock(pos, step_7_in_ch, 0, step_7_out_ch, flag=rm_early_skip)
    
    def forward_upsampling(self, image_size, upsamples, *x):
        up_idx = 0
        for step in range(self.length):
            x = self.decoder.blocks[f'step_{step}'](image_size, *x)
            # if isinstance(self.decoder.blocks[f'step_{step}'], UpSkipBlock):
            if step in self.up_stage_indices:
                if self.merge_dec[step]:
                    # print(self.merge_dec, step, up_idx)
                    x = unsqueeze_merge_squeeze(x, upsamples[up_idx], self.decoder.blocks[f'step_{step+1}'].pos)
                up_idx = up_idx + 1
        # exit(0)
        return x
    
    def forward(self, image_size, upsamples, *x):
        x = self.forward_upsampling(image_size, upsamples, *x)
        x = self.multires(*x[-1:])
        logits = x[-1]
        image = self.reconstruct(logits)
        return image

    def reconstruct(self, logits):
        # the Sigmoid-function transforms the input to [0, 1]-space
        x = torch.sigmoid(logits)

        # Perform zero-mean normalization to have the space as the input
        x = x - MEAN
        x = x / STD
        return x

    def _freeze_decoders(self, mode=True):
        """
        Freeze selected last reconstruction decoder layers. 
        """
        for name, param in self.named_parameters():
            if any(x in f'rec.decoder.{name}' for x in self.decoder_layers):
                param.requires_grad = False

        for name, module in self.named_modules():
            if any(x in f'rec.decoder.{name}' for x in self.decoder_layers):
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

class SemsegModel(nn.Module):
    def __init__(self, backbone, encoder='resnet18', route='fw', delta_d=0, idlc=[0,0,0,0], rec_decoder='md'):
        super().__init__()
        self.backbone = backbone
        num_outputs = 3 # (R, G, B) channel
        rm_early_skip = True if 'rm_early_skip' in rec_decoder else False
        print(f"remove early skip connection: {rm_early_skip}")
        self.rec = MonoDepthRec(self.backbone.common, num_outputs, route=route, delta_d=delta_d, idlc=idlc, rm_early_skip=rm_early_skip)
        self.delta_d = delta_d

        print('\nFreezing the complete segmentation network')
        print(f'Decoder architecture for the reconstruction decoder: {rec_decoder}')

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
            self.rec._freeze_decoders(mode=mode)

    def eval(self):
        """
        Override the default eval() to freeze the BN and/or entire backbone parameters
        """
        super().eval()

    def forward(self, x):
        image_size = x.shape[2:4]
        logits, upsamples, features = self.backbone(x)

        image = self.rec(image_size, upsamples, *features)
        # image = self.rec.reconstruct(pre_sigmoid)

        return {'logits': logits, 'image_reconstruction': image}
    
    def random_init_params(self):
        return self.rec.parameters()
    
    def fine_tune_params(self):
        return self.backbone.fine_tune_params()