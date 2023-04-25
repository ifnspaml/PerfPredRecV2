import torch
from ...seg_decoders.deeplab.decoder import Decoder
from ...util import upsample

MEAN = torch.tensor([[[[0.485, 0.456, 0.406]]]]).permute(0, 3, 1, 2).to(device='cuda')
STD = torch.tensor([[[[0.229, 0.224, 0.225]]]]).permute(0, 3, 1, 2).to(device='cuda')

class RecDecoder(Decoder):
    def __init__(self, num_classes, backbone, BatchNorm, idlc=[0,0]):
        super().__init__(num_classes, backbone, BatchNorm)
        self.idlc = idlc + [0]
    
    def forward(self, x, low_level_feat, image_size, idlc):
        # x = super().forward(x, low_level_feat)
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = upsample(self.training, x, low_level_feat.size()[2:])
        x = torch.cat((x, low_level_feat), dim=1)
        for i, (last_conv, merge) in enumerate(zip(self.last_conv, self.idlc)):
            x = last_conv(x)
            if merge:
                x = x + idlc[i]

        x = upsample(self.training, x, image_size)
        x = torch.sigmoid(x)
        x = x - MEAN
        x = x / STD
        return x

def build_rec_decoder(num_classes, backbone, BatchNorm, idlc=[0,0]):
    return RecDecoder(num_classes, backbone, BatchNorm, idlc=idlc)