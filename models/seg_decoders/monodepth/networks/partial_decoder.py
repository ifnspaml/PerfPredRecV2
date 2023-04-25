import torch
import torch.nn as nn
import torch.nn.functional as F

class PreConvBlock(nn.Module):
    """Decoder basic block
    """

    def __init__(self, pos, n_in, n_out):
        super().__init__()
        self.pos = pos

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(n_in, n_out, 3)
        self.nl = nn.ELU()

    def forward(self, image_size, *x):
        if self.pos == 0:
            x_pre = x[:self.pos]
            x_cur = x[self.pos]
            x_pst = x[self.pos + 1:]
        else:
            x_pre = x[:self.pos]
            x_cur = x[self.pos - 1]
            x_pst = x[self.pos + 1:]

        x_cur = self.pad(x_cur)
        x_cur = self.conv(x_cur)
        x_cur = self.nl(x_cur)

        return x_pre + (x_cur, ) + x_pst

class UpSkipBlock(nn.Module):
    """Decoder basic block

    Perform the following actions:
        - Upsample by factor 2
        - Concatenate skip connections (if any)
        - Convolve
    """

    def __init__(self, pos, ch_in, ch_skip, ch_out, flag=False):
        super().__init__()
        self.pos = pos

        self.up = nn.Upsample(scale_factor=2)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(ch_in + ch_skip, ch_out, 3)
        self.nl = nn.ELU()
        self.pos = pos
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.flag = flag

    def forward(self, image_size, *x):
        if self.pos == 5:
            x_pre = x[:self.pos - 1 ]
            x_new = x[self.pos - 1]
            x_skp = tuple()
            x_pst = x[self.pos:] 
        else:
            x_pre = x[:self.pos - 1]
            x_new = x[self.pos - 1]
            x_skp = x[self.pos]
            x_pst = x[self.pos:]

        x_new = F.interpolate(x_new, x_skp.size()[2:4]) if len(x_skp) > 0 else F.interpolate(x_new, image_size) # default: bilinear


        # Mix in skip connections from the encoder
        # (if there are any)
        if len(x_skp) > 0:
            if self.flag is False:
                x_new = torch.cat((x_new, x_skp), 1)

        # Combine up-scaled input and skip connections
        x_new = self.pad(x_new)
        x_new = self.conv(x_new)
        x_new = self.nl(x_new)

        return x_pre + (x_new, ) + x_pst

class PartialDecoder(nn.Module):
    """Decode some features encoded by a feature extractor

    Args:
        chs_dec: A list of decoder-internal channel counts
        chs_enc: A list of channel counts that we get from the encoder
        start: The first step of the decoding process this decoder should perform
        end: The last step of the decoding process this decoder should perform
    """

    def __init__(self, chs_dec, chs_enc, start=0, end=None):
        super().__init__()

        self.start = start
        self.end = (2 * len(chs_dec)) if (end is None) else end

        self.chs_dec = tuple(chs_dec) # 256, 128, 64, 32, 16
        self.chs_enc = tuple(chs_enc) # 512, 256, 128, 64, 64

        self.blocks = nn.ModuleDict()

        for step in range(self.start, self.end):
            macro_step = step // 2
            mini_step = step % 2
            pos_x = (step + 1) // 2

            # The decoder steps are interleaved ...
            if (mini_step == 0):
                n_in = self.chs_dec[macro_step - 1] if (macro_step > 0) else self.chs_enc[0]
                n_out = self.chs_dec[macro_step]

                # ... first there is a pre-convolution ...
                self.blocks[f'step_{step}'] = PreConvBlock(pos_x, n_in, n_out)

            else:
                # ... and then an upsampling and convolution with
                # the skip connections input.
                n_in = self.chs_dec[macro_step]
                n_skips = self.chs_enc[macro_step + 1] if ((macro_step + 1) < len(chs_enc)) else 0
                n_out = self.chs_dec[macro_step]

                self.blocks[f'step_{step}'] = UpSkipBlock(pos_x, n_in, n_skips, n_out)

    def chs_x(self):
        return self.chs_dec

    @classmethod
    def gen_head(cls, chs_dec, chs_enc, end=None):
        return cls(chs_dec, chs_enc, 0, end)

    @classmethod
    def gen_tail(cls, head, end=None):
        # print(head.chs_dec, head.chs_enc, head.end, end)
        return cls(head.chs_dec, head.chs_enc, head.end, end)

    def forward(self, image_size, *x):
        upsamples = []
        for step in range(self.start, self.end):
            x = self.blocks[f'step_{step}'](image_size, *x)
            if isinstance(self.blocks[f'step_{step}'], UpSkipBlock):
                upsamples.append(x[self.blocks[f'step_{step}'].pos - 1])
        return upsamples, x

if __name__ == '__main__':
    ch_dec = [256, 128, 64, 32, 16]
    ch_enc = [512, 256, 128, 64, 64]
    decoder = PartialDecoder(ch_dec, ch_enc, end=10)
    decoder.eval()
    for x in range(10):
        print(isinstance(decoder.blocks[f'step_{x}'], UpSkipBlock))

    torch.save(decoder.state_dict(), 'partial_decoder.pth', _use_new_zipfile_serialization=False)

    new_decoder = PartialDecoder(ch_dec, ch_enc, end=10)
    new_decoder_dict = new_decoder.state_dict()
    decoder_state_dict = torch.load('partial_decoder.pth', map_location=torch.device('cpu'))
    new_decoder.load_state_dict(decoder_state_dict)
    for name, param in decoder_state_dict.items():
        new_decoder_dict[name].copy_(param)
    new_decoder.load_state_dict(new_decoder_dict, strict=True)
    print("######################################################################################")

    for x in range(10):
        print(isinstance(new_decoder.blocks[f'step_{x}'], UpSkipBlock))