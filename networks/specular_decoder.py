from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import ConvBlock, Conv3x3, upsample


class SpecularDecoder(nn.Module):
    """Lightweight specular confidence head.

    Mirrors DepthDecoder but outputs a 1-channel sigmoid map S(u,v) ∈ [0,1]
    where high values indicate specular highlights.  Uses output key
    ("specular", scale) to match the ("disp", scale) convention.

    The decoder shares encoder features with the depth branch — no extra
    encoder pass needed.  Build it only when opt.use_specular_net=True.
    """

    def __init__(self, num_ch_enc, scales=range(4), use_skips=True):
        super(SpecularDecoder, self).__init__()

        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("specularconv", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("specular", i)] = self.sigmoid(self.convs[("specularconv", i)](x))
        return outputs
