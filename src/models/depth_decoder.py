from .blocks import ConvNeXtBlock, ConvBlock, Conv3x3
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch
from ..util import are_lists_equal


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class DepthDecoder(nn.Module):
    valid_methods = ["sum", "concat"]

    def __init__(
        self,
        enc_dims,
        pyramid_level_names,
        num_ch_dec=[
            1,
            16,
            32,
            64,
            128,
            256,
            768,
        ],
        num_output_channels=1,
        use_skips=True,
        method="sum",
        use_conv_next=True,
    ):
        super().__init__()

        assert len(num_ch_dec) == len(pyramid_level_names) == len(enc_dims)
        assert are_lists_equal(pyramid_level_names, list(enc_dims.keys()))
        assert method in self.valid_methods
        if method == "concat":
            raise NotImplementedError("'concat' method is not implemented")

        self.num_output_channels = num_output_channels
        self.method = method
        self.num_ch_dec = {k: v for k, v in zip(pyramid_level_names, num_ch_dec)}
        self.enc_dims = enc_dims
        self.pyramid_level_names = pyramid_level_names
        self.use_skips = use_skips
        self.use_conv_next = use_conv_next

        self.dispconvs = nn.ModuleDict()
        for l in self.pyramid_level_names:
            self.dispconvs[l] = Conv3x3(self.num_ch_dec[l], self.num_output_channels)

        self.sigmoid = nn.Sigmoid()

        self.net = OrderedDict()
        for scale in range(len(pyramid_level_names) - 1, 0, -1):
            ch_in = (
                self.num_ch_dec[pyramid_level_names[-1]]
                if scale == len(pyramid_level_names) - 1
                else self.num_ch_dec[pyramid_level_names[scale + 1]]
            )
            ch_out = self.num_ch_dec[pyramid_level_names[scale]]
            if use_conv_next:
                self.net[("upconv", pyramid_level_names[scale], 0)] = ConvNeXtBlock(
                    ch_in
                )
            else:
                self.net[("upconv", pyramid_level_names[scale], 0)] = ConvBlock(
                    ch_in, ch_out
                )
            if self.use_skips and scale > 1:
                ch_in += self.enc_dims[pyramid_level_names[scale - 1]]
            self.net[("upconv", pyramid_level_names[scale], 1)] = ConvBlock(
                ch_in, ch_out
            )
        self.decoder = nn.ModuleList(list(self.net.values()))

    def forward(self, input_features):
        x = input_features[self.pyramid_level_names[-1]]
        outputs = dict()
        for scale in range(len(self.pyramid_level_names) - 1, 0, -1):
            l = self.pyramid_level_names[scale]
            l_before = self.pyramid_level_names[scale - 1]
            x = self.net[("upconv", l, 0)](x)
            x = [upsample(x)]
            if self.use_skips and scale > 1:
                x += [input_features[l_before]]
            x = torch.cat(x, 1)
            x = self.net[("upconv", l, 1)](x)

            # get disp
            outputs[l_before] = self.sigmoid(self.dispconvs[l](x))

        return outputs
