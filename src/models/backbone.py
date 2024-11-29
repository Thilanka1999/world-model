from torch import nn, Tensor
from typing import Dict
from ..constants import valid_enc_Ds, valid_encs
from mt_pipe.src.encoders import VisionTransformerBase, ResNet50, ResNet18, ConvNeXt
import torch
from omegaconf import OmegaConf
import numpy as np


class BackBone(nn.Module):
    device_count = 1
    enc_Ds = valid_enc_Ds
    valid_encs = valid_encs

    def _init_encoder(self, enc_params: OmegaConf):
        if self.enc_name == "ResNet50":
            self.encoder = ResNet50(**dict(enc_params))
        elif self.enc_name == "ResNet18":
            self.encoder = ResNet18(**dict(enc_params))
        elif self.enc_name == "ViT-B":
            self.encoder = VisionTransformerBase(**dict(enc_params))
        elif self.enc_name == "ConvNeXt":
            self.encoder = ConvNeXt(**dict(enc_params))
        self.emb_D = self.enc_Ds[self.enc_name]
        self.dims = self.encoder.dims

    def __init__(
        self,
        enc_name: str,
        enc_params: Dict = {},
    ) -> None:
        super().__init__()
        if enc_name not in self.valid_encs:
            raise ValueError(
                f"Unsupported 'enc_name' definition. Supported encoders are {self.valid_encs}"
            )
        self.enc_name = enc_name
        self._init_encoder(enc_params)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        *rest_axs, C, H, W = x.shape
        stacked_B = np.prod(rest_axs)
        with torch.no_grad():
            stacked_tens = x.view(stacked_B, C, H, W)

        stacked_tens = stacked_tens
        out = self.encoder(stacked_tens)
        for k, v in out.items():
            v_trail_shape = v.shape[1:]
            out[k] = v.view(([*rest_axs, *v_trail_shape]))

        return out
