import torch
import torch.nn as nn
from typing import Sequence
from ..models.flow_decoder import FlowDecoder
from ..util import make_obj_from_conf
from omegaconf.dictconfig import DictConfig
from typing import Dict, Sequence, Any


class FlowLearner(nn.Module):
    device_count = 1

    def __init__(
        self,
        encoder: nn.Module | Dict[str, Any] | DictConfig,
    ):
        super(FlowLearner, self).__init__()

        self.device = None
        # self.fpyramid = PWCEncoder()
        # self.pwc_model = PWCDecoder()
        if type(encoder) in [dict, DictConfig]:
            encoder = make_obj_from_conf(encoder)
        else:
            assert isinstance(encoder, torch.nn.Module), type(encoder)
        self.encoder = encoder
        self.decoder = FlowDecoder(enc_dims=encoder.dims)

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def set_devices(self, devices: Sequence[int]) -> None:
        self.device = devices[0]
        self.encoder.cuda(devices[0])
        self.decoder.cuda(devices[0])

    def inference_flow(self, img1, img2):
        img_hw = [img1.shape[2], img1.shape[3]]
        feature_list_1, feature_list_2 = self.encoder(img1), self.encoder(img2)
        optical_flow = self.decoder(feature_list_1, feature_list_2, img_hw)[0]
        return optical_flow

    def inference_corres(self, batch):
        info = self.forward(batch)
        info.pop("feature_pyramid_one")
        info.pop("feature_pyramid_two")
        info.pop("flow_pred")
        info = {k: v[0] for k, v in info.items()}
        return info

    def forward(self, batch):

        img1 = batch["img1"].to(self.device)
        img2 = batch["img2"].to(self.device)
        img_hw = img1.shape[2:]

        # get the optical flows and reverse optical flows for each pair of adjacent images
        feature_list_1, feature_list_2 = self.encoder(img1), self.encoder(img2)
        decoding = self.decoder(feature_list_1, feature_list_2, img_hw)

        return {
            "feature_pyramid_one": feature_list_1,
            "feature_pyramid_two": feature_list_2,
            **decoding,
        }
