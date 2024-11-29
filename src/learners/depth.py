import torch
import torch.nn as nn
from ..models.backbone import BackBone
from ..models.depth_decoder import DepthDecoder
from ..util import load_class
from omegaconf.dictconfig import DictConfig
from typing import Dict, Sequence, Any


class DepthLearner(nn.Module):
    device_count = 2

    def __init__(
        self, encoder: BackBone | Dict[str, Any] | DictConfig, decoder_use_skip=True
    ) -> None:
        # TODO: selectable decoder
        super().__init__()

        if type(encoder) in [dict, DictConfig]:
            # TODO: bring these three lines to a single function in util
            bb_cls = load_class(encoder["target"])
            params = encoder["params"] if "params" in encoder else {}
            encoder = bb_cls(**params)
        else:
            assert isinstance(encoder, torch.nn.Module), type(encoder)

        self.encoder = encoder
        self.decoder = DepthDecoder(
            self.encoder.dims,
            self.encoder.pyramid_level_names,
            use_skips=decoder_use_skip,
        )

    def set_devices(self, devices: Sequence[int]) -> None:
        self.devices = devices
        self.encoder.cuda(devices[0])
        self.decoder.cuda(devices[1])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img = batch["img"]
        img = img.cuda(self.devices[0])
        features = self.encoder(img)
        for k, feat in features.items():
            features[k] = feat.cuda(self.devices[1])
        depth_pyr = self.decoder(features)
        depth_pyr["pred"] = depth_pyr["l7"]
        return depth_pyr
