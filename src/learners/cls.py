import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Softmax
from typing import Tuple, Sequence, Dict, Any
import logging
from ..models.backbone import BackBone
from ..util import make_obj_from_conf, are_lists_equal
from omegaconf.dictconfig import DictConfig

logger = logging.getLogger()


class ClassificationHead(nn.Module):
    def __init__(self, in_dims: int, n_classes: int) -> None:
        super(ClassificationHead, self).__init__()

        self.net = Sequential(
            Linear(in_dims, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, n_classes),
            Softmax(dim=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features)

        return logits


class ClassLearner(nn.Module):
    device_count = 2

    def __init__(
        self, encoder: BackBone | Dict[str, Any] | DictConfig, n_classes: int = None
    ) -> None:
        assert n_classes is not None, "'n_classes' should not be None"

        super().__init__()
        if type(encoder) in [dict, DictConfig]:
            encoder = make_obj_from_conf(encoder)
        else:
            assert isinstance(encoder, torch.nn.Module), type(encoder)

        self.encoder = encoder
        in_dims = encoder.dims["l6"]
        self.decoder = ClassificationHead(in_dims=in_dims, n_classes=n_classes)

    def set_devices(self, devices: Sequence[int]) -> None:
        self.devices = devices
        self.encoder.cuda(devices[0])
        self.decoder.cuda(devices[1])

    def forward(self, batch: Dict[str, Tuple[torch.Tensor, int]]) -> torch.Tensor:
        tens = batch["img"]
        tens = tens.cuda(self.devices[0])
        feature_pyramid = self.encoder(tens)
        features = feature_pyramid["emb"].cuda(self.devices[1])

        logits = self.decoder(features)
        return {"logits": logits}

    def load_ckeckpoint(self, ckpt_path: str) -> None:
        cur_param_names = [name for name in self.state_dict().keys()]
        sd = torch.load(ckpt_path)["learner"]
        sd_names = [nm for nm in sd.keys()]

        if are_lists_equal(sd_names, cur_param_names):
            res = self.load_state_dict(sd)
        else:
            encoder_sd = {
                k[len("encoder.") :]: v
                for k, v in sd.items()
                if k.startswith("encoder")
            }
            res = self.encoder.load_state_dict(encoder_sd)

        logger.info(f"{self._get_name()}: {res}")
