import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from typing import Sequence
from typing import Dict
import logging
from omegaconf.dictconfig import DictConfig
from ..models.backbone import BackBone
from ..util import make_obj_from_conf, are_lists_equal

logger = logging.getLogger()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegmentDecoder(nn.Module):
    def __init__(self, out_channels, pyramid_channels=[48, 48, 96, 192, 384, 768]):
        super(SegmentDecoder, self).__init__()

        self.Tconv1 = nn.ConvTranspose2d(
            pyramid_channels[-1], pyramid_channels[-2], kernel_size=2, stride=2
        )
        self.Tconv2 = nn.ConvTranspose2d(
            pyramid_channels[-2], pyramid_channels[-3], kernel_size=2, stride=2
        )
        self.Tconv3 = nn.ConvTranspose2d(
            pyramid_channels[-3], pyramid_channels[-4], kernel_size=2, stride=2
        )
        self.Tconv4 = nn.ConvTranspose2d(
            pyramid_channels[-4], pyramid_channels[-5], kernel_size=2, stride=2
        )
        self.Tconv5 = nn.ConvTranspose2d(
            pyramid_channels[-5], pyramid_channels[-6], kernel_size=2, stride=2
        )
        self.Tconv6 = nn.ConvTranspose2d(
            pyramid_channels[-6], pyramid_channels[-6], kernel_size=2, stride=2
        )

        self.Dconv1 = DoubleConv(pyramid_channels[-1], pyramid_channels[-2])
        self.Dconv2 = DoubleConv(pyramid_channels[-2], pyramid_channels[-3])
        self.Dconv3 = DoubleConv(pyramid_channels[-3], pyramid_channels[-4])
        self.Dconv4 = DoubleConv(pyramid_channels[-4], pyramid_channels[-5])
        self.Dconv5 = DoubleConv(pyramid_channels[-5] * 2, pyramid_channels[-6])

        self.final_conv = nn.Conv2d(pyramid_channels[-6], out_channels, kernel_size=1)

    def forward(self, feature_pyramid: Dict[str, torch.Tensor]):
        x = self.Tconv1(feature_pyramid["l6"])

        if (
            x.shape != feature_pyramid["l5"].shape
        ):  # shapes will not be equal when ever a specific feature level had an odd width or height when encoding
            x = TF.resize(x, size=feature_pyramid["l5"].shape[2:], antialias=True)
        x = torch.cat((feature_pyramid["l5"], x), dim=1)
        x = self.Dconv1(x)
        x = self.Tconv2(x)

        if x.shape != feature_pyramid["l4"].shape:
            x = TF.resize(x, size=feature_pyramid["l4"].shape[2:], antialias=True)
        x = torch.cat((feature_pyramid["l4"], x), dim=1)
        x = self.Dconv2(x)
        x = self.Tconv3(x)

        if x.shape != feature_pyramid["l3"].shape:
            x = TF.resize(x, size=feature_pyramid["l3"].shape[2:], antialias=True)
        x = torch.cat((feature_pyramid["l3"], x), dim=1)
        x = self.Dconv3(x)
        x = self.Tconv4(x)

        if x.shape != feature_pyramid["l2"].shape:
            x = TF.resize(x, size=feature_pyramid["l2"].shape[2:], antialias=True)
        x = torch.cat((feature_pyramid["l2"], x), dim=1)
        x = self.Dconv4(x)
        x = self.Tconv5(x)

        if x.shape != feature_pyramid["l1"].shape:
            x = TF.resize(x, size=feature_pyramid["l1"].shape[2:], antialias=True)
        x = torch.cat((feature_pyramid["l1"], x), dim=1)
        x = self.Dconv5(x)

        x = self.Tconv6(x)

        return self.final_conv(x)


class SegmentLearner(nn.Module):
    device_count = 2

    def __init__(
        self, encoder: BackBone | Dict[str, str | Dict], n_classes: int
    ) -> None:
        super().__init__()
        if type(encoder) in [dict, DictConfig]:
            encoder = make_obj_from_conf(encoder)
        else:
            assert isinstance(encoder, torch.nn.Module), type(encoder)
        self.encoder = encoder
        self.decoder = SegmentDecoder(n_classes)

    def set_devices(self, devices: Sequence[int]) -> None:
        self.devices = devices
        self.encoder.cuda(devices[0])
        self.decoder.cuda(devices[1])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tens = batch["img"]
        tens = tens.cuda(self.devices[0])
        feature_pyramid = self.encoder(tens)
        for k, v in feature_pyramid.items():
            feature_pyramid[k] = v.cuda(self.devices[1])
        seg = self.decoder(feature_pyramid)
        return {"seg": seg}

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
