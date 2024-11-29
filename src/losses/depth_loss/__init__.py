import torch
from typing import Dict
from .preprocessor import Preprocessor
from .pose_extractor import PoseExtractor
from .loss_fn import Loss
from ...util import set_device_nested_tens


class DepthLoss:
    def __init__(
        self,
        device=None,
    ) -> None:
        self.device = device
        self.pose_extractor = PoseExtractor(device)
        self.preprocessor = Preprocessor(device)
        self.loss_fn = Loss(device=device)

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.device is not None:
            info, batch = set_device_nested_tens([info, batch], self.device)

        flow_out = info["flow"]
        pose_out = self.pose_extractor(batch=batch, flow_out=flow_out)
        proc_out = self.preprocessor(
            batch=batch,
            pose_out=pose_out,
        )
        proc_out["disp1_list"] = info["depth1"]
        proc_out["disp2_list"] = info["depth2"]

        loss_pack = self.loss_fn(proc_out, batch)

        for k, v in loss_pack.items():
            if v is not None:
                loss_pack[k] = v

        return loss_pack
