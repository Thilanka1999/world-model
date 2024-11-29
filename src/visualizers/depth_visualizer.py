import torch
from typing import Dict
from mt_pipe.src.visualizers import BaseVisualizer


class DepthVisualizer(BaseVisualizer):
    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int,
        loop: str,
    ) -> None:
        pred = info["pred"].detach().cpu()
        gt = batch["depth_map"]

        img_ten = torch.cat((pred[0].unsqueeze(0), gt[0].unsqueeze(0)), dim=0)
        self._output(img_ten, loop)
