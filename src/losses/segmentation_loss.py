from typing import Sequence, Dict
import torch
import torch
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss
from .util import safe_merge


class SegmentationLoss:
    default_loss_weights = {"dice": 1, "bce": 1}

    def __init__(self, device: int = 1, loss_weights=None) -> None:
        self.device = device
        self.loss_fn_dice = DiceLoss(mode="multilabel")
        self.loss_fn_bce = nn.BCEWithLogitsLoss()  # TODO: from where?
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else safe_merge(self.default_loss_weights, loss_weights)
        )

    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        segments = batch["seg"]
        logits = info["seg"]

        segments = segments.cuda(self.device)
        logits = logits.cuda(self.device)
        loss_dice = self.loss_fn_dice(logits, segments) * self.loss_weights["dice"]
        loss_bce = self.loss_fn_bce(logits, segments) * self.loss_weights["bce"]
        loss = loss_dice + loss_bce

        return {"tot": loss, "Dice": loss_dice, "BCEWithLogits": loss_bce}
