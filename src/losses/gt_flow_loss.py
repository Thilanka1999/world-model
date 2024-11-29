from typing import Sequence, Dict
import torch
from torch.nn import MSELoss as TorchMSELoss
from .util import safe_merge


class GTFlowLoss:
    default_loss_weights = {"l2": 1, "smo": 1}

    def __init__(self, loss_weights, device: int = 1) -> None:
        self.device = device
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else safe_merge(self.default_loss_weights, loss_weights)
        )
        self.l2_loss = TorchMSELoss()

    def gradients(self, img):
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dx, dy

    def cal_grad2_error(self, flow, img):
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:, :, :, 1:] * torch.abs(dx2)).mean((1, 2, 3)) + (
            w_y[:, :, 1:, :] * torch.abs(dy2)
        ).mean((1, 2, 3))
        return error / 2.0

    def smooth_loss(self, pred, img):
        loss = self.cal_grad2_error(pred / 20.0, img)
        return sum(loss)

    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        predictions = info["flow_pred"].cuda(self.device)
        targets = batch["flow_map"].cuda(self.device)
        img = batch["img1"].cuda(self.device)

        smooth = self.smooth_loss(predictions, img)
        l2 = self.l2_loss(predictions, targets)
        loss = self.loss_weights["l2"] * l2 + self.loss_weights["smo"] * smooth

        loss_pack = {"tot": loss, "L1": l2, "Smooth": smooth}

        return loss_pack
