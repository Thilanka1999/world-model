from .vicreg_loss import VICRegLoss
import omegaconf
import torch
from typing import Dict
from .util import safe_merge


class ContentLoss:
    default_loss_weights = omegaconf.OmegaConf.create(
        {
            "vc_loss_X": [0.01, 0.04],
            "vic_loss_Y": [25, 1, 1],
        }
    )
    hparams = {
        "variance_loss_epsilon": 0.0001,
    }

    def __init__(
        self,
        loss_weights: omegaconf.dictconfig.DictConfig = None,
        device=1,
    ) -> None:
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else safe_merge(self.default_loss_weights, loss_weights)
        )
        self.loss_X = VICRegLoss(
            lambda_param=0,
            mu_param=self.loss_weights["vc_loss_X"][0],
            nu_param=self.loss_weights["vc_loss_X"][1],
            eps=self.hparams["variance_loss_epsilon"],
        )
        self.loss_Y = VICRegLoss(
            lambda_param=self.loss_weights["vic_loss_Y"][1],
            mu_param=self.loss_weights["vic_loss_Y"][0],
            nu_param=self.loss_weights["vic_loss_Y"][2],
            eps=self.hparams["variance_loss_epsilon"],
        )
        self.device = device

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for k, v in info.items():
            info[k] = v.cuda(self.device)
        (
            X_one,
            X_two,
            Y_one,
            Y_two,
        ) = (
            info["X_one"],
            info["X_two"],
            info["Y_one"],
            info["Y_two"],
        )

        pack_loss_X = self.loss_X(X_one, X_two)
        pack_loss_Y = self.loss_Y(Y_one, Y_two)

        loss_X = pack_loss_X["tot"]
        loss_Y = pack_loss_Y["tot"]
        total_loss = loss_X + loss_Y

        return {"tot": total_loss, "Content_X": pack_loss_X, "Content_Y": pack_loss_Y}
