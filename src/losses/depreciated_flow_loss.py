import torch.nn.functional as F
import torch
from pytorch_msssim import ssim
from .vicreg_loss import VICRegLoss
from ..models.flow_decoder.warp import Warp
from typing import Dict
import omegaconf
from .util import safe_merge


class FlowLoss:
    default_loss_weights = omegaconf.OmegaConf.create(
        {
            "regression_loss": 1,
            "reconstruction_loss": 1,
            "reconstruction_loss_coeffs": [1, 1, 1],
            "cycle_loss": 0.2,
            "smooth_loss": 75,
            "vc_loss": 0,
            "vc_loss_coeffs": {
                "l1": [0.01, 0.04],
                "l2": [0.01, 0.04],
                "l3": [0.01, 0.001],
                "l4": [0.01, 0],
                "l5": [0.001, 0],
                "l6": [0.0001, 0],
            },
            "gt_loss": 0,
        }
    )

    def __init__(
        self,
        loss_weights: omegaconf.dictconfig.DictConfig = None,
        device=1,
        num_levels: int = 6,
    ):
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else safe_merge(self.default_loss_weights, loss_weights)
        )
        self.var_cov_loss_fns = {
            k: VICRegLoss(
                lambda_param=0,
                mu_param=v[0],
                nu_param=v[1],
            )
            for (k, v) in self.loss_weights["vc_loss_coeffs"].items()
        }
        self.warp = Warp()
        self.device = device
        self.num_levels = num_levels
        self.scale_2_level_map = {i: num_levels + 1 - i for i in range(num_levels - 2)}
        self.pyr_keys = None

    def regression_loss(
        self,
        feat_pyramid: Dict[str, torch.Tensor],
        warped_feat_pyramid: Dict[str, torch.Tensor],
        occ_mask_pyramid: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss_pack = {}
        loss_list = []
        for key in self.pyr_keys:
            feat = feat_pyramid[key]
            feat_warped = warped_feat_pyramid[key]
            occ_mask = occ_mask_pyramid[key]
            divider = occ_mask.mean((1, 2, 3))
            img_diff = torch.abs((feat - feat_warped)) * occ_mask.repeat(
                1, feat.shape[1], 1, 1
            )
            loss_pixel = img_diff.mean((1, 2, 3)) / (divider + 1e-12)  # (B)
            loss_pack[key] = loss_pixel.sum()
            loss_list.append(loss_pixel)

        loss = torch.cat(loss_list).sum()
        loss_pack["tot"] = loss
        return loss_pack

    def recons_loss(
        self, im1: torch.Tensor, im2: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        l1_wei, l2_wei, ssim_wei = self.loss_weights["reconstruction_loss_coeffs"]
        # l1 = F.l1_loss(im1, im2, valid_mask)
        l1 = self.compute_loss_pixel(im1, im2, valid_mask)
        # l2 = F.mse_loss(im1, im2, valid_mask)
        l2 = 0
        # ssim_loss = 1 - ssim(im1, im2, valid_mask, data_range=255, size_average=True)
        ssim_loss = self.compute_loss_ssim(im1, im2, valid_mask)
        recons_loss = l1_wei * l1 + l2_wei * l2 + ssim_wei * ssim_loss

        return {"tot": recons_loss, "l1": l1, "l2": l2, "ssim": ssim_loss}

    def cycle_loss(
        self,
        feature_pyramid: torch.Tensor,
        rev_flow: torch.Tensor,
        warp_feature_pyramid: torch.Tensor,
    ) -> torch.Tensor:
        c_loss = 0.0
        loss_pack = {}
        for k in self.pyr_keys:
            feature_cyc = self.warp(
                warp_feature_pyramid[k],
                rev_flow[k],
            )
            loss = F.mse_loss(feature_pyramid[k], feature_cyc)
            loss_pack[k] = loss
            c_loss += loss
        loss_pack["tot"] = c_loss
        return loss_pack

    def gradients(self, img):
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        return dx, dy

    def smooth_loss(self, flow: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:, :, :, 1:] * torch.abs(dx2)).mean() + (
            w_y[:, :, 1:, :] * torch.abs(dy2)
        ).mean()

        return error / 2.0

    def compute_loss_pixel(self, im1, im2, valid_mask):
        divider = valid_mask.mean((1, 2, 3))
        img_diff = torch.abs((im1 - im2)) * valid_mask.repeat(1, 3, 1, 1)
        loss_pixel = img_diff.mean((1, 2, 3)) / (divider + 1e-12)  # (B)
        return loss_pixel.sum()

    def compute_loss_ssim(self, im1, im2, valid_mask):
        divider = valid_mask.mean((1, 2, 3))
        occ_mask_pad = valid_mask.repeat(1, 3, 1, 1)
        ssim_loss = ssim(im1 * occ_mask_pad, im2 * occ_mask_pad)
        loss_ssim = torch.clamp((1.0 - ssim_loss) / 2.0, 0, 1)
        loss_ssim = loss_ssim / (divider + 1e-12)
        return loss_ssim.sum()

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feature_pyramid_one = info["feature_pyramid_one"]
        feature_pyramid_two = info["feature_pyramid_two"]
        warp_feature_pyramid_fwd = info["warp_feature_pyramid_fwd"]
        warp_feature_pyramid_bwd = info["warp_feature_pyramid_bwd"]
        flow_fwd = info["flow_fwd"]
        flow_bwd = info["flow_bwd"]
        img1_valid_mask = info["img1_valid_mask"]

        feature_pyramid_one.pop("emb")
        feature_pyramid_two.pop("emb")

        if self.pyr_keys is None:
            keys1 = feature_pyramid_one.keys()
            keys2 = flow_fwd.keys()
            self.pyr_keys = list(set(keys1) - (set(keys1) - set(keys2)))

        # batch to loss device
        img1, img2 = batch["img1"], batch["img2"]
        (img1, img2) = (img1.cuda(self.device), img2.cuda(self.device))
        for k, v in info.items():
            for i, li in v.items():
                if li is not None:
                    info[k][i] = li.cuda(self.device)

        # info to loss device
        for dct in [
            feature_pyramid_one,
            feature_pyramid_two,
            warp_feature_pyramid_fwd,
            warp_feature_pyramid_bwd,
            flow_fwd,
            flow_bwd,
            img1_valid_mask,
        ]:
            for k, tens in dct.items():
                dct[k] = None if tens is None else tens.cuda(self.device)

        total_loss = 0
        recons_loss = None
        cyc_loss = None
        smooth_loss = None
        regression_loss = None
        vc_loss = None

        # Regression loss
        if self.loss_weights["regression_loss"] != 0:
            regression_loss = self.regression_loss(
                feature_pyramid_two, warp_feature_pyramid_fwd, img1_valid_mask
            )
            reg_loss = regression_loss["tot"] * self.loss_weights["regression_loss"]
            total_loss += reg_loss

        # Reconstruction loss
        if self.loss_weights["reconstruction_loss"] != 0:
            recons_loss = self.recons_loss(
                img2, warp_feature_pyramid_fwd["l7"], img1_valid_mask["l7"]
            )
            total_loss += recons_loss["tot"] * self.loss_weights["reconstruction_loss"]

        # Cycle loss
        if self.loss_weights["cycle_loss"] != 0:
            fwd_cyc_loss = self.cycle_loss(
                feature_pyramid_one,
                flow_bwd,
                warp_feature_pyramid_fwd,
            )
            bwd_cyc_loss = self.cycle_loss(
                feature_pyramid_two,
                flow_fwd,
                warp_feature_pyramid_bwd,
            )
            cyc_loss_tot = (
                fwd_cyc_loss["tot"] + bwd_cyc_loss["tot"]
            ) * self.loss_weights["cycle_loss"]
            total_loss += cyc_loss_tot
            cyc_loss = {"tot": cyc_loss_tot, "fwd": fwd_cyc_loss, "bwd": bwd_cyc_loss}

        # Smooth loss
        if self.loss_weights["smooth_loss"] != 0:
            smooth_loss = (
                self.smooth_loss(flow_fwd["l7"], img1)
                + self.smooth_loss(flow_bwd["l7"], img2)
            ) * self.loss_weights["smooth_loss"]
            total_loss += smooth_loss

        # VC Reg. loss
        if self.loss_weights["vc_loss"] != 0:
            vc_loss_pack = {}
            vc_loss = 0
            for k in feature_pyramid_one.keys():
                if k != "emb":
                    emb1, emb2 = feature_pyramid_one[k], feature_pyramid_two[k]
                    vc_level_loss = self.var_cov_loss_fns[k](emb1, emb2)
                    vc_loss += vc_level_loss["tot"]
                    vc_loss_pack[k] = vc_level_loss
            vc_loss = vc_loss * self.loss_weights["vc_loss"]
            total_loss += vc_loss
            vc_loss_pack["tot"] = vc_loss

        return {
            "tot": total_loss,
            "Flow/VC": vc_loss_pack,
            "Flow/Reconstruction": recons_loss,
            "Flow/Cycle": cyc_loss,
            "Flow/Smooth": smooth_loss,
            "Flow/Regression": regression_loss,
        }
