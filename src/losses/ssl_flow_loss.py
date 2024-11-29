import torch
from torch import nn
import torch.nn.functional as F
from .util import safe_merge
from ..models.flow_decoder import Warp


def SSIM(x, y):
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return SSIM


class SSLFlowLoss:
    default_loss_weights = {
        "loss_pixel": 0.15,
        "loss_ssim": 0.85,
        "loss_flow_smooth": 10,
        "loss_flow_consis": 0.01,
    }

    def __init__(self, device=None, loss_weights=None) -> None:
        self.device = device
        self.num_scales = 1
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else safe_merge(self.default_loss_weights, loss_weights)
        )
        self.warp = Warp()

    def generate_img_pyramid(self, img, num_pyramid):
        img_h, img_w = img.shape[2], img.shape[3]
        img_pyramid = []
        for s in range(num_pyramid):
            img_new = F.adaptive_avg_pool2d(
                img, [int(img_h / (2**s)), int(img_w / (2**s))]
            ).data
            img_pyramid.append(img_new)
        return img_pyramid

    def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
        img_warped_pyramid = []
        for img, flow in zip(img_pyramid, flow_pyramid):
            img_warped_pyramid.append(self.warp(img, flow))
        return img_warped_pyramid

    def compute_loss_pixel(self, img_pyramid, img_warped_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = (
                img_pyramid[scale],
                img_warped_pyramid[scale],
                occ_mask_list[scale],
            )
            divider = occ_mask.mean((1, 2, 3))
            img_diff = torch.abs((img - img_warped)) * occ_mask.repeat(1, 3, 1, 1)
            loss_pixel = img_diff.mean((1, 2, 3)) / (divider + 1e-12)  # (B)
            loss_list.append(loss_pixel[:, None])
        loss = torch.cat(loss_list, 1).sum(1)  # (B)
        return loss

    def compute_loss_ssim(self, img_pyramid, img_warped_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = (
                img_pyramid[scale],
                img_warped_pyramid[scale],
                occ_mask_list[scale],
            )
            divider = occ_mask.mean((1, 2, 3))
            occ_mask_pad = occ_mask.repeat(1, 3, 1, 1)
            ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
            loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1, 2, 3))
            loss_ssim = loss_ssim / (divider + 1e-12)
            loss_list.append(loss_ssim[:, None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

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

    def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
        loss_list = []
        for scale in range(self.num_scales):
            flow, img = optical_flows[scale], img_pyramid[scale]
            error = self.cal_grad2_error(flow / 20.0, img)
            loss_list.append(error[:, None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    # TODO: never invoked? @muditha
    def compute_loss_flow_consis(self, fwd_flow_diff_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            fwd_flow_diff, occ_mask = fwd_flow_diff_pyramid[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1, 2, 3))
            loss_consis = (fwd_flow_diff * occ_mask).mean((1, 2, 3))
            loss_consis = loss_consis / (divider + 1e-12)
            loss_list.append(loss_consis[:, None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    def __call__(self, info, batch):
        img1 = batch["img1"]
        img2 = batch["img2"]

        optical_flows_rev = info["optical_flows_rev"]
        optical_flows = info["optical_flows"]
        img1_valid_masks = info["img1_valid_masks"]
        img2_valid_masks = info["img2_valid_masks"]

        if self.device is not None:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            for lst in [
                optical_flows_rev,
                optical_flows,
                img1_valid_masks,
                img2_valid_masks,
            ]:
                for i, obj in enumerate(lst):
                    lst[i] = obj.to(self.device)

        # warp images
        img1_pyramid = self.generate_img_pyramid(img1, len(optical_flows_rev))
        img2_pyramid = self.generate_img_pyramid(img2, len(optical_flows))

        img1_warped_pyramid = self.warp_flow_pyramid(img2_pyramid, optical_flows)
        img2_warped_pyramid = self.warp_flow_pyramid(img1_pyramid, optical_flows_rev)

        loss_pack = {}

        loss_pack["loss_pixel"] = self.compute_loss_pixel(
            img1_pyramid, img1_warped_pyramid, img1_valid_masks
        ) + self.compute_loss_pixel(img2_pyramid, img2_warped_pyramid, img2_valid_masks)
        loss_pack["loss_ssim"] = self.compute_loss_ssim(
            img1_pyramid, img1_warped_pyramid, img1_valid_masks
        ) + self.compute_loss_ssim(img2_pyramid, img2_warped_pyramid, img2_valid_masks)
        loss_pack["loss_flow_smooth"] = self.compute_loss_flow_smooth(
            optical_flows, img1_pyramid
        ) + self.compute_loss_flow_smooth(optical_flows_rev, img2_pyramid)

        loss_pack["loss_flow_consis"] = (
            torch.zeros([2])
            .to(img1.get_device() if img1.is_cuda else "cpu")
            .requires_grad_()
        )

        # weight the losses
        loss_pack["loss_pixel"] = (
            loss_pack["loss_pixel"].mean() * self.loss_weights["loss_pixel"]
        )
        loss_pack["loss_ssim"] = (
            loss_pack["loss_ssim"].mean() * self.loss_weights["loss_ssim"]
        )
        loss_pack["loss_flow_smooth"] = (
            loss_pack["loss_flow_smooth"].mean() * self.loss_weights["loss_flow_smooth"]
        )
        loss_pack["loss_flow_consis"] = (
            loss_pack["loss_flow_consis"].mean() * self.loss_weights["loss_flow_consis"]
        )
        loss_pack["tot"] = sum(loss_pack.values())

        return loss_pack
