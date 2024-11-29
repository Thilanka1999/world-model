"""
----------
Cost Class
----------
Initiating parameters:
        device
        weights

forward/call function:
        inputs: info, batch
        #####
        Here info is the output of the model
        Unpack and use info and batch as required in the loss function
        #####
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import Dict
import numpy as np
import logging
from .util import safe_merge

logger = logging.getLogger()


class GTDepthLoss:
    default_weights = {"l1": 1, "smo": 0.0001}

    def __init__(self, loss_weights: Dict | OmegaConf = None, device: int = 0):
        #####################
        # Do not change
        self.device = device
        #####################
        self.l1_loss = nn.L1Loss()
        if loss_weights is None:
            self.loss_weights = self.default_weights
        else:
            self.loss_weights = safe_merge(self.default_weights, loss_weights)
        self.min_depth = 1e-3
        self.max_depth = 80

    def get_smooth_loss(self, img, disp):
        # img: [b,3,h,w] depth: [b,1,h,w]
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True
        )

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return sum(grad_disp_x.mean((1, 2, 3)) + grad_disp_y.mean((1, 2, 3)))

    def process_depth(self, gt_depth, pred_depth, min_depth, max_depth):
        mask = gt_depth > 0
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        gt_depth[gt_depth < min_depth] = min_depth
        gt_depth[gt_depth > max_depth] = max_depth

        return gt_depth, pred_depth, mask

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred = info["l7"].to(self.device)
        img = batch["img"].to(self.device)
        gt = batch["depth_map"].to(self.device)

        gt_depth = gt.cpu().numpy()
        pred_depth = pred.detach().cpu().numpy()
        mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
        nyu = False

        if not nyu:
            gt_height, gt_width = gt_depth.shape[2:]
            crop = np.array(
                [
                    0.40810811 * gt_height,
                    0.99189189 * gt_height,
                    0.03594771 * gt_width,
                    0.96405229 * gt_width,
                ]
            ).astype(np.int32)
            crop_mask = np.zeros(mask.shape)  # .to(self.device)
            crop_mask[:, :, crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]

        scale = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= scale

        gt_depth, pred_depth, mask = self.process_depth(
            gt_depth, pred_depth, self.min_depth, self.max_depth
        )
        l1 = self.l1_loss(
            torch.tensor(pred_depth, requires_grad=True).to(self.device),
            torch.tensor(gt_depth, requires_grad=True).to(self.device),
        )
        # smoo = self.get_smooth_loss(img, pred_depth)
        loss = self.loss_weights["l1"] * l1  # + self.weights["smo"] * smoo

        logger.warn("TODO: smooth loss to be implemented")
        loss_pack = {"tot": loss, "L1": l1, "Smooth": torch.tensor(0).to(torch.float32)}

        return loss_pack
