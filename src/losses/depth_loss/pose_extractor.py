import torch
import torch.nn as nn
import numpy as np
from .ransac import ReducedRansac
from typing import Dict


class PoseExtractor(nn.Module):
    def __init__(self, device=None, num_levels=6):
        self.inlier_thres = 0.1
        self.rigid_thres = 0.5
        self.filter = ReducedRansac(check_num=6000, thres=self.inlier_thres)
        self.num_levels = num_levels
        self.device = device

    def meshgrid(self, h, w):
        xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
        meshgrid = np.transpose(np.stack([xx, yy], axis=-1), [2, 0, 1])  # [2,h,w]
        meshgrid = torch.from_numpy(meshgrid)
        return meshgrid

    def compute_epipolar_loss(self, fmat, match, mask):
        # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
        num_batch = match.shape[0]
        match_num = match.shape[-1]

        points1 = match[:, :2, :]
        points2 = match[:, 2:, :]
        ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
        points1 = torch.cat([points1, ones], 1)  # [b,3,n]
        points2 = torch.cat([points2, ones], 1).transpose(1, 2)  # [b,n,3]

        # compute fundamental matrix loss
        fmat = fmat.unsqueeze(1)
        fmat_tiles = fmat.view([-1, 3, 3])
        epi_lines = fmat_tiles.bmm(points1)  # [b,3,n]  [b*n, 3, 1]
        dist_p2l = torch.abs(
            (epi_lines.permute([0, 2, 1]) * points2).sum(-1, keepdim=True)
        )  # [b,n,1]
        a = epi_lines[:, 0, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
        b = epi_lines[:, 1, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
        dist_div = torch.sqrt(a * a + b * b) + 1e-6
        dist_map = dist_p2l / dist_div  # [B, n, 1]
        loss = (dist_map * mask.transpose(1, 2)).mean([1, 2]) / mask.mean([1, 2])
        return loss, dist_map

    def get_rigid_mask(self, dist_map):
        rigid_mask = (dist_map < self.rigid_thres).float()
        inlier_mask = (dist_map < self.inlier_thres).float()
        rigid_score = rigid_mask * 1.0 / (1.0 + dist_map)
        return rigid_mask, inlier_mask, rigid_score

    def __call__(
        self, batch: Dict[str, torch.Tensor], flow_out
    ) -> Dict[str, torch.Tensor]:
        img1, img2 = batch["img1"], batch["img2"]

        img_h, img_w = img1.shape[2], img1.shape[3]
        batch_size = img1.shape[0]

        flow_fwd = flow_out["flow_fwd"]
        flow_bwd = flow_out["flow_bwd"]
        img1_valid_mask = flow_out["img1_valid_mask"]
        img2_valid_mask = flow_out["img2_valid_mask"]
        img1_flow_diff_mask = flow_out["img1_flow_diff_mask"]
        img2_flow_diff_mask = flow_out["img2_flow_diff_mask"]

        info = {
            "F_final": None,
            "flow": flow_out,
            "img1_rigid_mask": {},
            "img1_valid_mask": {},
            "geo_loss": 0,
            "fwd_match": {},
        }

        F_final = None
        for key in img1_valid_mask.keys():
            img_h, img_w = img1_valid_mask[key].shape[-2:]

            grid = (
                self.meshgrid(img_h, img_w)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
            )  # [b,2,h,w]

            if self.device is not None:
                grid = grid.to(self.device)

            fwd_corres = torch.cat(
                [
                    (grid[:, 0, :, :] + flow_fwd[key][:, 0, :, :]).unsqueeze(1),
                    (grid[:, 1, :, :] + flow_fwd[key][:, 1, :, :]).unsqueeze(1),
                ],
                1,
            )
            fwd_match = torch.cat([grid, fwd_corres], 1)  # [b,4,h,w]

            bwd_corres = torch.cat(
                [
                    (grid[:, 0, :, :] + flow_bwd[key][:, 0, :, :]).unsqueeze(1),
                    (grid[:, 1, :, :] + flow_bwd[key][:, 1, :, :]).unsqueeze(1),
                ],
                1,
            )
            bwd_match = torch.cat([grid, bwd_corres], 1)  # [b,4,h,w]

            # Use fwd-bwd consistency map for filter
            img1_score_mask = (
                img1_valid_mask[key]
                * 1.0
                / (0.1 + img1_flow_diff_mask[key].mean(1).unsqueeze(1))
            )
            img2_score_mask = (
                img2_valid_mask[key]
                * 1.0
                / (0.1 + img2_flow_diff_mask[key].mean(1).unsqueeze(1))
            )

            if key == (len(img1_valid_mask) - 1):
                F_final = self.filter(fwd_match, img1_score_mask)
                info["F_final"] = F_final

            if F_final is None:
                info["F_final"] = None
                info["img1_rigid_mask"][key] = None
                info["img1_valid_mask"][key] = img1_score_mask
                info["fwd_match"][key] = fwd_match

            else:
                _, dist_map_1 = self.compute_epipolar_loss(
                    F_final,
                    fwd_match.view([batch_size, 4, -1]),
                    img1_valid_mask[key].view([batch_size, 1, -1]),
                )
                dist_map_1 = dist_map_1.view([batch_size, img_h, img_w, 1])

                # Compute geo loss for regularize correspondence.
                rigid_mask_1, inlier_mask_1, rigid_score_1 = self.get_rigid_mask(
                    dist_map_1
                )

                # We only use rigid mask to filter out the moving objects for computing geo loss.
                geo_loss = (dist_map_1 * (rigid_mask_1 - inlier_mask_1)).mean(
                    (1, 2, 3)
                ) / (rigid_mask_1 - inlier_mask_1).mean((1, 2, 3))

                info["geo_loss"] += geo_loss
                info["img1_valid_mask"][key] = img1_score_mask
                info["img1_rigid_mask"][key] = rigid_score_1.permute(0, 3, 1, 2)
                info["fwd_match"][key] = fwd_match

        return info
