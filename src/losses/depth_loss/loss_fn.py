import torch
import torch.nn.functional as F
import numpy as np
import omegaconf
import copy
import logging
from typing import Dict

logger = logging.getLogger()


class Loss:
    default_loss_weights = omegaconf.OmegaConf.create(
        {
            "pt_depth_loss": 1,
            "pj_depth_loss": 1,
            "flow_loss": 1,
            "depth_smooth_loss": 1,
        }
    )

    def __init__(
        self,
        loss_weights: omegaconf.dictconfig.DictConfig = None,
        depth_scale=1,
        device=1,
        weight: float = 1.0,
    ) -> None:
        self.loss_weights = (
            self.default_loss_weights
            if loss_weights is None
            else omegaconf.OmegaConf.merge(self.default_loss_weights, loss_weights)
        )
        self.depth_scale = depth_scale
        self.device = device
        self.weight = weight

    def disp2depth(self, disp, min_depth=0.1, max_depth=100.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def scale_adapt(self, depth1, depth2, eps=1e-12):
        with torch.no_grad():
            A = torch.sum((depth1**2) / (depth2**2 + eps), dim=1)  # [b,1]
            C = torch.sum(depth1 / (depth2 + eps), dim=1)  # [b,1]
            a = C / (A + eps)
        return a

    def affine_adapt(self, depth1, depth2, use_translation=True, eps=1e-12):
        a_scale = self.scale_adapt(depth1, depth2, eps=eps)
        if not use_translation:  # only fit the scale parameter
            return a_scale, torch.zeros_like(a_scale)
        else:
            with torch.no_grad():
                A = torch.sum((depth1**2) / (depth2**2 + eps), dim=1)  # [b,1]
                B = torch.sum(depth1 / (depth2**2 + eps), dim=1)  # [b,1]
                C = torch.sum(depth1 / (depth2 + eps), dim=1)  # [b,1]
                D = torch.sum(1.0 / (depth2**2 + eps), dim=1)  # [b,1]
                E = torch.sum(1.0 / (depth2 + eps), dim=1)  # [b,1]
                a = (B * E - D * C) / (B * B - A * D + 1e-12)
                b = (B * C - A * E) / (B * B - A * D + 1e-12)

                # check ill condition
                cond = B * B - A * D
                valid = (torch.abs(cond) > 1e-4).float()
                a = a * valid + a_scale * (1 - valid)
                b = b * valid
            return a, b

    def register_depth(self, depth_pred, coord_tri, depth_tri):
        # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
        batch, _, h, w = (
            depth_pred.shape[0],
            depth_pred.shape[1],
            depth_pred.shape[2],
            depth_pred.shape[3],
        )
        n = depth_tri.shape[1]
        coord_tri_nor = torch.stack(
            [
                2.0 * coord_tri[:, :, 0] / (w - 1.0) - 1.0,
                2.0 * coord_tri[:, :, 1] / (h - 1.0) - 1.0,
            ],
            -1,
        )
        depth_inter = (
            F.grid_sample(
                depth_pred,
                coord_tri_nor.view([batch, n, 1, 2]),
                padding_mode="reflection",
                align_corners=True,
            )
            .squeeze(-1)
            .transpose(1, 2)
        )  # [b,n,1]

        # Normalize
        scale = torch.median(depth_inter, 1)[0] / (
            torch.median(depth_tri, 1)[0] + 1e-12
        )
        scale = scale.detach()  # [b,1]
        scale_depth_inter = depth_inter / (scale.unsqueeze(-1) + 1e-12)
        scale_depth_pred = depth_pred / (scale.unsqueeze(-1).unsqueeze(-1) + 1e-12)

        # affine adapt
        a, b = self.affine_adapt(scale_depth_inter, depth_tri, use_translation=False)
        affine_depth_inter = a.unsqueeze(1) * scale_depth_inter + b.unsqueeze(
            1
        )  # [b,n,1]
        affine_depth_pred = a.unsqueeze(-1).unsqueeze(
            -1
        ) * scale_depth_pred + b.unsqueeze(-1).unsqueeze(
            -1
        )  # [b,1,h,w]
        return affine_depth_pred, affine_depth_inter

    def get_trian_loss(self, tri_depth, pred_tri_depth):
        # depth: [b,n,1]
        loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean((1, 2))
        return loss

    def reproject(self, P, point3d):
        # P: [b,3,4] point3d: [b,n,4]
        point2d = P.bmm(point3d.transpose(1, 2))  # [b,4,n]
        point2d_coord = (
            point2d[:, :2, :] / (point2d[:, 2, :].unsqueeze(1) + 1e-12)
        ).transpose(
            1, 2
        )  # [b,n,2]
        point2d_depth = point2d[:, 2, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
        return point2d_coord, point2d_depth

    def get_reproj_fdp_loss(
        self, pred1, pred2, P2, K, K_inv, valid_mask, rigid_mask, flow
    ):
        # pred: [b,1,h,w] Rt: [b,3,4] K: [b,3,3] mask: [b,1,h,w] flow: [b,2,h,w]
        b, h, w = pred1.shape[0], pred1.shape[2], pred1.shape[3]
        xy = (
            self.meshgrid(h, w)
            .unsqueeze(0)
            .repeat(b, 1, 1, 1)
            .float()
            .to(flow.get_device())
        )  # [b,2,h,w]
        ones = torch.ones([b, 1, h, w]).float().to(flow.get_device())
        pts1_3d = K_inv.bmm(torch.cat([xy, ones], 1).view([b, 3, -1])) * pred1.view(
            [b, 1, -1]
        )  # [b,3,h*w]
        pts2_coord, pts2_depth = self.reproject(
            P2, torch.cat([pts1_3d, ones.view([b, 1, -1])], 1).transpose(1, 2)
        )  # [b,h*w, 2]
        # TODO Here some of the reprojection coordinates are invalid. (<0 or >max)
        reproj_valid_mask = (
            pts2_coord > torch.Tensor([0, 0]).to(pred1.get_device())
        ).all(-1, True).float() * (
            pts2_coord < torch.Tensor([w - 1, h - 1]).to(pred1.get_device())
        ).all(
            -1, True
        ).float()  # [b,h*w, 1]
        reproj_valid_mask = (
            valid_mask * reproj_valid_mask.view([b, h, w, 1]).permute([0, 3, 1, 2])
        ).detach()
        rigid_mask = rigid_mask.detach()
        pts2_depth = pts2_depth.transpose(1, 2).view([b, 1, h, w])

        # Get the interpolated depth prediction2
        pts2_coord_nor = torch.cat(
            [
                2.0 * pts2_coord[:, :, 0].unsqueeze(-1) / (w - 1.0) - 1.0,
                2.0 * pts2_coord[:, :, 1].unsqueeze(-1) / (h - 1.0) - 1.0,
            ],
            -1,
        )
        inter_depth2 = F.grid_sample(
            pred2,
            pts2_coord_nor.view([b, h, w, 2]),
            padding_mode="reflection",
            align_corners=True,
        )  # [b,1,h,w]
        pj_loss_map = (
            torch.abs(1.0 - pts2_depth / (inter_depth2 + 1e-12))
            * rigid_mask
            * reproj_valid_mask
        )
        pj_loss = pj_loss_map.mean((1, 2, 3)) / (
            (reproj_valid_mask * rigid_mask).mean((1, 2, 3)) + 1e-12
        )
        # pj_loss = (valid_mask * mask * torch.abs(pts2_depth - inter_depth2) / (torch.abs(pts2_depth + inter_depth2)+1e-12)).mean((1,2,3)) / ((valid_mask * mask).mean((1,2,3))+1e-12) # [b]
        flow_loss = (
            rigid_mask
            * torch.abs(
                flow + xy - pts2_coord.detach().permute(0, 2, 1).view([b, 2, h, w])
            )
        ).mean((1, 2, 3)) / (rigid_mask.mean((1, 2, 3)) + 1e-12)
        return pj_loss, flow_loss

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

        return grad_disp_x.mean((1, 2, 3)) + grad_disp_y.mean((1, 2, 3))

    def meshgrid(self, h, w):
        xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
        meshgrid = np.transpose(np.stack([xx, yy], axis=-1), [2, 0, 1])  # [2,h,w]
        meshgrid = torch.from_numpy(meshgrid)
        return meshgrid

    def get_intrinsics_per_scale(self, K, scale):
        K_new = copy.deepcopy(K)
        K_new[0, :] = K_new[0, :] / (2**scale)
        K_new[1, :] = K_new[1, :] / (2**scale)
        K_new_inv = torch.tensor(np.linalg.inv(K_new.cpu().numpy())).to(K_new)
        return K_new, K_new_inv

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_pack = {
            "tot": torch.tensor(0.0),
            "pt_depth_loss": None,
            "pj_depth_loss": None,
            "flow_loss": None,
            "depth_smooth_loss": None,
        }

        if (
            (self.loss_weights["pt_depth_loss"] != 0)
            and (self.loss_weights["pj_depth_loss"] != 0)
            and (self.loss_weights["flow_loss"] != 0)
            and (self.loss_weights["depth_smooth_loss"] != 0)
            and all(
                [
                    k in info
                    for k in [
                        "abort_early",
                        "point3d_1",
                        "disp1_list",
                        "disp2_list",
                        "points2d_list",
                        "P2",
                        "img1_valid_mask",
                        "img1_rigid_mask",
                        "flow_fwd",
                    ]
                ]
            )
        ):
            abort_early = info["abort_early"]
            point3d_1 = info["point3d_1"]
            disp1_list = info["disp1_list"]
            disp2_list = info["disp2_list"]
            points2d_list = info["points2d_list"]
            P2 = info["P2"]
            img1_valid_mask = info["img1_valid_mask"]
            img1_rigid_mask = info["img1_rigid_mask"]
            flow_fwd = info["flow_fwd"]

            if abort_early:
                logger.warn("Depth loss not calculated")
                return loss_pack

            batch_one, batch_two, K_ms, K_inv_ms = batch
            K, K_inv = K_ms[:, 0, :, :], K_inv_ms[:, 0, :, :]
            img_hw = (batch_one.shape[2], batch_one.shape[3])

            # batch to loss device
            batch_one = batch_one.cuda(self.device)
            batch_two = batch_two.cuda(self.device)
            K = K.cuda(self.device)
            K_inv = K_inv.cuda(self.device)

            # info to loss device
            point3d_1 = point3d_1.cuda(self.device)
            dicts = (disp1_list, disp2_list)
            for s, dic in enumerate(dicts):
                for k, v in dic.items():
                    dicts[s][k] = v.cuda(self.device)
            for s, pt in enumerate(points2d_list):
                points2d_list[s] = pt.cuda(self.device)
            P2 = P2.cuda(self.device)
            for s in img1_valid_mask.keys():
                img1_valid_mask[s] = img1_valid_mask[s].cuda(self.device)
                img1_rigid_mask[s] = img1_rigid_mask[s].cuda(self.device)
                flow_fwd[s] = flow_fwd[s].cuda(self.device)

            (
                point2d_1_coord,
                point2d_1_depth,
                point2d_2_coord,
                point2d_2_depth,
            ) = points2d_list

            pt_depth_loss = 0
            pj_depth_loss = 0
            flow_loss = 0
            depth_smooth_loss = 0

            for s in range(6):
                # pre-calculate
                disp_pred1 = F.interpolate(
                    disp1_list[5 - s], size=img_hw, mode="bilinear"
                )  # [b,1,h,w]
                disp_pred2 = F.interpolate(
                    disp2_list[5 - s], size=img_hw, mode="bilinear"
                )
                scaled_disp1, depth_pred1 = self.disp2depth(disp_pred1)
                scaled_disp2, depth_pred2 = self.disp2depth(disp_pred2)
                # Rescale predicted depth according to triangulated depth
                # [b,1,h,w], [b,n,1]
                rescaled_pred1, inter_pred1 = self.register_depth(
                    depth_pred1, point2d_1_coord, point2d_1_depth
                )
                rescaled_pred2, inter_pred2 = self.register_depth(
                    depth_pred2, point2d_2_coord, point2d_2_depth
                )

                # Get Losses
                if self.loss_weights["pt_depth_loss"] != 0:
                    pt_depth_loss += self.get_trian_loss(
                        point2d_1_depth, inter_pred1
                    ) + self.get_trian_loss(point2d_2_depth, inter_pred2)
                if (
                    self.loss_weights["pj_depth_loss"] != 0
                    or self.loss_weights["flow_loss"] != 0
                ):
                    pj_depth, flow_loss = self.get_reproj_fdp_loss(
                        rescaled_pred1,
                        rescaled_pred2,
                        P2,
                        K,
                        K_inv,
                        img1_valid_mask[0],
                        img1_rigid_mask[0],
                        flow_fwd[0],
                    )
                    pj_depth_loss += pj_depth
                    flow_loss += flow_loss
                if self.loss_weights["depth_smooth_loss"] != 0:
                    depth_smooth_loss += self.get_smooth_loss(
                        batch_one, disp_pred1 / (disp_pred1.mean((2, 3), True) + 1e-12)
                    ) + self.get_smooth_loss(
                        batch_two, disp_pred2 / (disp_pred2.mean((2, 3), True) + 1e-12)
                    )

            # pt loss
            if self.loss_weights["pt_depth_loss"] != 0:
                pt_depth_loss = (
                    pt_depth_loss.sum()
                    * self.loss_weights["pt_depth_loss"]
                    * self.weight
                )
                loss_pack["tot"] += pt_depth_loss
                loss_pack["pt_depth_loss"] = pt_depth_loss.detach().cpu().item()

            # pj loss
            if self.loss_weights["pj_depth_loss"] != 0:
                pj_depth_loss = (
                    pj_depth_loss.sum()
                    * self.loss_weights["pj_depth_loss"]
                    * self.weight
                )
                loss_pack["tot"] += pj_depth_loss
                loss_pack["pj_depth_loss"] = pj_depth_loss.detach().cpu().item()

            # flow loss
            if self.loss_weights["flow_loss"] != 0:
                flow_loss = (
                    flow_loss.sum() * self.loss_weights["flow_loss"] * self.weight
                )
                loss_pack["tot"] += flow_loss
                loss_pack["flow_loss"] = flow_loss.detach().cpu().item()

            # smooth loss
            if self.loss_weights["depth_smooth_loss"] != 0:
                depth_smooth_loss = (
                    depth_smooth_loss.sum()
                    * self.loss_weights["depth_smooth_loss"]
                    * self.weight
                )
                loss_pack["tot"] += depth_smooth_loss
                loss_pack["depth_smooth_loss"] = depth_smooth_loss.detach().cpu().item()

        return loss_pack
