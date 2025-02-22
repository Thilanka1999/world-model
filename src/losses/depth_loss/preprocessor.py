import torch
import numpy as np
from typing import Dict


class Preprocessor:
    def __init__(self, device=None):
        # self.depth_match_num = cfg.depth_match_num
        # self.depth_sample_ratio = cfg.depth_sample_ratio
        # self.depth_scale = cfg.scale
        # self.w_flow_error = cfg.w_flow_error
        # self.dataset = cfg.dataset
        self.depth_match_num = 6000
        self.depth_sample_ratio = 0.20
        self.depth_scale = 1
        self.w_flow_error = 0.0

        self.device = device

    def robust_rand_sample(self, match, mask, num):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, n = match.shape[0], match.shape[2]
        nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1))  # []
        if nonzeros_num.detach().cpu().numpy() == n:
            rand_int = torch.randint(0, n, [num])
            select_match = match[:, :, rand_int]
        else:
            # If there is zero score in match, sample the non-zero matches.
            num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
            select_idxs = []
            for i in range(b):
                nonzero_idx = torch.nonzero(mask[i, 0, :])  # [nonzero_num,1]
                rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
                select_idx = nonzero_idx[rand_int, :]  # [num, 1]
                select_idxs.append(select_idx)
            select_idxs = torch.stack(select_idxs, 0)  # [b,num,1]
            select_match = torch.gather(
                match.transpose(1, 2), index=select_idxs.repeat(1, 1, 4), dim=1
            ).transpose(
                1, 2
            )  # [b, 4, num]
        return select_match, num

    def top_ratio_sample(self, match, mask, ratio):
        # match: [b, 4, -1] mask: [b, 1, -1]
        b, total_num = match.shape[0], match.shape[-1]
        scores, indices = torch.topk(
            mask, int(ratio * total_num), dim=-1
        )  # [B, 1, ratio*tnum]
        select_match = torch.gather(
            match.transpose(1, 2),
            index=indices.squeeze(1).unsqueeze(-1).repeat(1, 1, 4),
            dim=1,
        ).transpose(
            1, 2
        )  # [b, 4, ratio*tnum]
        return select_match, scores

    def rand_sample(self, match, num):
        b, c, n = match.shape[0], match.shape[1], match.shape[2]
        rand_int = torch.randint(0, match.shape[-1], size=[num])
        select_pts = match[:, :, rand_int]
        return select_pts

    def filt_negative_depth(
        self, point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord
    ):
        # Filter out the negative projection depth.
        # point2d_1_depth: [b, n, 1]
        b, n = point2d_1_depth.shape[0], point2d_1_depth.shape[1]
        mask = (point2d_1_depth > 0.01).float() * (point2d_2_depth > 0.01).float()
        select_idxs = []
        flag = 0
        for i in range(b):
            if torch.sum(mask[i, :, 0]) == n:
                idx = torch.arange(n).to(mask.get_device())
            else:
                nonzero_idx = torch.nonzero(mask[i, :, 0]).squeeze(1)  # [k]
                if nonzero_idx.shape[0] < 0.1 * n:
                    idx = torch.arange(n).to(mask.get_device())
                    flag = 1
                else:
                    res = torch.randint(
                        0, nonzero_idx.shape[0], size=[n - nonzero_idx.shape[0]]
                    ).to(
                        mask.get_device()
                    )  # [n-nz]
                    idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
            select_idxs.append(idx)
        select_idxs = torch.stack(select_idxs, dim=0)  # [b,n]
        point2d_1_depth = torch.gather(
            point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1
        )  # [b,n,1]
        point2d_2_depth = torch.gather(
            point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1
        )  # [b,n,1]
        point2d_1_coord = torch.gather(
            point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1
        )  # [b,n,2]
        point2d_2_coord = torch.gather(
            point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1
        )  # [b,n,2]
        return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

    def filt_invalid_coord(
        self,
        point2d_1_depth,
        point2d_2_depth,
        point2d_1_coord,
        point2d_2_coord,
        max_h,
        max_w,
    ):
        # Filter out the negative projection depth.
        # point2d_1_depth: [b, n, 1]
        b, n = point2d_1_coord.shape[0], point2d_1_coord.shape[1]
        max_coord = torch.Tensor([max_w, max_h]).to(point2d_1_coord.get_device())
        mask = (
            (point2d_1_coord > 0).all(dim=-1, keepdim=True).float()
            * (point2d_2_coord > 0).all(dim=-1, keepdim=True).float()
            * (point2d_1_coord < max_coord).all(dim=-1, keepdim=True).float()
            * (point2d_2_coord < max_coord).all(dim=-1, keepdim=True).float()
        )

        flag = 0
        if torch.sum(1.0 - mask) == 0:
            return (
                point2d_1_depth,
                point2d_2_depth,
                point2d_1_coord,
                point2d_2_coord,
                flag,
            )

        select_idxs = []
        for i in range(b):
            if torch.sum(mask[i, :, 0]) == n:
                idx = torch.arange(n).to(mask.get_device())
            else:
                nonzero_idx = torch.nonzero(mask[i, :, 0]).squeeze(1)  # [k]
                if nonzero_idx.shape[0] < 0.1 * n:
                    idx = torch.arange(n).to(mask.get_device())
                    flag = 1
                else:
                    res = torch.randint(
                        0, nonzero_idx.shape[0], size=[n - nonzero_idx.shape[0]]
                    ).to(mask.get_device())
                    idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
            select_idxs.append(idx)
        select_idxs = torch.stack(select_idxs, dim=0)  # [b,n]
        point2d_1_depth = torch.gather(
            point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1
        )  # [b,n,1]
        point2d_2_depth = torch.gather(
            point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1
        )  # [b,n,1]
        point2d_1_coord = torch.gather(
            point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1
        )  # [b,n,2]
        point2d_2_coord = torch.gather(
            point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2), dim=1
        )  # [b,n,2]
        return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

    def ray_angle_filter(self, match, P1, P2, return_angle=False):
        # match: [b, 4, n] P: [B, 3, 4]
        b, n = match.shape[0], match.shape[2]
        K = P1[:, :, :3]  # P1 with identity rotation and zero translation
        K_inv = torch.inverse(K)
        RT1 = K_inv.bmm(P1)  # [b, 3, 4]
        RT2 = K_inv.bmm(P2)
        ones = torch.ones([b, 1, n]).to(match.get_device())
        pts1 = torch.cat([match[:, :2, :], ones], 1)
        pts2 = torch.cat([match[:, 2:, :], ones], 1)

        ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
        ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(
            RT1[:, :, 3].unsqueeze(-1)
        )  # [b, 3, 1]
        ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
        ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(
            RT2[:, :, 3].unsqueeze(-1)
        )  # [b, 3, 1]

        # We compute the angle betwwen vertical line from ray1 origin to ray2 and ray1.
        p1p2 = (ray1_origin - ray2_origin).repeat(1, 1, n)
        verline = (
            ray2_origin.repeat(1, 1, n)
            + torch.sum(p1p2 * ray2_dir, dim=1, keepdim=True) * ray2_dir
            - ray1_origin.repeat(1, 1, n)
        )  # [b,3,n]
        cosvalue = torch.sum(ray1_dir * verline, dim=1, keepdim=True) / (
            (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
            * (torch.norm(verline, dim=1, keepdim=True, p=2) + 1e-12)
        )  # [b,1,n]

        mask = (cosvalue > 0.001).float()  # we drop out angles less than 1' [b,1,n]
        flag = 0
        num = torch.min(torch.sum(mask, -1)).int()
        if num.cpu().detach().numpy() == 0:
            flag = 1
            filt_match = match[:, :, :100]
            if return_angle:
                return (
                    filt_match,
                    flag,
                    torch.zeros_like(mask).to(filt_match.get_device()),
                )
            else:
                return filt_match, flag
        nonzero_idx = []
        for i in range(b):
            idx = torch.nonzero(mask[i, 0, :])[:num]  # [num,1]
            nonzero_idx.append(idx)
        nonzero_idx = torch.stack(nonzero_idx, 0)  # [b,num,1]
        filt_match = torch.gather(
            match.transpose(1, 2), index=nonzero_idx.repeat(1, 1, 4), dim=1
        ).transpose(
            1, 2
        )  # [b,4,num]
        if return_angle:
            return filt_match, flag, mask
        else:
            return filt_match, flag

    def midpoint_triangulate(self, match, K_inv, P1, P2):
        # match: [b, 4, num] P1: [b, 3, 4]
        # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, M, 4]
        b, n = match.shape[0], match.shape[2]
        RT1 = K_inv.bmm(P1)  # [b, 3, 4]
        RT2 = K_inv.bmm(P2)
        ones = torch.ones([b, 1, n]).to(match.get_device())
        pts1 = torch.cat([match[:, :2, :], ones], 1)
        pts2 = torch.cat([match[:, 2:, :], ones], 1)

        ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
        ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(
            RT1[:, :, 3].unsqueeze(-1)
        )  # [b, 3, 1]
        ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
        ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
        ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(
            RT2[:, :, 3].unsqueeze(-1)
        )  # [b, 3, 1]

        dir_cross = torch.cross(ray1_dir, ray2_dir, dim=1)  # [b,3,n]
        denom = 1.0 / (
            torch.sum(dir_cross * dir_cross, dim=1, keepdim=True) + 1e-12
        )  # [b,1,n]
        origin_vec = (ray2_origin - ray1_origin).repeat(1, 1, n)  # [b,3,n]
        a1 = origin_vec.cross(ray2_dir, dim=1)  # [b,3,n]
        a1 = torch.sum(a1 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
        a2 = origin_vec.cross(ray1_dir, dim=1)  # [b,3,n]
        a2 = torch.sum(a2 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
        p1 = ray1_origin + a1 * ray1_dir
        p2 = ray2_origin + a2 * ray2_dir
        point = (p1 + p2) / 2.0  # [b,3,n]
        # Convert to homo coord to get consistent with other functions.
        point_homo = torch.cat([point, ones], dim=1).transpose(1, 2)  # [b,n,4]
        return point_homo

    def verifyRT(self, match, K_inv, P1, P2):
        # match: [b, 4, n] P1: [b,3,4] P2: [b,3,4]
        b, n = match.shape[0], match.shape[2]
        point3d = (
            self.midpoint_triangulate(match, K_inv, P1, P2)
            .reshape([-1, 4])
            .unsqueeze(-1)
        )  # [b*n, 4, 1]
        P1_ = P1.repeat(n, 1, 1)
        P2_ = P2.repeat(n, 1, 1)
        depth1 = P1_.bmm(point3d)[:, -1, :] / point3d[:, -1, :]  # [b*n, 1]
        depth2 = P2_.bmm(point3d)[:, -1, :] / point3d[:, -1, :]
        inlier_num = torch.sum(
            (depth1.view([b, n]) > 0).float() * (depth2.view([b, n]) > 0).float(), 1
        )  # [b]
        return inlier_num

    def rt_from_fundamental_mat(self, fmat, K, depth_match):
        # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
        verify_match = self.rand_sample(depth_match, 200)  # [b,4,100]
        K_inv = torch.inverse(K)
        b = fmat.shape[0]
        fmat_ = K.transpose(1, 2).bmm(fmat)
        essential_mat = fmat_.bmm(K)
        essential_mat_cpu = essential_mat.cpu()
        U, S, V = torch.svd(essential_mat_cpu)
        U, S, V = U.to(K.get_device()), S.to(K.get_device()), V.to(K.get_device())
        W = (
            torch.from_numpy(
                np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
            )
            .float()
            .repeat(b, 1, 1)
            .to(K.get_device())
        )
        # R = UWV^t or UW^tV^t t = U[:,2] the third column of U
        R1 = U.bmm(W).bmm(V.transpose(1, 2))  # Do we need matrix determinant sign?
        R1 = torch.sign(torch.det(R1)).unsqueeze(-1).unsqueeze(-1) * R1
        R2 = U.bmm(W.transpose(1, 2)).bmm(V.transpose(1, 2))
        R2 = torch.sign(torch.det(R2)).unsqueeze(-1).unsqueeze(-1) * R2
        t1 = U[:, :, 2].unsqueeze(-1)  # The third column
        t2 = -U[:, :, 2].unsqueeze(-1)  # Inverse direction

        iden = (
            torch.cat([torch.eye(3), torch.zeros([3, 1])], -1)
            .unsqueeze(0)
            .repeat(b, 1, 1)
            .to(K.get_device())
        )  # [b,3,4]
        P1 = K.bmm(iden)
        P2_1 = K.bmm(torch.cat([R1, t1], -1))
        P2_2 = K.bmm(torch.cat([R2, t1], -1))
        P2_3 = K.bmm(torch.cat([R1, t2], -1))
        P2_4 = K.bmm(torch.cat([R2, t2], -1))
        P2_c = [P2_1, P2_2, P2_3, P2_4]
        flags = []
        for i in range(4):
            with torch.no_grad():
                inlier_num = self.verifyRT(verify_match, K_inv, P1, P2_c[i])
                flags.append(inlier_num)
        P2_c = torch.stack(P2_c, dim=1)  # [B, 4, 3, 4]
        flags = torch.stack(flags, dim=1)  # [B, 4]
        idx = torch.argmax(flags, dim=-1, keepdim=True)  # [b,1]
        P2 = torch.gather(
            P2_c, index=idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 4), dim=1
        ).squeeze(
            1
        )  # [b,3,4]
        # pdb.set_trace()
        return P1, P2

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

    def __call__(self, batch, pose_out) -> Dict[str, torch.Tensor]:
        # initialization
        batch_one, batch_two, K_ms, K_inv_ms = (
            batch["img1"],
            batch["img2"],
            batch["K"],
            batch["K_inv"],
        )
        if len(batch_one.shape) == 3:
            raise IOError("This method only supports batch processing")

        K, K_inv = K_ms[:, 0, :, :], K_inv_ms[:, 0, :, :]
        assert batch_one.shape[1] == 3
        img_h, img_w = batch_one.shape[2], batch_one.shape[3]
        b = batch_one.shape[0]
        flag1, flag2, flag3 = 0, 0, 0

        F_final = pose_out["F_final"]
        flow_info = pose_out["flow"]
        geo_loss = pose_out["geo_loss"]  # TODO : What is the geo_loss?
        img1_valid_mask = pose_out["img1_valid_mask"]
        img1_rigid_mask = pose_out["img1_rigid_mask"]
        fwd_match = pose_out["fwd_match"]

        if F_final is None:
            return {
                "abort_early": True,
                "point3d_1": None,
                "points2d_list": None,
                "P2": None,
                "img1_valid_mask": None,
                "img1_rigid_mask": None,
                "flow_fwd": None,
            }
        else:
            # Get masks
            img1_depth_mask = img1_rigid_mask[0] * img1_valid_mask[0]

            # Select top score matches to triangulate depth.
            top_ratio_match, top_ratio_mask = self.top_ratio_sample(
                fwd_match[0].view([b, 4, -1]),
                img1_depth_mask.view([b, 1, -1]),
                ratio=self.depth_sample_ratio,
            )  # [b, 4, ratio*h*w]
            depth_match, depth_match_num = self.robust_rand_sample(
                top_ratio_match, top_ratio_mask, num=self.depth_match_num
            )

            K = K.to(depth_match)
            P1, P2 = self.rt_from_fundamental_mat(F_final.detach(), K, depth_match)
            P1 = P1.detach()
            P2 = P2.detach()

            # Get triangulated points
            filt_depth_match, flag1 = self.ray_angle_filter(
                depth_match, P1, P2, return_angle=False
            )  # [b, 4, filt_num]

            K_inv = K_inv.to(filt_depth_match)
            point3d_1 = self.midpoint_triangulate(filt_depth_match, K_inv, P1, P2)
            point2d_1_coord, point2d_1_depth = self.reproject(
                P1, point3d_1
            )  # [b,n,2], [b,n,1]
            point2d_2_coord, point2d_2_depth = self.reproject(P2, point3d_1)

            # Filter out some invalid triangulation results to stablize training.
            (
                point2d_1_depth,
                point2d_2_depth,
                point2d_1_coord,
                point2d_2_coord,
                flag2,
            ) = self.filt_negative_depth(
                point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord
            )
            (
                point2d_1_depth,
                point2d_2_depth,
                point2d_1_coord,
                point2d_2_coord,
                flag3,
            ) = self.filt_invalid_coord(
                point2d_1_depth,
                point2d_2_depth,
                point2d_1_coord,
                point2d_2_coord,
                max_h=img_h,
                max_w=img_w,
            )

            points2d_list = [
                point2d_1_coord,
                point2d_1_depth,
                point2d_2_coord,
                point2d_2_depth,
            ]

            return {
                "abort_early": flag1 + flag2 + flag3 > 0,
                "point3d_1": point3d_1,
                "points2d_list": points2d_list,
                "P2": P2,
                "img1_valid_mask": img1_valid_mask,
                "img1_rigid_mask": img1_rigid_mask,
                "flow_fwd": flow_info["flow_fwd"],
            }
