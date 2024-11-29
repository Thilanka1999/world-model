from typing import Any
import torch
import numpy as np
from .warp import Warp


def transformerFwd(U, flo, out_size, name="SpatialTransformerFwd"):
    """Forward Warping Layer described in
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = (
            torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1, 0)
        )
        x = x.view([-1, 1]).mm(rep)
        return x.view([-1]).int()

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = (
            im.shape[0],
            im.shape[1],
            im.shape[2],
            im.shape[3],
        )
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width - 1.0) / 2.0
        y = (y + 1.0) * (height - 1.0) / 2.0

        # do sampling
        x0 = (torch.floor(x)).int()
        x1 = x0 + 1
        y0 = (torch.floor(y)).int()
        y1 = y0 + 1

        x0_c = torch.clamp(x0, 0, max_x)
        x1_c = torch.clamp(x1, 0, max_x)
        y0_c = torch.clamp(y0, 0, max_y)
        y1_c = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)
        if im.is_cuda:
            base = base.to(im.get_device())

        base_y0 = base + y0_c * dim2
        base_y1 = base + y1_c * dim2
        idx_a = base_y0 + x0_c
        idx_b = base_y1 + x0_c
        idx_c = base_y0 + x1_c
        idx_d = base_y1 + x1_c

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.view([-1, channels])
        im_flat = im_flat.float()

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        zerof = torch.zeros_like(wa)
        wa = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof
        )
        wb = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof
        )
        wc = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof
        )
        wd = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof
        )

        zeros = torch.zeros(
            size=[int(num_batch) * int(height) * int(width), int(channels)],
            dtype=torch.float,
        )
        if im.is_cuda:
            output = zeros.to(im.get_device())
        else:
            output = zeros
        output = output.scatter_add(
            dim=0, index=idx_a.long().unsqueeze(1).repeat(1, channels), src=im_flat * wa
        )
        output = output.scatter_add(
            dim=0, index=idx_b.long().unsqueeze(1).repeat(1, channels), src=im_flat * wb
        )
        output = output.scatter_add(
            dim=0, index=idx_c.long().unsqueeze(1).repeat(1, channels), src=im_flat * wc
        )
        output = output.scatter_add(
            dim=0, index=idx_d.long().unsqueeze(1).repeat(1, channels), src=im_flat * wd
        )

        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()

    def _transform(flo, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape[0:4]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        x_s, y_s = _meshgrid(out_height, out_width)
        x_s = x_s.unsqueeze(0)
        x_s = x_s.repeat([num_batch, 1, 1])

        y_s = y_s.unsqueeze(0)
        y_s = y_s.repeat([num_batch, 1, 1])

        if flo.is_cuda:
            x_s = x_s.to(flo.get_device())
            y_s = y_s.to(flo.get_device())

        x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
        y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

        x_t_flat = x_t.view([-1])
        y_t_flat = y_t.view([-1])

        input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat, out_size)

        output = input_transformed.view(
            [num_batch, out_height, out_width, num_channels]
        )
        return output

    # out_size = int(out_size)
    output = _transform(flo, U, out_size)
    return output


class MaskRetriever:

    def __init__(self) -> None:
        self.warp = Warp()
        # hyperparameters
        self.flow_consist_alpha = 3
        self.flow_consist_beta = 0.05

    def get_occlusion_mask_from_flow(self, tensor_size, flow: torch.Tensor):
        mask = torch.ones(tensor_size)
        if flow.is_cuda:
            mask = mask.to(flow.get_device())
        h, w = mask.shape[2], mask.shape[3]
        occ_mask = transformerFwd(
            mask.permute(0, 2, 3, 1), flow.permute(0, 2, 3, 1), out_size=[h, w]
        ).permute(0, 3, 1, 2)
        with torch.no_grad():
            occ_mask = torch.clamp(occ_mask, 0.0, 1.0)
        return occ_mask

    def get_visible_masks(self, optical_flows, optical_flows_rev):
        # get occlusion masks
        batch_size, _, img_h, img_w = optical_flows[0].shape
        img2_visible_masks, img1_visible_masks = [], []
        for s, (optical_flow, optical_flow_rev) in enumerate(
            zip(optical_flows, optical_flows_rev)
        ):
            shape = [batch_size, 1, int(img_h / (2**s)), int(img_w / (2**s))]
            img2_visible_masks.append(
                self.get_occlusion_mask_from_flow(shape, optical_flow)
            )
            img1_visible_masks.append(
                self.get_occlusion_mask_from_flow(shape, optical_flow_rev)
            )
        return img2_visible_masks, img1_visible_masks

    def get_flow_norm(self, flow, p=2):
        """
        Inputs:
        flow (bs, 2, H, W)
        """
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        return flow_norm

    def get_consistent_masks(self, optical_flows, optical_flows_rev):
        # get consist masks
        batch_size, _, img_h, img_w = optical_flows[0].shape
        (
            img2_consis_masks,
            img1_consis_masks,
            fwd_flow_diff_pyramid,
            bwd_flow_diff_pyramid,
        ) = ([], [], [], [])
        for s, (optical_flow, optical_flow_rev) in enumerate(
            zip(optical_flows, optical_flows_rev)
        ):

            bwd2fwd_flow = self.warp(optical_flow_rev, optical_flow)
            fwd2bwd_flow = self.warp(optical_flow, optical_flow_rev)

            fwd_flow_diff = torch.abs(bwd2fwd_flow + optical_flow)
            fwd_flow_diff_pyramid.append(fwd_flow_diff)
            bwd_flow_diff = torch.abs(fwd2bwd_flow + optical_flow_rev)
            bwd_flow_diff_pyramid.append(bwd_flow_diff)

            # flow consistency condition
            bwd_consist_bound = torch.max(
                self.flow_consist_beta * self.get_flow_norm(optical_flow_rev),
                torch.from_numpy(np.array([self.flow_consist_alpha]))
                .float()
                .to(
                    optical_flow_rev.get_device() if optical_flow_rev.is_cuda else "cpu"
                ),
            )
            fwd_consist_bound = torch.max(
                self.flow_consist_beta * self.get_flow_norm(optical_flow),
                torch.from_numpy(np.array([self.flow_consist_alpha]))
                .float()
                .to(optical_flow.get_device() if optical_flow.is_cuda else "cpu"),
            )
            with torch.no_grad():
                noc_masks_img2 = (
                    self.get_flow_norm(bwd_flow_diff) < bwd_consist_bound
                ).float()
                noc_masks_img1 = (
                    self.get_flow_norm(fwd_flow_diff) < fwd_consist_bound
                ).float()
                img2_consis_masks.append(noc_masks_img2)
                img1_consis_masks.append(noc_masks_img1)
        return (
            img2_consis_masks,
            img1_consis_masks,
            fwd_flow_diff_pyramid,
            bwd_flow_diff_pyramid,
        )

    def __call__(self, optical_flows, optical_flows_rev) -> Any:

        # get occlusion masks
        img2_visible_masks, img1_visible_masks = self.get_visible_masks(
            optical_flows, optical_flows_rev
        )
        # get consistent masks
        (
            img2_consis_masks,
            img1_consis_masks,
            fwd_flow_diff_pyramid,
            bwd_flow_diff_pyramid,
        ) = self.get_consistent_masks(optical_flows, optical_flows_rev)
        # get final valid masks
        img2_valid_masks, img1_valid_masks = [], []

        for i, (
            img2_visible_mask,
            img1_visible_mask,
            img2_consis_mask,
            img1_consis_mask,
        ) in enumerate(
            zip(
                img2_visible_masks,
                img1_visible_masks,
                img2_consis_masks,
                img1_consis_masks,
            )
        ):
            img2_valid_masks.append(img2_visible_mask * img2_consis_mask)
            img1_valid_masks.append(img1_visible_mask * img1_consis_mask)

        return {
            "img1_valid_masks": img1_valid_masks,
            "img2_valid_masks": img2_valid_masks,
            "fwd_flow_diff_pyramid": fwd_flow_diff_pyramid,
            "bwd_flow_diff_pyramid": bwd_flow_diff_pyramid,
        }
