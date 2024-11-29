import torch
from torch import nn


class Warp:
    def __call__(
        self, ten_input: torch.Tensor, ten_flow: torch.Tensor, use_mask: bool = False
    ) -> torch.Tensor:

        batched = True
        if len(ten_input.shape) == 3:
            batched = False
            ten_input = ten_input.view([1, *ten_input.shape])
            ten_flow = ten_flow.view([1, *ten_flow.shape])

        B, C, H, W = ten_input.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if grid.shape != ten_flow.shape:
            raise ValueError(
                "the shape of grid {0} is not equal to the shape of flow {1}.".format(
                    grid.shape, ten_flow.shape
                )
            )
        if ten_input.is_cuda:
            grid = grid.to(ten_input.get_device())
        vgrid = grid + ten_flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        tenOutput = nn.functional.grid_sample(ten_input, vgrid, align_corners=True)
        if use_mask:
            mask = torch.autograd.Variable(torch.ones(ten_input.size())).to(
                ten_input.get_device()
            )
            mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1
            tenOutput = tenOutput * mask

        if not batched:
            tenOutput = tenOutput[0]

        return tenOutput
