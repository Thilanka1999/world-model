import torch
from typing import Dict
import numpy as np
import cv2
from ..util.visualize import flow2rgb
from mt_pipe.src.visualizers import BaseVisualizer


class FlowVisualizer(BaseVisualizer):

    def _build_image(
        self, gt: np.ndarray, pred: np.ndarray, img: np.ndarray, text: str
    ) -> np.ndarray:
        pred = pred.clip(gt.min(), gt.max())
        flow = np.concatenate([gt, pred], axis=1)
        flow = flow2rgb(flow)
        built_img = np.concatenate([img, flow], axis=1)

        # Put the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = built_img.shape[1] / 1000
        thickness = max(1, int(built_img.shape[1] / 800))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (built_img.shape[1] - text_size[0]) // 2
        text_y = (built_img.shape[0] + text_size[1]) // 2
        cv2.putText(
            built_img, text, (text_x, text_y), font, font_scale, (1, 0, 0), thickness
        )

        return built_img

    def __call__(
        self,
        info: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int = None,
        loop: str = None,
    ) -> None:
        samples = self._get_samples(info["flow_pred"].shape[0])

        flow_pred = (
            info["flow_pred"][samples].detach().cpu().numpy().transpose(0, 2, 3, 1)
        )
        flow_gt = batch["flow_gt"][samples].detach().cpu().numpy().transpose(0, 2, 3, 1)
        img = batch["img1"][samples].detach().cpu().numpy().transpose(0, 2, 3, 1)

        for idx in range(len(flow_pred)):
            text = f"Idx: {idx}" if epoch is None else f"Epoch: {epoch+1}, Idx: {idx}"
            vis = self._build_image(flow_gt[idx], flow_pred[idx], img[idx], text)
            self._output(vis, loop)
