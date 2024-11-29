import cv2
import matplotlib.pyplot as plt
from src.models.flow_decoder import Warp
import numpy as np
import torch
import random


def flow2rgb(flow: np.ndarray | torch.Tensor):

    if type(flow) == torch.Tensor:
        flow = flow.detach().cpu().numpy().transpose(1, 2, 0)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    hue = angle * 180 / np.pi / 2
    hue_normalized = cv2.normalize(hue, None, 0, 1, cv2.NORM_MINMAX)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = hue_normalized * 255
    hsv[..., 1] = magnitude_normalized * 255
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255

    return rgb


warp = Warp()


def plot_warp(sample):
    if len(sample) == 5:
        fig, ax = plt.subplots(2, 3, figsize=(12, 4))

        img1 = sample["img1"]
        img2 = sample["img2"]
        flow = sample["flow_gt"]
        valid = sample["valid"]

        warped_img2 = warp(img2.to(torch.float32), flow)
        flow_vis = flow2rgb(flow.numpy().transpose(1, 2, 0))
        if valid is None:
            diff = (warped_img2 - img1).abs()

            # create a "None" image
            valid = np.zeros((*img1.shape[1:], 3), dtype=np.uint8)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = valid.shape[0] / 128
            font_thickness = max(1, int(valid.shape[0] / 128))
            text_size, _ = cv2.getTextSize("None", font_face, font_scale, 1)
            text_x = (valid.shape[1] - text_size[0]) // 2
            text_y = (valid.shape[0] + text_size[1]) // 2
            cv2.putText(
                valid,
                "None",
                (text_x, text_y),
                font_face,
                font_scale,
                [255, 0, 0],
                font_thickness,
            )
        else:
            diff = (warped_img2 - img1).abs() * valid

        img1 = img1.numpy().transpose(1, 2, 0)
        img2 = img2.numpy().transpose(1, 2, 0)
        warped_img2 = warped_img2.numpy().transpose(1, 2, 0)
        diff = diff.numpy().transpose(1, 2, 0)

        ax[0][0].imshow(img1)
        ax[0][0].set_title("img1")
        ax[0][1].imshow(img2)
        ax[0][1].set_title("img2")
        ax[0][2].imshow(warped_img2)
        ax[0][2].set_title("warped img2")
        ax[1][0].imshow(flow_vis)
        ax[1][0].set_title("flow")
        ax[1][1].imshow(valid.squeeze(), cmap="gray")
        ax[1][1].set_title("valid mask")
        ax[1][2].imshow(diff)
        ax[1][2].set_title("difference")

        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))

        img1 = sample["img1"]
        img2 = sample["img2"]

        img1 = img1.numpy().transpose(1, 2, 0)
        img2 = img2.numpy().transpose(1, 2, 0)

        ax[0].imshow(img1)
        ax[0].set_title("img1")
        ax[1].imshow(img2)
        ax[1].set_title("img2")

        plt.tight_layout()
        plt.show()


def plot_segs(sample, classes, verbose=True):
    img, seg = sample["img"], sample["seg"]

    if verbose:
        print(
            f"Image shape: {img.shape}, Image min: {img.min()}, Image max: {img.max()}"
        )
        print(
            f"Segment shape: {seg.shape}, Segment min: {seg.min()}, Segment max: {seg.max()}"
        )

    img, seg = img.numpy(), seg.numpy()
    present_ids = np.unique(seg).astype(int)
    random.shuffle(present_ids)
    selected_ids = present_ids[:2]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img.transpose(1, 2, 0))
    ax[0].set_title("Image")
    for i, idx in enumerate(selected_ids):
        ax[i + 1].imshow(seg == idx)
        ax[i + 1].set_title(classes[idx])

    plt.show()
