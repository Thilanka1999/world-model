import torch
import numpy as np
import cv2


class FlowNormalize:
    def __init__(self, orig, trgt) -> None:
        self.orig = orig
        self.trgt = trgt

    def __call__(self, img):
        return img * (torch.tensor(self.trgt) / torch.tensor(self.orig)).view(2, 1, 1)


class ClipValues:
    """
    Transform clips pixel values between 0 and 1 after converting to PIL image.
    """

    def __call__(self, img):
        img = torch.clamp(img, 0, 1)  # Clip pixel values between 0 and 1
        return img


class ResizeNumpy:

    def __init__(self, wh, interpolation=cv2.INTER_NEAREST) -> None:
        self.wh = wh
        self.interpolation = interpolation

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        resized_array = cv2.resize(arr, dsize=self.wh, interpolation=self.interpolation)
        return resized_array
