from PIL import Image
import torch
import numpy as np
from typing import Tuple, Literal
from ..constants import flow_img_wh, content_img_wh
from mt_pipe.src.util import make_obj_from_conf, are_lists_equal, load_class


def load_image(
    path: str, reshape=False, target=Literal["flow", "content"]
) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(path)
    ten = torch.FloatTensor(
        np.array(img).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    )
    if reshape:
        _, origHeight, origWidth = ten.shape
        ten = ten.view(1, 3, origHeight, origWidth)
        if target == "flow":
            targetWidth = flow_img_wh[0]
            targetHeight = flow_img_wh[1]
        elif target == "content":
            targetWidth = content_img_wh[0]
            targetHeight = content_img_wh[1]
        ten = torch.nn.functional.interpolate(
            input=ten,
            size=(targetHeight, targetWidth),
            mode="bilinear",
            align_corners=False,
        )

    return img, ten


def set_device_nested_tens(obj, device) -> None:
    if type(obj) == torch.Tensor:
        return obj.to(device)
    else:
        if type(obj) in (list, tuple):
            new_obj = []
            for sub_obj in obj:
                new_obj.append(set_device_nested_tens(sub_obj, device))
            return new_obj
        elif type(obj) == dict:
            new_obj = dict()
            for k, v in obj.items():
                new_obj[k] = set_device_nested_tens(v, device)
            return new_obj
        elif obj is None:
            return None
        else:
            raise ValueError(type(obj))


__all__ = [
    "make_obj_from_conf",
    "are_lists_equal",
    "load_class",
    "load_image",
    "set_device_nested_tens",
]
