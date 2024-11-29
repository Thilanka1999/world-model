from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
from torchvision.datasets.utils import download_and_extract_archive
from ..constants import flow_img_wh
from typing import Sequence


class Cityscape(Dataset):
    valid_splits = ["train", "val", "test"]

    def download(self, root: str):
        raise NotImplementedError()  # TODO

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Sequence[int] = flow_img_wh,
        download: bool = False,
    ) -> None:
        if not os.path.exists(root):
            if download:
                self.download(root)
            else:
                raise FileNotFoundError(f"No such directory as {root}")
        if split not in self.valid_splits:
            raise ValueError(f"Valid splits are {self.valid_splits}")

        img_dir = "leftImg8bit"
        seg_dir = "gtFine"

        self.img_root = os.path.join(os.path.abspath(root), img_dir, split)
        self.seg_root = os.path.join(os.path.abspath(root), seg_dir, split)

        if not os.path.exists(self.img_root):
            raise FileNotFoundError(
                f"No such directory as {self.img_root}. The dataset directory might be corrupted"
            )
        if not os.path.exists(self.seg_root):
            raise FileNotFoundError(
                f"No such directory as {self.seg_root}. The dataset directory might be corrupted"
            )

        self.img_list = []
        img_items = sorted(os.listdir(self.img_root))
        for item in img_items:
            imgs = sorted(os.listdir(os.path.join(self.img_root, item)))
            for im in imgs:
                self.img_list.append(os.path.join(self.img_root, item, im))

        self.seg_list = []
        seg_items = sorted(os.listdir(self.seg_root))
        for item in seg_items:
            segs = sorted(os.listdir(os.path.join(self.seg_root, item)))
            for seg in segs:
                self.seg_list.append(os.path.join(self.seg_root, item, seg))

        self.class_names = [
            "unlabeled",
            "dynamic",
            "ground",
            "road",
            "sidewalk",
            "parking",
            "rail track",
            "building",
            "wall",
            "fence",
            "guard rail",
            "bridge",
            "tunnel",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "caravan",
            "trailer",
            "train",
            "motorcycle",
            "bicycle",
            "license plate",
        ]
        self.class_map = dict(zip(self.class_names, range(len(self.class_names))))
        self.color_map = [
            [0, 0, 0],
            [0, 74, 111],
            [81, 0, 81],
            [128, 64, 128],
            [232, 35, 244],
            [160, 170, 250],
            [140, 150, 230],
            [70, 70, 70],
            [156, 102, 102],
            [153, 153, 190],
            [180, 165, 180],
            [100, 100, 150],
            [90, 120, 150],
            [153, 153, 153],
            [30, 170, 250],
            [0, 220, 220],
            [35, 142, 107],
            [152, 251, 152],
            [180, 130, 70],
            [60, 20, 220],
            [0, 0, 255],
            [142, 0, 0],
            [70, 0, 0],
            [100, 60, 0],
            [90, 0, 0],
            [110, 0, 0],
            [100, 80, 0],
            [230, 0, 0],
            [32, 11, 119],
            [142, 0, 0],
        ]

        self.img_wh = img_wh

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, id: int):
        img_path = self.img_list[id]
        img = np.array(Image.open(img_path))

        mask_path = self.seg_list[id * 4]
        image = cv2.imread(mask_path)
        height = image.shape[0]  # 1024
        width = image.shape[1]  # 2048
        masks = np.zeros([*img.shape[:-1], 1])
        # masks = np.zeros([*img.shape[:-1], len(self.class_names)])
        s = set()
        # for i in range(width):
        #     for j in range(height):
        #         index = self.color_map.index(list(image[j][i]))
        #         s.add(index)
        #         masks[j][i] = index + 1
        #         # masks[j][i][index] = 1

        color_map_array = np.array(self.color_map)
        flattened_image = image.reshape(-1, image.shape[-1])
        indexes_in_color_map = np.argmax(
            (flattened_image[:, None, :] == color_map_array[None, :, :]).all(axis=-1),
            axis=-1,
        )
        masks = indexes_in_color_map.reshape(image.shape[:-1])
        masks += 1

        masks = cv2.resize(masks, self.img_wh, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self.img_wh)
        masks = torch.FloatTensor(masks.astype(np.float32))
        # masks = torch.FloatTensor(masks.transpose(2, 0, 1).astype(np.float32))
        img = torch.FloatTensor(
            img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
        return {"img": img, "seg": masks}
