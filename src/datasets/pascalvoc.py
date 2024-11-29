from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
from torchvision.datasets.utils import download_and_extract_archive
from .common import move_subdir_up
from ..constants import flow_img_wh
from typing import Sequence


class PascalVOC(Dataset):
    valid_splits = ["train", "val"]

    urls = {
        "img_archive": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    }

    def download(self, root: str) -> None:
        download_and_extract_archive(
            self.urls["img_archive"], root, remove_finished=True
        )
        move_subdir_up(root, "VOCdevkit")

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

        img_dir = "JPEGImages"
        seg_dir = "SegmentationClass"
        split_dir = "ImageSets/Segmentation"

        if split == "train":
            last = "train.txt"
        elif split == "val":
            last = "val.txt"

        split_path = os.path.join(os.path.abspath(root), split_dir, last)

        with open(split_path, "r") as file:
            self.file_contents = [line.strip() for line in file.readlines()]

        self.img_list = []
        self.seg_list = []
        for item in self.file_contents:
            img_item = str(item) + ".jpg"
            seg_item = str(item) + ".png"
            self.img_list.append(os.path.join(os.path.abspath(root), img_dir, img_item))
            self.seg_list.append(os.path.join(os.path.abspath(root), seg_dir, seg_item))

        self.class_names = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv/monitor",
        ]
        self.color_map = [
            [0, 0, 0],
            [0, 0, 128],
            [0, 128, 0],
            [0, 128, 128],
            [128, 0, 0],
            [128, 0, 128],
            [128, 128, 0],
            [128, 128, 128],
            [0, 0, 64],
            [0, 0, 192],
            [0, 128, 64],
            [0, 128, 192],
            [128, 0, 64],
            [128, 0, 192],
            [128, 128, 64],
            [128, 128, 192],
            [0, 64, 0],
            [0, 64, 128],
            [0, 192, 0],
            [0, 192, 128],
            [128, 64, 0],
        ]

        self.img_wh = img_wh

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, id: int):
        img_path = self.img_list[id]
        img = np.array(Image.open(img_path))

        mask_path = self.seg_list[id]
        image = cv2.imread(mask_path)
        height = image.shape[0]  # 281
        width = image.shape[1]  # 500
        # masks = np.zeros([*img.shape[:-1], len(self.class_names)])
        masks = np.zeros([height, width, 1])
        s = set()
        for i in range(width):
            for j in range(height):
                if list(image[j][i]) == [192, 224, 224]:
                    continue
                index = self.color_map.index(list(image[j][i]))
                s.add(index)
                masks[j][i][0] = index + 1  # (index+1) = first class
                # masks[j][i][index] = 1
        masks = cv2.resize(masks, self.img_wh, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self.img_wh)
        masks = torch.FloatTensor(masks.astype(np.float32))
        # masks = torch.FloatTensor(masks.transpose(2, 0, 1).astype(np.float32))
        img = torch.FloatTensor(
            img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
        return {"img": img, "seg": masks}
