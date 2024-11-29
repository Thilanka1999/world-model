from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import cv2
from torchvision.datasets.utils import download_and_extract_archive
from typing import Sequence
from ..constants import flow_img_wh


class Davis(Dataset):
    valid_splits = ["train", "val"]
    urls = {
        "img_archive": "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    }

    def download(self, root: str) -> None:
        download_and_extract_archive(
            self.urls["img_archive"], root, remove_finished=True
        )

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

        img_dir = "JPEGImages/480p"
        seg_dir = "Annotations/480p"
        split_dir = "ImageSets"

        if split == "train":
            last = "train.txt"
        elif split == "val":
            # last = "val.txt"
            raise NotImplementedError("Train and Validation classes mismatch")  # TODO

        split_path = os.path.join(os.path.abspath(root), split_dir, "2017", last)

        with open(split_path, "r") as file:
            self.file_contents = [line.strip() for line in file.readlines()]

        self.img_list = []
        self.seg_list = []
        for item in self.file_contents:
            split_img_path = os.path.join(os.path.abspath(root), img_dir, str(item))
            split_imgs = sorted(os.listdir(split_img_path))
            for im in split_imgs:
                self.img_list.append(os.path.join(split_img_path, im))

            split_seg_path = os.path.join(os.path.abspath(root), seg_dir, str(item))
            split_segs = sorted(os.listdir(split_seg_path))
            for seg in split_segs:
                self.seg_list.append(os.path.join(split_seg_path, seg))

        self.img_wh = img_wh
        # self.image_shape = (640,480)   # w=640, h=480

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, id: int):
        img_path = self.img_list[id]
        img = np.array(Image.open(img_path))

        mask_path = self.seg_list[id]
        seg_image = cv2.imread(mask_path)
        height = seg_image.shape[0]  # 480
        width = seg_image.shape[1]  # 854
        masks = np.zeros([*img.shape[:-1], 1])
        # masks = np.zeros([*img.shape[:-1], len(self.file_contents)])
        i = mask_path.split("/")
        index = self.file_contents.index(i[-2])
        s = set()
        for i in range(width):
            for j in range(height):
                if list(seg_image[j][i]) != [0, 0, 0]:
                    masks[j][i] = index + 1
                    # masks[j][i][index] = 1
                    s.add(index)
        masks = cv2.resize(masks, self.img_wh, interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, self.img_wh)
        masks = torch.FloatTensor(masks.astype(np.float32))
        # masks = torch.FloatTensor(masks.transpose(2, 0, 1).astype(np.float32))
        img = torch.FloatTensor(
            img.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
        return {"img": img, "seg": masks}
