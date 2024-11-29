from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle
import cv2
import json
from typing import Sequence
from ..constants import flow_img_wh


class ADE20k(Dataset):
    valid_splits = ["train", "val"]

    def download(self):
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

        if split == "train":
            split = "training"
        elif split == "val":
            split = "validation"

        self.img_list = []
        self.polygon_list = []
        self.seg_list = []

        data_path = os.path.join(os.path.abspath(root), "images", "ADE", split)
        sub = sorted(os.listdir(data_path))
        for s in sub:
            data = sorted(os.listdir(os.path.join(data_path, s)))
            for d in data:
                all = sorted(os.listdir(os.path.join(data_path, s, d)))
                for a in all:
                    seg_parts = a.split("_")
                    img_parts = a.split(".")
                    if seg_parts[-1] == "seg.png":
                        self.seg_list.append(os.path.join(data_path, s, d, a))
                    if img_parts[-1] == "jpg":
                        self.img_list.append(os.path.join(data_path, s, d, a))
                    if img_parts[-1] == "json":
                        json_file_path = os.path.join(data_path, s, d, a)
                        with open(
                            json_file_path, encoding="utf-8", errors="ignore"
                        ) as handler:
                            desc = json.load(handler)["annotation"]["object"]
                        polygon_dict = {}
                        for i in range(len(desc)):
                            polygon_dict[f"key_{i}"] = [
                                desc[i]["name"],
                                desc[i]["polygon"]["x"],
                                desc[i]["polygon"]["y"],
                            ]
                        self.polygon_list.append(polygon_dict)

        key_file_path = os.path.join(os.path.abspath(root), "index_ade20k.pkl")
        with open(key_file_path, "rb") as file:
            data = pickle.load(file)
        self.name_list = data["objectnames"]
        del self.name_list[0]

        self.img_wh = img_wh

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, id: int):
        img_path = self.img_list[id]
        img = np.array(Image.open(img_path))

        png_path = self.seg_list[id]
        polygon_dict = self.polygon_list[id]
        image = np.array(Image.open(png_path))
        masks = np.zeros([*image.shape[:-1], len(self.name_list)])
        s = set()

        for key in polygon_dict:
            name = polygon_dict[key][0]
            s.add(name)
            index = self.name_list.index(name)
            mask_to_fill = masks[:, :, index].copy()
            polygon = {"x": polygon_dict[key][1], "y": polygon_dict[key][2]}
            polygon_vertices = np.array(
                list(zip(polygon["x"], polygon["y"])), dtype=np.int32
            )
            polygon_vertices = polygon_vertices.reshape((-1, 1, 2))
            cv2.fillPoly(mask_to_fill, [polygon_vertices], color=255)
            masks[:, :, index] = mask_to_fill

        resized_img_channels = np.empty((self.img_wh[1], self.img_wh[0], img.shape[2]))
        for channel in range(img.shape[2]):
            single_channel = img[:, :, channel]
            resized_channel = cv2.resize(single_channel, self.img_wh)
            resized_img_channels[:, :, channel] = resized_channel

        resized_seg_channels = np.empty(
            (self.img_wh[1], self.img_wh[0], masks.shape[2])
        )
        for channel in range(masks.shape[2]):
            single_channel = masks[:, :, channel]
            resized_channel = cv2.resize(
                single_channel, self.img_wh, interpolation=cv2.INTER_NEAREST
            )
            resized_seg_channels[:, :, channel] = resized_channel

        masks = torch.FloatTensor(
            resized_seg_channels.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
        img = torch.FloatTensor(
            resized_img_channels.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        )
        return {"img": img, "seg": masks}
