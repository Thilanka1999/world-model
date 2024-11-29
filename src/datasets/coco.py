import json
from torch.utils.data import Dataset
import os
import shutil
import json
import numpy as np
from typing import Dict, Tuple, Sequence
from PIL import Image
import torch
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from torchvision.transforms import Resize, ToTensor, Compose, InterpolationMode, Lambda


class COCOSegment(Dataset):
    valid_splits = ["train", "val"]
    subsets = {
        "THINGS": [
            1,
            91,
        ],  # thing categories (80 total): individual instances (person, car, chair, etc.)
        "ORIGINAL_STUFF": [
            92,
            182,
        ],  # original stuff categories (36 total): materials and objects with no clear boundaries (sky, street, grass, etc.)
        "MERGED_STUFF": [183, 200],  # merged stuff categories (17 total)
    }
    urls = {
        "images": {
            "train": "http://images.cocodataset.org/zips/train2017.zip",  # 18.0GB
            "val": "http://images.cocodataset.org/zips/val2017.zip",  # 778MB
        },
        "annotations": {
            "panoptic": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",  # 821MB
            "stuff": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",  # 1.1GB
        },
    }

    def download(self, root: str):
        download_and_extract_archive(
            self.urls["images"]["train"],
            os.path.join(root, "images"),
            remove_finished=True,
        )
        download_and_extract_archive(
            self.urls["images"]["val"],
            os.path.join(root, "images"),
            remove_finished=True,
        )
        download_and_extract_archive(
            self.urls["annotations"]["panoptic"],
            os.path.join(root, "annotations"),
            remove_finished=True,
        )

        # format image directories
        shutil.move(
            os.path.join(root, "images", "train2017"),
            os.path.join(root, "images", "train"),
        )
        shutil.move(
            os.path.join(root, "images", "val2017"),
            os.path.join(root, "images", "val"),
        )

        # format annotations directories
        for split in ["train", "val"]:
            archive_path = os.path.abspath(
                os.path.join(root, f"annotations/annotations/panoptic_{split}2017.zip")
            )
            target_path = os.path.abspath(os.path.join(root, f"annotations/panoptic"))
            print(f"Extracting {archive_path} to {target_path}")
            extract_archive(archive_path, target_path)
            shutil.move(
                os.path.join(root, f"annotations/panoptic/panoptic_{split}2017"),
                os.path.join(root, f"annotations/panoptic/{split}"),
            )
            shutil.copy(
                os.path.join(
                    root, f"annotations/annotations/panoptic_{split}2017.json"
                ),
                os.path.join(root, f"annotations/panoptic/{split}.json"),
            )

        trash_dirs = [
            "annotations/panoptic/__MACOSX",
            "annotations/__MACOSX",
            "annotations/annotations",
        ]
        for trash_dir in trash_dirs:
            shutil.rmtree(os.path.join(root, trash_dir))

    def __init__(
        self,
        root: str,
        split: str = "train",
        subsets: Sequence[str] = ["THINGS"],
        img_wh: Sequence[int] = None,
        download: bool = False,
    ) -> None:
        if split not in self.valid_splits:
            raise ValueError(f"Valid splits are {self.valid_splits}")
        if not os.path.exists(root):
            if download:
                self.download(root)
            else:
                raise FileNotFoundError(f"No such directory as {root}")
        for subset in subsets:
            if subset not in self.subsets.keys():
                raise ValueError(
                    f"Invalid subset definition. Valid subsets are: {list(self.subsets.keys())}"
                )

        self.img_root = os.path.join(root, "images", split)
        self.seg_root = os.path.join(root, "annotations", "panoptic", split)

        if not os.path.exists(self.img_root):
            raise FileNotFoundError(
                f"No such directory as {self.img_root}. The dataset directory might be corrupted"
            )
        if not os.path.exists(self.seg_root):
            raise FileNotFoundError(
                f"No such directory as {self.seg_root}. The dataset directory might be corrupted"
            )

        img_items = sorted(
            [".".join(it.split(".")[:-1]) for it in os.listdir(self.img_root)]
        )
        seg_items = sorted(
            [".".join(it.split(".")[:-1]) for it in os.listdir(self.seg_root)]
        )

        if img_items != seg_items:
            raise FileNotFoundError(
                f"content of images({self.img_root}) and segmentations({self.seg_root}) does not coincide."
            )

        self.items = img_items

        with open(f"{self.seg_root}.json") as handler:
            panoptics = json.load(handler)
            annotations = panoptics["annotations"]
            categories = panoptics["categories"]
        del panoptics

        filtered_subsets = [v for (k, v) in self.subsets.items() if k in subsets]
        self.categories = {
            cat["id"]: cat["name"]
            for cat in categories
            if any(
                [(cat["id"] in range(ran[0], ran[1] + 1)) for ran in filtered_subsets]
            )
        }
        self.classes = tuple(self.categories.values())
        self.seg_cat_maps = {
            ant["file_name"].split(".")[0]: [
                (seg["id"], seg["category_id"]) for seg in ant["segments_info"]
            ]
            for ant in annotations
        }
        self.img_hw = None if img_wh is None else img_wh[::-1]
        self.trans = {
            "img": Compose(
                [
                    Lambda(lambda x: x) if img_wh is None else Resize(img_wh[::-1]),
                    ToTensor(),
                ]
            ),
            "seg": Compose(
                [
                    (
                        Lambda(lambda x: x)
                        if img_wh is None
                        else Resize(img_wh[::-1], InterpolationMode.NEAREST)
                    )
                ]
            ),
        }

    def encode_labels(self, lbls: np.ndarray, map: Tuple[Tuple[int]]) -> np.ndarray:
        masks = np.zeros(lbls.shape[:-1])
        lbl_ids = lbls[:, :, 0] + lbls[:, :, 1] * 256 + lbls[:, :, 2] * (256**2)
        for i, cat_id in enumerate(self.categories.keys()):
            valid_seg_ids = [seg[0] for seg in map if seg[1] == cat_id]
            masks[(lbl_ids[..., np.newaxis] == valid_seg_ids).any(2)] = i

        return masks

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, id: int) -> Dict[str, torch.Tensor]:
        item = self.items[id]
        img = Image.open(os.path.join(self.img_root, item + ".jpg")).convert("RGB")
        lbl = np.array(Image.open(os.path.join(self.seg_root, item + ".png")))

        seg_cat_map = self.seg_cat_maps[item]
        seg = self.encode_labels(lbl, seg_cat_map)

        seg = self.trans["seg"](torch.FloatTensor(seg).unsqueeze(0)).squeeze()
        img = self.trans["img"](img)

        return {"img": img, "seg": seg}
