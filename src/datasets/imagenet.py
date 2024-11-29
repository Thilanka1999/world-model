import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from lightly.transforms.vicreg_transform import VICRegTransform
from PIL import Image
from typing import Dict
import scipy.io
from ..constants import content_img_wh
from .common import get_valid_ds_len


class ImageNetBase(Dataset):
    valid_splits = ["train", "val"]
    urls = {
        "train": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        "val": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "annotations": "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
    }

    def download(self, root: str):
        # download and unpack archives
        for dir, url in self.urls.items():
            dir = os.path.join(root, dir)
            download_and_extract_archive(url, dir, remove_finished=True)

        # unpack training class archives
        print("Unpacking training class archives")
        training_archives = glob.glob(os.path.join(root, "train/*.tar"))
        for arch_path in training_archives:
            dst = arch_path[:-4]
            extract_archive(arch_path, dst, True)

        # format validation images
        print("Formating validation images")
        meta = scipy.io.loadmat(
            os.path.join(root, "annotations/ILSVRC2012_devkit_t12/data/meta.mat")
        )["synsets"]
        id_dir_map = {}
        for row in meta:
            id_dir_map[row[0][0].item()] = row[0][1].item()
        with open(
            os.path.join(
                root,
                "annotations/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            )
        ) as handler:
            ilsvrc_ids = [int(id) for id in handler.read().strip().split("\n")]
        for d in os.listdir(os.path.join(root, "train")):
            os.makedirs(os.path.join(root, "val", d))
        for i, ilsvrc_id in enumerate(ilsvrc_ids):
            f_name = f"ILSVRC2012_val_000{i+1:05d}.JPEG"
            val_dir = id_dir_map[ilsvrc_id]
            src = os.path.join(root, "val", f_name)
            dst = os.path.join(root, "val", val_dir, f_name)
            os.rename(src, dst)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: transforms = None,
        download: bool = False,
    ) -> None:
        if split not in self.valid_splits:
            raise ValueError(
                f"Invalid split definition. Valid splits are {self.valid_splits}"
            )
        if not os.path.isdir(root):
            if download:
                self.download(root)
            else:
                raise FileNotFoundError(f"No such directory as {root}")
        self.root = os.path.abspath(root)
        if transform is None:
            self.transform = transforms.Resize(content_img_wh)
        else:
            self.transform = transforms.Compose(
                [
                    *transform,
                    transforms.Resize(content_img_wh),
                ]
            )
        img_paths = []
        split_path = os.path.abspath(os.path.join(root, split))
        img_sub_folders = [
            dir
            for dir in sorted(os.listdir(split_path))
            if os.path.isdir(os.path.join(split_path, dir))
        ]
        for sub in img_sub_folders:
            ims = os.listdir(os.path.join(split_path, sub))
            new_img_paths = [
                os.path.join(os.path.join(split_path, sub), im)
                for im in ims
                if im.endswith(".JPEG")
            ]
            img_paths.extend(new_img_paths)

        new_len = get_valid_ds_len(len(img_paths))
        img_paths = img_paths[:new_len]

        self.split_path = split_path
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)


class ImageNetVICReg(ImageNetBase):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: transforms = None,
        img_wh=content_img_wh,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, download)
        self.transform = transforms.Compose(
            [
                self.transform,
                VICRegTransform(
                    input_size=img_wh,
                    cj_prob=1,
                    min_scale=0.6,
                    random_gray_scale=0,
                    solarize_prob=0,
                    gaussian_blur=0,
                    rr_prob=0,
                    hf_prob=0,
                    rr_degrees=0,
                    vf_prob=0,
                    normalize=None,
                ),
            ]
        )

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")

        view1, view2 = self.transform(img)

        return {"view1": view1, "view2": view2}


class ImageNetClassify(ImageNetBase):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: transforms = None,
        img_wh=content_img_wh,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, download)

        meta_data_path = os.path.join(
            root, "annotations/ILSVRC2012_devkit_t12/data/meta.mat"
        )
        mat_data = scipy.io.loadmat(meta_data_path)
        dir2txtlabel_map = {}
        dir2label_map = {}
        for row in mat_data["synsets"]:
            dir2txtlabel_map[row[0][1].item()] = row[0][2].item()
            dir2label_map[row[0][1].item()] = row[0][0].item() - 1

        available_dirs = os.listdir(self.split_path)
        excess_nms = []
        for nm in dir2txtlabel_map:
            if nm not in available_dirs:
                excess_nms.append(nm)
        for nm in excess_nms:
            dir2txtlabel_map.pop(nm)
            dir2label_map.pop(nm)

        self.dir2label_map = dir2label_map
        self.dir2txtlabel_map = dir2txtlabel_map
        self.label2txtlable_map = {
            dir2label_map[k]: v for k, v in dir2txtlabel_map.items()
        }
        self.n_classes = len(dir2txtlabel_map)

        if split == "train":
            rz_trans = transforms.RandomResizedCrop(
                img_wh[::-1]
            )  # TODO: find the appropriate scale and ratio
        else:
            rz_trans = transforms.Resize(img_wh[::-1])
        self.transform = transforms.Compose(
            [self.transform, rz_trans, transforms.ToTensor()]
        )

    def __getitem__(self, index) -> Dict[str, int | torch.Tensor]:
        img_path = self.img_paths[index]
        dir = img_path.split("/")[-2]
        lbl = self.dir2label_map[dir] if dir in self.dir2label_map.keys() else -1

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {"img": img, "lbl": lbl}
