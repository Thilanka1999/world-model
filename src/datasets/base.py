from torch.utils.data import Dataset
import os
from typing import Dict, Sequence, Tuple
from PIL import Image
import torch
from torchvision.transforms import Resize, ToTensor, Compose
from ..util.data.transforms import FlowNormalize, ResizeNumpy
from ..util.io import sparse_flow_from_filepath, flow_from_filepath, occ_from_filepath
from torchvision.datasets.utils import download_and_extract_archive, download_url
from abc import abstractmethod
from .common import move_subdir_up


class FlowBase(Dataset):
    valid_splits = ["train", "val"]
    sparse = None

    def download(self, root: str) -> None:
        for url in self.urls:
            if url["archive"]:
                download_and_extract_archive(url["url"], root, remove_finished=True)
                if "move_up" in url:
                    move_subdir_up(root, url["move_up"])
            else:
                download_url(url["url"], root)

    @abstractmethod
    def _get_paths(self):
        """Placeholder for load image paths."""
        pass

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Sequence[int] = None,
        download: bool = False,
        **get_path_kwargs,
    ) -> None:
        assert (
            self.sparse is not None
        ), "'sparse' attribute bust be overridden by the child class"

        if not os.path.exists(root):
            if download:
                self.download(root)
            else:
                raise FileNotFoundError(f"No such directory as {root}")
        if split not in self.valid_splits:
            raise ValueError(f"Valid splits are {self.valid_splits}")

        img1_paths, img2_paths, flow_paths, occs_paths = self._get_paths(
            root, split, get_path_kwargs
        )
        self.img1_paths = img1_paths
        self.img2_paths = img2_paths
        self.flow_paths = flow_paths
        self.occs_paths = occs_paths

        self.img_hw = None if img_wh is None else img_wh[::-1]
        self.trans = (
            {"img": ToTensor(), "mask": lambda x: x, "flow": {}}
            if img_wh is None
            else {
                "img": Compose([Resize(self.img_hw, antialias=True), ToTensor()]),
                "flow": {},
                "mask": ResizeNumpy(self.img_hw[::-1]),
            }
        )

    def _transform_flow(self, flow: torch.Tensor) -> torch.Tensor:
        flow_trans_key = str(flow.shape)
        if flow_trans_key in self.trans["flow"]:
            trans = self.trans["flow"][flow_trans_key]
        else:
            img_hw_orig = flow.shape[1:]
            if self.img_hw is None:
                trans = lambda x: x
            else:
                trans = Compose(
                    [
                        Resize(self.img_hw, antialias=True),
                        FlowNormalize(img_hw_orig, self.img_hw),
                    ]
                )
            self.trans["flow"][flow_trans_key] = trans
        flow = trans(flow)
        return flow

    def __len__(self) -> int:
        return len(self.img1_paths)

    def _read_flow(
        self, flow_path: str, occs_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sparse:
            flow, valid = sparse_flow_from_filepath(flow_path)
        else:
            flow = flow_from_filepath(flow_path)
            valid = None if occs_path is None else 1 - occ_from_filepath(occs_path)
        flow = torch.FloatTensor(flow.transpose(2, 0, 1).copy())
        return flow, valid

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:

        img1 = Image.open(self.img1_paths[index]).convert("RGB")
        img2 = Image.open(self.img2_paths[index]).convert("RGB")
        img1 = self.trans["img"](img1)
        img2 = self.trans["img"](img2)

        flow_path = self.flow_paths[index]
        occs_path = self.occs_paths[index]

        if flow_path is not None:
            flow, valid = self._read_flow(flow_path, occs_path)
        else:
            flow = valid = None

        img1 = Image.open(self.img1_paths[index]).convert("RGB")
        img2 = Image.open(self.img2_paths[index]).convert("RGB")
        img1 = self.trans["img"](img1)
        img2 = self.trans["img"](img2)
        valid = None if valid is None else self.trans["mask"](valid)
        flow = None if flow is None else self._transform_flow(flow)

        return {
            "img1": img1,
            "img2": img2,
            "valid": valid,
            "flow_gt": flow,
            "occ_gt": None if valid is None else 1 - valid,
        }
