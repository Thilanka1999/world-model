import os
from .common import get_valid_ds_len
import glob
from .base import FlowBase


class MPISintel(FlowBase):
    sparse = False
    urls = [
        {
            "url": "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",  # 7.8GB
            "archive": True,
        }
    ]

    def _get_paths(self, root, split, get_path_kwargs):
        split_dir = "training" if split == "train" else "test"
        scenes = os.listdir(os.path.join(root, split_dir, "clean"))
        img1_paths = []
        img2_paths = []
        for scene in scenes:
            imgs = sorted(
                glob.glob(os.path.join(root, split_dir, "clean", scene, "**.png"))
            )
            img1_paths.extend(imgs[:-1])
            img2_paths.extend(imgs[1:])

        new_len = get_valid_ds_len(len(img1_paths))
        img1_paths = img1_paths[:new_len]
        img2_paths = img2_paths[:new_len]

        if split == "train":
            flow_paths = [
                p.replace("/clean/", "/flow/").replace(".png", ".flo")
                for p in img1_paths
            ]
            occs_paths = [p.replace("/clean/", "/occlusions/") for p in img1_paths]
        else:
            flow_paths = [None] * new_len
            occs_paths = [None] * new_len

        return img1_paths, img2_paths, flow_paths, occs_paths
