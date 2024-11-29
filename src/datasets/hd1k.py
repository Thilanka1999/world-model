import os
from .common import get_valid_ds_len
import glob
from .base import FlowBase


class HD1K(FlowBase):
    train_val_split = (0.9, 0.1)
    sparse = True

    urls = [
        {
            "url": "http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_full_package.zip",  # 7.8GB
            "archive": True,
        }
    ]

    def _get_paths(self, root, split, get_path_kwargs):
        img_paths = sorted(
            glob.glob(os.path.join(root, "hd1k_input/image_2/**.png"), recursive=True)
        )
        img1_paths = img_paths[:-1]
        img2_paths = img_paths[1:]

        new_len = get_valid_ds_len(len(img1_paths))
        img1_paths = img1_paths[:new_len]
        img2_paths = img2_paths[:new_len]
        flow_paths = [
            p.replace("hd1k_input/image_2", "hd1k_flow_gt/flow_occ") for p in img1_paths
        ]
        occs_paths = [None] * len(flow_paths)

        return img1_paths, img2_paths, flow_paths, occs_paths
