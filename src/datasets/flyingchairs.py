import os
from .common import get_valid_ds_len
import glob
from .base import FlowBase


class FlyingChairs(FlowBase):
    sparse = False
    urls = [
        {
            "url": "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs2.zip",  # 31.2GB
            "archive": True,
            "move_up": "FlyingChairs2",
        },
    ]

    def _get_paths(self, root, split, get_path_kwargs):
        img1_paths = sorted(glob.glob(os.path.join(root, split, "**img_0.png")))
        new_len = get_valid_ds_len(len(img1_paths))
        img1_paths = img1_paths[:new_len]

        img2_paths = [p.replace("img_0", "img_1") for p in img1_paths]
        flow_paths = [p.replace("img_0.png", "flow_01.flo") for p in img1_paths]
        occs_paths = [p.replace("img_0", "mb_01") for p in img1_paths]

        return img1_paths, img2_paths, flow_paths, occs_paths
