import glob
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.datasets.utils import download_and_extract_archive
from PIL import ImageFile
from tqdm import tqdm
import threading
import torch
from typing import Dict, Tuple
import os
import numpy as np
from PIL import Image
from .utils.cam_caliberation import (
    get_scaled_intrinsic_matrix,
    rescale_intrinsics,
    get_intrinsics_per_scale,
)
from .utils.depth import generate_depth_map
from ..common import get_valid_ds_len
from ...util import are_lists_equal
from ...util.io import sparse_flow_from_filepath
from ...util.data.transforms import FlowNormalize, ResizeNumpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


class KITTIBase(Dataset):
    valid_splits = ["train", "val"]
    valid_subsets = ["2012", "2012_m", "2015", "2015_m", "2011"]
    raw_train_val_split = (0.9, 0.1)
    urls = {
        "2012": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip",  # 1.9GB
        "2012_m": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow_multiview.zip",  # 16.5GB
        "2015": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",  # 1.6GB
        "2015_m": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip",  # 12.5GB
        "2011": {  # 98GB
            "base_url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/",
            "files": [
                # caliberation matrices
                "2011_09_26_calib.zip",  # 3.97KB
                "2011_09_28_calib.zip",  # 3.98KB
                "2011_09_29_calib.zip",  # 3.98KB
                "2011_09_30_calib.zip",  # 3.98KB
                "2011_10_03_calib.zip",  # 3.98KB
                # city
                "2011_09_26_drive_0001",  # 437.40MB
                "2011_09_26_drive_0002",  # 304.96MB
                "2011_09_26_drive_0005",  # 616.02MB
                "2011_09_26_drive_0009",  # 1.67GB
                "2011_09_26_drive_0011",  # 896.28MB
                "2011_09_26_drive_0013",  # 600.86MB
                "2011_09_26_drive_0014",  # 1.18GB
                "2011_09_26_drive_0017",  # 447.84MB
                "2011_09_26_drive_0018",  # 1.04GB
                "2011_09_26_drive_0048",  # 79.78MB
                "2011_09_26_drive_0051",  # 1.59GB
                "2011_09_26_drive_0056",  # 1.13GB
                "2011_09_26_drive_0057",  # 1.23GB
                "2011_09_26_drive_0059",  # 1.41GB
                "2011_09_26_drive_0060",  # 284.89MB
                "2011_09_26_drive_0084",  # 1.47GB
                "2011_09_26_drive_0091",  # 1.32GB
                "2011_09_26_drive_0093",  # 1.60GB
                "2011_09_26_drive_0095",  # 1.02GB
                "2011_09_26_drive_0096",  # 1.88GB
                "2011_09_26_drive_0104",  # 1.22GB
                "2011_09_26_drive_0106",  # 893.47MB
                "2011_09_26_drive_0113",  # 355.19MB
                "2011_09_26_drive_0117",  # 2.60GB
                "2011_09_28_drive_0001",  # 404.75MB
                "2011_09_28_drive_0002",  # 1.36GB
                "2011_09_29_drive_0026",  # 563.48MB
                "2011_09_29_drive_0071",  # 4.04GB
                # residential
                "2011_09_26_drive_0019",  # 1.95GB
                "2011_09_26_drive_0020",  # 350.09MB
                "2011_09_26_drive_0022",  # 3.18GB
                "2011_09_26_drive_0023",  # 1.86GB
                "2011_09_26_drive_0035",  # 488.30MB
                "2011_09_26_drive_0036",  # 3.03GB
                "2011_09_26_drive_0039",  # 1.43GB
                "2011_09_26_drive_0046",  # 464.08MB
                "2011_09_26_drive_0061",  # 2.84GB
                "2011_09_26_drive_0064",  # 2.21GB
                "2011_09_26_drive_0079",  # 418.79MB
                "2011_09_26_drive_0086",  # 2.85GB
                "2011_09_26_drive_0087",  # 2.92GB
                "2011_09_30_drive_0018",  # 10.54GB
                "2011_09_30_drive_0020",  # 4.17GB
                "2011_09_30_drive_0027",  # 4.12GB
                "2011_09_30_drive_0028",  # 19.74GB
                "2011_09_30_drive_0033",  # 6.10GB
                "2011_09_30_drive_0034",  # 4.73GB
                "2011_10_03_drive_0027",  # 17.09GB
                "2011_10_03_drive_0034",  # 18.42GB
                # road
                "2011_09_26_drive_0015",  # 1.12GB
                "2011_09_26_drive_0027",  # 810.50MB
                "2011_09_26_drive_0028",  # 1.78GB
                "2011_09_26_drive_0029",  # 1.57GB
                "2011_09_26_drive_0032",  # 1.35GB
                "2011_09_26_drive_0052",  # 271.95MB
                "2011_09_26_drive_0070",  # 1.64GB
                "2011_09_26_drive_0101",  # 3.37GB
                "2011_09_29_drive_0004",  # 1.31GB
                "2011_09_30_drive_0016",  # 1.06GB
                "2011_10_03_drive_0042",  # 3.95GB
                "2011_10_03_drive_0047",  # 2.89GB
                # campus
                "2011_09_28_drive_0016",  # 721.41MB
                "2011_09_28_drive_0021",  # 816.20MB
                "2011_09_28_drive_0034",  # 184.55MB
                "2011_09_28_drive_0035",  # 125.62MB
                "2011_09_28_drive_0037",  # 352.59MB
                "2011_09_28_drive_0038",  # 430.92MB
                "2011_09_28_drive_0039",  # 1.37GB
                "2011_09_28_drive_0043",  # 576.40MB
                "2011_09_28_drive_0045",  # 165.68MB
                "2011_09_28_drive_0047",  # 117.40MB
                # person
                "2011_09_28_drive_0053",  # 281.66MB
                "2011_09_28_drive_0054",  # 186.38MB
                "2011_09_28_drive_0057",  # 305.59MB
                "2011_09_28_drive_0065",  # 160.16MB
                "2011_09_28_drive_0066",  # 119.99MB
                "2011_09_28_drive_0068",  # 277.20MB
                "2011_09_28_drive_0070",  # 161.45MB
                "2011_09_28_drive_0071",  # 178.03MB
                "2011_09_28_drive_0075",  # 289.10MB
                "2011_09_28_drive_0077",  # 173.87MB
                "2011_09_28_drive_0078",  # 153.25MB
                "2011_09_28_drive_0080",  # 164.76MB
                "2011_09_28_drive_0082",  # 309.10MB
                "2011_09_28_drive_0086",  # 124.68MB
                "2011_09_28_drive_0087",  # 340.64MB
                "2011_09_28_drive_0089",  # 157.89MB
                "2011_09_28_drive_0090",  # 191.08MB
                "2011_09_28_drive_0094",  # 355.03MB
                "2011_09_28_drive_0095",  # 169.58MB
                "2011_09_28_drive_0096",  # 186.21MB
                "2011_09_28_drive_0098",  # 181.85MB
                "2011_09_28_drive_0100",  # 308.21MB
                "2011_09_28_drive_0102",  # 184.11MB
                "2011_09_28_drive_0103",  # 152.13MB
                "2011_09_28_drive_0104",  # 176.08MB
                "2011_09_28_drive_0106",  # 299.98MB
                "2011_09_28_drive_0108",  # 191.61MB
                "2011_09_28_drive_0110",  # 251.96MB
                "2011_09_28_drive_0113",  # 296.13MB
                "2011_09_28_drive_0117",  # 143.52MB
                "2011_09_28_drive_0119",  # 307.68MB
                "2011_09_28_drive_0121",  # 183.83MB
                "2011_09_28_drive_0122",  # 171.92MB
                "2011_09_28_drive_0125",  # 235.86MB
                "2011_09_28_drive_0126",  # 127.84MB
                "2011_09_28_drive_0128",  # 115.87MB
                "2011_09_28_drive_0132",  # 292.01MB
                "2011_09_28_drive_0134",  # 219.91MB
                "2011_09_28_drive_0135",  # 168.00MB
                "2011_09_28_drive_0136",  # 124.02MB
                "2011_09_28_drive_0138",  # 273.43MB
                "2011_09_28_drive_0141",  # 284.10MB
                "2011_09_28_drive_0143",  # 127.99MB
                "2011_09_28_drive_0145",  # 144.02MB
                "2011_09_28_drive_0146",  # 283.94MB
                "2011_09_28_drive_0149",  # 183.99MB
                "2011_09_28_drive_0153",  # 359.33MB
                "2011_09_28_drive_0154",  # 171.90MB
                "2011_09_28_drive_0155",  # 192.01MB
                "2011_09_28_drive_0156",  # 119.89MB
                "2011_09_28_drive_0160",  # 163.75MB
                "2011_09_28_drive_0161",  # 147.72MB
                "2011_09_28_drive_0162",  # 151.75MB
                "2011_09_28_drive_0165",  # 331.57MB
                "2011_09_28_drive_0166",  # 155.81MB
                "2011_09_28_drive_0167",  # 215.73MB
                "2011_09_28_drive_0168",  # 227.66MB
                "2011_09_28_drive_0171",  # 111.94MB
                "2011_09_28_drive_0174",  # 215.42MB
                "2011_09_28_drive_0177",  # 311.58MB
                "2011_09_28_drive_0179",  # 171.81MB
                "2011_09_28_drive_0183",  # 155.91MB
                "2011_09_28_drive_0184",  # 343.65MB
                "2011_09_28_drive_0185",  # 319.45MB
                "2011_09_28_drive_0186",  # 163.77MB
                "2011_09_28_drive_0187",  # 219.80MB
                "2011_09_28_drive_0191",  # 150.33MB
                "2011_09_28_drive_0192",  # 335.49MB
                "2011_09_28_drive_0195",  # 155.27MB
                "2011_09_28_drive_0198",  # 251.12MB
                "2011_09_28_drive_0199",  # 139.12MB
                "2011_09_28_drive_0201",  # 329.75MB
                "2011_09_28_drive_0204",  # 202.36MB
                "2011_09_28_drive_0205",  # 139.22MB
                "2011_09_28_drive_0208",  # 213.97MB
                "2011_09_28_drive_0209",  # 342.30MB
                "2011_09_28_drive_0214",  # 167.34MB
                "2011_09_28_drive_0216",  # 235.20MB
                "2011_09_28_drive_0220",  # 296.72MB
                "2011_09_28_drive_0222",  # 214.22MB
                # calibration
                "2011_09_26_drive_0119",  # 5.45MB
                "2011_09_28_drive_0225",  # 13.69MB
                "2011_09_29_drive_0108",  # 13.49MB
                "2011_09_30_drive_0072",  # 4.86MB
                "2011_10_03_drive_0058",  # 57.44MB
            ],
        },
    }

    def download(self, root: str, subset: str) -> None:
        if subset not in self.valid_subsets:
            raise ValueError("Invalid subset specification for download")

        if subset == "2011":
            base_url = self.urls["2011"]["base_url"]
            for file in self.urls["2011"]["files"]:
                if file.endswith(".zip"):
                    url = base_url + file
                else:
                    url = base_url + file + "/" + file + "_sync.zip"
                download_and_extract_archive(url, root, remove_finished=True)
        else:
            download_and_extract_archive(self.urls[subset], root, remove_finished=True)

    def _get_set(self, root):
        """
        Automatically Determine the KITTI release
            1. KITTI 2011: 2011
            2. KITTI 2012: 2012
            3. KITTI 2012 multiview: 2012_m
            4. KITTI 2015: 2015
            5. KITTI 2015 multiview: 2015_m
        """
        if are_lists_equal(["training", "testing"], os.listdir(root)):
            # 2, 3, 4, or 5
            if are_lists_equal(
                [
                    "image_0",
                    "disp_refl_occ",
                    "image_1",
                    "colored_1",
                    "disp_occ",
                    "flow_noc",
                    "colored_0",
                    "disp_refl_noc",
                    "flow_occ",
                    "calib",
                    "disp_noc",
                ],
                os.listdir(os.path.join(root, "training")),
            ):
                return "2012"
            if are_lists_equal(
                ["image_0", "image_1", "image_3", "image_2"],
                os.listdir(os.path.join(root, "training")),
            ):
                return "2012_m"
            if are_lists_equal(
                [
                    "disp_noc_0",
                    "viz_flow_occ",
                    "obj_map",
                    "viz_flow_occ_dilate_1",
                    "flow_noc",
                    "flow_occ",
                    "disp_occ_1",
                    "disp_noc_1",
                    "image_3",
                    "disp_occ_0",
                    "image_2",
                ],
                os.listdir(os.path.join(root, "training")),
            ):
                return "2015"
            if are_lists_equal(
                ["image_3", "image_2"], os.listdir(os.path.join(root, "training"))
            ):
                return "2015_m"

            raise FileNotFoundError("Dataset corrupted")
        else:
            return "2011"

    def preprocess_velodyne(self, root):
        print("Exporting depth maps")
        src_list = glob.glob(
            os.path.join(root, "**/velodyne_points/data/*.bin"), recursive=True
        )
        dst_list = [
            p.replace("/velodyne_points/", "/velodyne_npy/").replace(".bin", ".npy")
            for p in src_list
        ]
        new_dirs = list(set([os.path.split(p)[0] for p in dst_list]))
        for d in new_dirs:
            os.makedirs(d)

        def worker(src_list, dst_list):
            for src_path, dst_path in tqdm(
                zip(src_list, dst_list), total=len(src_list)
            ):
                depth_map = generate_depth_map(src_path)
                np.save(dst_path, depth_map)

        thread_count = 10
        element_count = int(len(src_list) / thread_count)
        threads = []
        for i in range(thread_count):
            sl = (
                slice(i * element_count, -1)
                if i == thread_count - 1
                else slice(i * element_count, (i + 1) * element_count)
            )
            thread = threading.Thread(target=worker, args=(src_list[sl], dst_list[sl]))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Tuple[int, int] = [1242, 376],
        download: str = None,
    ) -> None:
        # find the directories
        if not os.path.exists(root):
            if download is not None:
                self.download(root, download)
                ds_set = self._get_set(root)
                if ds_set == "2011":
                    self.preprocess_velodyne(root)
            else:
                raise FileNotFoundError(f"No such directory as {root}")
        if split not in self.valid_splits:
            raise ValueError(
                f"Invalid split definition. Valid splits are {self.valid_splits}"
            )

        ds_set = self._get_set(root)
        self.set = ds_set
        if ds_set == "2011":
            split_dir = None
        else:
            split_dir = "training" if split == "train" else "testing"

        def get_10_11_dir_img_lst(img_dir: str):
            imgs = glob.glob(
                os.path.join(root, split_dir, f"{img_dir}/**.png"), recursive=True
            )
            img1_lst = [img for img in imgs if img.endswith("10.png")]
            img2_lst = [img for img in imgs if img.endswith("11.png")]
            return img1_lst, img2_lst

        if ds_set == "2011":
            img1_dir = "image_02"
            img1_lst_tmp = glob.glob(
                os.path.join(root, "**/**/image_02/data/**.png"), recursive=True
            )
            img2_from_1_lst = [
                img.replace("image_02", "image_03") for img in img1_lst_tmp
            ]
            img2_lst_tmp = glob.glob(
                os.path.join(root, "**/**/image_03/data/**.png"), recursive=True
            )
            mask = [img in img2_lst_tmp for img in img2_from_1_lst]
            img1_lst = [img for i, img in enumerate(img1_lst_tmp) if mask[i]]
            img2_lst = [img for i, img in enumerate(img2_from_1_lst) if mask[i]]
        if ds_set == "2012":
            img1_dir = "colored_0"
            img1_lst, img2_lst = get_10_11_dir_img_lst(img1_dir)
        if ds_set in ["2012_m", "2015_m"]:
            img1_dir = "image_2"
            img1_lst = glob.glob(os.path.join(root, split_dir, "image_2/**.png"))
            img2_lst = glob.glob(os.path.join(root, split_dir, "image_3/**.png"))
        if ds_set == "2015":
            img1_dir = "image_2"
            img1_lst, img2_lst = get_10_11_dir_img_lst(img1_dir)

        img1_lst = sorted(img1_lst)
        img2_lst = sorted(img2_lst)
        new_len = get_valid_ds_len(len(img1_lst))
        img1_lst = img1_lst[:new_len]
        img2_lst = img2_lst[:new_len]

        assert len(img1_lst) == len(img2_lst)
        self.img1_dir = img1_dir
        self.img1_lst = img1_lst
        self.img2_lst = img2_lst
        self.img_hw = None if img_wh is None else img_wh[::-1]
        self.trans = (
            {"img": ToTensor(), "mask": lambda x: x, "flow": {}, "depth": lambda x: x}
            if img_wh is None
            else {
                "img": Compose([Resize(self.img_hw, antialias=True), ToTensor()]),
                "flow": {},
                "depth": Resize(self.img_hw, antialias=True),
                "mask": ResizeNumpy(self.img_hw[::-1]),
            }
        )

    def __len__(self) -> int:
        return len(self.img1_lst)

    def __getitem__(
        self, index: int
    ) -> Dict[str, torch.Tensor | Tuple[int, int] | Compose]:
        img_one_path = self.img1_lst[index]
        img_two_path = self.img2_lst[index]

        img1 = Image.open(img_one_path).convert("RGB")
        img2 = Image.open(img_two_path).convert("RGB")
        img_hw_orig = img1.size[::-1]

        img1 = self.trans["img"](img1)
        img2 = self.trans["img"](img2)

        return {"img1": img1, "img2": img2, "img_hw_orig": img_hw_orig}


class KITTI(KITTIBase):
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        return {"img1": sample["img1"], "img2": sample["img2"]}


class KITTIWithCalibration(KITTIBase):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Tuple[int, int] = [1242, 376],
        download: str = None,
    ) -> None:
        super().__init__(root, split, img_wh, download)

        assert self.set in [
            "2011",
            "2012",
        ], "Calibration matrices are only available in the ['2011', '2012'] variants"

        if self.set == "2011":
            cam_intrinsic_paths = [
                os.path.join(*path.split("/")[:-4], "calib_cam_to_cam.txt")
                for path in self.img1_lst
            ]
        else:
            cam_intrinsic_paths = [
                os.path.join(
                    *splitted_path[:-2],
                    "calib",
                    splitted_path[-1].split("_")[0] + ".txt",
                )
                for splitted_path in [path.split("/") for path in self.img1_lst]
            ]

        cam_intrinsic = list(set(cam_intrinsic_paths))
        cam_intrinsic = {
            path: get_scaled_intrinsic_matrix(path, zoom_x=1.0, zoom_y=1.0)
            for path in cam_intrinsic
        }
        cam_intrinsic = [cam_intrinsic[path] for path in cam_intrinsic_paths]
        self.cam_intrinsics = cam_intrinsic
        self.calc_intrinsics = {}

    def _get_intrinsics(self, index, img_hw_orig, resize_hw):
        cam_intrinsic = self.cam_intrinsics[index]
        key = str(
            (
                *cam_intrinsic.reshape(-1).tolist(),
                *img_hw_orig,
                *resize_hw,
            )
        )
        if key in self.calc_intrinsics:
            return self.calc_intrinsics[key]
        else:
            cam_intrinsic = rescale_intrinsics(cam_intrinsic, img_hw_orig, resize_hw)
            K, K_inv = get_intrinsics_per_scale(
                cam_intrinsic, scale=0
            )  # (3, 3), (3, 3)
            K, K_inv = (
                K[np.newaxis, ...],
                K_inv[np.newaxis, ...],
            )  # (1, 3, 3), (1, 3, 3)
            K, K_inv = torch.Tensor(K), torch.Tensor(K_inv)
            self.calc_intrinsics[key] = K, K_inv
            return K, K_inv

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        img_hw_orig = sample["img_hw_orig"]
        resize_hw = sample["img1"].shape[1:]

        K, K_inv = self._get_intrinsics(index, img_hw_orig, resize_hw)

        # return *imgs, K, K_inv
        return {
            "img1": sample["img1"],
            "img2": sample["img2"],
            "K": K,
            "K_inv": K_inv,
        }


class KITTIWithFlow(KITTIBase):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Tuple[int, int] = [1242, 376],
        download: str = None,
        noc_gt=True,
    ) -> None:
        super().__init__(root, split, img_wh, download)
        assert self.set in [
            "2012",
            "2015",
        ], "Flow maps are only available in KITTI 2012 and KITTI 2015"
        assert split == "train", "Flow maps are only available in the 'train' split"

        flow_dir = "flow_noc" if noc_gt else "flow_occ"
        self.flow_lst = [
            path.replace(self.img1_dir, flow_dir) for path in self.img1_lst
        ]

    def _transform_flow(self, flow: np.ndarray) -> torch.Tensor:
        flow = torch.FloatTensor(flow.transpose(2, 0, 1))
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        flow_path = self.flow_lst[index]

        flow, valid = sparse_flow_from_filepath(flow_path)
        with torch.no_grad():
            flow = self._transform_flow(flow)
            valid = self.trans["mask"](valid)

        return {
            "img1": sample["img1"],
            "img2": sample["img2"],
            "flow_gt": flow,
            "valid": valid,
            "occ_gt": 1 - valid,
        }


class KITTIWithDepth(KITTIBase):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_wh: Tuple[int, int] = [1242, 376],
        download: str = None,
    ) -> None:
        super().__init__(root, split, img_wh, download)
        assert self.set in [
            "2012",
            "2015",
            "2011",
        ], "Depth maps are only available in 2011, 2012, and 2015 variants"
        assert split == "train", "Depth maps are only available in the 'train' split"
        if self.set == "2011":
            self.img1_lst = [
                path
                for path in self.img1_lst
                if not ("2011_10_03" in path or "2011_09_30_drive_0027_sync" in path)
            ]
            self.velo = [
                path.replace(self.img1_dir, "velodyne_npy") for path in self.img1_lst
            ]
            self.velo = [path.replace("png", "npy") for path in self.velo]

        elif self.set == "2012":
            self.img1_lst = [path for path in self.img1_lst if path.endswith("10.png")]
            self.depth_lst = [
                path.replace(self.img1_dir, "disp_noc") for path in self.img1_lst
            ]

        else:
            self.img1_lst = [path for path in self.img1_lst if path.endswith("10.png")]
            self.depth_lst = [
                path.replace(self.img1_dir, "disp_noc_0") for path in self.img1_lst
            ]

    def load_saved_depth_map(self, file):
        # Load the saved depth map from file
        loaded_depth_map = np.load(file)
        # depth_map_key = "{}".format(file.split("/")[-1].split(".")[0])
        # loaded_depth_map = loaded_depth_map[depth_map_key]
        return loaded_depth_map

    def load_and_preprocess_depth_map(self, index):

        if self.set == "2011":
            depth_map = self.load_saved_depth_map(self.velo[index])
            return torch.FloatTensor(depth_map).unsqueeze(0)
        else:
            depth_path = self.depth_lst[index]
            depth_map = Image.open(depth_path)
            return torch.FloatTensor(
                np.array(depth_map).astype(np.float32) * (1.0 / 256.0)
            ).unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        trans = self.trans["depth"]
        depth_map = self.load_and_preprocess_depth_map(index)
        depth_map = trans(depth_map)

        return {"img": sample["img1"], "depth_map": depth_map}
