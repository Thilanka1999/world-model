import os
from .common import get_valid_ds_len
import glob
from .base import FlowBase


class FlyingThings(FlowBase):
    sparse = False
    urls = [
        {
            "url": "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar",  # 42.1GB
            "archive": True,
            "move_up": "frames_finalpass",
        },
    ]

    def _get_paths(self, root, split, get_path_kwargs):
        split = "TRAIN" if split == "train" else "TEST"
        img1_paths = sorted(glob.glob(os.path.join(root, split, "*/*/left/*.png")))
        new_len = get_valid_ds_len(len(img1_paths))
        img1_paths = img1_paths[:new_len]

        img2_paths = [p.replace("/left/", "/right/") for p in img1_paths]
        flow_paths = [None] * len(img1_paths)
        occs_paths = [None] * len(img1_paths)

        return img1_paths, img2_paths, flow_paths, occs_paths


# class FlyingThings(Dataset):
#     valid_splits = ["train", "val"]
#     urls = {
#         "img_archive": "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar"  # 42.1GB
#     }

#     def download(self, root: str) -> None:
#         download_and_extract_archive(
#             self.urls["img_archive"], root, remove_finished=True
#         )
#         # {root}/frames_finalpass/* ->{root}/*
#         move_subdir_up(root, "frames_finalpass")

#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_wh: Sequence[int] = flow_img_wh,
#         download: bool = False,
#     ) -> None:
#         if not os.path.exists(root):
#             if download:
#                 self.download(root)
#             else:
#                 raise FileNotFoundError(f"No such directory as {root}")
#         if split not in self.valid_splits:
#             raise ValueError(f"Valid splits are {self.valid_splits}")
#         split_dir = "TRAIN" if split == "train" else "TEST"

#         left_img_paths = []
#         right_img_paths = []

#         split_path = os.path.abspath(os.path.join(root, split_dir))
#         img_sub_folders = sorted(os.listdir(split_path))

#         for sub in img_sub_folders:
#             ims = sorted(os.listdir(os.path.join(split_path, sub)))
#             for im in ims:
#                 l_im_path = os.path.join(os.path.join(split_path, sub), im, "left")
#                 r_im_path = os.path.join(os.path.join(split_path, sub), im, "right")

#                 left_img_paths.extend(
#                     os.path.join(l_im_path, i) for i in os.listdir(l_im_path)
#                 )
#                 right_img_paths.extend(
#                     os.path.join(r_im_path, i) for i in os.listdir(r_im_path)
#                 )

#         new_len = get_valid_ds_len(len(left_img_paths))
#         left_img_paths = left_img_paths[:new_len]
#         right_img_paths = right_img_paths[:new_len]

#         self.left_img_paths = left_img_paths
#         self.right_img_paths = right_img_paths
#         self.img_hw = img_wh[::-1]

#     def transform_(self, img_one):
#         target_aspect_ratio = self.img_hw[1] / self.img_hw[0]
#         img_hw_orig = img_one.size[::-1]
#         orig_aspect_ratio = img_hw_orig[1] / img_hw_orig[0]

#         if target_aspect_ratio > orig_aspect_ratio:
#             resize_hw = [
#                 round(self.img_hw[1] / img_hw_orig[1] * img_hw_orig[0]),
#                 self.img_hw[1],
#             ]
#         else:
#             resize_hw = [
#                 self.img_hw[0],
#                 round(self.img_hw[0] / img_hw_orig[0] * img_hw_orig[1]),
#             ]
#         trans = Compose([Resize(resize_hw), CenterCrop(self.img_hw)])
#         return trans(img_one)

#     def __len__(self) -> int:
#         return len(self.left_img_paths)

#     def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
#         left_im = self.left_img_paths[index]
#         right_im = self.right_img_paths[index]
#         img1 = Image.open(left_im).convert("RGB")
#         img2 = Image.open(right_im).convert("RGB")
#         img1 = torch.FloatTensor(
#             np.array(self.transform_(img1)).transpose(2, 0, 1).astype(np.float32)
#         ) * (1.0 / 255.0)
#         img2 = torch.FloatTensor(
#             np.array(self.transform_(img2)).transpose(2, 0, 1).astype(np.float32)
#         ) * (1.0 / 255.0)

#         return {"img1": img1, "img2": img2}
