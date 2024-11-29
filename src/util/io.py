import numpy as np
import cv2
from typing import Tuple
from io import BytesIO
import re


def sparse_flow_from_filepath(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the optical flow in KITTI datasets from bytes.

    Credits: based on mmvc.sparse_flow_from_bytes

    This function is modified from RAFT load the `KITTI datasets
    <https://github.com/princeton-vl/RAFT/blob/224320502d66c356d88e6c712f38129e60661e80/core/utils/frame_utils.py#L102>`_.

    Args:
        filepath (str): Path to the flow file.

    Returns:
        Tuple(ndarray, ndarray): Loaded optical flow with the shape (H, W, 2)
        and flow valid mask with the shape (H, W).
    """
    with open(filepath, "rb") as handler:
        content = handler.read()
    content = np.frombuffer(content, np.uint8)
    flow = cv2.imdecode(content, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def flow_from_filepath(filepath: str) -> np.ndarray:
    """Read dense optical flow from bytes.

    Credits: based on mmvc.flow_from_bytes

    .. note::
        This load optical flow function works for FlyingChairs, FlyingThings3D,
        Sintel, FlyingChairsOcc datasets, but cannot load the data from
        ChairsSDHom.

    Args:
        filepath (str): Path to the flow file.

    Returns:
        ndarray: Loaded optical flow with the shape (H, W, 2).
    """
    suffix = filepath[-3:]
    with open(filepath, "rb") as handler:
        content = handler.read()

    assert suffix in ("flo", "pfm"), (
        "suffix of flow file must be `flo` " f"or `pfm`, but got {suffix}"
    )

    if suffix == "flo":
        return flo_from_bytes(content)
    else:
        return pfm_from_bytes(content)


def flo_from_bytes(content: bytes):
    """Decode bytes based on flo file.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: Loaded optical flow with the shape (H, W, 2).
    """

    # header in first 4 bytes
    header = content[:4]
    if header != b"PIEH":
        raise Exception("Flow file header does not contain PIEH")
    # width in second 4 bytes
    width = np.frombuffer(content[4:], np.int32, 1).squeeze()
    # height in third 4 bytes
    height = np.frombuffer(content[8:], np.int32, 1).squeeze()
    # after first 12 bytes, all bytes are flow
    flow = np.frombuffer(content[12:], np.float32, width * height * 2).reshape(
        (height, width, 2)
    )

    return flow


def pfm_from_bytes(content: bytes) -> np.ndarray:
    """Load the file with the suffix '.pfm'.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: The loaded data
    """

    file = BytesIO(content)

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.frombuffer(file.read(), endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[:, :, :-1]


def occ_from_filepath(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as handler:
        content = handler.read()
    img_np = np.frombuffer(content, np.uint8)
    occ = (cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE) / 255).astype(np.float32)
    return occ
