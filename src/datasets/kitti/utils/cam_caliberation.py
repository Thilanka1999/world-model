import numpy as np
import copy

# Obtained from: https://github.com/B1ueber2y/TrianFlow/blob/master/core/evaluation/evaluate_flow.py


def get_scaled_intrinsic_matrix(calib_file, zoom_x=1.0, zoom_y=1.0):
    intrinsics = load_intrinsics_raw(calib_file)
    intrinsics = scale_intrinsics(intrinsics, zoom_x, zoom_y)

    intrinsics[0, 1] = 0.0
    intrinsics[1, 0] = 0.0
    intrinsics[2, 0] = 0.0
    intrinsics[2, 1] = 0.0
    return intrinsics


def load_intrinsics_raw(calib_file):
    filedata = read_raw_calib_file(calib_file)
    if "P_rect_02" in filedata:
        P_rect = filedata["P_rect_02"]
    else:
        P_rect = filedata["P2"]
    P_rect = np.reshape(P_rect, (3, 4))
    intrinsics = P_rect[:3, :3]
    return intrinsics


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()]).astype(
                    np.float32
                )
            except ValueError:
                pass
    return data


def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def rescale_intrinsics(K, img_hw_orig, img_hw_new):
    K[0, :] = K[0, :] * img_hw_new[1] / img_hw_orig[1]
    K[1, :] = K[1, :] * img_hw_new[0] / img_hw_orig[0]
    return K


def get_intrinsics_per_scale(K, scale):
    K_new = copy.deepcopy(K)
    K_new[0, :] = K_new[0, :] / (2**scale)
    K_new[1, :] = K_new[1, :] / (2**scale)
    K_new_inv = np.linalg.inv(K_new)
    return K_new, K_new_inv
