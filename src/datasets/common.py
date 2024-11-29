import os
import shutil


def get_valid_ds_len(cur_len: int):
    """
    Returns a valid length for a dataset when the current length is given.
    Since our methods uses vicreg always, there must be alteast two elements in each batch
    Assumption: batch size is always an even number
    """
    return cur_len // 2 * 2


def move_subdir_up(root: str, subdir: str) -> None:
    subdirectory_path = os.path.join(root, subdir)
    subdirectory_contents = os.listdir(subdirectory_path)
    for item in subdirectory_contents:
        item_path = os.path.join(subdirectory_path, item)
        destination_path = os.path.join(root, item)
        shutil.move(item_path, destination_path)
    os.rmdir(subdirectory_path)
