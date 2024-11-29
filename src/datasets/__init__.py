from .kitti import KITTI, KITTIWithCalibration, KITTIWithFlow, KITTIWithDepth
from .imagenet import ImageNetVICReg, ImageNetClassify
from .coco import COCOSegment
from .flyingthings import FlyingThings
from .flyingchairs import FlyingChairs
from .hd1k import HD1K
from .mpi_sintel import MPISintel
from .cityscape import Cityscape
from .davis2017 import Davis
from .pascalvoc import PascalVOC
from .ade20k import ADE20k

__all__ = [
    "KITTI",
    "KITTIWithCalibration",
    "KITTIWithFlow",
    "KITTIWithDepth",
    "ImageNetVICReg",
    "ImageNetClassify",
    "COCOSegment",
    "FlyingThings",
    "FlyingChairs",
    "HD1K",
    "MPISintel",
    "Cityscape",
    "Davis",
    "PascalVOC",
    "ADE20k",
]
