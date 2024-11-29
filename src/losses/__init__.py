# from .depreciated_flow_loss import FlowLoss
from .content_loss import ContentLoss
from .ssl_flow_loss import SSLFlowLoss
from .gt_flow_loss import GTFlowLoss
from .depth_loss import DepthLoss  # TODO: make this SSLDepthLoss
from .gt_depth_loss import GTDepthLoss
from .segmentation_loss import SegmentationLoss
from .vicreg_loss import VICRegLoss

__all__ = [
    "ContentLoss",
    "SSLFlowLoss",
    "GTFlowLoss",
    "DepthLoss",
    "GTDepthLoss",
    "SegmentationLoss",
    "VICRegLoss",
]
