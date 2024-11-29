"""
# ChildLearners

- The models that can participate in the main Learner (src.learner)
- Each expects a BackBone (src.models.backbone) model to use as it's encoder
- Main Learner is expected to combine these together, while having the shared encoder

## ContentLearner
Learns content information

## FlowLearner
Predicts the flow in a pyramid fashion

## DepthLearner
Predicts the depth in a pyramid fashion

"""

from .content import ContentLearner
from .flow import FlowLearner
from .depth import DepthLearner
from .cls import ClassLearner
from .segment import SegmentLearner

__all__ = [
    "ContentLearner",
    "FlowLearner",
    "DepthLearner",
    "ClassLearner",
    "SegmentLearner",
]
