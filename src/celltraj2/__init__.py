"""celltraj2 frame-based microscopy trajectory analysis backend."""

from celltraj2.schema import (
    ChannelSpec,
    ImageSourceSpec,
    RoiBounds,
    RoiSpec,
    SegmentationRunSpec,
    TrajectoryMetadata,
)
from celltraj2.store import TrajectoryStore
from celltraj2.trajectory import Trajectory

__all__ = [
    "ChannelSpec",
    "ImageSourceSpec",
    "RoiBounds",
    "RoiSpec",
    "SegmentationRunSpec",
    "Trajectory",
    "TrajectoryMetadata",
    "TrajectoryStore",
]

__version__ = "0.1.0"
