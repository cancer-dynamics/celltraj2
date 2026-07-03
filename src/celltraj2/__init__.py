"""celltraj2 frame-based microscopy trajectory analysis backend."""

from celltraj2.batch import (
    BatchSegmentationSummary,
    JsonlReporter,
    SegmentationBatchJob,
    SegmentationFileJob,
    SegmentationResult,
    run_batch_segmentation,
)
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
    "BatchSegmentationSummary",
    "ChannelSpec",
    "ImageSourceSpec",
    "JsonlReporter",
    "RoiBounds",
    "RoiSpec",
    "SegmentationBatchJob",
    "SegmentationFileJob",
    "SegmentationResult",
    "SegmentationRunSpec",
    "Trajectory",
    "TrajectoryMetadata",
    "TrajectoryStore",
    "run_batch_segmentation",
]

__version__ = "0.1.0"
