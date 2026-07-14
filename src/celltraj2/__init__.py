"""celltraj2 frame-based microscopy trajectory analysis backend."""

from celltraj2.batch import (
    BatchSegmentationSummary,
    JsonlReporter,
    SegmentationBatchJob,
    SegmentationFileJob,
    SegmentationResult,
    run_batch_segmentation,
)
from celltraj2.boundaries import (
    BoundaryGeometryResult,
    BoundaryLibraryResult,
    BoundaryLibraryView,
    BoundaryNeighborResult,
    BoundarySourceSpec,
    BoundaryTransportPlan,
    build_boundary_library,
    compute_boundary_geometry,
    compute_boundary_neighbors,
    optimal_transport_plan,
)
from celltraj2.feature_extraction import (
    BatchFeatureExtractionSummary,
    FeatureExtractionBatchJob,
    FeatureExtractionFileJob,
    run_batch_feature_extraction,
)
from celltraj2.features import (
    FeatureExtractionResult,
    FeatureSetSpec,
    extract_feature_set,
    regionprops_v1_spec,
    site_signaling_v1_spec,
)
from celltraj2.object_indexing import (
    BatchObjectIndexSummary,
    ObjectIndexBatchJob,
    ObjectIndexFileJob,
    run_batch_object_indexing,
)
from celltraj2.objects import ObjectIndexResult, index_object_set
from celltraj2.registration import (
    RegistrationResult,
    RegistrationSet,
    estimate_pair_translation,
    register_global_translation,
)
from celltraj2.registration_batch import (
    BatchRegistrationSummary,
    RegistrationBatchJob,
    RegistrationFileJob,
    run_batch_registration,
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
from celltraj2.tracking import (
    SparseAdjacency,
    TrackGraph,
    TrackingResult,
    track_minimum_boundary_ot_cost,
    track_minimum_centroid_distance,
)
from celltraj2.tracking_batch import (
    BatchTrackingSummary,
    TrackingBatchJob,
    TrackingFileJob,
    run_batch_tracking,
)
from celltraj2.trajectory import Trajectory

__all__ = [
    "BatchSegmentationSummary",
    "BatchTrackingSummary",
    "BatchFeatureExtractionSummary",
    "BatchObjectIndexSummary",
    "BatchRegistrationSummary",
    "BoundaryGeometryResult",
    "BoundaryLibraryResult",
    "BoundaryLibraryView",
    "BoundaryNeighborResult",
    "BoundarySourceSpec",
    "BoundaryTransportPlan",
    "ChannelSpec",
    "FeatureExtractionBatchJob",
    "FeatureExtractionFileJob",
    "FeatureExtractionResult",
    "FeatureSetSpec",
    "ImageSourceSpec",
    "JsonlReporter",
    "ObjectIndexBatchJob",
    "ObjectIndexFileJob",
    "ObjectIndexResult",
    "RoiBounds",
    "RoiSpec",
    "RegistrationBatchJob",
    "RegistrationFileJob",
    "RegistrationResult",
    "RegistrationSet",
    "SegmentationBatchJob",
    "SegmentationFileJob",
    "SegmentationResult",
    "SegmentationRunSpec",
    "SparseAdjacency",
    "TrackGraph",
    "TrackingBatchJob",
    "TrackingFileJob",
    "TrackingResult",
    "Trajectory",
    "TrajectoryMetadata",
    "TrajectoryStore",
    "build_boundary_library",
    "compute_boundary_geometry",
    "compute_boundary_neighbors",
    "extract_feature_set",
    "estimate_pair_translation",
    "index_object_set",
    "optimal_transport_plan",
    "regionprops_v1_spec",
    "register_global_translation",
    "run_batch_registration",
    "run_batch_object_indexing",
    "run_batch_feature_extraction",
    "run_batch_segmentation",
    "run_batch_tracking",
    "site_signaling_v1_spec",
    "track_minimum_boundary_ot_cost",
    "track_minimum_centroid_distance",
]

__version__ = "0.1.0"
