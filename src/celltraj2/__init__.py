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
    resolve_boundary_source_ids,
)
from celltraj2.boundary_batch import (
    BatchBoundarySummary,
    BoundaryBatchJob,
    BoundaryFileJob,
    BoundaryGeometryJob,
    BoundaryNeighborJob,
    run_batch_boundaries,
)
from celltraj2.boundary_features import (
    boundary_multipole_magnitudes,
    compute_boundary_feature_frame,
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
from celltraj2.h5_access import H5AccessTimeout, H5DependencyChangedError, open_h5
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
from celltraj2.surface_motion_batch import (
    BatchSurfaceMotionSummary,
    SurfaceMotionBatchJob,
    SurfaceMotionFileJob,
    run_batch_surface_motion,
)
from celltraj2.tracking import (
    BoundaryMotionResult,
    SparseAdjacency,
    TrackGraph,
    TrackingResult,
    compute_boundary_motion,
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
    "BatchBoundarySummary",
    "BatchTrackingSummary",
    "BatchSurfaceMotionSummary",
    "BatchFeatureExtractionSummary",
    "BatchObjectIndexSummary",
    "BatchRegistrationSummary",
    "BoundaryGeometryResult",
    "BoundaryBatchJob",
    "BoundaryFileJob",
    "BoundaryGeometryJob",
    "BoundaryLibraryResult",
    "BoundaryLibraryView",
    "BoundaryNeighborResult",
    "BoundaryNeighborJob",
    "BoundaryMotionResult",
    "BoundarySourceSpec",
    "BoundaryTransportPlan",
    "ChannelSpec",
    "FeatureExtractionBatchJob",
    "FeatureExtractionFileJob",
    "FeatureExtractionResult",
    "FeatureSetSpec",
    "ImageSourceSpec",
    "H5AccessTimeout",
    "H5DependencyChangedError",
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
    "SurfaceMotionBatchJob",
    "SurfaceMotionFileJob",
    "TrackGraph",
    "TrackingBatchJob",
    "TrackingFileJob",
    "TrackingResult",
    "Trajectory",
    "TrajectoryMetadata",
    "TrajectoryStore",
    "build_boundary_library",
    "boundary_multipole_magnitudes",
    "compute_boundary_feature_frame",
    "compute_boundary_geometry",
    "compute_boundary_neighbors",
    "compute_boundary_motion",
    "extract_feature_set",
    "estimate_pair_translation",
    "index_object_set",
    "optimal_transport_plan",
    "open_h5",
    "resolve_boundary_source_ids",
    "regionprops_v1_spec",
    "register_global_translation",
    "run_batch_registration",
    "run_batch_boundaries",
    "run_batch_object_indexing",
    "run_batch_feature_extraction",
    "run_batch_segmentation",
    "run_batch_tracking",
    "run_batch_surface_motion",
    "site_signaling_v1_spec",
    "track_minimum_boundary_ot_cost",
    "track_minimum_centroid_distance",
]

__version__ = "0.1.0"
