"""User-facing Trajectory facade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from celltraj2.boundaries import (
    BoundaryGeometryResult,
    BoundaryLibraryResult,
    BoundaryLibraryView,
    BoundaryNeighborResult,
    BoundarySourceSpec,
)
from celltraj2.objects import ObjectIndexResult, index_object_set
from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.sources import ImageSource, image_source_from_spec
from celltraj2.store import TrajectoryStore


class ObjectSet:
    """Convenience view over one named object set in a trajectory H5."""

    def __init__(self, trajectory: "Trajectory", name: str) -> None:
        self.trajectory = trajectory
        self.name = name

    def index_observations(
        self,
        *,
        source_label_set: str | None = None,
        frames: Sequence[int] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectIndexResult:
        """Index source label frames into stable observations for this object set."""

        return index_object_set(
            self.trajectory,
            self.name,
            source_label_set=source_label_set,
            frames=frames,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def read_observations(self) -> Any:
        return self.trajectory.store.read_observations(self.name)

    def read_observations_schema(self) -> Any:
        return self.trajectory.store.read_observations_schema(self.name)

    def has_observations(self) -> bool:
        return self.trajectory.store.has_observations(self.name)

    def observation_count(self) -> int:
        return self.trajectory.store.observation_count(self.name)

    def read_observation(self, observation_id: int) -> Any:
        value = int(observation_id)
        if value < 1:
            raise ValueError("observation_id is one-based and must be >= 1")
        observations = self.read_observations()
        index = value - 1
        if index >= int(observations.shape[0]):
            raise IndexError(f"observation_id {value} is outside 1..{int(observations.shape[0])}")
        return observations[index]

    def read_lookup_frame(self, frame: int) -> Any:
        return self.trajectory.store.read_observation_lookup_frame(self.name, frame)

    def lookup_frames(self) -> list[int]:
        return self.trajectory.store.list_observation_lookup_frames(self.name)

    def observation_id_for_label(self, *, frame: int, label_id: int) -> int:
        lookup = self.read_lookup_frame(frame)
        value = int(label_id)
        if value < 0 or value >= int(lookup.shape[0]):
            return 0
        return int(lookup[value])

    def extract_features(
        self,
        spec: Any,
        *,
        frames: Sequence[int] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Extract and optionally save a row-aligned feature set for this object set."""

        from celltraj2.features import FeatureSetSpec, extract_feature_set

        if isinstance(spec, Mapping):
            payload = dict(spec)
            payload.setdefault("object_set", self.name)
            feature_spec = FeatureSetSpec.from_dict(payload)
        elif isinstance(spec, FeatureSetSpec):
            feature_spec = spec
            if feature_spec.object_set != self.name:
                raise ValueError(f"Feature spec object_set {feature_spec.object_set!r} does not match {self.name!r}")
        else:
            raise TypeError("spec must be a FeatureSetSpec or mapping")
        return extract_feature_set(
            self.trajectory,
            feature_spec,
            frames=frames,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def read_features(self, feature_set: str) -> Any:
        return self.trajectory.store.read_feature_values(self.name, feature_set)

    def read_feature_schema(self, feature_set: str) -> Any:
        return self.trajectory.store.read_feature_schema(self.name, feature_set)

    def feature_sets(self) -> list[str]:
        return self.trajectory.store.list_feature_sets(self.name)

    def track_minimum_centroid_distance(
        self,
        *,
        max_distance: float,
        track_set: str = "centroid_mindist",
        coordinate_scale: Sequence[float] | None = None,
        registration_set: str | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Track observations with the legacy-compatible nearest-centroid rule."""

        from celltraj2.tracking import track_minimum_centroid_distance

        return track_minimum_centroid_distance(
            self.trajectory,
            self.name,
            max_distance=max_distance,
            track_set=track_set,
            coordinate_scale=coordinate_scale,
            registration_set=registration_set,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def track_sets(self) -> list[str]:
        return self.trajectory.store.list_track_sets(self.name)

    def read_tracks(self, track_set: str) -> Any:
        return self.trajectory.store.read_track_graph(self.name, track_set)

    def build_boundary_library(
        self,
        boundary_set: str = "native",
        *,
        frames: Sequence[int] | None = None,
        coordinate_scale: Sequence[float] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> BoundaryLibraryResult:
        """Build native boundaries for this indexed object set."""

        return self.trajectory.build_boundary_library(
            boundary_set,
            object_set=self.name,
            frames=frames,
            coordinate_scale=coordinate_scale,
            overwrite=overwrite,
            save_outputs=save_outputs,
            metadata=metadata,
        )

    def track_minimum_boundary_ot_cost(
        self,
        *,
        boundary_set: str,
        max_distance: float,
        ot_cost_cutoff: float = float("inf"),
        track_set: str = "boundary_ot",
        motion_set: str | None = None,
        registration_set: str | None = None,
        ot_method: str = "emd",
        sinkhorn_regularization: float = 0.05,
        max_boundary_points: int | None = 512,
        mass_tolerance: float = 1e-12,
        save_motion: bool = True,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Track this object set using registered boundary OT costs."""

        from celltraj2.tracking import track_minimum_boundary_ot_cost

        return track_minimum_boundary_ot_cost(
            self.trajectory,
            self.name,
            boundary_set=boundary_set,
            max_distance=max_distance,
            ot_cost_cutoff=ot_cost_cutoff,
            track_set=track_set,
            motion_set=motion_set,
            registration_set=registration_set,
            ot_method=ot_method,
            sinkhorn_regularization=sinkhorn_regularization,
            max_boundary_points=max_boundary_points,
            mass_tolerance=mass_tolerance,
            save_motion=save_motion,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )


class Trajectory:
    """Open and interact with one celltraj2 per-ROI analysis H5."""

    def __init__(self, path: str | Path, mode: str = "r+") -> None:
        self.store = TrajectoryStore.open(path, mode=mode)
        self.path = self.store.path
        self.metadata = self.store.read_metadata()
        self.image_source = self._load_image_source()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        metadata: TrajectoryMetadata,
        overwrite: bool = False,
        site_manifest: dict[str, Any] | None = None,
        roi_record: dict[str, Any] | None = None,
    ) -> "Trajectory":
        store = TrajectoryStore.create(
            path,
            metadata=metadata,
            site_manifest=site_manifest,
            roi_record=roi_record,
            overwrite=overwrite,
        )
        store.close()
        return cls(path, mode="r+")

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Trajectory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _load_image_source(self) -> ImageSource:
        spec = self.metadata.image_source
        if spec is None:
            try:
                spec = self.store.read_image_source()
            except Exception:
                spec = ImageSourceSpec(source_type="embedded_h5")
        return image_source_from_spec(spec, store=self.store)

    def get_image_data(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        """Return image data for one local one-based frame."""

        return self.image_source.read_frame(frame=frame, channels=channels, z=z, y=y, x=x)

    def channel_index_map(self) -> dict[int, int] | None:
        """Return raw source-channel index to local image C-axis index mapping."""

        return self.image_source.channel_index_map()

    def frame_axes(self, ndim: int | None = None) -> tuple[str, ...]:
        """Return axes for the most recently read image frame."""

        return self.image_source.frame_axes(ndim)

    def write_raw_frame(self, frame: int, image: Any, *, overwrite: bool = False) -> str:
        return self.store.write_raw_frame(frame, image, overwrite=overwrite)

    def write_label_frame(self, label_set: str, frame: int, labels: Any, *, overwrite: bool = False) -> str:
        return self.store.write_label_frame(label_set, frame, labels, overwrite=overwrite)

    def read_label_frame(self, label_set: str, frame: int) -> Any:
        return self.store.read_label_frame(label_set, frame)

    def write_mask_frame(self, mask_set: str, frame: int, mask: Any, *, overwrite: bool = False) -> str:
        return self.store.write_mask_frame(mask_set, frame, mask, overwrite=overwrite)

    def read_mask_frame(self, mask_set: str, frame: int) -> Any:
        return self.store.read_mask_frame(mask_set, frame)

    def label_frames(self, label_set: str) -> list[int]:
        return self.store.list_label_frames(label_set)

    def mask_frames(self, mask_set: str) -> list[int]:
        return self.store.list_mask_frames(mask_set)

    def label_sets(self) -> list[str]:
        return self.store.list_label_sets()

    def mask_sets(self) -> list[str]:
        return self.store.list_mask_sets()

    def register_global_translation(
        self,
        object_set: str,
        *,
        registration_set: str = "global_registration",
        max_shift_per_frame: Sequence[float] | float = 10.0,
        grid_step: Sequence[float] | float = 1.0,
        coordinate_scale: Sequence[float] | None = None,
        distance_unit: str | None = None,
        contact_transform: bool = False,
        overwrite: bool = False,
        save_outputs: bool = True,
        set_active: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Register ROI frames from one indexed object's centroid point clouds."""

        from celltraj2.registration import register_global_translation

        return register_global_translation(
            self,
            object_set,
            registration_set=registration_set,
            max_shift_per_frame=max_shift_per_frame,
            grid_step=grid_step,
            coordinate_scale=coordinate_scale,
            distance_unit=distance_unit,
            contact_transform=contact_transform,
            overwrite=overwrite,
            save_outputs=save_outputs,
            set_active=set_active,
            run_id=run_id,
            metadata=metadata,
        )

    def registration_sets(self) -> list[str]:
        return self.store.list_registration_sets()

    def read_registration(self, registration_set: str) -> Any:
        return self.store.read_registration_set(registration_set)

    def active_registration_name(self) -> str | None:
        return self.store.active_registration_name()

    def active_registration(self) -> Any:
        return self.store.read_active_registration()

    def set_active_registration(self, registration_set: str) -> str:
        return self.store.set_active_registration(registration_set, reason="trajectory_api")

    def registration_runs(self) -> list[str]:
        return self.store.list_registration_runs()

    def object_set(self, object_set: str) -> ObjectSet:
        return ObjectSet(self, object_set)

    def object_sets(self) -> list[str]:
        return self.store.list_object_sets()

    def build_boundary_library(
        self,
        boundary_set: str,
        *,
        sources: Sequence[BoundarySourceSpec | Mapping[str, Any]] | None = None,
        object_set: str | None = None,
        frames: Sequence[int] | None = None,
        coordinate_scale: Sequence[float] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> BoundaryLibraryResult:
        """Build a native boundary library from indexed objects, labels, or masks."""

        from celltraj2.boundaries import build_boundary_library

        return build_boundary_library(
            self,
            boundary_set,
            sources=sources,
            object_set=object_set,
            frames=frames,
            coordinate_scale=coordinate_scale,
            overwrite=overwrite,
            save_outputs=save_outputs,
            metadata=metadata,
        )

    def boundary_library(self, boundary_set: str) -> BoundaryLibraryView:
        return self.store.boundary_library(boundary_set)

    def boundary_sets(self) -> list[str]:
        return self.store.list_boundary_sets()

    def compute_boundary_geometry(
        self,
        boundary_set: str,
        *,
        geometry_set: str = "surface_v1",
        knn: int = 40,
        backend: str = "auto",
        overwrite: bool = False,
        save_outputs: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> BoundaryGeometryResult:
        from celltraj2.boundaries import compute_boundary_geometry

        return compute_boundary_geometry(
            self,
            boundary_set,
            geometry_set=geometry_set,
            knn=knn,
            backend=backend,  # type: ignore[arg-type]
            overwrite=overwrite,
            save_outputs=save_outputs,
            metadata=metadata,
        )

    def compute_boundary_neighbors(
        self,
        boundary_set: str,
        *,
        neighbor_set: str = "nearest_external_v1",
        k: int = 1,
        source_entity_ids: Sequence[int] | None = None,
        target_entity_ids: Sequence[int] | None = None,
        same_frame: bool = True,
        exclude_same_entity: bool = True,
        max_distance: float | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> BoundaryNeighborResult:
        from celltraj2.boundaries import compute_boundary_neighbors

        return compute_boundary_neighbors(
            self,
            boundary_set,
            neighbor_set=neighbor_set,
            k=k,
            source_entity_ids=source_entity_ids,
            target_entity_ids=target_entity_ids,
            same_frame=same_frame,
            exclude_same_entity=exclude_same_entity,
            max_distance=max_distance,
            overwrite=overwrite,
            save_outputs=save_outputs,
            metadata=metadata,
        )

    def write_boundary_entity_attributes(
        self,
        boundary_set: str,
        attribute_set: str,
        values: Any,
        schema: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> str:
        return self.store.write_boundary_entity_attributes(
            boundary_set,
            attribute_set,
            values,
            schema,
            overwrite=overwrite,
        )

    def write_observations(
        self,
        object_set: str,
        observations: Any,
        schema: dict[str, Any],
        *,
        source_label_set: str | None = None,
        overwrite: bool = False,
    ) -> str:
        return self.store.write_observations(
            object_set,
            observations,
            schema,
            source_label_set=source_label_set,
            overwrite=overwrite,
        )

    def read_observations(self, object_set: str) -> Any:
        return self.store.read_observations(object_set)

    def index_observations(
        self,
        object_set: str,
        *,
        source_label_set: str | None = None,
        frames: Sequence[int] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ObjectIndexResult:
        return self.object_set(object_set).index_observations(
            source_label_set=source_label_set,
            frames=frames,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def write_segmentation_run(self, run_id: str, data: dict[str, Any], *, overwrite: bool = True) -> str:
        return self.store.write_segmentation_run(run_id, data, overwrite=overwrite)

    def write_segmentation_frame_result(
        self,
        run_id: str,
        frame: int,
        data: dict[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        return self.store.write_segmentation_frame_result(run_id, frame, data, overwrite=overwrite)

    def segmentation_runs(self) -> list[str]:
        return self.store.list_segmentation_runs()

    def object_indexing_runs(self) -> list[str]:
        return self.store.list_object_indexing_runs()

    def write_feature_set(
        self,
        object_set: str,
        feature_set: str,
        values: Any,
        schema: dict[str, Any],
        *,
        overwrite: bool = False,
        qc: dict[str, Any] | None = None,
    ) -> str:
        return self.store.write_feature_set(object_set, feature_set, values, schema, overwrite=overwrite, qc=qc)

    def read_features(self, object_set: str, feature_set: str) -> Any:
        return self.store.read_feature_values(object_set, feature_set)

    def read_feature_schema(self, object_set: str, feature_set: str) -> Any:
        return self.store.read_feature_schema(object_set, feature_set)

    def feature_sets(self, object_set: str) -> list[str]:
        return self.store.list_feature_sets(object_set)

    def track_minimum_centroid_distance(
        self,
        object_set: str,
        *,
        max_distance: float,
        track_set: str = "centroid_mindist",
        coordinate_scale: Sequence[float] | None = None,
        registration_set: str | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.object_set(object_set).track_minimum_centroid_distance(
            max_distance=max_distance,
            track_set=track_set,
            coordinate_scale=coordinate_scale,
            registration_set=registration_set,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def track_sets(self, object_set: str) -> list[str]:
        return self.store.list_track_sets(object_set)

    def read_tracks(self, object_set: str, track_set: str) -> Any:
        return self.store.read_track_graph(object_set, track_set)

    def tracking_runs(self) -> list[str]:
        return self.store.list_tracking_runs()

    def track_minimum_boundary_ot_cost(
        self,
        object_set: str,
        *,
        boundary_set: str,
        max_distance: float,
        ot_cost_cutoff: float = float("inf"),
        track_set: str = "boundary_ot",
        motion_set: str | None = None,
        registration_set: str | None = None,
        ot_method: str = "emd",
        sinkhorn_regularization: float = 0.05,
        max_boundary_points: int | None = 512,
        mass_tolerance: float = 1e-12,
        save_motion: bool = True,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return self.object_set(object_set).track_minimum_boundary_ot_cost(
            boundary_set=boundary_set,
            max_distance=max_distance,
            ot_cost_cutoff=ot_cost_cutoff,
            track_set=track_set,
            motion_set=motion_set,
            registration_set=registration_set,
            ot_method=ot_method,
            sinkhorn_regularization=sinkhorn_regularization,
            max_boundary_points=max_boundary_points,
            mass_tolerance=mass_tolerance,
            save_motion=save_motion,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def extract_features(
        self,
        spec: Any,
        *,
        frames: Sequence[int] | None = None,
        overwrite: bool = False,
        save_outputs: bool = True,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        from celltraj2.features import extract_feature_set

        return extract_feature_set(
            self,
            spec,
            frames=frames,
            overwrite=overwrite,
            save_outputs=save_outputs,
            run_id=run_id,
            metadata=metadata,
        )

    def write_feature_extraction_run(self, run_id: str, data: dict[str, Any], *, overwrite: bool = True) -> str:
        return self.store.write_feature_extraction_run(run_id, data, overwrite=overwrite)

    def feature_extraction_runs(self) -> list[str]:
        return self.store.list_feature_extraction_runs()
