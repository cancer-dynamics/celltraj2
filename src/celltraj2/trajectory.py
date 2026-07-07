"""User-facing Trajectory facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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

    def object_set(self, object_set: str) -> ObjectSet:
        return ObjectSet(self, object_set)

    def object_sets(self) -> list[str]:
        return self.store.list_object_sets()

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
