"""User-facing Trajectory facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from celltraj2.schema import ImageSourceSpec, TrajectoryMetadata
from celltraj2.sources import ImageSource, image_source_from_spec
from celltraj2.store import TrajectoryStore


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
