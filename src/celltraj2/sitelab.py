"""SITE handoff helpers.

These helpers intentionally avoid importing ``sitelab``. They work with the
JSON files SITE already writes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from celltraj2.schema import (
    ImageSourceSpec,
    RoiSpec,
    TrajectoryMetadata,
    channels_from_site,
)
from celltraj2.store import TrajectoryStore


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def infer_project_root_from_roi_json(roi_json_path: str | Path) -> Path:
    """Return the SITE project root for a ROI JSON path when possible."""

    path = Path(roi_json_path)
    if path.parent.name == "rois":
        return path.parent.parent
    return path.parent


def dataset_id_from_roi_json(roi_json_path: str | Path, roi_set: Mapping[str, Any] | None = None) -> str:
    """Return the dataset id/stem associated with a SITE ROI JSON."""

    if roi_set is not None and roi_set.get("dataset_id"):
        return str(roi_set["dataset_id"])
    name = Path(roi_json_path).name
    if name.endswith(".rois.json"):
        return name[: -len(".rois.json")]
    return Path(name).stem


def default_cell_file_path(project_root: str | Path, dataset_id: str, roi_id: str) -> Path:
    """Return the canonical per-ROI celltraj2 H5 path for SITE projects."""

    return Path(project_root) / "cell_files" / str(dataset_id) / f"{roi_id}.ct2.h5"


def find_roi_record(roi_set: Mapping[str, Any], roi_id: str) -> dict[str, Any]:
    for roi in roi_set.get("rois", []) or []:
        if str(roi.get("roi_id")) == str(roi_id):
            return dict(roi)
    raise KeyError(f"ROI id not found: {roi_id}")


def resolve_site_path(path: str | Path | None, *, project_root: str | Path) -> Path | None:
    if path in (None, ""):
        return None
    value = Path(path)
    return value if value.is_absolute() else Path(project_root) / value


def frame_count_from_roi(roi: Mapping[str, Any], roi_set: Mapping[str, Any]) -> int:
    """Return local frame count for one ROI, treating snapshots as one frame."""

    t_start = int(roi.get("time_start") or 0)
    t_stop = roi.get("time_stop")
    if t_stop is None:
        sizes = dict(roi_set.get("source_sizes", {}) or {})
        t_stop = sizes.get("T", 1)
    try:
        count = int(t_stop) - t_start
    except Exception:
        count = 1
    return max(1, count)


def image_source_from_site_roi(
    *,
    roi_set: Mapping[str, Any],
    roi_record: Mapping[str, Any],
    project_root: str | Path,
) -> ImageSourceSpec:
    """Create an image source spec from SITE ROI JSON fields."""

    roi = RoiSpec.from_dict(
        {
            **dict(roi_record),
            "dataset_id": roi_set.get("dataset_id"),
            "source_path": roi_set.get("source_path"),
        }
    )
    storage_mode = str(roi_record.get("storage_mode") or "linked_nd2")
    artifact_path = resolve_site_path(roi_record.get("artifact_path"), project_root=project_root)
    source_path = resolve_site_path(roi_set.get("source_path"), project_root=project_root)
    axes = tuple(str(axis) for axis in roi_set.get("source_axes", ()) or ())
    sizes = {str(key): int(value) for key, value in dict(roi_set.get("source_sizes", {}) or {}).items()}

    if storage_mode == "roi_ome_zarr":
        return ImageSourceSpec(
            source_type="roi_ome_zarr",
            path=artifact_path,
            axes=("T", "C", "Z", "Y", "X"),
            roi=roi,
            metadata={"site_storage_mode": storage_mode},
        )
    if storage_mode == "roi_tiff":
        return ImageSourceSpec(
            source_type="roi_tiff",
            path=artifact_path,
            axes=("T", "Z", "Y", "X", "C"),
            roi=roi,
            metadata={"site_storage_mode": storage_mode},
        )
    return ImageSourceSpec(
        source_type="linked_nd2",
        path=source_path,
        axes=axes,
        sizes=sizes,
        roi=roi,
        metadata={"site_storage_mode": storage_mode},
    )


def create_metadata_from_site_roi(
    *,
    roi_json_path: str | Path,
    roi_id: str,
    manifest: Mapping[str, Any] | None = None,
    roi_set: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> tuple[TrajectoryMetadata, dict[str, Any], dict[str, Any], Path, str]:
    """Return trajectory metadata plus source SITE payloads for one ROI."""

    roi_set_data = dict(roi_set or load_json(roi_json_path))
    roi_record = find_roi_record(roi_set_data, roi_id)
    root = Path(project_root) if project_root is not None else infer_project_root_from_roi_json(roi_json_path)
    dataset_id = dataset_id_from_roi_json(roi_json_path, roi_set_data)
    image_source = image_source_from_site_roi(roi_set=roi_set_data, roi_record=roi_record, project_root=root)

    image = {}
    if manifest is not None:
        images = manifest.get("images") if isinstance(manifest, Mapping) else None
        if isinstance(images, list) and images:
            image = dict(images[0])
    channels = channels_from_site(image.get("channels", []))
    acquisition = dict(image.get("acquisition", {}) or {})
    treatments = list(image.get("treatments", []) or [])
    roi_spec = RoiSpec.from_dict(
        {
            **roi_record,
            "dataset_id": dataset_id,
            "source_path": roi_set_data.get("source_path"),
        }
    )
    metadata = TrajectoryMetadata(
        roi_id=roi_id,
        dataset_id=dataset_id,
        frame_count=frame_count_from_roi(roi_record, roi_set_data),
        channels=channels,
        roi=roi_spec,
        image_source=image_source,
        acquisition=acquisition,
        treatments=treatments,
    )
    return metadata, roi_set_data, roi_record, root, dataset_id


def create_analysis_h5_from_site_roi(
    *,
    roi_json_path: str | Path,
    roi_id: str,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a frame-based celltraj2 H5 for one SITE ROI."""

    manifest = load_json(manifest_path) if manifest_path is not None and Path(manifest_path).exists() else {}
    metadata, roi_set, roi_record, root, dataset_id = create_metadata_from_site_roi(
        roi_json_path=roi_json_path,
        roi_id=roi_id,
        manifest=manifest,
        project_root=project_root,
    )
    out = Path(output_path) if output_path is not None else default_cell_file_path(root, dataset_id, roi_id)
    source_links = {
        "roi_json_path": str(Path(roi_json_path)),
        "manifest_path": str(Path(manifest_path)) if manifest_path is not None else None,
        "project_root": str(root),
        "roi_set_source_path": roi_set.get("source_path"),
    }
    with TrajectoryStore.create(
        out,
        metadata=metadata,
        site_manifest=manifest,
        roi_record=roi_record,
        source_links=source_links,
        overwrite=overwrite,
    ):
        pass
    return out
