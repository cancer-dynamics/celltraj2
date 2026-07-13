"""H5 storage API for celltraj2 analysis files."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from celltraj2.paths import frame_key, frame_sort_key, raw_frame_path, validate_name
from celltraj2.schema import SCHEMA_VERSION, ImageSourceSpec, TrajectoryMetadata


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 H5 storage requires h5py. Install with "
            "`python -m pip install -e .[analysis]`."
        ) from exc
    return h5py


def _json_text(data: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _as_json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    return value


class TrajectoryStore:
    """Low-level frame-based H5 store."""

    def __init__(self, path: str | Path, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        h5py = _require_h5py()
        self._h5 = h5py.File(self.path, mode)

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        metadata: TrajectoryMetadata,
        site_manifest: Mapping[str, Any] | None = None,
        roi_record: Mapping[str, Any] | None = None,
        source_links: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> "TrajectoryStore":
        """Create a new analysis H5 and initialize metadata groups."""

        out = Path(path)
        if out.exists() and not overwrite:
            raise FileExistsError(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        store = cls(out, mode="w")
        store._h5.attrs["celltraj2_schema_version"] = metadata.schema_version or SCHEMA_VERSION
        store._init_groups()
        store.write_json("/metadata/celltraj2.json", metadata.to_dict(), overwrite=True)
        store.write_json("/metadata/site_manifest.json", dict(site_manifest or {}), overwrite=True)
        store.write_json("/metadata/roi.json", dict(roi_record or (metadata.roi.to_dict() if metadata.roi else {})), overwrite=True)
        store.write_json("/metadata/source_links.json", dict(source_links or {}), overwrite=True)
        store.write_json("/metadata/channels.json", [channel.to_dict() for channel in metadata.channels], overwrite=True)
        store.write_json("/metadata/acquisition.json", metadata.acquisition, overwrite=True)
        store.write_json("/metadata/treatments.json", metadata.treatments, overwrite=True)
        if metadata.image_source is not None:
            store.write_image_source(metadata.image_source)
        from celltraj2.registration import initialize_identity_registration

        initialize_identity_registration(store, metadata)
        return store

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> "TrajectoryStore":
        """Open an existing analysis H5."""

        return cls(path, mode=mode)

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "TrajectoryStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def h5(self) -> Any:
        return self._h5

    def _init_groups(self) -> None:
        for name in (
            "metadata",
            "sources",
            "images/raw",
            "labels",
            "masks",
            "registrations",
            "object_sets",
            "cells",
            "features",
            "runs/segmentation",
            "runs/object_indexing",
            "runs/feature_extraction",
            "runs/registration",
            "runs/tracking",
        ):
            self._h5.require_group(name)
        self.write_json("/images/raw/metadata.json", {"storage": "frame_based", "frame_index_base": 1}, overwrite=True)

    def write_json(self, path: str, data: Mapping[str, Any] | list[Any], *, overwrite: bool = False) -> None:
        """Write a JSON dataset."""

        clean_path = path.strip("/")
        if clean_path in self._h5:
            if not overwrite:
                raise FileExistsError(path)
            del self._h5[clean_path]
        parent_path, _, name = clean_path.rpartition("/")
        parent = self._h5.require_group(parent_path) if parent_path else self._h5
        h5py = _require_h5py()
        parent.create_dataset(name, data=_json_text(data), dtype=h5py.string_dtype(encoding="utf-8"))

    def read_json(self, path: str) -> Any:
        """Read a JSON dataset."""

        value = self._h5[path.strip()][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(str(value))

    def write_image_source(self, spec: ImageSourceSpec) -> None:
        self.write_json("/sources/image_source.json", spec.to_dict(), overwrite=True)

    def read_image_source(self) -> ImageSourceSpec:
        return ImageSourceSpec.from_dict(self.read_json("/sources/image_source.json"))

    def read_metadata(self) -> TrajectoryMetadata:
        return TrajectoryMetadata.from_dict(self.read_json("/metadata/celltraj2.json"))

    def _write_frame_dataset(
        self,
        group_path: str,
        frame: int,
        data: Any,
        *,
        overwrite: bool = False,
        attrs: Mapping[str, Any] | None = None,
        compression: str | None = "gzip",
    ) -> str:
        key = frame_key(frame)
        group = self._h5.require_group(group_path.strip("/"))
        if key in group:
            if not overwrite:
                raise FileExistsError(f"/{group_path.strip('/')}/{key}")
            del group[key]
        try:
            dataset = group.create_dataset(key, data=data, compression=compression)
        except TypeError:
            dataset = group.create_dataset(key, data=data)
        dataset.attrs["frame"] = int(frame)
        for attr_key, attr_value in dict(attrs or {}).items():
            dataset.attrs[str(attr_key)] = json.dumps(_as_json_safe(attr_value)) if isinstance(attr_value, (dict, list)) else attr_value
        return f"/{group_path.strip('/')}/{key}"

    def _read_frame_dataset(self, group_path: str, frame: int) -> Any:
        return self._h5[f"{group_path.strip('/')}/{frame_key(frame)}"][()]

    def write_raw_frame(self, frame: int, image: Any, *, overwrite: bool = False, attrs: Mapping[str, Any] | None = None) -> str:
        """Write one embedded raw image frame."""

        return self._write_frame_dataset("/images/raw", frame, image, overwrite=overwrite, attrs=attrs)

    def read_raw_frame(self, frame: int) -> Any:
        return self._read_frame_dataset("/images/raw", frame)

    def write_label_frame(
        self,
        label_set: str,
        frame: int,
        labels: Any,
        *,
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Write one frame of integer labels for a named label set."""

        name = validate_name(label_set, kind="label set")
        group_path = f"/labels/{name}"
        group = self._h5.require_group(group_path.strip("/"))
        if "metadata.json" not in group:
            self.write_json(
                f"{group_path}/metadata.json",
                {"label_set": name, "storage": "frame_based", "frame_index_base": 1},
                overwrite=True,
            )
        attrs = {"label_set": name}
        if metadata:
            attrs["metadata"] = dict(metadata)
        return self._write_frame_dataset(group_path, frame, labels, overwrite=overwrite, attrs=attrs)

    def read_label_frame(self, label_set: str, frame: int) -> Any:
        name = validate_name(label_set, kind="label set")
        return self._read_frame_dataset(f"/labels/{name}", frame)

    def write_mask_frame(
        self,
        mask_set: str,
        frame: int,
        mask: Any,
        *,
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Write one frame of a named binary/uint mask set."""

        name = validate_name(mask_set, kind="mask set")
        group_path = f"/masks/{name}"
        group = self._h5.require_group(group_path.strip("/"))
        if "metadata.json" not in group:
            self.write_json(
                f"{group_path}/metadata.json",
                {"mask_set": name, "storage": "frame_based", "frame_index_base": 1},
                overwrite=True,
            )
        attrs = {"mask_set": name}
        if metadata:
            attrs["metadata"] = dict(metadata)
        return self._write_frame_dataset(group_path, frame, mask, overwrite=overwrite, attrs=attrs)

    def read_mask_frame(self, mask_set: str, frame: int) -> Any:
        name = validate_name(mask_set, kind="mask set")
        return self._read_frame_dataset(f"/masks/{name}", frame)

    def require_object_set(
        self,
        object_set: str,
        *,
        source_label_set: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite_metadata: bool = False,
    ) -> str:
        """Ensure a named object-set analysis group exists."""

        name = validate_name(object_set, kind="object set")
        group_path = f"/object_sets/{name}"
        group = self._h5.require_group(group_path.strip("/"))
        group.require_group("lookup")
        group.require_group("features")
        group.require_group("tracks")
        json_path = f"{group_path}/object_set.json"
        if overwrite_metadata or json_path.strip("/") not in self._h5:
            payload = {
                "schema": "celltraj2.object_set.v1",
                "object_set": name,
                "source_label_set": source_label_set or name,
                "observation_id_base": 1,
                "row_alignment": "row_index_zero_based_maps_to_observation_id_minus_1",
                "metadata": dict(metadata or {}),
            }
            self.write_json(json_path, payload, overwrite=True)
        return group_path

    def write_observations(
        self,
        object_set: str,
        observations: Any,
        schema: Mapping[str, Any],
        *,
        source_label_set: str | None = None,
        overwrite: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Write the canonical observation table for one object set."""

        name = validate_name(object_set, kind="object set")
        group_path = self.require_object_set(
            name,
            source_label_set=source_label_set,
            metadata=metadata,
            overwrite_metadata=overwrite,
        )
        dataset_path = f"{group_path}/observations".strip("/")
        if dataset_path in self._h5:
            if not overwrite:
                raise FileExistsError(f"/{dataset_path}")
            del self._h5[dataset_path]
        dataset = self._h5.create_dataset(dataset_path, data=observations, compression="gzip")
        dataset.attrs["object_set"] = name
        dataset.attrs["observation_id_base"] = 1
        dataset.attrs["row_alignment"] = "row_index_zero_based_maps_to_observation_id_minus_1"
        self.write_json(f"{group_path}/observations_schema.json", dict(schema), overwrite=True)
        return f"/{dataset_path}"

    def read_observations(self, object_set: str) -> Any:
        name = validate_name(object_set, kind="object set")
        return self._h5[f"object_sets/{name}/observations"][()]

    def has_observations(self, object_set: str) -> bool:
        name = validate_name(object_set, kind="object set")
        return f"object_sets/{name}/observations" in self._h5

    def observation_count(self, object_set: str) -> int:
        if not self.has_observations(object_set):
            return 0
        name = validate_name(object_set, kind="object set")
        return int(self._h5[f"object_sets/{name}/observations"].shape[0])

    def read_observations_schema(self, object_set: str) -> Any:
        name = validate_name(object_set, kind="object set")
        return self.read_json(f"/object_sets/{name}/observations_schema.json")

    def write_observation_lookup_frame(
        self,
        object_set: str,
        frame: int,
        lookup: Any,
        *,
        overwrite: bool = False,
    ) -> str:
        """Write a frame-local label-id to observation-id lookup array."""

        name = validate_name(object_set, kind="object set")
        group_path = f"/object_sets/{name}/lookup"
        self.require_object_set(name)
        return self._write_frame_dataset(
            group_path,
            frame,
            lookup,
            overwrite=overwrite,
            attrs={"object_set": name, "lookup": "label_id_to_observation_id", "observation_id_base": 1},
            compression="gzip",
        )

    def clear_observation_lookup_frames(self, object_set: str) -> None:
        """Remove all per-frame observation lookup arrays for one object set."""

        name = validate_name(object_set, kind="object set")
        path = f"object_sets/{name}/lookup"
        if path not in self._h5:
            return
        group = self._h5[path]
        for key in list(group.keys()):
            if str(key).startswith("frame_"):
                del group[key]

    def read_observation_lookup_frame(self, object_set: str, frame: int) -> Any:
        name = validate_name(object_set, kind="object set")
        return self._read_frame_dataset(f"/object_sets/{name}/lookup", frame)

    def list_observation_lookup_frames(self, object_set: str) -> list[int]:
        name = validate_name(object_set, kind="object set")
        path = f"object_sets/{name}/lookup"
        if path not in self._h5:
            return []
        keys = [key for key in self._h5[path].keys() if str(key).startswith("frame_")]
        return [int(frame_sort_key(key)) for key in sorted(keys, key=frame_sort_key)]

    def list_object_sets(self) -> list[str]:
        if "object_sets" not in self._h5:
            return []
        return sorted(str(key) for key in self._h5["object_sets"].keys())

    def write_track_graph(
        self,
        object_set: str,
        track_set: str,
        *,
        adjacency: Any,
        links: Any,
        assignments: Any,
        schema: Mapping[str, Any],
        overwrite: bool = False,
    ) -> str:
        """Write a canonical CSR lineage graph and its derived caches."""

        object_name = validate_name(object_set, kind="object set")
        track_name = validate_name(track_set, kind="track set")
        self.require_object_set(object_name)
        group_path = f"object_sets/{object_name}/tracks/{track_name}"
        if group_path in self._h5:
            if not overwrite:
                raise FileExistsError(f"/{group_path}")
            del self._h5[group_path]
        group = self._h5.require_group(group_path)
        adjacency_group = group.require_group("adjacency")
        adjacency_group.attrs["format"] = "csr"
        adjacency_group.attrs["orientation"] = "row_parent_column_child"
        adjacency_group.attrs["index_base"] = 0
        adjacency_group.attrs["shape"] = tuple(int(value) for value in adjacency.shape)
        adjacency_group.create_dataset("indptr", data=adjacency.indptr, compression="gzip")
        adjacency_group.create_dataset("indices", data=adjacency.indices, compression="gzip")
        adjacency_group.create_dataset("data", data=adjacency.data, compression="gzip")
        links_dataset = group.create_dataset("links", data=links, compression="gzip")
        links_dataset.attrs["observation_id_base"] = 1
        assignments_dataset = group.create_dataset("assignments", data=assignments, compression="gzip")
        assignments_dataset.attrs["row_alignment"] = f"/object_sets/{object_name}/observations"
        self.write_json(f"/{group_path}/schema.json", dict(schema), overwrite=True)
        return f"/{group_path}"

    def read_track_graph(self, object_set: str, track_set: str) -> Any:
        """Load one stored lineage graph without requiring scipy."""

        from celltraj2.tracking import SparseAdjacency, TrackGraph

        object_name = validate_name(object_set, kind="object set")
        track_name = validate_name(track_set, kind="track set")
        group_path = f"object_sets/{object_name}/tracks/{track_name}"
        group = self._h5[group_path]
        adjacency_group = group["adjacency"]
        shape = tuple(int(value) for value in adjacency_group.attrs["shape"])
        adjacency = SparseAdjacency(
            indptr=adjacency_group["indptr"][()],
            indices=adjacency_group["indices"][()],
            data=adjacency_group["data"][()],
            shape=(shape[0], shape[1]),
        )
        return TrackGraph(
            adjacency=adjacency,
            links=group["links"][()],
            assignments=group["assignments"][()],
            schema=self.read_json(f"/{group_path}/schema.json"),
        )

    def list_track_sets(self, object_set: str) -> list[str]:
        object_name = validate_name(object_set, kind="object set")
        path = f"object_sets/{object_name}/tracks"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def has_track_set(self, object_set: str, track_set: str) -> bool:
        object_name = validate_name(object_set, kind="object set")
        track_name = validate_name(track_set, kind="track set")
        return f"object_sets/{object_name}/tracks/{track_name}/adjacency/indptr" in self._h5

    def write_registration_set(self, registration: Any, *, overwrite: bool = False) -> str:
        """Write one ROI-level native-to-registered transform set."""

        name = validate_name(registration.name, kind="registration set")
        group_path = f"registrations/{name}"
        if group_path in self._h5:
            if not overwrite:
                raise FileExistsError(f"/{group_path}")
            del self._h5[group_path]
        group = self._h5.require_group(group_path)
        frames = group.create_dataset("frames", data=registration.frames, compression="gzip")
        frames.attrs["frame_index_base"] = 1
        transforms = group.create_dataset("transforms", data=registration.transforms, compression="gzip")
        transforms.attrs["direction"] = "native_roi_physical_to_registered_roi_physical"
        transforms.attrs["matrix_convention"] = "homogeneous_column_vector"
        group.create_dataset("frame_status", data=registration.frame_status, compression="gzip")
        group.create_dataset("pairwise_results", data=registration.pairwise_results, compression="gzip")
        self.write_json(f"/{group_path}/schema.json", dict(registration.schema), overwrite=True)
        self.write_json(f"/{group_path}/canvas.json", dict(registration.canvas), overwrite=True)
        return f"/{group_path}"

    def read_registration_set(self, registration_set: str) -> Any:
        """Read one stored registration set."""

        from celltraj2.registration import RegistrationSet

        name = validate_name(registration_set, kind="registration set")
        group_path = f"registrations/{name}"
        group = self._h5[group_path]
        return RegistrationSet(
            name=name,
            frames=group["frames"][()],
            transforms=group["transforms"][()],
            frame_status=group["frame_status"][()],
            pairwise_results=group["pairwise_results"][()],
            schema=self.read_json(f"/{group_path}/schema.json"),
            canvas=self.read_json(f"/{group_path}/canvas.json"),
        )

    def list_registration_sets(self) -> list[str]:
        path = "registrations"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys() if str(key) != "active.json")

    def has_registration_set(self, registration_set: str) -> bool:
        name = validate_name(registration_set, kind="registration set")
        return f"registrations/{name}/transforms" in self._h5

    def set_active_registration(
        self,
        registration_set: str,
        *,
        reason: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Select the registration used by default for viewing and analysis."""

        name = validate_name(registration_set, kind="registration set")
        if not self.has_registration_set(name):
            raise FileNotFoundError(f"/registrations/{name}")
        payload = {"registration_set": name}
        if reason:
            payload["reason"] = str(reason)
        if run_id:
            payload["run_id"] = str(run_id)
        self.write_json("/registrations/active.json", payload, overwrite=True)
        return "/registrations/active.json"

    def active_registration_name(self) -> str | None:
        path = "registrations/active.json"
        if path not in self._h5:
            return "identity" if self.has_registration_set("identity") else None
        value = self.read_json(f"/{path}")
        name = str(value.get("registration_set") or "") if isinstance(value, Mapping) else ""
        return name or None

    def read_active_registration(self) -> Any:
        name = self.active_registration_name()
        if name is None:
            raise FileNotFoundError("No active registration set is stored")
        return self.read_registration_set(name)

    def write_registration_run(
        self,
        run_id: str,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write top-level provenance for one registration run."""

        name = validate_name(run_id, kind="registration run")
        group = self._h5.require_group(f"runs/registration/{name}")
        group.require_group("frames")
        self.write_json(f"/runs/registration/{name}/run.json", dict(data), overwrite=overwrite)
        return f"/runs/registration/{name}/run.json"

    def read_registration_run(self, run_id: str) -> Any:
        name = validate_name(run_id, kind="registration run")
        return self.read_json(f"/runs/registration/{name}/run.json")

    def write_registration_frame_result(
        self,
        run_id: str,
        frame: int,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write one frame's registration status and absolute shift."""

        name = validate_name(run_id, kind="registration run")
        self._h5.require_group(f"runs/registration/{name}/frames")
        path = f"/runs/registration/{name}/frames/{frame_key(frame)}.json"
        self.write_json(path, dict(data), overwrite=overwrite)
        return path

    def read_registration_frame_result(self, run_id: str, frame: int) -> Any:
        name = validate_name(run_id, kind="registration run")
        return self.read_json(f"/runs/registration/{name}/frames/{frame_key(frame)}.json")

    def list_registration_runs(self) -> list[str]:
        path = "runs/registration"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def write_tracking_run(
        self,
        run_id: str,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write top-level metadata for one tracking run."""

        name = validate_name(run_id, kind="tracking run")
        group = self._h5.require_group(f"runs/tracking/{name}")
        group.require_group("frames")
        self.write_json(f"/runs/tracking/{name}/run.json", dict(data), overwrite=overwrite)
        return f"/runs/tracking/{name}/run.json"

    def read_tracking_run(self, run_id: str) -> Any:
        name = validate_name(run_id, kind="tracking run")
        return self.read_json(f"/runs/tracking/{name}/run.json")

    def write_tracking_frame_result(
        self,
        run_id: str,
        frame: int,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write tracking summary metadata for one target frame."""

        name = validate_name(run_id, kind="tracking run")
        self._h5.require_group(f"runs/tracking/{name}/frames")
        path = f"/runs/tracking/{name}/frames/{frame_key(frame)}.json"
        self.write_json(path, dict(data), overwrite=overwrite)
        return path

    def read_tracking_frame_result(self, run_id: str, frame: int) -> Any:
        name = validate_name(run_id, kind="tracking run")
        return self.read_json(f"/runs/tracking/{name}/frames/{frame_key(frame)}.json")

    def list_tracking_runs(self) -> list[str]:
        path = "runs/tracking"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def write_feature_set(
        self,
        object_set: str,
        feature_set: str,
        values: Any,
        schema: Mapping[str, Any],
        *,
        overwrite: bool = False,
        qc: Mapping[str, Any] | None = None,
    ) -> str:
        """Write one row-aligned feature table for an object set."""

        object_name = validate_name(object_set, kind="object set")
        feature_name = validate_name(feature_set, kind="feature set")
        self.require_object_set(object_name)
        group_path = f"object_sets/{object_name}/features/{feature_name}"
        if group_path in self._h5:
            if not overwrite:
                raise FileExistsError(f"/{group_path}")
            del self._h5[group_path]
        group = self._h5.require_group(group_path)
        dataset = group.create_dataset("values", data=values, compression="gzip")
        dataset.attrs["object_set"] = object_name
        dataset.attrs["feature_set"] = feature_name
        dataset.attrs["row_alignment"] = f"/object_sets/{object_name}/observations"
        self.write_json(f"/{group_path}/schema.json", dict(schema), overwrite=True)
        if qc is not None:
            self.write_json(f"/{group_path}/qc.json", dict(qc), overwrite=True)
        return f"/{group_path}/values"

    def read_feature_values(self, object_set: str, feature_set: str) -> Any:
        object_name = validate_name(object_set, kind="object set")
        feature_name = validate_name(feature_set, kind="feature set")
        return self._h5[f"object_sets/{object_name}/features/{feature_name}/values"][()]

    def read_feature_schema(self, object_set: str, feature_set: str) -> Any:
        object_name = validate_name(object_set, kind="object set")
        feature_name = validate_name(feature_set, kind="feature set")
        return self.read_json(f"/object_sets/{object_name}/features/{feature_name}/schema.json")

    def has_feature_set(self, object_set: str, feature_set: str) -> bool:
        object_name = validate_name(object_set, kind="object set")
        feature_name = validate_name(feature_set, kind="feature set")
        return f"object_sets/{object_name}/features/{feature_name}/values" in self._h5

    def list_feature_sets(self, object_set: str) -> list[str]:
        object_name = validate_name(object_set, kind="object set")
        path = f"object_sets/{object_name}/features"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def write_segmentation_run(
        self,
        run_id: str,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write top-level metadata for one segmentation run."""

        name = validate_name(run_id, kind="segmentation run")
        group = self._h5.require_group(f"runs/segmentation/{name}")
        group.require_group("frames")
        self.write_json(f"/runs/segmentation/{name}/run.json", dict(data), overwrite=overwrite)
        return f"/runs/segmentation/{name}/run.json"

    def read_segmentation_run(self, run_id: str) -> Any:
        name = validate_name(run_id, kind="segmentation run")
        return self.read_json(f"/runs/segmentation/{name}/run.json")

    def write_segmentation_frame_result(
        self,
        run_id: str,
        frame: int,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write metadata for one frame processed by a segmentation run."""

        name = validate_name(run_id, kind="segmentation run")
        self._h5.require_group(f"runs/segmentation/{name}/frames")
        path = f"/runs/segmentation/{name}/frames/{frame_key(frame)}.json"
        self.write_json(path, dict(data), overwrite=overwrite)
        return path

    def read_segmentation_frame_result(self, run_id: str, frame: int) -> Any:
        name = validate_name(run_id, kind="segmentation run")
        return self.read_json(f"/runs/segmentation/{name}/frames/{frame_key(frame)}.json")

    def list_segmentation_runs(self) -> list[str]:
        path = "runs/segmentation"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def write_object_indexing_run(
        self,
        run_id: str,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write top-level metadata for one object-indexing run."""

        name = validate_name(run_id, kind="object-indexing run")
        group = self._h5.require_group(f"runs/object_indexing/{name}")
        group.require_group("frames")
        self.write_json(f"/runs/object_indexing/{name}/run.json", dict(data), overwrite=overwrite)
        return f"/runs/object_indexing/{name}/run.json"

    def read_object_indexing_run(self, run_id: str) -> Any:
        name = validate_name(run_id, kind="object-indexing run")
        return self.read_json(f"/runs/object_indexing/{name}/run.json")

    def write_object_indexing_frame_result(
        self,
        run_id: str,
        frame: int,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write metadata for one frame indexed by an object-indexing run."""

        name = validate_name(run_id, kind="object-indexing run")
        self._h5.require_group(f"runs/object_indexing/{name}/frames")
        path = f"/runs/object_indexing/{name}/frames/{frame_key(frame)}.json"
        self.write_json(path, dict(data), overwrite=overwrite)
        return path

    def read_object_indexing_frame_result(self, run_id: str, frame: int) -> Any:
        name = validate_name(run_id, kind="object-indexing run")
        return self.read_json(f"/runs/object_indexing/{name}/frames/{frame_key(frame)}.json")

    def list_object_indexing_runs(self) -> list[str]:
        path = "runs/object_indexing"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def write_feature_extraction_run(
        self,
        run_id: str,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write top-level metadata for one feature-extraction run."""

        name = validate_name(run_id, kind="feature-extraction run")
        group = self._h5.require_group(f"runs/feature_extraction/{name}")
        group.require_group("frames")
        self.write_json(f"/runs/feature_extraction/{name}/run.json", dict(data), overwrite=overwrite)
        return f"/runs/feature_extraction/{name}/run.json"

    def read_feature_extraction_run(self, run_id: str) -> Any:
        name = validate_name(run_id, kind="feature-extraction run")
        return self.read_json(f"/runs/feature_extraction/{name}/run.json")

    def write_feature_extraction_frame_result(
        self,
        run_id: str,
        frame: int,
        data: Mapping[str, Any],
        *,
        overwrite: bool = True,
    ) -> str:
        """Write metadata for one frame processed by a feature-extraction run."""

        name = validate_name(run_id, kind="feature-extraction run")
        self._h5.require_group(f"runs/feature_extraction/{name}/frames")
        path = f"/runs/feature_extraction/{name}/frames/{frame_key(frame)}.json"
        self.write_json(path, dict(data), overwrite=overwrite)
        return path

    def read_feature_extraction_frame_result(self, run_id: str, frame: int) -> Any:
        name = validate_name(run_id, kind="feature-extraction run")
        return self.read_json(f"/runs/feature_extraction/{name}/frames/{frame_key(frame)}.json")

    def list_feature_extraction_runs(self) -> list[str]:
        path = "runs/feature_extraction"
        if path not in self._h5:
            return []
        return sorted(str(key) for key in self._h5[path].keys())

    def list_label_sets(self) -> list[str]:
        if "labels" not in self._h5:
            return []
        return sorted(str(key) for key in self._h5["labels"].keys())

    def list_mask_sets(self) -> list[str]:
        if "masks" not in self._h5:
            return []
        return sorted(str(key) for key in self._h5["masks"].keys())

    def list_label_frames(self, label_set: str) -> list[int]:
        name = validate_name(label_set, kind="label set")
        path = f"labels/{name}"
        if path not in self._h5:
            return []
        keys = [key for key in self._h5[path].keys() if str(key).startswith("frame_")]
        return [int(frame_sort_key(key)) for key in sorted(keys, key=frame_sort_key)]

    def list_mask_frames(self, mask_set: str) -> list[int]:
        name = validate_name(mask_set, kind="mask set")
        path = f"masks/{name}"
        if path not in self._h5:
            return []
        keys = [key for key in self._h5[path].keys() if str(key).startswith("frame_")]
        return [int(frame_sort_key(key)) for key in sorted(keys, key=frame_sort_key)]

    def has_label_frame(self, label_set: str, frame: int) -> bool:
        name = validate_name(label_set, kind="label set")
        return f"labels/{name}/{frame_key(frame)}" in self._h5

    def has_mask_frame(self, mask_set: str, frame: int) -> bool:
        name = validate_name(mask_set, kind="mask set")
        return f"masks/{name}/{frame_key(frame)}" in self._h5

    def raw_frame_path(self, frame: int) -> str:
        return raw_frame_path(frame)
