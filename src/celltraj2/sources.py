"""Image source abstractions for celltraj2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from celltraj2.paths import frame_key
from celltraj2.schema import ImageSourceSpec


PROJECT_PATH_ANCHORS = ("roi_files", "rois", "cell_files", "segmentation", "analysis", "outputs", "manifests")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("celltraj2 image access requires numpy.") from exc
    return np


def _slice_or_full(value: slice | int | Sequence[int] | None) -> Any:
    return slice(None) if value is None else value


def _spatial_channel_order(data: Any, axes: Sequence[str]) -> tuple[Any, tuple[str, ...]]:
    """Move remaining axes into Z,Y,X,C order when possible."""

    np = _require_numpy()
    array = np.asarray(data)
    if array.ndim != len(axes):
        return array, tuple(axes[-array.ndim :])
    desired = [axis for axis in ("Z", "Y", "X", "C") if axis in axes]
    desired.extend(axis for axis in axes if axis not in desired and axis not in {"T", "P"})
    if len(desired) != array.ndim:
        return array, tuple(axes)
    permutation = [axes.index(axis) for axis in desired]
    return array.transpose(permutation), tuple(desired)


def _select_by_axes(
    data: Any,
    axes: Sequence[str],
    *,
    frame: int,
    channels: Sequence[int] | int | None = None,
    z: slice | int | Sequence[int] | None = None,
    y: slice | int | Sequence[int] | None = None,
    x: slice | int | Sequence[int] | None = None,
    position: int | None = None,
) -> tuple[Any, tuple[str, ...]]:
    """Select axis-by-axis while preserving the order of remaining axes."""

    np = _require_numpy()
    selected = data
    remaining_axes = list(axes)
    requests = {
        "T": int(frame) - 1,
        "P": None if position is None else int(position),
        "C": channels,
        "Z": z,
        "Y": y,
        "X": x,
    }
    if "P" in remaining_axes and position is None:
        requests["P"] = 0
    for axis in list(axes):
        if axis not in remaining_axes:
            continue
        request = requests.get(axis)
        if request is None and axis not in {"T"}:
            continue
        axis_index = remaining_axes.index(axis)
        if isinstance(request, int):
            selected = np.take(selected, int(request), axis=axis_index)
            remaining_axes.pop(axis_index)
        elif isinstance(request, slice):
            slicer = [slice(None)] * len(remaining_axes)
            slicer[axis_index] = request
            selected = selected[tuple(slicer)]
        elif request is not None:
            selected = np.take(selected, [int(value) for value in request], axis=axis_index)
    return selected, tuple(remaining_axes)


class ImageSource:
    """Base image source."""

    def __init__(self, spec: ImageSourceSpec) -> None:
        self.spec = spec
        self._last_frame_axes: tuple[str, ...] | None = None

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        raise NotImplementedError

    def channel_index_map(self) -> dict[int, int] | None:
        """Return raw source-channel index to local C-axis index mapping."""

        return None

    def frame_axes(self, ndim: int | None = None) -> tuple[str, ...]:
        """Return axes for the most recently read frame."""

        if self._last_frame_axes is not None and (ndim is None or len(self._last_frame_axes) == int(ndim)):
            return self._last_frame_axes
        axes = tuple(axis for axis in self.spec.axes if axis not in {"T", "P"})
        if ndim is not None:
            axes = _axes_for_array(int(ndim), axes)
        return axes


class InMemoryImageSource(ImageSource):
    """Simple in-memory source for tests and notebooks."""

    def __init__(self, data: Any, axes: Sequence[str]) -> None:
        spec = ImageSourceSpec(source_type="in_memory", axes=tuple(axes))
        super().__init__(spec)
        self.data = data
        self.axes = tuple(axes)

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        selected, selected_axes = _select_by_axes(
            self.data,
            self.axes,
            frame=frame,
            channels=channels,
            z=z,
            y=y,
            x=x,
        )
        ordered, axes = _spatial_channel_order(selected, selected_axes)
        self._last_frame_axes = tuple(axes)
        return ordered


class EmbeddedH5ImageSource(ImageSource):
    """Read raw image frames embedded in the trajectory H5."""

    def __init__(self, store: Any, spec: ImageSourceSpec | None = None) -> None:
        super().__init__(spec or ImageSourceSpec(source_type="embedded_h5"))
        self.store = store

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        data = self.store.read_raw_frame(frame)
        axes = tuple(self.spec.axes or ("Z", "Y", "X", "C")[-getattr(data, "ndim", 0) :])
        selected, selected_axes = _select_by_axes(
            data,
            axes,
            frame=1,
            channels=channels,
            z=z,
            y=y,
            x=x,
        )
        ordered, axes = _spatial_channel_order(selected, selected_axes)
        self._last_frame_axes = tuple(axes)
        return ordered


class OmeZarrRoiImageSource(ImageSource):
    """Read frames from a ROI OME-Zarr cache."""

    def __init__(self, spec: ImageSourceSpec) -> None:
        super().__init__(spec)
        self._root = None
        self._array = None
        self._axes: tuple[str, ...] | None = None
        self._channel_index_map: dict[int, int] | None = None

    def _open(self) -> tuple[Any, tuple[str, ...]]:
        if self._array is not None and self._axes is not None:
            return self._array, self._axes
        try:
            import zarr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OME-Zarr image access requires zarr. Install with "
                "`python -m pip install -e .[analysis]`."
            ) from exc
        if self.spec.path is None:
            raise ValueError("OME-Zarr image source requires a path")
        source_path = Path(self.spec.path)
        if not source_path.exists():
            raise FileNotFoundError(f"OME-Zarr image source path does not exist: {source_path}")
        try:
            self._root = _open_zarr_group_for_read(zarr, source_path)
            dataset_path = self.spec.dataset_path or _ome_zarr_dataset_path(self._root)
            self._array = self._root[dataset_path]
            attrs_source = self._root
        except Exception as group_exc:
            try:
                self._array = _open_zarr_array_for_read(zarr, source_path)
                self._root = self._array
                attrs_source = self._array
            except Exception as array_exc:
                raise RuntimeError(_ome_zarr_open_failure_message(source_path, group_exc, array_exc)) from group_exc
        ndim = len(getattr(self._array, "shape", ()))
        attrs_axes = _ome_zarr_axes_from_attrs(attrs_source, ndim)
        self._axes = _axes_for_array(
            ndim,
            attrs_axes,
            self.spec.axes,
            _default_axes_for_ndim(ndim),
        )
        self._channel_index_map = _ome_zarr_channel_index_map(
            attrs_source,
            axes=self._axes,
            shape=getattr(self._array, "shape", ()),
        )
        return self._array, self._axes

    def channel_index_map(self) -> dict[int, int] | None:
        self._open()
        return None if self._channel_index_map is None else dict(self._channel_index_map)

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        array, axes = self._open()
        selected, selected_axes = _select_by_axes(
            array,
            axes,
            frame=frame,
            channels=channels,
            z=z,
            y=y,
            x=x,
        )
        ordered, axes = _spatial_channel_order(selected, selected_axes)
        self._last_frame_axes = tuple(axes)
        return ordered


class TiffRoiImageSource(ImageSource):
    """Read frames from a ROI TIFF fallback cache."""

    def __init__(self, spec: ImageSourceSpec) -> None:
        super().__init__(spec)
        self._data = None
        self._axes: tuple[str, ...] | None = None

    def _open(self) -> tuple[Any, tuple[str, ...]]:
        if self._data is not None and self._axes is not None:
            return self._data, self._axes
        try:
            import tifffile  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TIFF image access requires tifffile. Install with "
                "`python -m pip install -e .[analysis]`."
            ) from exc
        if self.spec.path is None:
            raise ValueError("TIFF image source requires a path")
        with tifffile.TiffFile(self.spec.path) as tif:
            series = tif.series[0]
            self._data = series.asarray()
            axes = tuple(str(axis) for axis in getattr(series, "axes", "") or "")
        self._axes = _axes_for_array(
            self._data.ndim,
            axes,
            self.spec.axes,
            _default_axes_for_ndim(self._data.ndim),
        )
        return self._data, self._axes

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        data, axes = self._open()
        selected, selected_axes = _select_by_axes(
            data,
            axes,
            frame=frame,
            channels=channels,
            z=z,
            y=y,
            x=x,
        )
        ordered, axes = _spatial_channel_order(selected, selected_axes)
        self._last_frame_axes = tuple(axes)
        return ordered


class LinkedNd2RoiImageSource(ImageSource):
    """Read ROI frames directly from a linked ND2 using stored ROI coordinates."""

    def __init__(self, spec: ImageSourceSpec) -> None:
        super().__init__(spec)
        self._arr = None

    def _open(self) -> Any:
        if self._arr is not None:
            return self._arr
        try:
            import nd2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Linked ND2 image access requires nd2, dask, and xarray. Install with "
                "`python -m pip install -e .[analysis,nd2]`."
            ) from exc
        if self.spec.path is None:
            raise ValueError("Linked ND2 image source requires a path")
        self._arr = nd2.imread(self.spec.path, dask=True, xarray=True)
        return self._arr

    def read_frame(
        self,
        frame: int = 1,
        *,
        channels: Sequence[int] | int | None = None,
        z: slice | int | Sequence[int] | None = None,
        y: slice | int | Sequence[int] | None = None,
        x: slice | int | Sequence[int] | None = None,
    ) -> Any:
        np = _require_numpy()
        arr = self._open()
        dims = tuple(getattr(arr, "dims", ()) or ())
        roi = self.spec.roi
        indexers: dict[str, Any] = {}
        if "P" in dims and roi is not None:
            indexers["P"] = int(roi.position_index)
        if "T" in dims:
            start = int(roi.time_start if roi is not None else 0)
            indexers["T"] = start + int(frame) - 1
        if "C" in dims and channels is not None:
            indexers["C"] = channels
        bounds = roi.bounds if roi is not None else None
        if "Z" in dims:
            base = slice(bounds.z_start, bounds.z_stop) if bounds is not None else slice(None)
            indexers["Z"] = _combine_slice(base, z)
        if "Y" in dims:
            base = slice(bounds.y_start, bounds.y_stop) if bounds is not None else slice(None)
            indexers["Y"] = _combine_slice(base, y)
        if "X" in dims:
            base = slice(bounds.x_start, bounds.x_stop) if bounds is not None else slice(None)
            indexers["X"] = _combine_slice(base, x)
        selected = arr.isel(**indexers)
        axes = [axis for axis in ("Z", "Y", "X", "C") if axis in getattr(selected, "dims", ())]
        if hasattr(selected, "transpose") and axes:
            selected = selected.transpose(*axes)
        data = getattr(selected, "data", selected)
        if hasattr(data, "compute"):
            data = data.compute()
        self._last_frame_axes = tuple(axes)
        return np.asarray(data)


def image_source_from_spec(spec: ImageSourceSpec, *, store: Any | None = None) -> ImageSource:
    """Create an image source from a stored specification."""

    if spec.source_type == "embedded_h5":
        if store is None:
            raise ValueError("embedded_h5 image source requires a TrajectoryStore")
        return EmbeddedH5ImageSource(store, spec)
    spec = _spec_with_resolved_path(spec, store=store)
    if spec.source_type == "roi_ome_zarr":
        return OmeZarrRoiImageSource(spec)
    if spec.source_type == "roi_tiff":
        return TiffRoiImageSource(spec)
    if spec.source_type == "linked_nd2":
        return LinkedNd2RoiImageSource(spec)
    raise ValueError(f"Unsupported image source type: {spec.source_type}")


def _spec_with_resolved_path(spec: ImageSourceSpec, *, store: Any | None = None) -> ImageSourceSpec:
    if spec.path is None:
        return spec
    resolved = resolve_image_source_path(spec.path, store=store, source_type=spec.source_type)
    if resolved == spec.path:
        return spec
    payload = spec.to_dict()
    payload["path"] = str(resolved)
    return ImageSourceSpec.from_dict(payload)


def resolve_image_source_path(
    path: str | Path,
    *,
    store: Any | None = None,
    source_type: str | None = None,
) -> Path:
    """Resolve a stored image-source path without breaking standalone H5 use.

    Relative project paths are resolved against the inferred SITE project root
    when the H5 is still in ``cell_files/<dataset>/``. If the H5 is exported
    elsewhere, the H5 parent and current working directory are also tried.
    Legacy absolute ROI-cache paths that no longer exist are remapped by their
    ``roi_files/...`` suffix when that suffix exists beside the current H5.
    """

    value = Path(path)
    bases = _candidate_base_dirs(store)
    if value.is_absolute() or value.root:
        if value.exists():
            return value
        suffix = _project_relative_suffix(value)
        if suffix is not None:
            for base in bases:
                candidate = base / suffix
                if candidate.exists():
                    return candidate
        return value
    for base in bases:
        candidate = base / value
        if candidate.exists():
            return candidate
    if bases:
        return bases[0] / value
    return value


def _candidate_base_dirs(store: Any | None = None) -> list[Path]:
    candidates: list[Path] = []
    store_path = Path(getattr(store, "path", "")) if store is not None and getattr(store, "path", None) is not None else None
    if store_path is not None:
        inferred = _project_root_from_h5_path(store_path)
        if inferred is not None:
            candidates.append(inferred)
        candidates.append(store_path.parent)
    if store is not None:
        try:
            links = store.read_json("/metadata/source_links.json")
        except Exception:
            links = {}
        if isinstance(links, dict):
            for key in ("project_root", "creation_project_root"):
                value = links.get(key)
                if value not in (None, ""):
                    candidates.append(Path(str(value)))
    candidates.append(Path.cwd())
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved in seen:
            continue
        out.append(candidate)
        seen.add(resolved)
    return out


def _project_root_from_h5_path(path: str | Path) -> Path | None:
    value = Path(path)
    parts = value.parts
    for index, part in enumerate(parts):
        if part == "cell_files" and index > 0:
            return Path(*parts[:index])
    return None


def _project_relative_suffix(path: str | Path) -> Path | None:
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in PROJECT_PATH_ANCHORS:
            return Path(*parts[index:])
    return None


def _default_axes_for_ndim(ndim: int) -> tuple[str, ...]:
    defaults = {
        2: ("Y", "X"),
        3: ("Z", "Y", "X"),
        4: ("T", "C", "Y", "X"),
        5: ("T", "Z", "Y", "X", "C"),
    }
    return defaults.get(int(ndim), tuple(f"D{index}" for index in range(int(ndim))))


def _normalize_axes(axes: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(str(axis).upper() for axis in (axes or ()) if str(axis))


def _axes_for_array(ndim: int, *candidates: Sequence[str] | None) -> tuple[str, ...]:
    """Return axes that match an array, repairing common stale 2D SITE specs."""

    count = int(ndim)
    normalized = [_normalize_axes(candidate) for candidate in candidates if candidate]
    for axes in normalized:
        if len(axes) == count:
            return axes
    for axes in normalized:
        repaired = _drop_optional_axes_to_ndim(axes, count)
        if repaired is not None:
            return repaired
    return _default_axes_for_ndim(count)


def _drop_optional_axes_to_ndim(axes: Sequence[str], ndim: int) -> tuple[str, ...] | None:
    current = list(_normalize_axes(axes))
    if len(current) <= int(ndim):
        return None
    for optional_axis in ("Z", "T", "P"):
        if len(current) == int(ndim):
            break
        if optional_axis in current:
            current.remove(optional_axis)
    if len(current) == int(ndim):
        return tuple(current)
    return None


def _ome_zarr_dataset_path(root: Any) -> str:
    attrs = dict(getattr(root, "attrs", {}) or {})
    multiscales = attrs.get("multiscales")
    if not multiscales and isinstance(attrs.get("ome"), dict):
        multiscales = attrs["ome"].get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets") if isinstance(multiscales[0], dict) else None
        if isinstance(datasets, list) and datasets:
            path = datasets[0].get("path") if isinstance(datasets[0], dict) else None
            if path:
                return str(path)
    return "0"


def _zarr_open_kwargs() -> tuple[dict[str, int], ...]:
    return ({}, {"zarr_format": 2}, {"zarr_version": 2}, {"zarr_format": 3}, {"zarr_version": 3})


def _open_zarr_group_for_read(zarr: Any, path: Path) -> Any:
    errors = []
    for kwargs in _zarr_open_kwargs():
        try:
            return zarr.open_group(str(path), mode="r", **kwargs)
        except Exception as exc:
            errors.append((kwargs, exc))
    raise RuntimeError(_zarr_attempts_message("group", path, errors))


def _open_zarr_array_for_read(zarr: Any, path: Path) -> Any:
    open_array = getattr(zarr, "open_array", None)
    if not callable(open_array):
        raise RuntimeError("Installed zarr package does not provide open_array().")
    errors = []
    for kwargs in _zarr_open_kwargs():
        try:
            return open_array(str(path), mode="r", **kwargs)
        except Exception as exc:
            errors.append((kwargs, exc))
    raise RuntimeError(_zarr_attempts_message("array", path, errors))


def _zarr_attempts_message(kind: str, path: Path, errors: Sequence[tuple[dict[str, int], Exception]]) -> str:
    lines = [f"Could not open OME-Zarr {kind} at {path}."]
    for kwargs, exc in errors:
        label = ", ".join(f"{key}={value}" for key, value in kwargs.items()) or "default"
        lines.append(f"- {label}: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _ome_zarr_open_failure_message(path: Path, group_exc: Exception, array_exc: Exception) -> str:
    message = [
        f"Could not open OME-Zarr ROI cache at {path} as either a group or root array.",
        "",
        "Group open attempts:",
        str(group_exc),
        "",
        "Root-array open attempts:",
        str(array_exc),
    ]
    hint = _ome_zarr_layout_hint(path)
    if hint:
        message.extend(["", hint])
    return "\n".join(message)


def _ome_zarr_layout_hint(path: Path) -> str | None:
    if not path.exists():
        return f"Hint: the path does not exist from this Python environment: {path}"
    if (path / "zarr.json").exists() and not (path / ".zgroup").exists() and not (path / ".zarray").exists():
        return (
            "Hint: this looks like a Zarr v3 store. A worker environment with zarr 2.x "
            "cannot read it. Install zarr 3.x in the worker environment or re-extract "
            "the ROI cache with the updated SITE writer, which requests a v2-compatible "
            "OME-Zarr layout."
        )
    if path.is_dir() and not (path / ".zgroup").exists() and not (path / ".zarray").exists():
        return (
            "Hint: the path exists but does not contain .zgroup or .zarray metadata. "
            "Confirm the ROI artifact path points at the .ome.zarr directory itself."
        )
    return None


def _ome_zarr_axes_from_attrs(root: Any, ndim: int) -> tuple[str, ...] | None:
    attrs = dict(getattr(root, "attrs", {}) or {})
    site = attrs.get("site") if isinstance(attrs.get("site"), dict) else {}
    site_axes = site.get("axes") if isinstance(site, dict) else None
    if isinstance(site_axes, list) and len(site_axes) == ndim:
        return tuple(str(axis) for axis in site_axes)
    multiscales = attrs.get("multiscales")
    if not multiscales and isinstance(attrs.get("ome"), dict):
        multiscales = attrs["ome"].get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        axes = multiscales[0].get("axes") if isinstance(multiscales[0], dict) else None
        if isinstance(axes, list) and len(axes) == ndim:
            names = [axis.get("name") if isinstance(axis, dict) else axis for axis in axes]
            if all(name is not None for name in names):
                return tuple(str(name) for name in names)
    return None


def _ome_zarr_axes(root: Any, ndim: int) -> tuple[str, ...]:
    return _ome_zarr_axes_from_attrs(root, ndim) or _default_axes_for_ndim(ndim)


def _ome_zarr_channel_index_map(root: Any, *, axes: Sequence[str], shape: Sequence[int]) -> dict[int, int] | None:
    current_axes = tuple(str(axis).upper() for axis in axes)
    if "C" not in current_axes:
        return None
    attrs = dict(getattr(root, "attrs", {}) or {})
    site = attrs.get("site") if isinstance(attrs.get("site"), dict) else {}
    values = site.get("channel_indices") if isinstance(site, dict) else None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    c_axis = current_axes.index("C")
    c_size = int(shape[c_axis]) if len(shape) > c_axis else len(values)
    raw_indices = [int(value) for value in values]
    if len(raw_indices) != c_size:
        raise ValueError(
            "OME-Zarr site.channel_indices length "
            f"{len(raw_indices)} does not match C axis size {c_size}"
        )
    return {raw_index: local_index for local_index, raw_index in enumerate(raw_indices)}


def _combine_slice(base: slice, local: slice | int | Sequence[int] | None) -> Any:
    if local is None:
        return base
    start = 0 if base.start is None else int(base.start)
    if isinstance(local, int):
        return start + int(local)
    if isinstance(local, slice):
        local_start = 0 if local.start is None else int(local.start)
        step = local.step
        stop = None if local.stop is None else start + int(local.stop)
        return slice(start + local_start, stop, step)
    return [start + int(value) for value in local]
