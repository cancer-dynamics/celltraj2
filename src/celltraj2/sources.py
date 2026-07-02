"""Image source abstractions for celltraj2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from celltraj2.paths import frame_key
from celltraj2.schema import ImageSourceSpec


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
        ordered, _axes = _spatial_channel_order(selected, selected_axes)
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
        ordered, _axes = _spatial_channel_order(selected, selected_axes)
        return ordered


class OmeZarrRoiImageSource(ImageSource):
    """Read frames from a ROI OME-Zarr cache."""

    def __init__(self, spec: ImageSourceSpec) -> None:
        super().__init__(spec)
        self._root = None
        self._array = None
        self._axes: tuple[str, ...] | None = None

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
        self._root = zarr.open_group(str(self.spec.path), mode="r")
        dataset_path = self.spec.dataset_path or _ome_zarr_dataset_path(self._root)
        self._array = self._root[dataset_path]
        self._axes = tuple(self.spec.axes or _ome_zarr_axes(self._root, len(self._array.shape)))
        return self._array, self._axes

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
        ordered, _axes = _spatial_channel_order(selected, selected_axes)
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
        self._axes = tuple(self.spec.axes or axes or _default_axes_for_ndim(self._data.ndim))
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
        ordered, _axes = _spatial_channel_order(selected, selected_axes)
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
        return np.asarray(data)


def image_source_from_spec(spec: ImageSourceSpec, *, store: Any | None = None) -> ImageSource:
    """Create an image source from a stored specification."""

    if spec.source_type == "embedded_h5":
        if store is None:
            raise ValueError("embedded_h5 image source requires a TrajectoryStore")
        return EmbeddedH5ImageSource(store, spec)
    if spec.source_type == "roi_ome_zarr":
        return OmeZarrRoiImageSource(spec)
    if spec.source_type == "roi_tiff":
        return TiffRoiImageSource(spec)
    if spec.source_type == "linked_nd2":
        return LinkedNd2RoiImageSource(spec)
    raise ValueError(f"Unsupported image source type: {spec.source_type}")


def _default_axes_for_ndim(ndim: int) -> tuple[str, ...]:
    defaults = {
        2: ("Y", "X"),
        3: ("Z", "Y", "X"),
        4: ("T", "Z", "Y", "X"),
        5: ("T", "Z", "Y", "X", "C"),
    }
    return defaults.get(int(ndim), tuple(f"D{index}" for index in range(int(ndim))))


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


def _ome_zarr_axes(root: Any, ndim: int) -> tuple[str, ...]:
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
    return _default_axes_for_ndim(ndim)


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
