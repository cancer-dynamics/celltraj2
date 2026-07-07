"""Model-input composition for headless segmentation workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal


NormalizationMode = Literal["raw", "lut_full_uint16", "full_uint16"]
CombinationMode = Literal["single", "mean", "max"]


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("celltraj2 model-input composition requires numpy.") from exc
    return np


def normalized_frame_axes(axes: Sequence[str] | None, ndim: int) -> tuple[str, ...]:
    """Return the axis order used by ``Trajectory.get_image_data`` frames."""

    raw_axes = tuple(str(axis).upper() for axis in (axes or ()))
    frame_axes = tuple(axis for axis in raw_axes if axis not in {"T", "P"})
    desired = tuple(axis for axis in ("Z", "Y", "X", "C") if axis in frame_axes)
    desired += tuple(axis for axis in frame_axes if axis not in desired)
    if len(desired) == int(ndim):
        return desired
    defaults = {
        2: ("Y", "X"),
        3: ("Z", "Y", "X"),
        4: ("Z", "Y", "X", "C"),
    }
    return defaults.get(int(ndim), tuple(f"D{index}" for index in range(int(ndim))))


def channel_indices_from_spec(spec: Mapping[str, Any]) -> list[int]:
    """Return raw channel indices from a SITE/celltraj2 channel spec."""

    values = spec.get("channel_indices", spec.get("channels"))
    if values is None:
        source_channels = spec.get("source_channels")
        if isinstance(source_channels, Sequence) and not isinstance(source_channels, (str, bytes)):
            values = [
                item.get("raw_index")
                for item in source_channels
                if isinstance(item, Mapping) and item.get("raw_index") is not None
            ]
    if values is None:
        return []
    if isinstance(values, int):
        return [int(values)]
    return [int(index) for index in values]


def compose_model_input(
    frame_data: Any,
    *,
    channel_specs: Sequence[Mapping[str, Any]],
    axes: Sequence[str] | None = None,
    do_3d: bool = True,
    z_index: int | None = None,
    channel_luts: Mapping[int, Any] | None = None,
    channel_index_map: Mapping[int, int] | None = None,
) -> Any:
    """Compose Cellpose-ready input from a trajectory frame.

    ``do_3d=True`` returns ``Z,Y,X`` for one output channel or ``Z,C,Y,X`` for
    multiple output channels. ``do_3d=False`` returns ``Y,X`` or ``C,Y,X``.
    """

    np = _require_numpy()
    arr = np.asarray(frame_data)
    frame_axes = normalized_frame_axes(axes, arr.ndim)
    specs = [dict(spec) for spec in channel_specs]
    if not specs:
        raise ValueError("At least one model-input channel specification is required")

    outputs = [
        _compose_one_channel(
            arr,
            frame_axes,
            spec,
            do_3d=bool(do_3d),
            z_index=z_index,
            channel_luts=channel_luts,
            channel_index_map=channel_index_map,
            np=np,
        )
        for spec in specs
    ]
    if len(outputs) == 1:
        return outputs[0]
    if do_3d:
        return np.stack(outputs, axis=1)
    return np.stack(outputs, axis=0)


def model_input_summary(data: Any, *, channel_axis: int | None = None, max_channels: int = 6) -> dict[str, Any]:
    """Return compact numeric metadata for logs and previews."""

    np = _require_numpy()
    arr = np.asarray(data)
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": None,
        "max": None,
        "mean": None,
        "p1": None,
        "p99": None,
        "channels": [],
    }
    if arr.size:
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.reshape(-1)
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "p1": float(np.percentile(finite, 1)),
                    "p99": float(np.percentile(finite, 99)),
                }
            )
    if channel_axis is not None and arr.ndim > int(channel_axis):
        count = int(arr.shape[int(channel_axis)])
        for index in range(min(count, int(max_channels))):
            slicer = [slice(None)] * arr.ndim
            slicer[int(channel_axis)] = index
            channel = np.asarray(arr[tuple(slicer)])
            item: dict[str, Any] = {
                "index": index,
                "shape": list(channel.shape),
                "min": None,
                "max": None,
                "mean": None,
                "p1": None,
                "p99": None,
            }
            if channel.size:
                finite = channel[np.isfinite(channel)] if np.issubdtype(channel.dtype, np.floating) else channel.reshape(-1)
                if finite.size:
                    item.update(
                        {
                            "min": float(np.min(finite)),
                            "max": float(np.max(finite)),
                            "mean": float(np.mean(finite)),
                            "p1": float(np.percentile(finite, 1)),
                            "p99": float(np.percentile(finite, 99)),
                        }
                    )
            summary["channels"].append(item)
    return summary


def _compose_one_channel(
    arr: Any,
    axes: Sequence[str],
    spec: Mapping[str, Any],
    *,
    do_3d: bool,
    z_index: int | None,
    channel_luts: Mapping[int, Any] | None,
    channel_index_map: Mapping[int, int] | None,
    np: Any,
) -> Any:
    combination = str(spec.get("combination", spec.get("channel_projection", "mean")))
    if combination not in {"single", "mean", "max"}:
        raise ValueError(f"Unsupported channel combination: {combination}")
    normalization = str(spec.get("normalization", "lut_full_uint16"))
    if normalization not in {"raw", "lut_full_uint16", "full_uint16"}:
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    selected_channels = channel_indices_from_spec(spec)
    if "C" not in axes:
        image = _spatial_data(arr, axes, do_3d=do_3d, z_index=z_index, np=np)
        return image if normalization == "raw" else _scale_to_uint16(image, window=None, np=np)

    if not selected_channels:
        selected_channels = [0]
    outputs = []
    for raw_index in selected_channels:
        storage_index = _storage_channel_index(int(raw_index), channel_index_map)
        channel = _take_axis(arr, axes, "C", storage_index, np=np)
        channel_axes = tuple(axis for axis in axes if axis != "C")
        image = _spatial_data(channel, channel_axes, do_3d=do_3d, z_index=z_index, np=np)
        if normalization == "raw":
            scaled = np.asarray(image)
        elif normalization == "lut_full_uint16":
            scaled = _scale_to_uint16(
                image,
                window=_lut_window(_lut_for_channel(spec, int(raw_index), channel_luts)),
                np=np,
            )
        else:
            scaled = _scale_to_uint16(image, window=None, np=np)
        outputs.append(scaled)

    if len(outputs) == 1 or combination == "single":
        return outputs[0]
    stack = np.stack([np.asarray(image, dtype=float) for image in outputs], axis=0)
    if combination == "mean":
        return np.round(np.mean(stack, axis=0)).astype(outputs[0].dtype, copy=False)
    if combination == "max":
        return np.max(stack, axis=0).astype(outputs[0].dtype, copy=False)
    raise ValueError(f"Unsupported channel combination: {combination}")


def _storage_channel_index(raw_index: int, channel_index_map: Mapping[int, int] | None) -> int:
    if channel_index_map is None:
        return int(raw_index)
    mapping = {int(key): int(value) for key, value in dict(channel_index_map).items()}
    if int(raw_index) not in mapping:
        available = ", ".join(str(key) for key in sorted(mapping))
        raise IndexError(f"Raw channel index {raw_index} is not present in this image source; available raw channels: {available}")
    return int(mapping[int(raw_index)])


def _spatial_data(arr: Any, axes: Sequence[str], *, do_3d: bool, z_index: int | None, np: Any) -> Any:
    data = np.asarray(arr)
    current_axes = tuple(str(axis).upper() for axis in axes)
    if do_3d:
        if "Z" not in current_axes:
            data = data[np.newaxis, ...]
            current_axes = ("Z",) + current_axes
        return _transpose_to(data, current_axes, ("Z", "Y", "X"), np=np)

    if "Z" in current_axes:
        z_axis = current_axes.index("Z")
        z_size = int(data.shape[z_axis])
        if z_index is None:
            if z_size != 1:
                raise ValueError("2D model input from a Z stack requires z_index")
            z_index = 0
        data = np.take(data, int(z_index), axis=z_axis)
        current_axes = tuple(axis for axis in current_axes if axis != "Z")
    return _transpose_to(data, current_axes, ("Y", "X"), np=np)


def _transpose_to(data: Any, axes: Sequence[str], desired: Sequence[str], *, np: Any) -> Any:
    current_axes = tuple(str(axis).upper() for axis in axes)
    desired_axes = tuple(axis for axis in desired if axis in current_axes)
    if desired_axes != tuple(desired):
        raise ValueError(f"Cannot compose spatial image with axes {current_axes}; expected {tuple(desired)}")
    permutation = [current_axes.index(axis) for axis in desired_axes]
    result = np.asarray(data).transpose(permutation)
    if result.ndim != len(desired):
        raise ValueError(f"Expected {tuple(desired)} image; got shape {result.shape} from axes {current_axes}")
    return result


def _take_axis(arr: Any, axes: Sequence[str], axis: str, index: int, *, np: Any) -> Any:
    current_axes = tuple(str(item).upper() for item in axes)
    if axis not in current_axes:
        return arr
    axis_index = current_axes.index(axis)
    size = int(np.asarray(arr).shape[axis_index])
    if int(index) < 0 or int(index) >= size:
        raise IndexError(f"Channel index {index} is outside available C axis size {size}")
    return np.take(arr, int(index), axis=axis_index)


def _lut_for_channel(spec: Mapping[str, Any], raw_index: int, channel_luts: Mapping[int, Any] | None) -> Any:
    if channel_luts is not None and int(raw_index) in channel_luts:
        return channel_luts[int(raw_index)]
    source_channels = spec.get("source_channels")
    if isinstance(source_channels, Sequence) and not isinstance(source_channels, (str, bytes)):
        for item in source_channels:
            if not isinstance(item, Mapping) or int(item.get("raw_index", -1)) != int(raw_index):
                continue
            if item.get("lut") is not None:
                return item.get("lut")
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("lut") is not None:
                return metadata.get("lut")
    return None


def _lut_window(settings: Any) -> tuple[float, float] | None:
    if settings is None:
        return None
    if isinstance(settings, Mapping):
        low = settings.get("low_cutoff")
        high = settings.get("high_cutoff")
    else:
        low = getattr(settings, "low_cutoff", None)
        high = getattr(settings, "high_cutoff", None)
    try:
        low_value = float(low)
        high_value = float(high)
    except Exception:
        return None
    if high_value <= low_value:
        return None
    return low_value, high_value


def _scale_to_uint16(values: Any, *, window: tuple[float, float] | None, np: Any) -> Any:
    data = np.asarray(values, dtype=float)
    if window is None:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            low, high = 0.0, 1.0
        else:
            low = float(np.min(finite))
            high = float(np.max(finite))
            if high <= low:
                high = low + 1.0
    else:
        low, high = window
    scaled = np.clip((data - low) / max(high - low, 1e-12), 0.0, 1.0)
    return np.round(scaled * 65535.0).astype(np.uint16)
