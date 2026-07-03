"""Run a celltraj2 batch segmentation job with Cellpose."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from celltraj2.batch import JsonlReporter, SegmentationFileJob, SegmentationResult, load_batch_job, run_batch_segmentation


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _version_from_job(file_job: SegmentationFileJob, installed_major: int | None) -> int | None:
    params = file_job.parameters
    for value in (
        params.get("version_major"),
        file_job.backend.get("version_major"),
        file_job.backend.get("required_version_major"),
    ):
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    backend_text = " ".join(
        str(file_job.backend.get(key, ""))
        for key in ("backend_id", "backend_type", "label")
    ).lower()
    if "cellpose4" in backend_text or "cellpose 4" in backend_text:
        return 4
    if "cellpose3" in backend_text or "cellpose 3" in backend_text:
        return 3
    return installed_major


def _default_model_name(version_major: int | None) -> str:
    return "cpsam_v2" if version_major == 4 else "cyto3"


def _pad_cellpose4_channels(data: Any, do_3d: bool, np: Any) -> tuple[Any, int | None]:
    arr = np.asarray(data)
    channel_axis = 1 if do_3d and arr.ndim == 4 else 0 if (not do_3d and arr.ndim == 3) else None
    if channel_axis is None:
        if do_3d and arr.ndim == 3:
            return np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=1), 1
        if (not do_3d) and arr.ndim == 2:
            return np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=0), 0
        return arr, channel_axis
    channels = int(arr.shape[channel_axis])
    if channels == 3:
        return arr, channel_axis
    if channels > 3:
        slicer = [slice(None)] * arr.ndim
        slicer[channel_axis] = slice(0, 3)
        return arr[tuple(slicer)], channel_axis
    pad_shape = list(arr.shape)
    pad_shape[channel_axis] = 3 - channels
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=channel_axis), channel_axis


class CellposeSegmenter:
    """Callable Cellpose adapter used by ``run_batch_segmentation``."""

    def __init__(self) -> None:
        self._model_cache: dict[tuple[Any, ...], Any] = {}

    def __call__(self, image: Any, file_job: SegmentationFileJob, frame: int) -> SegmentationResult:
        import importlib.metadata

        import numpy as np
        from cellpose import models

        params = dict(file_job.parameters)
        do_3d = bool(file_job.do_3d)
        try:
            version = importlib.metadata.version("cellpose")
        except Exception:
            import cellpose

            version = getattr(cellpose, "__version__", "unknown")
        try:
            installed_major = int(str(version).split(".", 1)[0])
        except Exception:
            installed_major = None
        version_major = _version_from_job(file_job, installed_major)
        model_ref = file_job.model or params.get("model") or _default_model_name(version_major)
        use_gpu = bool(params.get("use_gpu", True))

        x = np.asarray(image)
        channel_axis = None
        z_axis = 0 if do_3d else None
        if do_3d:
            if x.ndim == 4:
                channel_axis = 1
                z_axis = 0
            elif x.ndim == 3:
                channel_axis = None
                z_axis = 0
            else:
                raise ValueError(f"3D Cellpose input must be ZYX or ZCYX; got {x.shape}")
        else:
            if x.ndim == 3:
                channel_axis = 0
            elif x.ndim == 2:
                channel_axis = None
            else:
                raise ValueError(f"2D Cellpose input must be YX or CYX; got {x.shape}")

        if version_major == 4:
            x, channel_axis = _pad_cellpose4_channels(x, do_3d, np)
            init_kwargs = {"gpu": use_gpu, "pretrained_model": str(model_ref)}
            cache_key = ("cellpose4", str(model_ref), use_gpu)
            model = self._model_cache.get(cache_key)
            if model is None:
                model = models.CellposeModel(**init_kwargs)
                self._model_cache[cache_key] = model
        else:
            model_path = str(model_ref)
            model_nchan = _optional_int(params.get("cp3_model_nchan")) or 2
            init_kwargs = {"gpu": use_gpu, "nchan": model_nchan}
            cache_key = ("cellpose3", model_path, use_gpu, model_nchan)
            model = self._model_cache.get(cache_key)
            if model is None:
                if model_path and any(sep in model_path for sep in ("/", "\\")):
                    init_kwargs["pretrained_model"] = model_path
                    model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path, nchan=model_nchan)
                else:
                    init_kwargs["model_type"] = model_path or "cyto3"
                    model = models.CellposeModel(gpu=use_gpu, model_type=model_path or "cyto3", nchan=model_nchan)
                self._model_cache[cache_key] = model

        eval_kwargs = {
            "channel_axis": channel_axis,
            "z_axis": z_axis,
            "normalize": bool(params.get("normalize", True)),
            "diameter": _optional_float(params.get("diameter")),
            "flow_threshold": float(params.get("flow_threshold", 0.4)),
            "cellprob_threshold": float(params.get("cellprob_threshold", 0.0)),
            "do_3D": do_3d,
            "anisotropy": _optional_float(params.get("anisotropy")),
            "flow3D_smooth": params.get("flow3D_smooth", 0.0),
            "stitch_threshold": float(params.get("stitch_threshold", 0.0)),
            "min_size": int(params.get("min_size", 15)),
            "batch_size": int(params.get("batch_size", 8)),
        }
        if version_major == 3:
            eval_kwargs["channels"] = [
                int(params.get("cp3_segmentation_channel", 0)),
                int(params.get("cp3_seed_channel", 0)),
            ]
        signature = inspect.signature(model.eval)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        clean_kwargs = {
            key: value
            for key, value in eval_kwargs.items()
            if value is not None and (accepts_kwargs or key in signature.parameters)
        }
        output = model.eval(x, **clean_kwargs)
        masks = output[0] if isinstance(output, tuple) else output
        masks = np.asarray(masks).astype(np.int32, copy=False)
        metadata = {
            "cellpose_version": version,
            "version_major": version_major,
            "frame": int(frame),
            "model": str(model_ref),
            "model_init_kwargs": {key: str(value) for key, value in init_kwargs.items()},
            "eval_kwargs": {key: str(value) for key, value in clean_kwargs.items()},
            "channel_axis": channel_axis,
            "z_axis": z_axis,
            "mask_shape": list(masks.shape),
            "mask_unique_count": int(len(np.unique(masks))) if masks.size else 0,
            "mask_max_label": int(np.max(masks)) if masks.size else 0,
        }
        return SegmentationResult(labels=masks, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a celltraj2 Cellpose batch segmentation job.")
    parser.add_argument("job", type=Path, help="Path to a celltraj2 segmentation batch job JSON file.")
    args = parser.parse_args(argv)
    job = load_batch_job(args.job)
    run_batch_segmentation(job, CellposeSegmenter(), reporter=JsonlReporter())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
