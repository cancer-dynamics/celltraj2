"""Global translation registration for celltraj2 ROI trajectories."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from celltraj2.paths import validate_name
from celltraj2.schema import utc_now_iso


FRAME_STATUS = {
    "reference": 1,
    "estimated": 2,
    "identity": 3,
    "inherited": 4,
    "failed": 5,
}
FRAME_STATUS_NAMES = {value: key for key, value in FRAME_STATUS.items()}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("celltraj2 registration requires numpy.") from exc
    return np


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    np = _require_numpy()
    return number if np.isfinite(number) and number > 0 else None


def registration_calibration(metadata: Any) -> dict[str, Any]:
    """Return automatic Z/Y/X physical calibration or explicit pixel fallback."""

    acquisition = dict(getattr(metadata, "acquisition", {}) or {})
    micron_per_pixel = _positive_float(acquisition.get("micron_per_pixel"))
    if micron_per_pixel is None:
        return {
            "distance_unit": "pixel",
            "coordinate_scale": [1.0, 1.0, 1.0],
            "micron_per_pixel": None,
            "zscale": None,
            "source": "pixel_fallback_missing_micron_per_pixel",
        }
    voxel_size = {
        str(key).upper(): value
        for key, value in dict(acquisition.get("voxel_size_um") or {}).items()
    }
    x_um = _positive_float(voxel_size.get("X")) or micron_per_pixel
    y_um = _positive_float(voxel_size.get("Y")) or micron_per_pixel
    zscale = _positive_float(acquisition.get("zscale"))
    z_um = _positive_float(voxel_size.get("Z")) or micron_per_pixel * (zscale or 1.0)
    return {
        "distance_unit": "um",
        "coordinate_scale": [z_um, y_um, x_um],
        "micron_per_pixel": micron_per_pixel,
        "zscale": zscale,
        "source": "h5_acquisition_metadata",
    }


def registration_spatial_axes(metadata: Any) -> tuple[str, ...]:
    """Return native spatial axes without synthesizing a Z axis for 2D data."""

    image_source = getattr(metadata, "image_source", None)
    axes = tuple(str(axis).upper() for axis in getattr(image_source, "axes", ()) or ())
    if "Z" in axes:
        return ("Z", "Y", "X")
    return ("Y", "X")


def registration_native_shape(metadata: Any, spatial_axes: Sequence[str]) -> tuple[int, ...] | None:
    """Return native ROI shape when it can be obtained from stored source metadata."""

    image_source = getattr(metadata, "image_source", None)
    sizes = {str(key).upper(): int(value) for key, value in dict(getattr(image_source, "sizes", {}) or {}).items()}
    if all(axis in sizes and sizes[axis] > 0 for axis in spatial_axes):
        return tuple(sizes[axis] for axis in spatial_axes)
    roi = getattr(metadata, "roi", None)
    bounds = getattr(roi, "bounds", None)
    if bounds is None:
        return None
    shape_zyx = tuple(int(value) for value in bounds.shape_zyx)
    if tuple(spatial_axes) == ("Z", "Y", "X") and all(value > 0 for value in shape_zyx):
        return shape_zyx
    if tuple(spatial_axes) == ("Y", "X") and all(value > 0 for value in shape_zyx[-2:]):
        return shape_zyx[-2:]
    return None


def registration_digest(frames: Any, transforms: Any, frame_status: Any) -> str:
    """Return a stable digest for downstream registration dependencies."""

    np = _require_numpy()
    digest = hashlib.sha256()
    for value in (frames, transforms, frame_status):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def registration_canvas(
    transforms: Any,
    *,
    spatial_axes: Sequence[str],
    coordinate_scale_zyx: Sequence[float],
    native_shape: Sequence[int] | None,
) -> dict[str, Any]:
    """Return an uncropped common canvas without changing canonical transforms."""

    np = _require_numpy()
    axes = tuple(str(axis).upper() for axis in spatial_axes)
    scale_zyx = np.asarray(coordinate_scale_zyx, dtype=float)
    axis_indices = ["ZYX".index(axis) for axis in axes]
    scale = scale_zyx[axis_indices]
    matrices = np.asarray(transforms, dtype=float)
    shifts = matrices[:, :-1, -1]
    if native_shape is None:
        return {
            "spatial_axes": list(axes),
            "native_shape": None,
            "output_shape": None,
            "coordinate_scale": scale.tolist(),
            "origin_registered": np.min(shifts, axis=0).tolist() if shifts.size else [0.0] * len(axes),
            "canvas_offset": (-np.min(shifts, axis=0)).tolist() if shifts.size else [0.0] * len(axes),
        }
    shape = np.asarray(native_shape, dtype=int)
    if shape.shape != scale.shape or np.any(shape <= 0):
        raise ValueError("native_shape must contain one positive value per spatial axis")
    shift_index = shifts / scale[np.newaxis, :]
    origin_index = np.floor(np.min(shift_index, axis=0)).astype(int)
    stop_index = np.ceil(np.max(shift_index + shape[np.newaxis, :], axis=0)).astype(int)
    output_shape = np.maximum(1, stop_index - origin_index)
    origin = origin_index * scale
    return {
        "spatial_axes": list(axes),
        "native_shape": shape.tolist(),
        "output_shape": output_shape.tolist(),
        "coordinate_scale": scale.tolist(),
        "origin_registered": origin.tolist(),
        "canvas_offset": (-origin).tolist(),
    }


@dataclass(frozen=True)
class RegistrationSet:
    """One stored set of frame-to-common-coordinate transforms."""

    name: str
    frames: Any
    transforms: Any
    frame_status: Any
    pairwise_results: Any
    schema: dict[str, Any]
    canvas: dict[str, Any]

    @property
    def digest(self) -> str:
        return str(self.schema.get("registration_digest") or registration_digest(self.frames, self.transforms, self.frame_status))

    @property
    def spatial_axes(self) -> tuple[str, ...]:
        return tuple(str(value).upper() for value in self.schema.get("spatial_axes", ()))

    def frame_index(self, frame: int) -> int:
        np = _require_numpy()
        matches = np.flatnonzero(np.asarray(self.frames, dtype=int) == int(frame))
        if not matches.size:
            raise KeyError(f"Registration set {self.name!r} has no frame {int(frame)}")
        return int(matches[0])

    def transform_for_frame(self, frame: int) -> Any:
        return self.transforms[self.frame_index(frame)]

    def status_for_frame(self, frame: int) -> str:
        code = int(self.frame_status[self.frame_index(frame)])
        return FRAME_STATUS_NAMES.get(code, f"unknown_{code}")

    def translation_zyx(self, frame: int) -> Any:
        np = _require_numpy()
        translation = np.zeros(3, dtype=float)
        matrix = np.asarray(self.transform_for_frame(frame), dtype=float)
        for index, axis in enumerate(self.spatial_axes):
            translation["ZYX".index(axis)] = matrix[index, -1]
        return translation

    def apply_zyx(self, points_zyx: Any, frames: Any) -> Any:
        """Apply physical-coordinate transforms to Z/Y/X points."""

        np = _require_numpy()
        points = np.asarray(points_zyx, dtype=float)
        frame_values = np.asarray(frames, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3 or frame_values.shape != (points.shape[0],):
            raise ValueError("points_zyx must be N x 3 and frames must contain N values")
        result = points.copy()
        axis_indices = ["ZYX".index(axis) for axis in self.spatial_axes]
        for frame in np.unique(frame_values):
            rows = np.flatnonzero(frame_values == frame)
            matrix = np.asarray(self.transform_for_frame(int(frame)), dtype=float)
            native = points[np.ix_(rows, axis_indices)]
            homogeneous = np.column_stack([native, np.ones(rows.size, dtype=float)])
            transformed = homogeneous @ matrix.T
            result[np.ix_(rows, axis_indices)] = transformed[:, :-1]
        return result


@dataclass(frozen=True)
class RegistrationResult:
    """Result of one global translation registration run."""

    registration: RegistrationSet
    registration_path: str | None
    run_id: str
    saved: bool
    active: bool

    def to_dict(self) -> dict[str, Any]:
        np = _require_numpy()
        return {
            "registration_set": self.registration.name,
            "registration_path": self.registration_path,
            "run_id": self.run_id,
            "frame_count": int(np.asarray(self.registration.frames).size),
            "estimated_frame_count": int(np.sum(np.asarray(self.registration.frame_status) == FRAME_STATUS["estimated"])),
            "registration_digest": self.registration.digest,
            "canvas": dict(self.registration.canvas),
            "saved": bool(self.saved),
            "active": bool(self.active),
        }


def identity_registration(metadata: Any, *, name: str = "identity") -> RegistrationSet:
    """Build an identity transform for every local frame."""

    np = _require_numpy()
    axes = registration_spatial_axes(metadata)
    frame_count = max(1, int(getattr(metadata, "frame_count", 1)))
    frames = np.arange(1, frame_count + 1, dtype=np.int32)
    transforms = np.repeat(np.eye(len(axes) + 1, dtype=float)[np.newaxis, :, :], frame_count, axis=0)
    status = np.full(frame_count, FRAME_STATUS["identity"], dtype=np.uint8)
    status[0] = FRAME_STATUS["reference"]
    calibration = registration_calibration(metadata)
    canvas = registration_canvas(
        transforms,
        spatial_axes=axes,
        coordinate_scale_zyx=calibration["coordinate_scale"],
        native_shape=registration_native_shape(metadata, axes),
    )
    digest = registration_digest(frames, transforms, status)
    schema = {
        "schema": "celltraj2.registration.v1",
        "registration_set": validate_name(name, kind="registration set"),
        "method": "identity",
        "spatial_axes": list(axes),
        "frame_index_base": 1,
        "reference_frame": 1,
        "coordinate_space_from": "native_roi_physical",
        "coordinate_space_to": "registered_roi_physical",
        "matrix_convention": "homogeneous_column_vector_output_equals_matrix_times_input",
        "coordinate_unit": calibration["distance_unit"],
        "coordinate_scale_zyx": list(calibration["coordinate_scale"]),
        "calibration_source": calibration["source"],
        "frame_status_codes": dict(FRAME_STATUS),
        "registration_digest": digest,
        "registration_complete": True,
        "created_at": utc_now_iso(),
    }
    return RegistrationSet(
        name=str(schema["registration_set"]),
        frames=frames,
        transforms=transforms,
        frame_status=status,
        pairwise_results=np.zeros(0, dtype=pairwise_result_dtype()),
        schema=schema,
        canvas=canvas,
    )


def initialize_identity_registration(store: Any, metadata: Any) -> str:
    """Persist the default identity registration in a newly created H5."""

    registration = identity_registration(metadata)
    path = store.write_registration_set(registration, overwrite=True)
    store.set_active_registration(registration.name, reason="default_identity")
    return path


def pairwise_result_dtype() -> Any:
    np = _require_numpy()
    return np.dtype(
        [
            ("source_frame", "<i4"),
            ("target_frame", "<i4"),
            ("frame_gap", "<i4"),
            ("source_count", "<i8"),
            ("target_count", "<i8"),
            ("coarse_score", "<f8"),
            ("refined_score", "<f8"),
            ("objective_score", "<f8"),
            ("shift_z", "<f8"),
            ("shift_y", "<f8"),
            ("shift_x", "<f8"),
            ("success", "u1"),
            ("optimizer_nit", "<i4"),
            ("quality_flags", "<u4"),
        ]
    )


def _contact_value(distance: Any, *, r0: float, d0: float, n: int, m: int) -> Any:
    np = _require_numpy()
    result = np.ones_like(distance, dtype=float)
    outside = distance >= float(d0)
    w = (distance[outside] - float(d0)) / float(r0)
    numerator = 1.0 - np.power(w, int(n))
    denominator = 1.0 - np.power(w, int(m))
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, float(n) / float(m)),
        where=np.abs(denominator) > 1e-12,
    )
    result[outside] = ratio
    return result


def pairwise_distance_score(
    shift: Sequence[float],
    reference_points: Any,
    moving_points: Any,
    *,
    contact_transform: bool = False,
    contact_r0: float = 100.0,
    contact_d0: float = 100.0,
    contact_n: int = 6,
    contact_m: int = 12,
    reference_tree: Any | None = None,
    moving_tree: Any | None = None,
) -> float:
    """Return the symmetric nearest-neighbor score for one candidate shift."""

    np = _require_numpy()
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Global registration requires scipy.") from exc
    reference = np.asarray(reference_points, dtype=float)
    moving = np.asarray(moving_points, dtype=float)
    delta = np.asarray(shift, dtype=float)
    if reference.ndim != 2 or moving.ndim != 2 or reference.shape[1:] != moving.shape[1:]:
        raise ValueError("reference_points and moving_points must be non-empty N x D arrays")
    if not reference.shape[0] or not moving.shape[0] or delta.shape != (reference.shape[1],):
        return float("inf")
    if contact_transform and (contact_r0 <= 0 or contact_n <= 0 or contact_m <= 0):
        raise ValueError("Contact-transform r0, n, and m must be positive")
    ref_tree = reference_tree or cKDTree(reference)
    mov_tree = moving_tree or cKDTree(moving)
    forward = np.asarray(ref_tree.query(moving + delta, k=1)[0], dtype=float)
    backward = np.asarray(mov_tree.query(reference - delta, k=1)[0], dtype=float)
    if contact_transform:
        forward = _contact_value(forward, r0=contact_r0, d0=contact_d0, n=contact_n, m=contact_m)
        backward = _contact_value(backward, r0=contact_r0, d0=contact_d0, n=contact_n, m=contact_m)
        return -float(np.mean(forward) + np.mean(backward))
    return float(np.sum(forward) + np.sum(backward))


def _axis_values(maximum: float, step: float, *, np: Any) -> Any:
    if maximum <= 0:
        return np.asarray([0.0])
    count = max(3, int(np.ceil((2.0 * maximum) / step)) + 1)
    return np.linspace(-maximum, maximum, count)


def estimate_pair_translation(
    reference_points: Any,
    moving_points: Any,
    *,
    max_shift: Sequence[float] | float,
    grid_step: Sequence[float] | float,
    contact_transform: bool = False,
    contact_r0: float = 100.0,
    contact_d0: float = 100.0,
    contact_n: int = 6,
    contact_m: int = 12,
) -> dict[str, Any]:
    """Estimate one relative translation with grid search and continuous refinement."""

    np = _require_numpy()
    try:
        from scipy.optimize import minimize  # type: ignore
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Global registration requires scipy.") from exc
    reference = np.asarray(reference_points, dtype=float)
    moving = np.asarray(moving_points, dtype=float)
    ndim = int(reference.shape[1])
    maximum = np.full(ndim, float(max_shift), dtype=float) if np.isscalar(max_shift) else np.asarray(max_shift, dtype=float)
    step = np.full(ndim, float(grid_step), dtype=float) if np.isscalar(grid_step) else np.asarray(grid_step, dtype=float)
    if maximum.shape != (ndim,) or step.shape != (ndim,) or np.any(maximum < 0) or np.any(step <= 0):
        raise ValueError("max_shift and grid_step must contain one valid value per spatial axis")
    ref_tree = cKDTree(reference)
    mov_tree = cKDTree(moving)

    def objective(value: Any) -> float:
        return pairwise_distance_score(
            value,
            reference,
            moving,
            contact_transform=contact_transform,
            contact_r0=contact_r0,
            contact_d0=contact_d0,
            contact_n=contact_n,
            contact_m=contact_m,
            reference_tree=ref_tree,
            moving_tree=mov_tree,
        )

    axes = [_axis_values(float(maximum[index]), float(step[index]), np=np) for index in range(ndim)]
    grid_size = 1
    for values in axes:
        grid_size *= int(values.size)
    if grid_size > 2_000_000:
        raise ValueError(f"Registration grid contains {grid_size:,} shifts; increase grid_step or reduce max_shift")
    coarse_shift = np.zeros(ndim, dtype=float)
    coarse_score = float("inf")
    for candidate in itertools.product(*axes):
        score = objective(candidate)
        if score < coarse_score:
            coarse_score = score
            coarse_shift = np.asarray(candidate, dtype=float)

    bounds = [(-float(value), float(value)) for value in maximum]
    if np.all(maximum == 0):
        refined = SimpleNamespace(
            x=coarse_shift,
            fun=coarse_score,
            success=True,
            message="fixed zero-shift bounds",
            nit=0,
        )
        optimizer_method = "fixed"
        refined_score = coarse_score
    else:
        refined = minimize(objective, coarse_shift, method="L-BFGS-B", bounds=bounds)
        optimizer_method = "L-BFGS-B"
        refined_score = float(refined.fun) if np.isfinite(getattr(refined, "fun", np.nan)) else float("inf")
        if not bool(refined.success) or refined_score > coarse_score:
            fallback = minimize(objective, coarse_shift, method="Powell", bounds=bounds)
            fallback_score = float(fallback.fun) if np.isfinite(getattr(fallback, "fun", np.nan)) else float("inf")
            if fallback_score < refined_score:
                refined = fallback
                refined_score = fallback_score
                optimizer_method = "Powell"
    use_refined = np.isfinite(refined_score) and refined_score <= coarse_score
    shift = np.asarray(refined.x if use_refined else coarse_shift, dtype=float)
    score = refined_score if use_refined else coarse_score
    return {
        "shift": shift,
        "coarse_shift": coarse_shift,
        "coarse_score": float(coarse_score),
        "refined_score": float(refined_score),
        "objective_score": float(score),
        "success": bool(np.all(np.isfinite(shift)) and np.isfinite(score)),
        "optimizer_success": bool(getattr(refined, "success", False)),
        "optimizer_method": optimizer_method,
        "optimizer_message": str(getattr(refined, "message", "")),
        "optimizer_nit": int(getattr(refined, "nit", 0) or 0),
        "grid_size": int(grid_size),
    }


def _source_label_shape(trajectory: Any, object_set: str, axes: Sequence[str]) -> tuple[int, ...] | None:
    try:
        metadata = trajectory.store.read_json(f"/object_sets/{object_set}/object_set.json")
        label_set = str(metadata.get("source_label_set") or object_set)
        frames = trajectory.label_frames(label_set)
        if not frames:
            return None
        shape = tuple(int(value) for value in trajectory.read_label_frame(label_set, frames[0]).shape)
        return shape[-len(tuple(axes)) :]
    except Exception:
        return None


def default_registration_run_id() -> str:
    stamp = utc_now_iso().replace("-", "").replace(":", "").replace("+", "_").replace(".", "_")
    return f"register_{stamp}"


def register_global_translation(
    trajectory: Any,
    object_set: str,
    *,
    registration_set: str = "global_registration",
    max_shift_per_frame: Sequence[float] | float = 10.0,
    grid_step: Sequence[float] | float = 1.0,
    coordinate_scale: Sequence[float] | None = None,
    distance_unit: str | None = None,
    contact_transform: bool = False,
    contact_r0: float = 100.0,
    contact_d0: float = 100.0,
    contact_n: int = 6,
    contact_m: int = 12,
    overwrite: bool = False,
    save_outputs: bool = True,
    set_active: bool = True,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> RegistrationResult:
    """Register all ROI frames from one indexed object's centroid point clouds."""

    np = _require_numpy()
    object_name = validate_name(object_set, kind="object set")
    registration_name = validate_name(registration_set, kind="registration set")
    if registration_name == "identity":
        raise ValueError("The reserved 'identity' registration set cannot store an estimated transform")
    observations = trajectory.store.read_observations(object_name)
    if int(observations.shape[0]) == 0:
        raise ValueError(f"Object set {object_name!r} has no observations to register")
    frames = np.arange(1, max(1, int(trajectory.metadata.frame_count)) + 1, dtype=np.int32)
    axes = registration_spatial_axes(trajectory.metadata)
    axis_indices = ["ZYX".index(axis) for axis in axes]
    calibration = registration_calibration(trajectory.metadata)
    scale_zyx = np.asarray(coordinate_scale or calibration["coordinate_scale"], dtype=float)
    if scale_zyx.shape != (3,) or np.any(~np.isfinite(scale_zyx)) or np.any(scale_zyx <= 0):
        raise ValueError("coordinate_scale must contain three finite positive Z/Y/X values")
    unit = str(distance_unit or calibration["distance_unit"])
    centroid_zyx = np.column_stack(
        [observations["centroid_z"], observations["centroid_y"], observations["centroid_x"]]
    ).astype(float)
    centroid_zyx *= scale_zyx[np.newaxis, :]
    observation_frames = np.asarray(observations["frame"], dtype=int)
    points_by_frame = {
        int(frame): centroid_zyx[observation_frames == int(frame)][:, axis_indices]
        for frame in np.unique(observation_frames)
        if np.any(observation_frames == int(frame))
    }
    ndim = len(axes)
    transforms = np.repeat(np.eye(ndim + 1, dtype=float)[np.newaxis, :, :], frames.size, axis=0)
    status = np.full(frames.size, FRAME_STATUS["identity"], dtype=np.uint8)
    pair_records: list[tuple[Any, ...]] = []
    anchors = sorted(points_by_frame)
    reference_frame = int(anchors[0]) if anchors else 1
    status[reference_frame - 1] = FRAME_STATUS["reference"]
    previous_anchor = reference_frame
    previous_translation = np.zeros(ndim, dtype=float)
    optimizer_methods: set[str] = set()

    def report_frame(frame: int, **details: Any) -> None:
        if progress is None:
            return
        translation_zyx = np.zeros(3, dtype=float)
        translation_zyx[axis_indices] = transforms[int(frame) - 1, :-1, -1]
        progress(
            {
                "frame": int(frame),
                "status": FRAME_STATUS_NAMES.get(int(status[int(frame) - 1]), "unknown"),
                "translation_zyx": translation_zyx.tolist(),
                "coordinate_unit": unit,
                "object_count": int(points_by_frame.get(int(frame), np.zeros((0, ndim))).shape[0]),
                **details,
            }
        )

    for identity_frame in range(1, reference_frame):
        report_frame(identity_frame, note="before first available registration anchor")
    report_frame(reference_frame, note="registration reference frame")
    for target_frame in anchors[1:]:
        gap = int(target_frame - previous_anchor)
        for inherited_frame in range(previous_anchor + 1, target_frame):
            if status[inherited_frame - 1] == FRAME_STATUS["identity"]:
                transforms[inherited_frame - 1, :-1, -1] = previous_translation
                status[inherited_frame - 1] = FRAME_STATUS["inherited"]
                report_frame(
                    inherited_frame,
                    inherited_from_frame=int(previous_anchor),
                    note="no indexed observations; inherited prior absolute transform",
                )
        max_shift = np.asarray(
            [float(max_shift_per_frame)] * ndim if np.isscalar(max_shift_per_frame) else max_shift_per_frame,
            dtype=float,
        ) * max(1, gap)
        estimate = estimate_pair_translation(
            points_by_frame[previous_anchor],
            points_by_frame[target_frame],
            max_shift=max_shift,
            grid_step=grid_step,
            contact_transform=contact_transform,
            contact_r0=contact_r0,
            contact_d0=contact_d0,
            contact_n=contact_n,
            contact_m=contact_m,
        )
        delta = np.asarray(estimate["shift"], dtype=float)
        success = bool(estimate["success"])
        optimizer_methods.add(str(estimate["optimizer_method"]))
        target_translation = previous_translation + delta if success else previous_translation.copy()
        transforms[target_frame - 1, :-1, -1] = target_translation
        status[target_frame - 1] = FRAME_STATUS["estimated"] if success else FRAME_STATUS["failed"]
        shift_zyx = np.zeros(3, dtype=float)
        shift_zyx[axis_indices] = delta
        pair_records.append(
            (
                previous_anchor,
                target_frame,
                gap,
                int(points_by_frame[previous_anchor].shape[0]),
                int(points_by_frame[target_frame].shape[0]),
                float(estimate["coarse_score"]),
                float(estimate["refined_score"]),
                float(estimate["objective_score"]),
                float(shift_zyx[0]),
                float(shift_zyx[1]),
                float(shift_zyx[2]),
                int(success),
                int(estimate["optimizer_nit"]),
                0 if success else 1,
            )
        )
        report_frame(
            target_frame,
            source_frame=int(previous_anchor),
            frame_gap=int(gap),
            relative_translation_zyx=shift_zyx.tolist(),
            source_count=int(points_by_frame[previous_anchor].shape[0]),
            target_count=int(points_by_frame[target_frame].shape[0]),
            coarse_score=float(estimate["coarse_score"]),
            refined_score=float(estimate["refined_score"]),
            objective_score=float(estimate["objective_score"]),
            optimizer_method=str(estimate["optimizer_method"]),
            optimizer_nit=int(estimate["optimizer_nit"]),
            note="pairwise transform estimated" if success else "pairwise estimation failed; retained prior transform",
        )
        if success:
            previous_anchor = target_frame
            previous_translation = target_translation
    for inherited_frame in range(previous_anchor + 1, int(frames[-1]) + 1):
        if status[inherited_frame - 1] == FRAME_STATUS["identity"]:
            transforms[inherited_frame - 1, :-1, -1] = previous_translation
            status[inherited_frame - 1] = FRAME_STATUS["inherited"]
            report_frame(
                inherited_frame,
                inherited_from_frame=int(previous_anchor),
                note="no later indexed observations; inherited prior absolute transform",
            )

    pairwise = np.asarray(pair_records, dtype=pairwise_result_dtype())
    native_shape = _source_label_shape(trajectory, object_name, axes) or registration_native_shape(trajectory.metadata, axes)
    canvas = registration_canvas(
        transforms,
        spatial_axes=axes,
        coordinate_scale_zyx=scale_zyx,
        native_shape=native_shape,
    )
    digest = registration_digest(frames, transforms, status)
    complete = bool(anchors and anchors == frames.tolist() and np.all(status != FRAME_STATUS["failed"]))
    schema = {
        "schema": "celltraj2.registration.v1",
        "registration_set": registration_name,
        "method": "pairwise_symmetric_nearest_neighbor_translation",
        "objective_transform": "contact" if contact_transform else "distance_sum",
        "source_object_set": object_name,
        "source_observations": f"/object_sets/{object_name}/observations",
        "spatial_axes": list(axes),
        "frame_index_base": 1,
        "reference_frame": reference_frame,
        "coordinate_space_from": "native_roi_physical",
        "coordinate_space_to": "registered_roi_physical",
        "matrix_convention": "homogeneous_column_vector_output_equals_matrix_times_input",
        "coordinate_unit": unit,
        "coordinate_scale_zyx": scale_zyx.tolist(),
        "calibration_source": str(dict(metadata or {}).get("calibration_source") or calibration["source"]),
        "max_shift_per_frame": (
            float(max_shift_per_frame) if np.isscalar(max_shift_per_frame) else [float(value) for value in max_shift_per_frame]
        ),
        "grid_step": float(grid_step) if np.isscalar(grid_step) else [float(value) for value in grid_step],
        "contact_parameters": {
            "r0": float(contact_r0),
            "d0": float(contact_d0),
            "n": int(contact_n),
            "m": int(contact_m),
        },
        "optimizer_methods": sorted(optimizer_methods),
        "missing_frame_policy": "inherit_previous_absolute_transform",
        "frame_status_codes": dict(FRAME_STATUS),
        "registration_digest": digest,
        "registration_complete": complete,
        "created_at": utc_now_iso(),
        "metadata": dict(metadata or {}),
    }
    registration = RegistrationSet(
        name=registration_name,
        frames=frames,
        transforms=transforms,
        frame_status=status,
        pairwise_results=pairwise,
        schema=schema,
        canvas=canvas,
    )
    run_name = validate_name(run_id or default_registration_run_id(), kind="registration run")
    path = None
    active = False
    if save_outputs:
        path = trajectory.store.write_registration_set(registration, overwrite=overwrite)
        if set_active:
            trajectory.store.set_active_registration(registration_name, reason="registration_run", run_id=run_name)
            active = True
        run_record = {
            "schema": "celltraj2.registration_run.v1",
            "run_id": run_name,
            "status": "completed",
            "started_at": utc_now_iso(),
            "completed_at": utc_now_iso(),
            "h5_path": str(trajectory.path),
            "roi_id": trajectory.metadata.roi_id,
            "dataset_id": trajectory.metadata.dataset_id,
            "object_set": object_name,
            "registration_set": registration_name,
            "registration_digest": digest,
            "registration_path": path,
            "set_active": bool(set_active),
            "overwrite": bool(overwrite),
            "save_outputs": True,
            "schema_record": schema,
        }
        trajectory.store.write_registration_run(run_name, run_record, overwrite=True)
        for frame in frames.tolist():
            translation = registration.translation_zyx(frame)
            trajectory.store.write_registration_frame_result(
                run_name,
                frame,
                {
                    "frame": int(frame),
                    "status": registration.status_for_frame(frame),
                    "translation_zyx": translation.tolist(),
                    "coordinate_unit": unit,
                },
                overwrite=True,
            )
    return RegistrationResult(
        registration=registration,
        registration_path=path,
        run_id=run_name,
        saved=bool(save_outputs),
        active=active,
    )


__all__ = [
    "FRAME_STATUS",
    "FRAME_STATUS_NAMES",
    "RegistrationResult",
    "RegistrationSet",
    "default_registration_run_id",
    "estimate_pair_translation",
    "identity_registration",
    "initialize_identity_registration",
    "pairwise_distance_score",
    "register_global_translation",
    "registration_calibration",
    "registration_canvas",
    "registration_digest",
    "registration_native_shape",
    "registration_spatial_axes",
]
