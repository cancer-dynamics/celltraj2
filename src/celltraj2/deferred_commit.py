"""Durable, safe deferred commit bundles for blocked canonical H5 files.

Bundles contain JSON plus NumPy arrays in a compressed NPZ.  They never use
pickle, and commit execution is limited to the allow-listed store operations
below.  This preserves an expensive calculation without cloning the source H5.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from celltraj2.h5_access import H5AccessTimeout, file_lease, validate_revisions
from celltraj2.schema import utc_now_iso
from celltraj2.trajectory import Trajectory


BUNDLE_SCHEMA = "celltraj2.deferred_h5_commit_bundle.v1"
PLAN_SCHEMA = "celltraj2.h5_commit_plan.v1"
QUEUE_SCHEMA = "sitelab.workflow_job.v1"


@dataclass(frozen=True)
class CommitOutcome:
    status: str
    operation_results: dict[str, Any]
    reason: str | None = None
    bundle_path: str | None = None
    queue_job_id: str | None = None

    @property
    def deferred(self) -> bool:
        return self.status == "deferred"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Deferred H5 commits require numpy") from exc
    return np


def _safe_name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return text or fallback


def _encode_value(value: Any, arrays: dict[str, Any]) -> Any:
    np = _require_numpy()
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):06d}"
        arrays[key] = value
        return {"__ndarray__": key}
    if isinstance(value, np.generic):
        return _encode_value(value.item(), arrays)
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, arrays) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _encode_value(item, arrays) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _encode_value(value.to_dict(), arrays)
    raise TypeError(f"Unsupported deferred-commit value: {type(value).__name__}")


def _decode_value(value: Any, arrays: Mapping[str, Any]) -> Any:
    if isinstance(value, list):
        return [_decode_value(item, arrays) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            return arrays[str(value["__ndarray__"])]
        if set(value) == {"__path__"}:
            return Path(str(value["__path__"]))
        if set(value) == {"__bytes__"}:
            return base64.b64decode(str(value["__bytes__"]).encode("ascii"))
        if set(value) == {"__tuple__"}:
            return tuple(_decode_value(item, arrays) for item in value["__tuple__"])
        return {str(key): _decode_value(item, arrays) for key, item in value.items()}
    return value


def write_bundle(path: str | Path, plan: Mapping[str, Any]) -> tuple[Path, str]:
    """Atomically write a non-pickle commit bundle and return its SHA-256."""

    np = _require_numpy()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "created_at": utc_now_iso(),
        "plan": _encode_value(dict(plan), arrays),
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    arrays["__manifest__"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, _sha256(destination)


def read_bundle(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Read and validate a deferred commit bundle without enabling pickle."""

    np = _require_numpy()
    source = Path(path)
    if expected_sha256 and _sha256(source) != str(expected_sha256):
        raise ValueError(f"Deferred commit bundle digest mismatch: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if "__manifest__" not in archive:
            raise ValueError(f"Deferred commit bundle has no manifest: {source}")
        manifest = json.loads(bytes(archive["__manifest__"].tolist()).decode("utf-8"))
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"Unsupported deferred commit bundle schema: {manifest.get('schema')!r}")
        arrays = {
            str(key): archive[key].copy()
            for key in archive.files
            if str(key) != "__manifest__"
        }
    plan = _decode_value(manifest.get("plan"), arrays)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("Deferred commit bundle contains an invalid commit plan")
    return plan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_commit_plan(
    plan: Mapping[str, Any],
    *,
    reporter: Callable[[Mapping[str, Any]], None] | None = None,
    timeout: float | None = None,
) -> CommitOutcome:
    """Validate dependencies and execute one allow-listed canonical H5 commit."""

    payload = dict(plan)
    if payload.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"Unsupported H5 commit plan schema: {payload.get('schema')!r}")
    emit = reporter or (lambda _event: None)
    h5_path = Path(str(payload["h5_path"]))
    job_id = str(payload.get("job_id") or "deferred_commit")
    operation = str(payload.get("operation") or "deferred_commit")
    with Trajectory(
        h5_path,
        mode="r+",
        timeout=timeout,
        reporter=emit,
        operation=operation,
        job_id=job_id,
    ) as trajectory:
        validate_revisions(trajectory.store, dict(payload.get("dependencies") or {}))
        for requirement in payload.get("absent_unless_overwrite", []) or []:
            path = str(requirement.get("path") or "")
            overwrite = bool(requirement.get("overwrite", False))
            if path and trajectory.store.resource_revision(path) >= 0 and not overwrite:
                reason = str(requirement.get("reason") or f"Output already exists: {path}")
                emit(
                    {
                        "event": "deferred_commit_skipped",
                        "job_id": job_id,
                        "h5_path": str(h5_path),
                        "path": path,
                        "reason": reason,
                    }
                )
                return CommitOutcome(status="skipped", operation_results={}, reason=reason)
        results: dict[str, Any] = {}
        for index, operation_payload in enumerate(payload.get("operations", []) or []):
            skip_path = str(operation_payload.get("skip_if_exists") or "")
            if skip_path and trajectory.store.resource_revision(skip_path) >= 0:
                emit(
                    {
                        "event": "deferred_commit_operation_skipped",
                        "job_id": job_id,
                        "h5_path": str(h5_path),
                        "path": skip_path,
                        "operation": operation_payload.get("op"),
                        "reason": "output already exists",
                    }
                )
                continue
            result = _execute_operation(trajectory, dict(operation_payload))
            key = str(operation_payload.get("result_key") or f"operation_{index}")
            results[key] = result
        emit(
            {
                "event": "h5_commit_completed",
                "job_id": job_id,
                "h5_path": str(h5_path),
                "operation": operation,
                "operation_count": len(payload.get("operations", []) or []),
            }
        )
        return CommitOutcome(status="committed", operation_results=results)


def _execute_operation(trajectory: Trajectory, operation: Mapping[str, Any]) -> Any:
    store = trajectory.store
    name = str(operation.get("op") or "")
    data = dict(operation.get("args") or {})
    if name == "write_json":
        return store.write_json(data["path"], data["data"], overwrite=bool(data.get("overwrite", False)))
    if name == "write_label_frame":
        return store.write_label_frame(
            data["label_set"], int(data["frame"]), data["labels"],
            overwrite=bool(data.get("overwrite", False)), metadata=data.get("metadata"),
        )
    if name == "write_mask_frame":
        return store.write_mask_frame(
            data["mask_set"], int(data["frame"]), data["mask"],
            overwrite=bool(data.get("overwrite", False)), metadata=data.get("metadata"),
        )
    if name == "write_observations":
        return store.write_observations(
            data["object_set"], data["observations"], data["schema"],
            source_label_set=data["source_label_set"],
            overwrite=bool(data.get("overwrite", False)), metadata=data.get("metadata"),
        )
    if name == "clear_observation_lookup_frames":
        return store.clear_observation_lookup_frames(data["object_set"])
    if name == "write_observation_lookup_frame":
        return store.write_observation_lookup_frame(
            data["object_set"], int(data["frame"]), data["lookup"],
            overwrite=bool(data.get("overwrite", False)),
        )
    if name == "write_feature_set":
        return store.write_feature_set(
            data["object_set"], data["feature_set"], data["values"], data["schema"],
            overwrite=bool(data.get("overwrite", False)), qc=data.get("qc"),
        )
    if name == "write_registration_set":
        registration = SimpleNamespace(**dict(data["registration"]))
        return store.write_registration_set(registration, overwrite=bool(data.get("overwrite", False)))
    if name == "set_active_registration":
        return store.set_active_registration(
            data["registration_set"], reason=str(data.get("reason") or "deferred_commit"),
            run_id=data.get("run_id"),
        )
    if name == "write_track_graph":
        adjacency = SimpleNamespace(**dict(data["adjacency"]))
        return store.write_track_graph(
            data["object_set"], data["track_set"], adjacency=adjacency,
            links=data["links"], assignments=data["assignments"], schema=data["schema"],
            overwrite=bool(data.get("overwrite", False)),
        )
    if name == "write_boundary_library":
        return store.write_boundary_library(
            data["boundary_set"], entities=data["entities"], points=data["points"],
            sources=data["sources"], schema=data["schema"], overwrite=bool(data.get("overwrite", False)),
        )
    if name == "write_boundary_geometry":
        return store.write_boundary_geometry(
            data["boundary_set"], data["geometry_set"], values=data["values"],
            topology_indptr=data["topology_indptr"], topology_indices=data["topology_indices"],
            schema=data["schema"], overwrite=bool(data.get("overwrite", False)),
        )
    if name == "write_boundary_neighbors":
        return store.write_boundary_neighbors(
            data["boundary_set"], data["neighbor_set"], indptr=data["indptr"],
            indices=data["indices"], distance=data["distance"],
            displacement_zyx=data["displacement_zyx"], schema=data["schema"],
            overwrite=bool(data.get("overwrite", False)),
        )
    if name == "write_boundary_motion":
        return store.write_boundary_motion(
            data["boundary_set"], data["motion_set"], links=data["links"],
            transport=data["transport"], point_summaries=data.get("point_summaries"),
            schema=data["schema"], overwrite=bool(data.get("overwrite", False)),
        )
    raise ValueError(f"Deferred commit operation is not allowed: {name!r}")


def defer_commit_plan(
    plan: Mapping[str, Any],
    *,
    reporter: Callable[[Mapping[str, Any]], None] | None = None,
    parent_job_id: str | None = None,
    retry_number: int = 1,
    bundle_path: str | Path | None = None,
    bundle_sha256: str | None = None,
) -> CommitOutcome:
    """Persist a plan and append a runnable deferred-commit queue record."""

    emit = reporter or (lambda _event: None)
    payload = dict(plan)
    commit_id = f"h5_commit_{uuid.uuid4().hex[:12]}"
    root = _deferred_root(payload)
    job_dir = root / commit_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if bundle_path is None:
        bundle, digest = write_bundle(job_dir / "commit_bundle.ct2commit.npz", payload)
    else:
        bundle = Path(bundle_path)
        digest = str(bundle_sha256 or _sha256(bundle))
    job_payload = {
        "schema": "celltraj2.deferred_h5_commit_job.v1",
        "job_id": commit_id,
        "parent_job_id": str(parent_job_id or payload.get("job_id") or ""),
        "retry_number": int(retry_number),
        "bundle_path": str(bundle),
        "bundle_sha256": digest,
        "h5_path": str(payload["h5_path"]),
        "created_at": utc_now_iso(),
    }
    job_path = job_dir / "deferred_commit_job.json"
    _write_json_atomic(job_path, job_payload)
    queue_record = {
        "schema": QUEUE_SCHEMA,
        "job_id": commit_id,
        "workflow": "deferred_h5_commit",
        "kind": "celltraj2.deferred_h5_commit",
        "status": "queued",
        "created_at": utc_now_iso(),
        "project_root": str(payload.get("project_root") or os.environ.get("CELLTRAJ2_PROJECT_ROOT") or ""),
        "python_executable": str(sys.executable),
        "job_path": str(job_path),
        "output_dir": str(job_dir),
        "visualization_output_dir": "",
        "log_path": str(job_dir / "deferred_commit_output.log"),
        "events_path": str(job_dir / "events.jsonl"),
        "h5_path": str(payload["h5_path"]),
        "parent_job_id": job_payload["parent_job_id"],
        "retry_number": int(retry_number),
        "bundle_path": str(bundle),
        "bundle_sha256": digest,
        "command": [
            str(sys.executable), "-m", "celltraj2.runners.apply_deferred_commit", str(job_path),
        ],
    }
    queue_path_value = os.environ.get("CELLTRAJ2_WORKFLOW_QUEUE_PATH")
    queued = False
    if queue_path_value not in (None, ""):
        append_queue_record(Path(str(queue_path_value)), queue_record)
        queued = True
    emit(
        {
            "event": "deferred_commit_created",
            "job_id": str(payload.get("job_id") or ""),
            "deferred_job_id": commit_id,
            "parent_job_id": job_payload["parent_job_id"],
            "h5_path": str(payload["h5_path"]),
            "bundle_path": str(bundle),
            "bundle_sha256": digest,
            "deferred_job_path": str(job_path),
            "queued": queued,
            "retry_number": int(retry_number),
        }
    )
    return CommitOutcome(
        status="deferred", operation_results={}, bundle_path=str(bundle), queue_job_id=commit_id
    )


def append_queue_record(queue_path: Path, record: Mapping[str, Any]) -> None:
    """Cross-process-safe append to SITE's JSONL workflow queue."""

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lease(queue_path, exclusive=True, timeout=60.0, operation="append_workflow_queue"):
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _deferred_root(plan: Mapping[str, Any]) -> Path:
    configured = os.environ.get("CELLTRAJ2_DEFERRED_ROOT")
    if configured not in (None, ""):
        root = Path(str(configured))
    elif plan.get("project_root") not in (None, ""):
        root = Path(str(plan["project_root"])) / "outputs" / "workflows" / "deferred_h5_commit"
    else:
        root = Path(str(plan["h5_path"])).parent / ".deferred_h5_commits"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(data), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def apply_deferred_job(
    job: Mapping[str, Any],
    *,
    reporter: Callable[[Mapping[str, Any]], None] | None = None,
) -> CommitOutcome:
    """Apply a saved bundle, or enqueue another retry if write access times out."""

    payload = dict(job)
    plan = read_bundle(payload["bundle_path"], expected_sha256=payload.get("bundle_sha256"))
    emit = reporter or (lambda _event: None)
    emit(
        {
            "event": "deferred_commit_started",
            "job_id": payload.get("job_id"),
            "parent_job_id": payload.get("parent_job_id"),
            "h5_path": plan.get("h5_path"),
            "bundle_path": payload.get("bundle_path"),
            "retry_number": int(payload.get("retry_number") or 1),
        }
    )
    try:
        outcome = execute_commit_plan(plan, reporter=emit)
    except H5AccessTimeout:
        return defer_commit_plan(
            plan,
            reporter=emit,
            parent_job_id=str(payload.get("parent_job_id") or plan.get("job_id") or ""),
            retry_number=int(payload.get("retry_number") or 1) + 1,
            bundle_path=payload["bundle_path"],
            bundle_sha256=payload.get("bundle_sha256"),
        )
    if outcome.status in {"committed", "skipped"}:
        bundle = Path(str(payload["bundle_path"]))
        if bundle.exists():
            bundle.unlink()
        emit(
            {
                "event": "deferred_commit_completed",
                "job_id": payload.get("job_id"),
                "parent_job_id": payload.get("parent_job_id"),
                "h5_path": plan.get("h5_path"),
                "status": outcome.status,
                "bundle_removed": True,
            }
        )
    return outcome


__all__ = [
    "BUNDLE_SCHEMA",
    "PLAN_SCHEMA",
    "CommitOutcome",
    "apply_deferred_job",
    "append_queue_record",
    "defer_commit_plan",
    "execute_commit_plan",
    "read_bundle",
    "write_bundle",
]
