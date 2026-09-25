"""Versioned scientific inputs, separate from observation-row identity.

Call with an H5 handle whose caller owns a read or write lease. Numeric tables
and schemas are content hashed; image/label/mask volumes use resource revisions
to avoid reading image pixels during interpretation. This is a snapshot of the
available inputs, not proof that an older upstream computation used those inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from celltraj2.interpretation import canonical_json_bytes, canonical_json_digest
from celltraj2.paths import validate_name


SOURCE_DEPENDENCIES_SCHEMA = "celltraj2.source_dependencies.v1"


def _json_value(dataset: Any) -> Any:
    value = dataset[()]
    return json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))


def _metadata_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    # Paths locate files; changing a locator is not a scientific transformation.
    result = {key: value.get(key) for key in ("frame_count", "acquisition", "treatments")}
    result["channels"] = [
        {key: channel.get(key) for key in (
            "raw_index", "raw_name", "display_name", "role", "target", "readout",
            "fluorophore", "category",
        )}
        for channel in value.get("channels") or []
    ]
    source = value.get("image_source") or {}
    result["image_source"] = {key: source.get(key) for key in (
        "source_type", "axes", "sizes", "dtype", "dataset_path", "metadata",
    )}
    roi = value.get("roi") or source.get("roi") or {}
    result["roi"] = {key: roi.get(key) for key in (
        "position_index", "time_start", "time_stop", "bounds",
    )}
    return result


def _content_digest(node: Any) -> str:
    """Hash tables in bounded row blocks, including scientific H5 attributes."""
    import numpy as np

    digest = hashlib.sha256()

    def visit(item: Any, relative: str) -> None:
        attrs = {}
        for key, value in item.attrs.items():
            if key == "celltraj2_revision":
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif hasattr(value, "tolist"):
                value = value.tolist()
            attrs[str(key)] = value
        digest.update(canonical_json_bytes({"path": relative, "attrs": attrs}))
        if hasattr(item, "keys"):
            for key in sorted(item.keys()):
                visit(item[key], f"{relative}/{key}")
            return
        if item.name.endswith(".json"):
            digest.update(canonical_json_bytes(_json_value(item)))
            return
        digest.update(canonical_json_bytes({"shape": item.shape, "dtype": item.dtype.descr}))
        if not item.shape:
            blocks = (item[()],)
        else:
            row_bytes = max(1, item.dtype.itemsize * int(np.prod(item.shape[1:])))
            block_rows = max(1, (4 * 1024 * 1024) // row_bytes)
            blocks = (item[start:start + block_rows] for start in range(0, item.shape[0], block_rows))
        for block in blocks:
            array = np.asarray(block)
            if array.dtype.hasobject:
                raise ValueError(f"Unsupported variable-length scientific table: {item.name}")
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(node, "")
    return digest.hexdigest()


def _fingerprint(h5: Any, path: str, mode: str) -> dict[str, Any]:
    record: dict[str, Any] = {"path": path, "mode": mode, "exists": path in h5}
    if path not in h5:
        return record
    node = h5[path]
    if mode == "revision":
        record["revision"] = int(node.attrs.get("celltraj2_revision", 0))
    elif mode == "scientific_metadata":
        record["content_digest"] = canonical_json_digest(_metadata_projection(_json_value(node)))
    elif mode == "content":
        record["content_digest"] = _content_digest(node)
    else:
        raise ValueError(f"Unsupported source fingerprint mode: {mode}")
    return record


def capture_source_dependencies(
    h5: Any, *, object_set: str, feature_sets: Sequence[str] = (), track_set: str | None = None,
    include_acquisition: bool = False,
) -> dict[str, Any]:
    """Capture the selected tables, schemas, geometry, and declared source versions."""
    base = f"/object_sets/{validate_name(object_set, kind='object set')}"
    paths = {f"{base}/{name}": "content" for name in (
        "observations", "observations_schema.json", "object_set.json",
    )}
    paths["/metadata/celltraj2.json"] = "scientific_metadata"
    if include_acquisition:
        paths["/metadata/acquisition.json"] = "content"
    # Embedded raw pixels and segmentation are large; writers maintain revisions.
    paths["/images/raw"] = "revision"
    schema_paths = [f"{base}/observations_schema.json", f"{base}/object_set.json"]
    for name in sorted(set(feature_sets)):
        path = f"{base}/features/{validate_name(name, kind='feature set')}"
        paths[path] = "content"
        schema_paths.append(f"{path}/schema.json")
    if track_set:
        path = f"{base}/tracks/{validate_name(track_set, kind='track set')}"
        paths[path] = "content"
        schema_paths.append(f"{path}/schema.json")

    def declared_sources(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"source_label_set", "label_set", "mask_set", "registration_set"} and isinstance(child, str) and child:
                    kind = "registrations" if key == "registration_set" else "masks" if key == "mask_set" else "labels"
                    name = validate_name(child, kind=key)
                    paths[f"/{kind}/{name}"] = "content" if kind == "registrations" else "revision"
                else:
                    declared_sources(child)
        elif isinstance(value, list):
            for child in value:
                declared_sources(child)

    for path in schema_paths:
        if path in h5:
            declared_sources(_json_value(h5[path]))
    resources = [_fingerprint(h5, path, mode) for path, mode in sorted(paths.items())]
    seed = {"schema": SOURCE_DEPENDENCIES_SCHEMA, "complete": True, "resources": resources}
    return {**seed, "content_digest": canonical_json_digest(seed)}


def merge_source_dependencies(*manifests: Mapping[str, Any] | None) -> dict[str, Any]:
    """Retain every input version, including conflicting versions and legacy gaps."""
    records = {}
    complete = bool(manifests)
    for manifest in manifests:
        if not manifest:
            complete = False
            continue
        if manifest.get("schema") != SOURCE_DEPENDENCIES_SCHEMA:
            raise ValueError("Unsupported source dependency schema")
        seed = {key: value for key, value in manifest.items() if key != "content_digest"}
        if manifest.get("content_digest") != canonical_json_digest(seed):
            raise ValueError("Source dependency metadata digest does not match its contents")
        complete = complete and bool(manifest.get("complete"))
        for resource in manifest.get("resources") or []:
            records[canonical_json_digest(resource)] = dict(resource)
    seed = {
        "schema": SOURCE_DEPENDENCIES_SCHEMA, "complete": complete,
        "resources": [records[key] for key in sorted(records)],
    }
    return {**seed, "content_digest": canonical_json_digest(seed)}


def check_source_dependencies(h5: Any, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Report current/stale/unverified without modifying the artifact or H5 file."""
    if not manifest:
        return {"status": "unverified", "changes": [], "reason": "Source dependency metadata is absent"}
    if manifest.get("schema") != SOURCE_DEPENDENCIES_SCHEMA:
        raise ValueError("Unsupported source dependency schema")
    seed = {key: value for key, value in manifest.items() if key != "content_digest"}
    if manifest.get("content_digest") != canonical_json_digest(seed):
        raise ValueError("Source dependency metadata digest does not match its contents")
    changes = []
    for expected in manifest.get("resources") or []:
        actual = _fingerprint(h5, str(expected["path"]), str(expected["mode"]))
        if actual != expected:
            changes.append({"path": expected["path"], "expected": expected, "actual": actual})
    status = "stale" if changes else "current" if manifest.get("complete") else "unverified"
    return {"status": status, "changes": changes}


def require_current_sources(h5: Any, manifest: Mapping[str, Any] | None, *,
                            object_set: str | None = None, policy: str = "require_current") -> dict[str, Any]:
    """Validate captured inputs, optionally retaining historical feature results.

    Feature drift never changes the saved fingerprint or its stale status.
    Non-feature changes remain blocking; legacy gaps remain unverified.
    The caller must separately validate the observation spine before writing.
    """
    if policy not in {"require_current", "allow_feature_drift"}:
        raise ValueError(f"Unknown source dependency policy: {policy}")
    if policy == "allow_feature_drift" and not object_set:
        raise ValueError("Feature drift policy requires an object set")
    report = check_source_dependencies(h5, manifest)
    prefix = f"/object_sets/{validate_name(object_set, kind='object set')}/features/" if object_set else ""
    feature_changes = [c for c in report["changes"] if prefix and str(c["path"]).startswith(prefix)]
    blocking = [c for c in report["changes"] if policy == "require_current" or c not in feature_changes]
    report.update(policy=policy, h5_path=str(getattr(h5, "filename", None) or "<unknown H5>"),
                  feature_drift_paths=sorted({c["path"] for c in feature_changes}))
    if blocking:
        details = []
        for change in blocking:
            expected, actual = change["expected"], change["actual"]
            if expected["exists"] != actual["exists"]:
                reason = "resource missing" if not actual["exists"] else "resource now exists"
            elif expected.get("mode") == "revision":
                reason = f"revision changed: expected {expected.get('revision')}, actual {actual.get('revision')}"
            else:
                reason = f"content fingerprint differs: expected {expected.get('content_digest')}, actual {actual.get('content_digest')}"
            versions = {canonical_json_digest(r) for r in (manifest or {}).get("resources", []) if r["path"] == change["path"]}
            if len(versions) > 1:
                reason += f"; artifact retains {len(versions)} historical versions of this resource"
            details.append(f"{change['path']} -- {reason}")
        filename = str(getattr(h5, "filename", None) or "<unknown H5>")
        raise ValueError(
            f"Stale interpretation source dependencies in H5 {filename}:\n"
            + "\n".join(details)
            + "\nA freshness mismatch is not evidence of H5 corruption. Group fingerprints include tables, schemas, QC metadata, and attributes."
        )
    return report
