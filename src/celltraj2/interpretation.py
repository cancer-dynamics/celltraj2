"""Dependency-light biological interpretation contracts for celltraj2.

This module contains the stable identities and array-level rules shared by the
SITE project interpretation store and per-ROI H5 materializations.  It avoids
GUI, Arrow, and query-engine dependencies so that classifications remain
readable in a minimal celltraj2 analysis environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from celltraj2.schema import utc_now_iso


PROJECT_OBSERVATION_KEY_SCHEMA = "site.project_observation_key.v1"
TYPE_TAXONOMY_SCHEMA = "site.type_taxonomy.v1"
STATE_SPACE_SCHEMA = "site.state_space.v1"
CLASSIFICATION_MANIFEST_SCHEMA = "site.classification_manifest.v1"
BIOLOGY_RELEASE_SCHEMA = "site.biology_release.v1"
MATERIALIZATION_RECEIPT_SCHEMA = "site.interpretation_materialization_receipt.v1"
CLASSIFICATION_VALUES_SCHEMA = "celltraj2.classification_values.v1"

CLASSIFICATION_STATUS_CODES: dict[str, int] = {
    "not_evaluated": 0,
    "assigned": 1,
    "unknown": 2,
    "ambiguous": 3,
    "excluded": 4,
}
CLASSIFICATION_STATUS_NAMES = {
    value: key for key, value in CLASSIFICATION_STATUS_CODES.items()
}

QC_TRACK_CONSTRAINT_DERIVED = 1 << 0
QC_TRACK_INCOMPLETE_SUPPORT = 1 << 1
QC_TRACK_TYPE_CONFLICT = 1 << 2
QC_POSTERIOR_SWITCH = 1 << 3
QC_SUSPICIOUS_BRANCH = 1 << 4

_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "celltraj2 interpretation arrays require numpy. Install with "
            "`python -m pip install -e .[h5]`."
        ) from exc
    return np


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {str(key): _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a JSON-safe value."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uuid_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    try:
        return str(UUID(text))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID; received {value!r}") from exc


def _optional_uuid_text(value: Any, *, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    return _uuid_text(value, field_name=field_name)


def _validate_slug(value: str, *, field_name: str = "slug") -> str:
    text = str(value or "").strip()
    if not _SLUG_RE.match(text):
        raise ValueError(
            f"{field_name} must start with a lowercase letter and contain only "
            "lowercase letters, numbers, dots, dashes, or underscores"
        )
    return text


def _validate_color(value: str) -> str:
    text = str(value or "").strip()
    if not _COLOR_RE.match(text):
        raise ValueError(f"color must be #RRGGBB; received {value!r}")
    return text.upper()


@dataclass(frozen=True, order=True)
class ProjectObservationKey:
    """Stable identity for one object observation across a SITE project."""

    project_uuid: str
    dataset_uuid: str
    roi_uuid: str
    object_set: str
    observation_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_uuid", _uuid_text(self.project_uuid, field_name="project_uuid"))
        object.__setattr__(self, "dataset_uuid", _uuid_text(self.dataset_uuid, field_name="dataset_uuid"))
        object.__setattr__(self, "roi_uuid", _uuid_text(self.roi_uuid, field_name="roi_uuid"))
        object_set = str(self.object_set or "").strip()
        if not object_set or "/" in object_set or "\\" in object_set:
            raise ValueError("object_set must be a non-empty H5 path segment")
        object.__setattr__(self, "object_set", object_set)
        observation_id = int(self.observation_id)
        if observation_id < 1:
            raise ValueError("observation_id must be one-based and >= 1")
        object.__setattr__(self, "observation_id", observation_id)

    @property
    def partition(self) -> tuple[str, str, str]:
        return self.dataset_uuid, self.roi_uuid, self.object_set

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROJECT_OBSERVATION_KEY_SCHEMA,
            "project_uuid": self.project_uuid,
            "dataset_uuid": self.dataset_uuid,
            "roi_uuid": self.roi_uuid,
            "object_set": self.object_set,
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectObservationKey":
        return cls(
            project_uuid=str(data["project_uuid"]),
            dataset_uuid=str(data["dataset_uuid"]),
            roi_uuid=str(data["roi_uuid"]),
            object_set=str(data["object_set"]),
            observation_id=int(data["observation_id"]),
        )


@dataclass(frozen=True)
class TypeNode:
    """One node in a project-defined hierarchical type taxonomy."""

    type_id: str
    name: str
    slug: str
    color: str
    parent_type_id: str | None = None
    description: str = ""
    ontology_refs: tuple[str, ...] = ()
    marker_expectations: dict[str, Any] = field(default_factory=dict)
    allowed_object_sets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "type_id", _uuid_text(self.type_id, field_name="type_id"))
        object.__setattr__(
            self,
            "parent_type_id",
            _optional_uuid_text(self.parent_type_id, field_name="parent_type_id"),
        )
        if not str(self.name or "").strip():
            raise ValueError("type node name must not be empty")
        object.__setattr__(self, "slug", _validate_slug(self.slug, field_name="type slug"))
        object.__setattr__(self, "color", _validate_color(self.color))
        object.__setattr__(self, "ontology_refs", tuple(str(item) for item in self.ontology_refs))
        object.__setattr__(
            self,
            "allowed_object_sets",
            tuple(str(item) for item in self.allowed_object_sets),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypeNode":
        payload = dict(data)
        payload["ontology_refs"] = tuple(payload.get("ontology_refs") or ())
        payload["allowed_object_sets"] = tuple(payload.get("allowed_object_sets") or ())
        return cls(**payload)


def _validate_hierarchy(
    nodes: Sequence[Any],
    *,
    id_attr: str,
    parent_attr: str,
    slug_attr: str,
) -> None:
    ids = [str(getattr(node, id_attr)) for node in nodes]
    slugs = [str(getattr(node, slug_attr)) for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {id_attr} values are not allowed")
    if len(slugs) != len(set(slugs)):
        raise ValueError(f"Duplicate {slug_attr} values are not allowed")
    known = set(ids)
    parents = {str(getattr(node, id_attr)): getattr(node, parent_attr) for node in nodes}
    for node_id, parent_id in parents.items():
        if parent_id is not None and str(parent_id) not in known:
            raise ValueError(f"Node {node_id} references missing parent {parent_id}")
        seen = {node_id}
        current = parent_id
        while current is not None:
            current_text = str(current)
            if current_text in seen:
                raise ValueError(f"Hierarchy contains a cycle at {current_text}")
            seen.add(current_text)
            current = parents.get(current_text)


@dataclass(frozen=True)
class TypeTaxonomy:
    """Immutable project type hierarchy."""

    taxonomy_id: str
    name: str
    nodes: tuple[TypeNode, ...]
    version: int = 1
    description: str = ""
    parent_taxonomy_id: str | None = None
    created_by: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    schema: str = TYPE_TAXONOMY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "taxonomy_id", _uuid_text(self.taxonomy_id, field_name="taxonomy_id"))
        object.__setattr__(
            self,
            "parent_taxonomy_id",
            _optional_uuid_text(self.parent_taxonomy_id, field_name="parent_taxonomy_id"),
        )
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not str(self.name or "").strip():
            raise ValueError("taxonomy name must not be empty")
        if int(self.version) < 1:
            raise ValueError("taxonomy version must be >= 1")
        if self.schema != TYPE_TAXONOMY_SCHEMA:
            raise ValueError(f"Unsupported taxonomy schema {self.schema!r}")
        _validate_hierarchy(
            self.nodes,
            id_attr="type_id",
            parent_attr="parent_type_id",
            slug_attr="slug",
        )

    @property
    def leaf_nodes(self) -> tuple[TypeNode, ...]:
        parents = {node.parent_type_id for node in self.nodes if node.parent_type_id}
        return tuple(node for node in self.nodes if node.type_id not in parents)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypeTaxonomy":
        payload = dict(data)
        payload["nodes"] = tuple(TypeNode.from_dict(item) for item in payload.get("nodes") or ())
        return cls(**payload)


@dataclass(frozen=True)
class StateNode:
    """One node in a hierarchical, type-scoped state space."""

    state_id: str
    name: str
    slug: str
    color: str
    parent_state_id: str | None = None
    description: str = ""
    lifecycle_role: str | None = None
    terminal: bool = False
    absorbing: bool = False
    allowed_transitions: tuple[str, ...] = ()
    simulation_role_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _uuid_text(self.state_id, field_name="state_id"))
        object.__setattr__(
            self,
            "parent_state_id",
            _optional_uuid_text(self.parent_state_id, field_name="parent_state_id"),
        )
        if not str(self.name or "").strip():
            raise ValueError("state node name must not be empty")
        object.__setattr__(self, "slug", _validate_slug(self.slug, field_name="state slug"))
        object.__setattr__(self, "color", _validate_color(self.color))
        object.__setattr__(
            self,
            "allowed_transitions",
            tuple(_uuid_text(item, field_name="allowed transition") for item in self.allowed_transitions),
        )
        object.__setattr__(
            self,
            "simulation_role_refs",
            tuple(str(item) for item in self.simulation_role_refs),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateNode":
        payload = dict(data)
        payload["allowed_transitions"] = tuple(payload.get("allowed_transitions") or ())
        payload["simulation_role_refs"] = tuple(payload.get("simulation_role_refs") or ())
        return cls(**payload)


@dataclass(frozen=True)
class StateSpace:
    """Immutable hierarchical state definition scoped to a type by default."""

    state_space_id: str
    name: str
    nodes: tuple[StateNode, ...]
    type_taxonomy_id: str | None = None
    scope_type_id: str | None = None
    version: int = 1
    description: str = ""
    parent_state_space_id: str | None = None
    created_by: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    schema: str = STATE_SPACE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_space_id",
            _uuid_text(self.state_space_id, field_name="state_space_id"),
        )
        for name in ("type_taxonomy_id", "scope_type_id", "parent_state_space_id"):
            object.__setattr__(
                self,
                name,
                _optional_uuid_text(getattr(self, name), field_name=name),
            )
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if not str(self.name or "").strip():
            raise ValueError("state-space name must not be empty")
        if int(self.version) < 1:
            raise ValueError("state-space version must be >= 1")
        if self.schema != STATE_SPACE_SCHEMA:
            raise ValueError(f"Unsupported state-space schema {self.schema!r}")
        _validate_hierarchy(
            self.nodes,
            id_attr="state_id",
            parent_attr="parent_state_id",
            slug_attr="slug",
        )
        known = {node.state_id for node in self.nodes}
        for node in self.nodes:
            missing = set(node.allowed_transitions).difference(known)
            if missing:
                raise ValueError(
                    f"State {node.state_id} has transitions to missing states {sorted(missing)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateSpace":
        payload = dict(data)
        payload["nodes"] = tuple(StateNode.from_dict(item) for item in payload.get("nodes") or ())
        return cls(**payload)


ClassificationKind = Literal["cell_type", "cell_state"]


@dataclass(frozen=True)
class ClassificationManifest:
    """Project manifest for one immutable row-aligned classification."""

    artifact_id: str
    project_uuid: str
    classification_kind: ClassificationKind
    definition_id: str
    class_ids: tuple[str, ...]
    population_snapshot_id: str | None = None
    role: str = ""
    source_artifact_ids: tuple[str, ...] = ()
    algorithm: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    creator: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    content_digest: str | None = None
    schema: str = CLASSIFICATION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        for name in ("artifact_id", "project_uuid", "definition_id"):
            object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "population_snapshot_id",
            _optional_uuid_text(self.population_snapshot_id, field_name="population_snapshot_id"),
        )
        class_ids = tuple(_uuid_text(item, field_name="class_id") for item in self.class_ids)
        if not class_ids or len(class_ids) != len(set(class_ids)):
            raise ValueError("class_ids must be a non-empty ordered set of UUIDs")
        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(
            self,
            "source_artifact_ids",
            tuple(_uuid_text(item, field_name="source_artifact_id") for item in self.source_artifact_ids),
        )
        if self.classification_kind not in {"cell_type", "cell_state"}:
            raise ValueError(f"Unsupported classification_kind {self.classification_kind!r}")
        if self.schema != CLASSIFICATION_MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported classification manifest schema {self.schema!r}")
        if self.content_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", self.content_digest):
            raise ValueError("content_digest must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClassificationManifest":
        payload = dict(data)
        payload["class_ids"] = tuple(payload.get("class_ids") or ())
        payload["source_artifact_ids"] = tuple(payload.get("source_artifact_ids") or ())
        return cls(**payload)


@dataclass(frozen=True)
class BiologyRelease:
    """Immutable selection of named biological interpretation roles."""

    release_id: str
    project_uuid: str
    name: str
    type_taxonomy_id: str
    type_classifications: dict[str, str]
    state_spaces: tuple[str, ...] = ()
    state_classifications: dict[str, str] = field(default_factory=dict)
    representations: tuple[str, ...] = ()
    dependencies: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    created_by: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    schema: str = BIOLOGY_RELEASE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("release_id", "project_uuid", "type_taxonomy_id"):
            object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        if not str(self.name or "").strip():
            raise ValueError("release name must not be empty")
        object.__setattr__(
            self,
            "type_classifications",
            {
                str(role): _uuid_text(value, field_name=f"type classification role {role!r}")
                for role, value in self.type_classifications.items()
            },
        )
        object.__setattr__(
            self,
            "state_classifications",
            {
                str(role): _uuid_text(value, field_name=f"state classification role {role!r}")
                for role, value in self.state_classifications.items()
            },
        )
        object.__setattr__(
            self,
            "state_spaces",
            tuple(_uuid_text(item, field_name="state_space") for item in self.state_spaces),
        )
        object.__setattr__(
            self,
            "representations",
            tuple(_uuid_text(item, field_name="representation") for item in self.representations),
        )
        if self.schema != BIOLOGY_RELEASE_SCHEMA:
            raise ValueError(f"Unsupported biology-release schema {self.schema!r}")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BiologyRelease":
        payload = dict(data)
        payload["state_spaces"] = tuple(payload.get("state_spaces") or ())
        payload["representations"] = tuple(payload.get("representations") or ())
        return cls(**payload)


@dataclass(frozen=True)
class MaterializationReceipt:
    """Receipt linking a project artifact to one H5 object-set materialization."""

    artifact_id: str
    artifact_digest: str
    project_uuid: str
    dataset_uuid: str
    roi_uuid: str
    object_set: str
    observation_spine_digest: str
    h5_path: str
    h5_interpretation_path: str
    writer_version: str
    written_at: str = field(default_factory=utc_now_iso)
    schema: str = MATERIALIZATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("artifact_id", "project_uuid", "dataset_uuid", "roi_uuid"):
            object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        for name in ("artifact_digest", "observation_spine_digest"):
            value = str(getattr(self, name))
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if self.schema != MATERIALIZATION_RECEIPT_SCHEMA:
            raise ValueError(f"Unsupported receipt schema {self.schema!r}")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterializationReceipt":
        return cls(**dict(data))


def observation_spine_digest(
    observations: Any,
    *,
    object_set: str,
    observation_schema: str | Mapping[str, Any] = "celltraj2.observations.v1",
) -> str:
    """Hash identity-relevant ordered observation columns and schema metadata."""

    np = _require_numpy()
    values = np.asarray(observations)
    names = set(values.dtype.names or ())
    required = ("observation_id", "frame", "label_id")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Observation table is missing identity columns: {missing}")
    count = int(values.shape[0])
    ids = np.asarray(values["observation_id"], dtype="<u8")
    if count and not np.array_equal(ids, np.arange(1, count + 1, dtype="<u8")):
        raise ValueError("Observation rows must align to one-based observation_id values")
    header = {
        "schema": "celltraj2.observation_spine_digest.v1",
        "object_set": str(object_set),
        "row_count": count,
        "observation_schema": _json_safe(observation_schema),
        "columns": [
            ["observation_id", "uint64"],
            ["frame", "int64"],
            ["label_id", "int64"],
        ],
    }
    digest = hashlib.sha256(canonical_json_bytes(header))
    digest.update(ids.tobytes(order="C"))
    digest.update(np.asarray(values["frame"], dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(values["label_id"], dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def classification_values_dtype() -> Any:
    """Return the canonical H5 value dtype for a classification fragment."""

    np = _require_numpy()
    return np.dtype(
        [
            ("observation_id", "<u8"),
            ("class_index", "<u4"),
            ("assignment_status", "u1"),
            ("confidence", "<f4"),
            ("qc_flags", "<u4"),
        ]
    )


@dataclass(frozen=True)
class ClassificationResult:
    """Validated in-memory classification values and dense posterior matrix."""

    values: Any
    probabilities: Any
    review_records: tuple[dict[str, Any], ...] = ()


def validate_classification_arrays(
    values: Any,
    probabilities: Any,
    *,
    class_count: int,
    expected_observation_ids: Any | None = None,
) -> tuple[Any, Any]:
    """Validate and normalize canonical classification arrays."""

    np = _require_numpy()
    table = np.asarray(values)
    required = set(classification_values_dtype().names or ())
    missing = required.difference(table.dtype.names or ())
    if missing:
        raise ValueError(f"Classification values are missing fields: {sorted(missing)}")
    count = int(table.shape[0])
    if table.ndim != 1:
        raise ValueError("Classification values must be a one-dimensional structured array")
    class_count = int(class_count)
    if class_count < 1:
        raise ValueError("class_count must be >= 1")
    ids = np.asarray(table["observation_id"], dtype=np.uint64)
    if ids.size != np.unique(ids).size or np.any(ids < 1):
        raise ValueError("classification observation_id values must be unique and one-based")
    if expected_observation_ids is not None and not np.array_equal(
        ids,
        np.asarray(expected_observation_ids, dtype=np.uint64),
    ):
        raise ValueError("Classification observation IDs do not match the target observation spine")
    indices = np.asarray(table["class_index"], dtype=np.uint64)
    if np.any(indices > class_count):
        raise ValueError("classification class_index exceeds the ordered class registry")
    statuses = np.asarray(table["assignment_status"], dtype=np.uint8)
    valid_statuses = np.asarray(sorted(CLASSIFICATION_STATUS_NAMES), dtype=np.uint8)
    if np.any(~np.isin(statuses, valid_statuses)):
        raise ValueError("classification contains an unknown assignment_status code")
    assigned = statuses == CLASSIFICATION_STATUS_CODES["assigned"]
    if np.any(assigned & (indices == 0)):
        raise ValueError("assigned classification rows require a nonzero class_index")
    if np.any((~assigned) & (indices != 0)):
        raise ValueError("only assigned classification rows may carry a hard class_index")
    matrix = np.asarray(probabilities, dtype=np.float32)
    if matrix.shape != (count, class_count):
        raise ValueError(
            f"Classification probabilities have shape {matrix.shape}; expected {(count, class_count)}"
        )
    if np.any(np.isinf(matrix)):
        raise ValueError("classification probabilities may not contain infinities")
    finite = np.isfinite(matrix)
    if np.any(finite & ((matrix < 0.0) | (matrix > 1.0))):
        raise ValueError("classification probabilities must lie in [0, 1]")
    mixed_nan = np.any(np.isnan(matrix), axis=1) & ~np.all(np.isnan(matrix), axis=1)
    if np.any(mixed_nan):
        raise ValueError("a posterior row must be entirely present or entirely NaN")
    supported = assigned | (statuses == CLASSIFICATION_STATUS_CODES["ambiguous"])
    sums = np.nansum(matrix, axis=1)
    if np.any(supported & ~np.isclose(sums, 1.0, atol=1e-5)):
        raise ValueError("assigned and ambiguous posterior rows must sum to one")
    normalized = np.empty(count, dtype=classification_values_dtype())
    for name in normalized.dtype.names or ():
        normalized[name] = table[name]
    return normalized, np.ascontiguousarray(matrix, dtype=np.float32)


def compile_gate_memberships(
    observation_ids: Any,
    memberships: Any,
    *,
    excluded: Any | None = None,
) -> ClassificationResult:
    """Compile deterministic leaf-gate memberships into a classification."""

    np = _require_numpy()
    ids = np.asarray(observation_ids, dtype=np.uint64)
    matrix = np.asarray(memberships, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != ids.size or matrix.shape[1] < 1:
        raise ValueError("memberships must have shape (observation_count, class_count)")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("gate memberships must be finite and nonnegative")
    selected = matrix > 0.0
    selected_count = np.sum(selected, axis=1)
    probabilities = np.zeros_like(matrix, dtype=np.float32)
    supported = selected_count > 0
    probabilities[supported] = selected[supported].astype(np.float32) / selected_count[supported, None]
    values = np.zeros(ids.size, dtype=classification_values_dtype())
    values["observation_id"] = ids
    values["assignment_status"] = CLASSIFICATION_STATUS_CODES["unknown"]
    values["confidence"] = np.nan
    exactly_one = selected_count == 1
    if np.any(exactly_one):
        values["assignment_status"][exactly_one] = CLASSIFICATION_STATUS_CODES["assigned"]
        values["class_index"][exactly_one] = np.argmax(selected[exactly_one], axis=1).astype(np.uint32) + 1
        values["confidence"][exactly_one] = 1.0
    overlapping = selected_count > 1
    if np.any(overlapping):
        values["assignment_status"][overlapping] = CLASSIFICATION_STATUS_CODES["ambiguous"]
        values["confidence"][overlapping] = np.max(probabilities[overlapping], axis=1)
    if excluded is not None:
        excluded_mask = np.asarray(excluded, dtype=bool)
        if excluded_mask.shape != (ids.size,):
            raise ValueError("excluded mask must have shape (observation_count,)")
        values["assignment_status"][excluded_mask] = CLASSIFICATION_STATUS_CODES["excluded"]
        values["class_index"][excluded_mask] = 0
        values["confidence"][excluded_mask] = np.nan
        probabilities[excluded_mask] = 0.0
    normalized, probabilities = validate_classification_arrays(
        values,
        probabilities,
        class_count=matrix.shape[1],
        expected_observation_ids=ids,
    )
    return ClassificationResult(values=normalized, probabilities=probabilities)


def type_class_ancestors(
    taxonomy: TypeTaxonomy, class_ids: Sequence[str]
) -> dict[int, frozenset[int]]:
    """Map one-based posterior columns to their registered strict ancestors."""

    registry = list(class_ids)
    parents = {node.type_id: node.parent_type_id for node in taxonomy.nodes}
    if len(set(registry)) != len(registry) or set(registry).difference(parents):
        raise ValueError("Class IDs must be unique members of the type taxonomy")
    indices = {class_id: index + 1 for index, class_id in enumerate(registry)}
    ancestors: dict[int, frozenset[int]] = {}
    for class_id, index in indices.items():
        found: set[int] = set()
        parent = parents[class_id]
        while parent is not None:
            if parent in indices:
                found.add(indices[parent])
            parent = parents[parent]
        ancestors[index] = frozenset(found)
    return ancestors


def reconcile_type_candidates(
    candidates: Sequence[int] | set[int], ancestors: Mapping[int, frozenset[int]]
) -> set[int]:
    """Keep deepest supported candidates; incompatible branches remain separate."""

    supported = set(candidates)
    superseded = {ancestor for index in supported for ancestor in ancestors.get(index, ())}
    return supported.difference(superseded)


def enforce_track_type_constancy(
    values: Any,
    probabilities: Any,
    assignments: Any,
    *,
    links: Any | None = None,
    taxonomy: TypeTaxonomy | None = None,
    class_ids: Sequence[str] | None = None,
) -> ClassificationResult:
    """Compile one hard type per accepted rooted lineage while retaining posteriors.

    The input probabilities remain observation-level evidence.  The returned
    hard calls are constant over every connected accepted lineage.  Conflicting
    support produces an ambiguous lineage and review records instead of a type
    transition. With a taxonomy, compatible ancestor/descendant support resolves
    to the deepest supported type. Omitting the taxonomy preserves flat behavior.
    Supported posterior rows are unchanged; filled rows are constraint-derived.
    """

    np = _require_numpy()
    table = np.asarray(values)
    matrix = np.asarray(probabilities, dtype=np.float32)
    track_rows = np.asarray(assignments)
    required = {"observation_id", "lineage_id", "n_children"}
    missing = required.difference(track_rows.dtype.names or ())
    if missing:
        raise ValueError(f"Track assignments are missing fields: {sorted(missing)}")
    normalized, matrix = validate_classification_arrays(
        table,
        matrix,
        class_count=matrix.shape[1],
        expected_observation_ids=track_rows["observation_id"],
    )
    result = normalized.copy()
    raw_matrix = matrix.copy()
    matrix = matrix.copy()
    registry = list(class_ids) if class_ids is not None else (
        [node.type_id for node in taxonomy.nodes] if taxonomy is not None else []
    )
    if taxonomy is not None and len(registry) != matrix.shape[1]:
        raise ValueError("Class IDs must match posterior columns")
    if taxonomy is None and class_ids is not None:
        raise ValueError("Class IDs require a type taxonomy")
    ancestors = type_class_ancestors(taxonomy, registry) if taxonomy is not None else {}
    status_assigned = CLASSIFICATION_STATUS_CODES["assigned"]
    status_not_evaluated = CLASSIFICATION_STATUS_CODES["not_evaluated"]
    status_unknown = CLASSIFICATION_STATUS_CODES["unknown"]
    status_ambiguous = CLASSIFICATION_STATUS_CODES["ambiguous"]
    status_excluded = CLASSIFICATION_STATUS_CODES["excluded"]
    for lineage_id in np.unique(track_rows["lineage_id"]):
        rows = np.flatnonzero(track_rows["lineage_id"] == lineage_id)
        active_rows = rows[result["assignment_status"][rows] != status_excluded]
        if not active_rows.size:
            continue
        candidate_classes: set[int] = set(
            int(value)
            for value in result["class_index"][active_rows]
            if int(value) > 0
        )
        ambiguous_rows = active_rows[result["assignment_status"][active_rows] == status_ambiguous]
        for row in ambiguous_rows.tolist():
            candidate_classes.update(int(item) + 1 for item in np.flatnonzero(matrix[row] > 0.0))
        candidate_classes = reconcile_type_candidates(candidate_classes, ancestors)
        had_unknown = bool(
            np.any(
                np.isin(
                    result["assignment_status"][active_rows],
                    np.asarray([status_unknown, status_not_evaluated], dtype=np.uint8),
                )
            )
        )
        suspicious_branch = bool(np.any(track_rows["n_children"][rows] > 1))
        if len(candidate_classes) == 1:
            class_index = next(iter(candidate_classes))
            derived = result["class_index"][active_rows] != class_index
            unsupported = active_rows[
                ~np.isclose(np.nansum(matrix[active_rows], axis=1), 1.0, atol=1e-5)
            ]
            if unsupported.size:
                matrix[unsupported] = 0.0
                matrix[unsupported, class_index - 1] = 1.0
            result["assignment_status"][active_rows] = status_assigned
            result["class_index"][active_rows] = class_index
            # Derived fills must not inflate the evidence-based support summary.
            supported = active_rows[np.isclose(np.nansum(raw_matrix[active_rows], axis=1), 1.0)]
            class_support = raw_matrix[supported, class_index - 1]
            finite_support = class_support[np.isfinite(class_support)]
            result["confidence"][active_rows] = (
                np.float32(np.mean(finite_support)) if finite_support.size else np.nan
            )
            result["qc_flags"][active_rows[derived]] |= QC_TRACK_CONSTRAINT_DERIVED
            if had_unknown:
                result["qc_flags"][active_rows] |= QC_TRACK_INCOMPLETE_SUPPORT
        elif len(candidate_classes) > 1:
            unsupported = active_rows[
                ~np.isclose(np.nansum(matrix[active_rows], axis=1), 1.0, atol=1e-5)
            ]
            if unsupported.size:
                supported_rows = active_rows[
                    np.isclose(np.nansum(matrix[active_rows], axis=1), 1.0, atol=1e-5)
                ]
                if supported_rows.size:
                    consensus = np.nansum(matrix[supported_rows], axis=0)
                    consensus_sum = float(np.nansum(consensus))
                else:
                    consensus = np.zeros(matrix.shape[1], dtype=np.float32)
                    consensus_sum = 0.0
                if consensus_sum <= 0.0:
                    consensus[list(index - 1 for index in candidate_classes)] = 1.0
                    consensus_sum = float(len(candidate_classes))
                matrix[unsupported] = consensus / consensus_sum
                result["qc_flags"][unsupported] |= QC_TRACK_CONSTRAINT_DERIVED
            result["assignment_status"][active_rows] = status_ambiguous
            result["class_index"][active_rows] = 0
            for row in active_rows.tolist():
                finite_row = matrix[row][np.isfinite(matrix[row])]
                result["confidence"][row] = (
                    np.float32(np.max(finite_row)) if finite_row.size else np.nan
                )
            result["qc_flags"][active_rows] |= QC_TRACK_TYPE_CONFLICT
            if had_unknown:
                result["qc_flags"][active_rows] |= QC_TRACK_INCOMPLETE_SUPPORT
        else:
            result["assignment_status"][active_rows] = status_unknown
            result["class_index"][active_rows] = 0
            result["confidence"][active_rows] = np.nan
        if suspicious_branch:
            result["qc_flags"][rows] |= QC_SUSPICIOUS_BRANCH
    review_records = tuple(track_type_review_records(
        raw_matrix, track_rows, links=links, class_ancestors=ancestors
    ))
    switch_ids: set[int] = set()
    for record in review_records:
        if record.get("kind") == "posterior_switch":
            switch_ids.add(int(record["parent_observation_id"]))
            switch_ids.add(int(record["child_observation_id"]))
    if switch_ids:
        switch_rows = np.isin(result["observation_id"], np.asarray(sorted(switch_ids), dtype=np.uint64))
        result["qc_flags"][switch_rows] |= QC_POSTERIOR_SWITCH
    result, matrix = validate_classification_arrays(
        result,
        matrix,
        class_count=matrix.shape[1],
        expected_observation_ids=track_rows["observation_id"],
    )
    return ClassificationResult(
        values=result,
        probabilities=matrix,
        review_records=review_records,
    )


def track_type_review_records(
    probabilities: Any,
    assignments: Any,
    *,
    links: Any | None = None,
    class_ancestors: Mapping[int, frozenset[int]] | None = None,
) -> list[dict[str, Any]]:
    """Return posterior-discordance and suspicious-branch review records."""

    np = _require_numpy()
    matrix = np.asarray(probabilities, dtype=np.float32)
    track_rows = np.asarray(assignments)
    records: list[dict[str, Any]] = []
    row_by_id = {int(row["observation_id"]): index for index, row in enumerate(track_rows)}
    ancestors = class_ancestors or {}
    if links is not None:
        link_rows = np.asarray(links)
        required = {"parent_observation_id", "child_observation_id"}
        missing = required.difference(link_rows.dtype.names or ())
        if missing:
            raise ValueError(f"Track links are missing fields: {sorted(missing)}")
        for link in link_rows:
            parent_id = int(link["parent_observation_id"])
            child_id = int(link["child_observation_id"])
            if parent_id not in row_by_id or child_id not in row_by_id:
                raise ValueError("Track link references an observation outside the assignments")
            parent = matrix[row_by_id[parent_id]]
            child = matrix[row_by_id[child_id]]
            if not np.all(np.isfinite(parent)) or not np.all(np.isfinite(child)):
                continue
            parent_sum = float(np.sum(parent))
            child_sum = float(np.sum(child))
            if parent_sum <= 0.0 or child_sum <= 0.0:
                continue
            parent_norm = parent / parent_sum
            child_norm = child / child_sum
            parent_argmax = int(np.argmax(parent_norm)) + 1
            child_argmax = int(np.argmax(child_norm)) + 1
            distance = float(0.5 * np.sum(np.abs(parent_norm - child_norm)))
            if parent_argmax != child_argmax:
                records.append(
                    {
                        "schema": "celltraj2.type_track_review.v1",
                        "kind": (
                            "hierarchy_refinement"
                            if parent_argmax in ancestors.get(child_argmax, ())
                            or child_argmax in ancestors.get(parent_argmax, ())
                            else "posterior_switch"
                        ),
                        "parent_observation_id": parent_id,
                        "child_observation_id": child_id,
                        "parent_class_index": parent_argmax,
                        "child_class_index": child_argmax,
                        "posterior_total_variation": distance,
                    }
                )
            elif distance > 0.25:
                records.append(
                    {
                        "schema": "celltraj2.type_track_review.v1",
                        "kind": "posterior_discordance",
                        "parent_observation_id": parent_id,
                        "child_observation_id": child_id,
                        "class_index": parent_argmax,
                        "posterior_total_variation": distance,
                    }
                )
    for row in track_rows[np.asarray(track_rows["n_children"], dtype=np.int64) > 1]:
        records.append(
            {
                "schema": "celltraj2.type_track_review.v1",
                "kind": "suspicious_branch",
                "parent_observation_id": int(row["observation_id"]),
                "child_count": int(row["n_children"]),
                "interpretation": "probable_segmentation_or_tracking_error",
                "biological_event_created": False,
            }
        )
    return records


def classification_content_digest(
    values: Any,
    probabilities: Any,
    schema: Mapping[str, Any],
) -> str:
    """Return a deterministic content digest for one classification fragment."""

    np = _require_numpy()
    table = np.asarray(values)
    matrix = np.asarray(probabilities, dtype="<f4")
    digest = hashlib.sha256(canonical_json_bytes(dict(schema)))
    digest.update(str(table.dtype.descr).encode("utf-8"))
    digest.update(np.ascontiguousarray(table).tobytes(order="C"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def confidence_is_missing(value: Any) -> bool:
    """Return whether a scalar confidence is intentionally absent."""

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False
