"""Physical-time trajectory eligibility for State exploration.

This module builds keyed windows and exact-leaf lag pairs from an explicitly
accepted graph. It never infers tracking edges or estimates a kinetic operator.
All input anchors survive, with explicit invalidity instead of padding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid5

from celltraj2.interpretation import ProjectObservationKey, TypeTaxonomy, canonical_json_digest


TRAJECTORY_WINDOW_SPEC_SCHEMA = "site.trajectory_window_spec.v1"
TRAJECTORY_WINDOW_RESULT_SCHEMA = "site.trajectory_window_result.v1"
PHYSICAL_TIMEBASE_SCHEMA = "site.physical_timebase.v1"
_NAMESPACE = UUID("a3f88904-5b59-4dbe-89a8-d7a80a629ccd")
_UNIT_SECONDS = {"s": 1.0, "ms": .001, "min": 60.0, "h": 3600.0}


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if not _finite(value) or (float(value) < 0 if allow_zero else float(value) <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def _key(value: Any) -> ProjectObservationKey:
    if isinstance(value, ProjectObservationKey):
        return value
    if isinstance(value, Mapping) and "key" in value:
        return _key(value["key"])
    return ProjectObservationKey.from_dict(value)


def _json(value: Any) -> Any:
    if isinstance(value, ProjectObservationKey):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _identifier(kind: str, payload: Mapping[str, Any]) -> str:
    return str(uuid5(_NAMESPACE, kind + ":" + canonical_json_digest(_json(payload))))


@dataclass(frozen=True)
class TrajectoryWindowSpec:
    """Past-only delay samples and optional physical-lag pairs, in seconds."""

    length: int = 1
    stride: int = 1
    sample_interval_s: float | None = None
    lag_s: float | None = None
    tolerance_s: float = 0.0
    time_start_s: float | None = None
    time_stop_s: float | None = None
    required_features: tuple[str, ...] = ()
    anchor: str = "last"
    mode: str = "past_only"
    branch_policy: str = "stop"
    gap_policy: str = "stop"
    interpolation: str = "none"
    padding: str = "none"
    schema: str = TRAJECTORY_WINDOW_SPEC_SCHEMA

    def __post_init__(self) -> None:
        for name in ("length", "stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("sample_interval_s", "lag_s"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "tolerance_s", _positive(self.tolerance_s, "tolerance_s", allow_zero=True))
        if self.length > 1 and self.sample_interval_s is None:
            raise ValueError("Delay windows require an explicit physical sample_interval_s")
        if self.sample_interval_s is not None and self.tolerance_s >= self.sample_interval_s / 2:
            raise ValueError("tolerance_s must be less than half the physical sample interval")
        if self.lag_s is not None and self.tolerance_s >= self.lag_s:
            raise ValueError("tolerance_s must be less than the physical lag")
        for name in ("time_start_s", "time_stop_s"):
            value = getattr(self, name)
            if value is not None:
                if not _finite(value):
                    raise ValueError(f"{name} must be finite")
                object.__setattr__(self, name, float(value))
        if self.time_start_s is not None and self.time_stop_s is not None and self.time_start_s > self.time_stop_s:
            raise ValueError("Physical time interval must have start <= stop")
        if (self.anchor, self.mode, self.branch_policy, self.gap_policy, self.interpolation, self.padding) != (
            "last", "past_only", "stop", "stop", "none", "none"
        ):
            raise ValueError("Checkpoint B supports past-only last anchors, stop policies, no interpolation or padding")
        features = tuple(str(value) for value in self.required_features)
        if any(not value for value in features) or len(set(features)) != len(features):
            raise ValueError("required_features must contain unique nonempty feature names")
        object.__setattr__(self, "required_features", features)
        if self.schema != TRAJECTORY_WINDOW_SPEC_SCHEMA:
            raise ValueError("Unsupported trajectory window spec schema")

    def to_dict(self) -> dict[str, Any]:
        return _json(vars(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryWindowSpec:
        return cls(**dict(data))


@dataclass(frozen=True)
class TimebaseSpec:
    """Recorded times or an explicitly declared frame-to-physical conversion."""

    source: str
    unit: str = "s"
    interval: float | None = None
    frame_origin: int = 1
    time_origin: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = PHYSICAL_TIMEBASE_SCHEMA

    def __post_init__(self) -> None:
        if self.source not in {"recorded_timestamps", "declared_interval"}:
            raise ValueError("Timebase source must be recorded_timestamps or declared_interval")
        if self.unit not in _UNIT_SECONDS:
            raise ValueError("Timebase requires physical units; frames are not seconds")
        if self.interval is not None:
            object.__setattr__(self, "interval", _positive(self.interval, "interval"))
        if self.source == "declared_interval" and self.interval is None:
            raise ValueError("Declared acquisition timebase requires an interval")
        if isinstance(self.frame_origin, bool) or not isinstance(self.frame_origin, int):
            raise ValueError("frame_origin must be an explicit integer")
        if not _finite(self.time_origin):
            raise ValueError("time_origin must be finite")
        object.__setattr__(self, "time_origin", float(self.time_origin))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.schema != PHYSICAL_TIMEBASE_SCHEMA:
            raise ValueError("Unsupported physical timebase schema")

    def to_dict(self) -> dict[str, Any]:
        return _json(vars(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimebaseSpec:
        return cls(**dict(data))


def resolve_physical_time(row: Mapping[str, Any], timebase: TimebaseSpec | None) -> tuple[float | None, str]:
    """Return seconds and a reason; missing recorded times never use frame values."""
    if timebase is None:
        return None, "timebase_unavailable"
    if timebase.source == "recorded_timestamps":
        if "time_s" in row:
            value, factor = row["time_s"], 1.0
        else:
            value, factor = row.get("physical_time"), _UNIT_SECONDS[timebase.unit]
        if not _finite(value):
            return None, "missing_recorded_time"
        resolved = float(value) * factor
        return (resolved, "") if math.isfinite(resolved) else (None, "nonfinite_physical_time")
    frame = row.get("frame")
    if not _finite(frame) or float(frame) != int(float(frame)):
        return None, "missing_frame_for_declared_timebase"
    if int(frame) < timebase.frame_origin:
        return None, "frame_before_time_origin"
    value = ((int(frame) - timebase.frame_origin) * timebase.interval + timebase.time_origin)
    resolved = value * _UNIT_SECONDS[timebase.unit]
    return (resolved, "") if math.isfinite(resolved) else (None, "nonfinite_physical_time")


@dataclass(frozen=True)
class TrajectoryWindowResult:
    observations: tuple[dict[str, Any], ...]
    windows: tuple[dict[str, Any], ...]
    lag_pairs: tuple[dict[str, Any], ...]
    edge_reviews: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...]
    spec: TrajectoryWindowSpec
    timebases: tuple[dict[str, Any], ...] = ()
    schema: str = TRAJECTORY_WINDOW_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAJECTORY_WINDOW_RESULT_SCHEMA:
            raise ValueError("Unsupported trajectory window result schema")
        if not isinstance(self.spec, TrajectoryWindowSpec):
            object.__setattr__(self, "spec", TrajectoryWindowSpec.from_dict(self.spec))
        for name in ("observations", "windows", "lag_pairs", "edge_reviews", "segments", "timebases"):
            object.__setattr__(self, name, tuple(dict(value) for value in getattr(self, name)))
        keys = [_key(row) for row in self.observations]
        if len(keys) != len(set(keys)) or [_key(row["anchor"]) for row in self.windows] != keys:
            raise ValueError("Window anchors must preserve the unique full observation spine and order")
        known = set(keys)
        times = {tuple(record["partition"]): TimebaseSpec.from_dict(record["timebase"])
                 for record in self.timebases}
        row_by_key = dict(zip(keys, self.observations))
        if len({record["edge_id"] for record in self.edge_reviews}) != len(self.edge_reviews):
            raise ValueError("Saved graph reviews require unique stable edge identities")
        valid_edges = {record["edge_id"]: (_key(record["source"]), _key(record["target"]))
                       for record in self.edge_reviews if record["valid"]}

        def validate_segment(record: Mapping[str, Any]) -> None:
            segment = [_key(key) for key in record["segment"]]
            if len(segment) != len(set(segment)) or set(segment) - known:
                raise ValueError("A valid segment requires unique known observation keys")
            if len({key.partition for key in segment}) != 1 or len({row_by_key[key].get("type_id") for key in segment}) != 1:
                raise ValueError("A valid segment cannot cross files or Type assignments")
            edge_ids = record["accepted_edge_ids"]
            expected = list(zip(segment, segment[1:]))
            if len(edge_ids) != len(expected) or [valid_edges.get(identifier) for identifier in edge_ids] != expected:
                raise ValueError("A valid segment must preserve all reviewed accepted edges")
            if [[a.to_dict(), b.to_dict()] for a, b in expected] != record["accepted_edges"]:
                raise ValueError("Segment edge keys disagree with accepted edge identities")
            if len(segment) > 1:
                for field_name in ("group_id", "split_id"):
                    values = [row_by_key[key].get(field_name) for key in segment]
                    if not all(values) or len(set(values)) != 1:
                        raise ValueError("A valid segment cannot cross missing or changed group/split membership")

        segment_by_key = {}
        for record in self.segments:
            members = [_key(key) for key in record["members"]]
            if not members or len(set(members)) != len(members) or set(members) - known:
                raise ValueError("Saved segments require unique known observation keys")
            if any(key in segment_by_key for key in members):
                raise ValueError("Saved segments must partition the full observation spine")
            expected_id = _identifier("segment", {"members": members, "edges": record["accepted_edge_ids"]})
            if record["segment_id"] != expected_id or record["graph_component"] != expected_id:
                raise ValueError("Segment content differs from its deterministic identity")
            if record["times_s"] != [row_by_key[key]["time_s"] for key in members]:
                raise ValueError("Segment times must match the frozen observation spine")
            if record["valid"] != all(row_by_key[key]["valid"] for key in members):
                raise ValueError("Segment validity must match the frozen observation spine")
            for field_name in ("type_id", "group_id", "split_id"):
                if record.get(field_name) != row_by_key[members[0]].get(field_name):
                    raise ValueError("Segment context differs from the frozen observation spine")
            validate_segment({**record, "segment": record["members"]})
            segment_by_key.update({key: expected_id for key in members})
        if set(segment_by_key) != known:
            raise ValueError("Saved segments must partition the full observation spine")

        for window in self.windows:
            if window["graph_component"] != segment_by_key[_key(window["anchor"])]:
                raise ValueError("Window graph component disagrees with its saved segment")
            if window["valid"] and (len(window["members"]) != self.spec.length or
                _key(window["members"][-1]) != _key(window["anchor"])):
                raise ValueError("Valid windows require exact length and last-observation anchor")
            if {_key(key) for key in window["members"]} - known:
                raise ValueError("Window references an unknown observation")
            if window["valid"]:
                validate_segment(window)
                segment = [_key(key) for key in window["segment"]]
                positions = {key: index for index, key in enumerate(segment)}
                member_positions = [positions[_key(key)] for key in window["members"]]
                if member_positions != sorted(member_positions) or segment[-1] != _key(window["anchor"]):
                    raise ValueError("Window samples must preserve chronological segment order")
            partition = _key(window["anchor"]).partition
            expected_id = _identifier("window", {"spec": self.spec.to_dict(),
                "record": {key: value for key, value in window.items() if key != "window_id"},
                "timebase": times[partition].to_dict() if partition in times else None})
            if window["window_id"] != expected_id:
                raise ValueError("Window content differs from its deterministic identity")
        for pair in self.lag_pairs:
            if pair["valid"]:
                segment = [_key(key) for key in pair["segment"]]
                if (len(segment) < 2 or segment[0] != _key(pair["source_anchor"])
                        or segment[-1] != _key(pair["target_anchor"]) or set(segment) - known):
                    raise ValueError("Valid lag pairs require their complete known source-to-target segment")
                if not _finite(pair["elapsed_time"]) or pair["elapsed_time"] <= 0 or not pair.get("leaf_type_id"):
                    raise ValueError("Valid lag pairs require physical elapsed time and an exact leaf")
                validate_segment(pair)
                if any(row_by_key[key].get("resolved_leaf_type_id") != pair["leaf_type_id"] for key in segment):
                    raise ValueError("Valid lag pairs require one resolved exact leaf throughout")
            partition = _key(pair["source_anchor"]).partition
            expected_id = _identifier("lag_pair", {"spec": self.spec.to_dict(),
                "record": {key: value for key, value in pair.items() if key != "pair_id"},
                "timebase": times[partition].to_dict() if partition in times else None})
            if pair["pair_id"] != expected_id:
                raise ValueError("Lag-pair content differs from its deterministic identity")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "spec": self.spec.to_dict(), **{
            name: _json(getattr(self, name)) for name in (
                "observations", "windows", "lag_pairs", "edge_reviews", "segments", "timebases")}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrajectoryWindowResult:
        return cls(**dict(data))


def _feature_valid(row: Mapping[str, Any], required: Sequence[str]) -> bool:
    mask = row.get("feature_valid")
    if isinstance(mask, Mapping):
        return all(bool(mask.get(name, False)) for name in required) if required else all(bool(v) for v in mask.values())
    if mask is not None:
        return bool(mask)
    features = row.get("features", row)
    return all(_finite(features.get(name)) for name in required)


def build_trajectory_windows(
    rows: Sequence[Mapping[str, Any]], accepted_edges: Sequence[Mapping[str, Any]],
    spec: TrajectoryWindowSpec, *, taxonomy: TypeTaxonomy,
    timebases: Mapping[tuple[str, str, str], TimebaseSpec | Mapping[str, Any]] | None = None,
    kinetic_leaf_id: str | None = None, cancelled: Callable[[], bool] | None = None,
    max_expanded_members: int = 250_000,
) -> TrajectoryWindowResult:
    """Build reproducible delays/leaf-local pair eligibility without interpolation.

    Rows contain exact observation-key fields (or ``key``), ``frame``,
    ``type_id``, ``assignment_status``, ``group_id`` and ``split_id``. Optional
    ``eligible``, ``feature_valid`` and ``censoring_barrier`` mark restrictions.
    Recorded ``time_s`` is seconds; ``physical_time`` uses the TimebaseSpec unit.
    Edges require source/target keys, a stable edge_id and ``accepted=True``.
    Supply the full reviewed graph, including scoped-out endpoints, so branches
    cannot disappear when the State membership changes.
    """
    if not isinstance(spec, TrajectoryWindowSpec):
        spec = TrajectoryWindowSpec.from_dict(spec)
    if isinstance(max_expanded_members, bool) or not isinstance(max_expanded_members, int) or max_expanded_members < 1:
        raise ValueError("max_expanded_members must be a positive integer")
    expanded_members = 0

    def reserve_members(count: int) -> None:
        # Include duplicated segment keys and accepted-edge endpoints, not just
        # sampled delays. Long lag intervals otherwise grow quadratically.
        nonlocal expanded_members
        expanded_members += count
        if expanded_members > max_expanded_members:
            raise ValueError("Trajectory expansion exceeds max_expanded_members="
                             f"{max_expanded_members:,}; reduce the population/window/lag or increase the explicit limit")
    leaves = {node.type_id for node in taxonomy.leaf_nodes}
    nodes = {node.type_id for node in taxonomy.nodes}
    if kinetic_leaf_id is not None and kinetic_leaf_id not in leaves:
        raise ValueError("The kinetic selector requires one exact Type leaf")
    timebases = {tuple(partition): (value if isinstance(value, TimebaseSpec) else TimebaseSpec.from_dict(value))
                 for partition, value in (timebases or {}).items()}

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise RuntimeError("cancelled")

    observations, keyed, reasons = [], {}, {}
    for source in rows:
        check_cancelled()
        reserve_members(1)
        key = _key(source)
        if key in keyed:
            raise ValueError("Duplicate project observation key in trajectory source")
        row = {**dict(source), **key.to_dict()}
        row.pop("key", None)
        issues = []
        if not row.get("eligible", True):
            issues.append("outside_membership")
        if row.get("assignment_status", "assigned") not in (1, "assigned") or row.get("type_id") not in nodes:
            issues.append("unresolved_type_assignment")
        if row.get("type_incompatible", False):
            issues.append("incompatible_type_assignment")
        if not row.get("valid", True) or not _feature_valid(row, spec.required_features):
            issues.append("missing_required_features")
        time_s, time_issue = resolve_physical_time(row, timebases.get(key.partition))
        row.update(time_s=time_s, time_valid=time_s is not None, time_exclusion_reason=time_issue,
                   valid=not issues, exclusion_reason=issues[0] if issues else "", exclusion_reasons=issues)
        row["resolved_leaf_type_id"] = row.get("type_id") if row.get("type_id") in leaves and not issues else None
        observations.append(row)
        keyed[key], reasons[key] = row, issues
    if len({key.project_uuid for key in keyed}) > 1:
        raise ValueError("Trajectory source cannot mix projects")

    # Degrees use all reviewed edges before membership/validity filtering.
    edges, edge_ids, pairs_seen = [], set(), set()
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for original in accepted_edges:
        check_cancelled()
        reserve_members(2)
        source, target = _key(original["source"]), _key(original["target"])
        edge_id = str(original.get("edge_id") or "")
        if not edge_id or edge_id in edge_ids or (source, target) in pairs_seen:
            raise ValueError("Graph edges require unique stable edge IDs and source/target pairs")
        edge_ids.add(edge_id)
        pairs_seen.add((source, target))
        edge = {**dict(original), "source": source, "target": target, "edge_id": edge_id}
        edges.append(edge)
        if original.get("accepted") is True:
            outgoing[source].append(edge)
            incoming[target].append(edge)

    reviews, parents, children, permitted = [], {}, {}, {}
    blocked_incoming, blocked_outgoing = defaultdict(list), defaultdict(list)
    for edge in sorted(edges, key=lambda item: (item["source"], item["target"], item["edge_id"])):
        check_cancelled()
        source, target = edge["source"], edge["target"]
        issues = []
        if edge.get("accepted") is not True:
            issues.append("edge_not_accepted")
        if source not in keyed or target not in keyed:
            issues.append("edge_endpoint_unavailable")
        if source.project_uuid != target.project_uuid or source.partition != target.partition:
            issues.append("cross_file_edge")
        if len(outgoing[source]) > 1 or len(incoming[source]) > 1 or len(incoming[target]) > 1 or len(outgoing[target]) > 1:
            issues.append("branch_or_merge")
        if edge.get("suspect") or edge.get("censoring_barrier"):
            issues.append("suspect_or_censoring_edge")
        if source in keyed and target in keyed:
            left, right = keyed[source], keyed[target]
            if reasons[source] or reasons[target]:
                issues.append("ineligible_edge_member")
            if left.get("type_id") != right.get("type_id"):
                issues.append("type_conflict")
            if any(row.get("censoring_barrier") or row.get("censored") for row in (left, right)):
                issues.append("censoring_barrier")
            for field_name in ("group_id", "split_id"):
                if not left.get(field_name) or not right.get(field_name):
                    issues.append(f"missing_{field_name}")
                elif left[field_name] != right[field_name]:
                    issues.append(f"cross_{field_name}")
            if not _finite(left.get("frame")) or not _finite(right.get("frame")):
                issues.append("missing_frame")
            elif float(right["frame"]) - float(left["frame"]) != 1:
                issues.append("frame_gap_or_reverse")
            if left["time_s"] is None or right["time_s"] is None:
                issues.append("missing_physical_time")
            elif right["time_s"] <= left["time_s"]:
                issues.append("nonincreasing_physical_time")
            elif spec.sample_interval_s is not None and right["time_s"] - left["time_s"] > spec.sample_interval_s + spec.tolerance_s + 1e-12:
                issues.append("physical_time_gap")
        review = {"edge_id": edge["edge_id"], "source": source.to_dict(), "target": target.to_dict(),
                  "valid": not issues, "exclusion_reason": issues[0] if issues else "", "exclusion_reasons": issues}
        reviews.append(review)
        if issues:
            blocked_outgoing[source].extend(issues)
            blocked_incoming[target].extend(issues)
        else:
            parents[target], children[source] = source, target
            permitted[(source, target)] = edge["edge_id"]

    segments, segment_by_key = [], {}
    for root in sorted(key for key in keyed if key not in parents):
        check_cancelled()
        members, visited, current = [], set(), root
        while current is not None:
            check_cancelled()
            if current in segment_by_key or current in visited:
                raise ValueError("Accepted trajectory graph contains a cycle")
            reserve_members(3)
            members.append(current)
            visited.add(current)
            current = children.get(current)
        segment_id = _identifier("segment", {"members": members,
            "edges": [permitted[(a, b)] for a, b in zip(members, members[1:])]})
        for member in members:
            segment_by_key[member] = segment_id
        segments.append({"segment_id": segment_id, "members": [key.to_dict() for key in members],
            "times_s": [keyed[key]["time_s"] for key in members], "type_id": keyed[root].get("type_id"),
            "group_id": keyed[root].get("group_id"), "split_id": keyed[root].get("split_id"),
            "graph_component": segment_id,
            "valid": all(keyed[key]["valid"] for key in members),
            "accepted_edges": [[a.to_dict(), b.to_dict()] for a, b in zip(members, members[1:])],
            "accepted_edge_ids": [permitted[(a, b)] for a, b in zip(members, members[1:])]})
    if len(segment_by_key) != len(keyed):
        raise ValueError("Accepted trajectory graph contains an unresolved cycle")

    def window_record(anchor: ProjectObservationKey) -> dict[str, Any]:
        reserve_members(3)
        row, issues = keyed[anchor], list(reasons[anchor])
        members, segment = [anchor], [anchor]
        temporal = spec.length > 1 or spec.time_start_s is not None or spec.time_stop_s is not None
        if temporal and row["time_s"] is None:
            issues.append(row["time_exclusion_reason"])
        if row["time_s"] is not None:
            if spec.time_start_s is not None and row["time_s"] < spec.time_start_s:
                issues.append("outside_physical_interval")
            if spec.time_stop_s is not None and row["time_s"] > spec.time_stop_s:
                issues.append("outside_physical_interval")
        current = anchor
        for sample in range(1, spec.length):
            if issues:
                break
            expected = row["time_s"] - sample * spec.sample_interval_s
            candidate = parents.get(current)
            while candidate is not None and keyed[candidate]["time_s"] > expected + spec.tolerance_s + 1e-12:
                reserve_members(3)
                segment.append(candidate)
                current, candidate = candidate, parents.get(candidate)
                check_cancelled()
            if candidate is None:
                issues.extend(blocked_incoming.get(current, ()))
                issues.append("insufficient_history")
                break
            if abs(keyed[candidate]["time_s"] - expected) > spec.tolerance_s + 1e-12:
                issues.append("missing_delay_sample")
                break
            if spec.time_start_s is not None and keyed[candidate]["time_s"] < spec.time_start_s:
                issues.append("window_crosses_physical_interval")
                break
            reserve_members(4)
            segment.append(candidate)
            members.append(candidate)
            current = candidate
        members.reverse()
        segment.reverse()
        return {"anchor": anchor.to_dict(), "members": [key.to_dict() for key in members],
            "segment": [key.to_dict() for key in segment], "times_s": [keyed[key]["time_s"] for key in members],
            "valid": not issues, "exclusion_reason": issues[0] if issues else "",
            "exclusion_reasons": list(dict.fromkeys(issues)), "group_id": row.get("group_id"),
            "split_id": row.get("split_id"), "type_id": row.get("type_id"),
            "graph_component": segment_by_key[anchor],
            "accepted_edges": [[a.to_dict(), b.to_dict()] for a, b in zip(segment, segment[1:])],
            "accepted_edge_ids": [permitted[(a, b)] for a, b in zip(segment, segment[1:])]}

    windows = {}
    for key in keyed:
        check_cancelled()
        windows[key] = window_record(key)
    for segment in segments:
        eligible_index = 0
        for key_dict in segment["members"]:
            key = _key(key_dict)
            window = windows[key]
            if window["valid"]:
                if eligible_index % spec.stride:
                    window.update(valid=False, exclusion_reason="stride_excluded", exclusion_reasons=["stride_excluded"])
                eligible_index += 1
    for key, window in windows.items():
        window["window_id"] = _identifier("window", {"spec": spec.to_dict(), "record": window,
            "timebase": timebases[key.partition].to_dict() if key.partition in timebases else None})

    lag_pairs = []
    if spec.lag_s is not None:
        for source, row in keyed.items():
            check_cancelled()
            issues = list(windows[source]["exclusion_reasons"])
            leaf = row.get("type_id") if row.get("type_id") in leaves else None
            if leaf is None:
                issues.append("unresolved_exact_leaf")
            elif kinetic_leaf_id is not None and leaf != kinetic_leaf_id:
                issues.append("outside_kinetic_leaf")
            if row["time_s"] is None:
                issues.append(row["time_exclusion_reason"])
            reserve_members(1)
            segment, target = [source], None
            current = source
            if not issues:
                expected = row["time_s"] + spec.lag_s
                candidate = children.get(current)
                while candidate is not None and keyed[candidate]["time_s"] < expected - spec.tolerance_s - 1e-12:
                    reserve_members(3)
                    segment.append(candidate)
                    current, candidate = candidate, children.get(candidate)
                    check_cancelled()
                if candidate is None:
                    issues.extend(blocked_outgoing.get(current, ()))
                    issues.append("insufficient_future")
                elif abs(keyed[candidate]["time_s"] - expected) > spec.tolerance_s + 1e-12:
                    issues.append("missing_lag_target")
                else:
                    target = candidate
                    reserve_members(3)
                    segment.append(target)
                    if not windows[target]["valid"]:
                        issues.append("invalid_target_window")
            pair = {"source_anchor": source.to_dict(), "target_anchor": None if target is None else target.to_dict(),
                "segment": [key.to_dict() for key in segment], "leaf_type_id": leaf,
                "source_window_id": windows[source]["window_id"],
                "target_window_id": None if target is None else windows[target]["window_id"],
                "elapsed_time": None if target is None else keyed[target]["time_s"] - row["time_s"],
                "time_unit": "s", "weight": 1.0,
                "valid": not issues, "exclusion_reason": issues[0] if issues else "", "exclusion_reasons": list(dict.fromkeys(issues)),
                "split_ids": [keyed[key].get("split_id") for key in segment],
                "group_ids": [keyed[key].get("group_id") for key in segment],
                "graph_components": [segment_by_key[key] for key in segment],
                "accepted_edges": [[a.to_dict(), b.to_dict()] for a, b in zip(segment, segment[1:])],
                "accepted_edge_ids": [permitted[(a, b)] for a, b in zip(segment, segment[1:])]}
            pair["pair_id"] = _identifier("lag_pair", {"spec": spec.to_dict(), "record": pair,
                "timebase": timebases[source.partition].to_dict() if source.partition in timebases else None})
            lag_pairs.append(pair)
    return TrajectoryWindowResult(tuple(observations), tuple(windows.values()), tuple(lag_pairs),
        tuple(reviews), tuple(segments), spec,
        tuple({"partition": list(partition), "timebase": value.to_dict()} for partition, value in sorted(timebases.items())))
