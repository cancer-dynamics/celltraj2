"""Validated, dependency-light State membership and release contracts.

Shared State vocabulary/predictions and exact-leaf kinetic cohorts are separate
objects. These contracts validate saved scope; they do not estimate dynamics.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from celltraj2.interpretation import (
    ProjectObservationKey, TypeTaxonomy, _json_safe, _uuid_text,
    canonical_json_digest,
)


STATE_MEMBERSHIP_SCHEMA = "site.state_membership.v1"
STATE_COMPONENT_BINDING_SCHEMA = "site.state_component_binding.v1"
KINETIC_MODEL_BINDING_SCHEMA = "site.kinetic_model_binding.v1"
STATE_DEPENDENCY_SCHEMA = "site.state_dependency.v1"
BIOLOGY_RELEASE_V2_SCHEMA = "site.biology_release.v2"


def _digest(value: str, name: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return str(value)


def _keys(values: Sequence[Any], name: str) -> tuple[ProjectObservationKey, ...]:
    result = tuple(
        value if isinstance(value, ProjectObservationKey)
        else ProjectObservationKey.from_dict(value) for value in values
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate observation keys")
    return result


def _partition(value: Sequence[str]) -> tuple[str, str, str]:
    if len(value) != 3:
        raise ValueError("A portable partition requires dataset_uuid, roi_uuid, object_set")
    key = ProjectObservationKey(str(value[0]), str(value[0]), str(value[1]), str(value[2]), 1)
    return key.partition


def observation_key_token(key: ProjectObservationKey) -> str:
    """Portable JSON key for mappings indexed by exact observation identity."""
    return "/".join((key.project_uuid, key.dataset_uuid, key.roi_uuid,
                     key.object_set, str(key.observation_id)))


@dataclass(frozen=True)
class MembershipExpression:
    """Boolean AST over stable taxonomy UUIDs; no executable expressions."""

    op: str
    type_id: str | None = None
    children: tuple[MembershipExpression, ...] = ()

    def __post_init__(self) -> None:
        children = tuple(child if isinstance(child, MembershipExpression)
                         else MembershipExpression.from_dict(child) for child in self.children)
        object.__setattr__(self, "children", children)
        if self.op in {"in_subtree", "assigned_exact"}:
            if children:
                raise ValueError("Taxonomy predicates cannot have children")
            object.__setattr__(self, "type_id", _uuid_text(self.type_id, field_name="type_id"))
        elif self.op in {"and", "or", "not"}:
            if self.type_id is not None:
                raise ValueError("Boolean groups cannot have a type_id")
            if self.op == "not" and len(children) != 1:
                raise ValueError("NOT requires exactly one child")
            if self.op in {"and", "or"} and len(children) < 2:
                raise ValueError("AND/OR require at least two children")
        else:
            raise ValueError(f"Unsupported membership operator {self.op!r}")

    @property
    def canonical_expression(self) -> str:
        if self.type_id is not None:
            return f"{self.op}({self.type_id})"
        if self.op == "not":
            return f"(NOT {self.children[0].canonical_expression})"
        return "(" + f" {self.op.upper()} ".join(
            child.canonical_expression for child in self.children) + ")"

    @property
    def referenced_type_ids(self) -> frozenset[str]:
        if self.type_id is not None:
            return frozenset((self.type_id,))
        return frozenset().union(*(child.referenced_type_ids for child in self.children))

    def validate_taxonomy(self, taxonomy: TypeTaxonomy) -> None:
        missing = self.referenced_type_ids - {node.type_id for node in taxonomy.nodes}
        if missing:
            raise ValueError(f"Membership references missing taxonomy nodes: {sorted(missing)}")

    def matches(self, type_id: str | None, taxonomy: TypeTaxonomy) -> bool:
        """Evaluate within the known assignment universe, including under NOT."""
        self.validate_taxonomy(taxonomy)
        parents = {node.type_id: node.parent_type_id for node in taxonomy.nodes}
        if type_id not in parents:
            return False

        def evaluate(node: MembershipExpression) -> bool:
            if node.op == "assigned_exact":
                return type_id == node.type_id
            if node.op == "in_subtree":
                current = type_id
                while current is not None:
                    if current == node.type_id:
                        return True
                    current = parents[current]
                return False
            if node.op == "not":
                return not evaluate(node.children[0])
            if node.op == "and":
                return all(evaluate(child) for child in node.children)
            return any(evaluate(child) for child in node.children)

        return evaluate(self)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MembershipExpression:
        return cls(**dict(data))


@dataclass(frozen=True)
class StateDependency:
    """An input actually consumed by a fit, application or kinetic estimate."""

    kind: str
    artifact_id: str
    digest: str
    role: str = "application"
    observation_keys: tuple[ProjectObservationKey, ...] = ()
    partitions: tuple[tuple[str, str, str], ...] = ()
    schema: str = STATE_DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        if self.kind not in {
            "observation_spine", "features", "graph", "type_assignments",
            "type_probabilities", "membership", "definition", "representation",
            "model", "partition", "assignment", "assessment", "window", "pair",
            "timebase", "release", "artifact",
        }:
            raise ValueError(f"Unsupported State dependency kind {self.kind!r}")
        if self.role not in {"fit", "application", "kinetic", "definition"}:
            raise ValueError(f"Unsupported dependency role {self.role!r}")
        if self.schema != STATE_DEPENDENCY_SCHEMA:
            raise ValueError(f"Unsupported State dependency schema {self.schema!r}")
        object.__setattr__(self, "artifact_id", _uuid_text(self.artifact_id, field_name="artifact_id"))
        object.__setattr__(self, "digest", _digest(self.digest, "digest"))
        object.__setattr__(self, "observation_keys", _keys(self.observation_keys, "observation_keys"))
        partitions = tuple(_partition(item) for item in self.partitions)
        if len(partitions) != len(set(partitions)):
            raise ValueError("Duplicate dependency partitions")
        object.__setattr__(self, "partitions", partitions)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateDependency:
        return cls(**dict(data))


def _dependencies(values: Sequence[Any]) -> tuple[StateDependency, ...]:
    return tuple(item if isinstance(item, StateDependency) else StateDependency.from_dict(item)
                 for item in values)


@dataclass(frozen=True)
class StateMembershipSpec:
    """Frozen Boolean request plus effective portable files and exact keys."""

    membership_id: str
    taxonomy_id: str
    type_release_id: str
    type_classification_id: str
    type_role: str
    expression: MembershipExpression
    resolved_observations: tuple[ProjectObservationKey, ...] = ()
    resolved_partitions: tuple[tuple[str, str, str], ...] = ()
    treatment_ids: tuple[str, ...] = ()
    allowed_assignment_statuses: tuple[str, ...] = ("assigned",)
    resolved_leaves_only: bool = False
    source_dependencies: tuple[StateDependency, ...] = ()
    excluded_reasons: dict[str, str] = field(default_factory=dict)
    membership_summary: dict[str, Any] = field(default_factory=dict)
    conditioning_digest: str | None = None
    schema: str = STATE_MEMBERSHIP_SCHEMA

    def __post_init__(self) -> None:
        for name in ("membership_id", "taxonomy_id", "type_release_id", "type_classification_id"):
            object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        if not self.type_role.strip():
            raise ValueError("type_role must not be empty")
        if self.schema != STATE_MEMBERSHIP_SCHEMA:
            raise ValueError(f"Unsupported State membership schema {self.schema!r}")
        expr = self.expression
        if not isinstance(expr, MembershipExpression):
            expr = MembershipExpression.from_dict(expr)
        object.__setattr__(self, "expression", expr)
        # Hard-call eligibility is the first implementation's only supported policy.
        if tuple(self.allowed_assignment_statuses) != ("assigned",):
            raise ValueError("State membership currently admits only valid assigned hard calls")
        object.__setattr__(self, "allowed_assignment_statuses", ("assigned",))
        observations = tuple(sorted(_keys(self.resolved_observations, "resolved_observations")))
        partitions = tuple(sorted(_partition(item) for item in self.resolved_partitions))
        if len(partitions) != len(set(partitions)):
            raise ValueError("Duplicate resolved partitions")
        if len({key.project_uuid for key in observations}) > 1:
            raise ValueError("A membership cannot span projects")
        if any(key.partition not in partitions for key in observations):
            raise ValueError("Membership observations escape the frozen resolved file set")
        object.__setattr__(self, "resolved_observations", observations)
        object.__setattr__(self, "resolved_partitions", partitions)
        object.__setattr__(self, "treatment_ids", tuple(sorted(set(self.treatment_ids))))
        object.__setattr__(self, "source_dependencies", _dependencies(self.source_dependencies))
        # Exact Type leaf identity is deliberately absent: a shared prediction may
        # survive a retype that stays inside its requested membership.
        expected = canonical_json_digest([key.to_dict() for key in observations])
        if self.conditioning_digest is not None and self.conditioning_digest != expected:
            raise ValueError("conditioning_digest does not match the frozen observation membership")
        object.__setattr__(self, "conditioning_digest", expected)

    @property
    def canonical_expression(self) -> str:
        return self.expression.canonical_expression

    def eligible_assignment(self, type_id: str | None, assignment_status: str,
                            taxonomy: TypeTaxonomy, *, incompatible: bool = False) -> bool:
        if taxonomy.taxonomy_id != self.taxonomy_id:
            raise ValueError("Membership taxonomy revision does not match")
        self.expression.validate_taxonomy(taxonomy)
        if incompatible or assignment_status not in self.allowed_assignment_statuses:
            return False
        if self.resolved_leaves_only and type_id not in {node.type_id for node in taxonomy.leaf_nodes}:
            return False
        return self.expression.matches(type_id, taxonomy)

    def to_dict(self) -> dict[str, Any]:
        result = _json_safe(self)
        result["canonical_expression"] = self.canonical_expression
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateMembershipSpec:
        payload = dict(data)
        readable = payload.pop("canonical_expression", None)
        result = cls(**payload)
        if readable is not None and readable != result.canonical_expression:
            raise ValueError("Saved canonical expression disagrees with membership AST")
        return result


@dataclass(frozen=True)
class StateComponentBinding:
    """An authoritative channel output on a membership, without an owner leaf."""

    binding_id: str
    channel: str
    membership_id: str
    definition_id: str
    assignment_id: str
    conditioning_digest: str
    partition_id: str | None = None
    representation_id: str | None = None
    model_id: str | None = None
    assessment_id: str | None = None
    dependencies: tuple[StateDependency, ...] = ()
    coverage_observations: tuple[ProjectObservationKey, ...] | None = None
    compatibility_status: str = "unchanged"
    assignment_basis: str = "unspecified"
    schema: str = STATE_COMPONENT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in ("binding_id", "membership_id", "definition_id", "assignment_id",
                     "partition_id", "representation_id", "model_id", "assessment_id"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        if not self.channel.strip():
            raise ValueError("State channel must not be empty")
        if self.schema != STATE_COMPONENT_BINDING_SCHEMA:
            raise ValueError(f"Unsupported State binding schema {self.schema!r}")
        if self.compatibility_status not in {
            "unchanged", "subset_reusable", "needs_application", "needs_refit",
            "incompatible", "unverified", "unavailable",
        }:
            raise ValueError(f"Unsupported State compatibility status {self.compatibility_status!r}")
        if self.assignment_basis not in {"unspecified", "declared_homogeneous", "model", "rule", "manual"}:
            raise ValueError(f"Unsupported State assignment basis {self.assignment_basis!r}")
        object.__setattr__(self, "conditioning_digest", _digest(self.conditioning_digest, "conditioning_digest"))
        object.__setattr__(self, "dependencies", _dependencies(self.dependencies))
        if self.coverage_observations is not None:
            object.__setattr__(self, "coverage_observations", tuple(sorted(
                _keys(self.coverage_observations, "coverage_observations"))))

    @property
    def artifact_ids(self) -> frozenset[str]:
        return frozenset(value for value in (
            self.definition_id, self.assignment_id, self.partition_id,
            self.representation_id, self.model_id, self.assessment_id,
            *(dep.artifact_id for dep in self.dependencies),
        ) if value is not None)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateComponentBinding:
        return cls(**dict(data))


@dataclass(frozen=True)
class KineticModelBinding:
    """One exact Type leaf's cohort; no pooled operator can be represented."""

    binding_id: str
    state_binding_id: str
    leaf_type_id: str
    observation_cohort: tuple[ProjectObservationKey, ...]
    cohort_digest: str
    timebase: dict[str, Any]
    window_ids: tuple[str, ...] = ()
    pair_ids: tuple[str, ...] = ()
    partition_id: str | None = None
    score_mapping_id: str | None = None
    operator_id: str | None = None
    flux_id: str | None = None
    assessment_id: str | None = None
    dependencies: tuple[StateDependency, ...] = ()
    schema: str = KINETIC_MODEL_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in ("binding_id", "state_binding_id", "leaf_type_id", "partition_id",
                     "score_mapping_id", "operator_id", "flux_id", "assessment_id"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        if self.schema != KINETIC_MODEL_BINDING_SCHEMA:
            raise ValueError(f"Unsupported kinetic binding schema {self.schema!r}")
        observations = tuple(sorted(_keys(self.observation_cohort, "observation_cohort")))
        if not observations:
            raise ValueError("A kinetic cohort must not be empty")
        object.__setattr__(self, "observation_cohort", observations)
        for name in ("window_ids", "pair_ids"):
            values = tuple(_uuid_text(value, field_name=name) for value in getattr(self, name))
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {name}")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "dependencies", _dependencies(self.dependencies))
        if self.timebase.get("unit") not in {"s", "ms", "min", "h"} or not self.timebase.get("source"):
            raise ValueError("Kinetics require a declared physical timebase unit and source")
        expected = kinetic_cohort_digest(self.leaf_type_id, observations,
                                         self.window_ids, self.pair_ids, self.timebase)
        if self.cohort_digest != expected:
            raise ValueError("cohort_digest does not match the exact leaf/cohort/timebase")
        if (self.operator_id or self.flux_id) and not self.pair_ids:
            raise ValueError("A kinetic operator/flux requires an explicit lag-pair cohort")
        if self.operator_id or self.flux_id:
            if not (self.partition_id or self.score_mapping_id) or not self.assessment_id:
                raise ValueError("A kinetic operator/flux requires a partition/score mapping and assessment")

    def validate_context(self, taxonomy: TypeTaxonomy, membership: StateMembershipSpec,
                         exact_type_assignments: Mapping[ProjectObservationKey, str], *,
                         windows: Mapping[str, Mapping[str, Any]] | None = None,
                         pairs: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        """Check all admitted members, not only pair endpoints, against one leaf.

        Window records have ``anchor`` and ``members``. Pair records have
        ``source_anchor``, ``target_anchor``, ``segment``, ``elapsed_time`` and
        ``time_unit``, per-member ``split_ids`` and ``graph_components``, and
        ``accepted_edges`` (ordered pairs of observation keys). These are
        validation inputs, not window-building APIs.
        """
        if taxonomy.taxonomy_id != membership.taxonomy_id:
            raise ValueError("Kinetic taxonomy differs from the membership revision")
        if self.leaf_type_id not in {node.type_id for node in taxonomy.leaf_nodes}:
            raise ValueError("Kinetics require one exact Type leaf, never a parent node")
        cohort = set(self.observation_cohort)
        if not cohort.issubset(membership.resolved_observations):
            raise ValueError("Kinetic cohort escapes shared State membership")
        if any(exact_type_assignments.get(key) != self.leaf_type_id for key in cohort):
            raise ValueError("Kinetic cohort contains unresolved or mixed Type leaves")

        def key(value: Any) -> ProjectObservationKey:
            return value if isinstance(value, ProjectObservationKey) else ProjectObservationKey.from_dict(value)

        for window_id in self.window_ids:
            if windows is None or window_id not in windows:
                raise ValueError(f"Missing kinetic window reference {window_id}")
            window = windows[window_id]
            members = _keys(window["members"], "window members")
            segment = _keys(window.get("segment", window["members"]), "window segment")
            if not members or key(window["anchor"]) not in members or not set(members).issubset(cohort):
                raise ValueError("Kinetic window members/anchor must belong to its exact leaf cohort")
            if not set(members).issubset(segment) or not set(segment).issubset(cohort):
                raise ValueError("Entire kinetic window segment must belong to its exact leaf cohort")
            if len({member.partition for member in segment}) != 1:
                raise ValueError("Kinetic window crosses files")
        for pair_id in self.pair_ids:
            if pairs is None or pair_id not in pairs:
                raise ValueError(f"Missing kinetic pair reference {pair_id}")
            pair = pairs[pair_id]
            segment = _keys(pair["segment"], "pair segment")
            source, target = key(pair["source_anchor"]), key(pair["target_anchor"])
            if (len(segment) < 2 or segment[0] != source or segment[-1] != target
                    or not set(segment).issubset(cohort)):
                raise ValueError("Entire kinetic pair segment must belong to its exact leaf cohort")
            if len({member.partition for member in segment}) != 1:
                raise ValueError("Kinetic pair crosses files")
            elapsed = float(pair["elapsed_time"])
            if not 0 < elapsed < float("inf") or pair["time_unit"] != self.timebase["unit"]:
                raise ValueError("Kinetic pair requires positive physical elapsed time in the declared unit")
            for field_name in ("split_ids", "graph_components"):
                if field_name not in pair or len(pair[field_name]) != len(segment):
                    raise ValueError(f"Kinetic pair requires per-member {field_name}")
                if len(set(pair[field_name])) != 1 or not all(pair[field_name]):
                    raise ValueError(f"Kinetic pair crosses {field_name}")
            accepted_edges = pair.get("accepted_edges", ())
            if len(accepted_edges) != len(segment) - 1 or any(
                len(edge) != 2 or (key(edge[0]), key(edge[1])) != (source_key, target_key)
                for edge, source_key, target_key in zip(accepted_edges, segment, segment[1:])
            ):
                raise ValueError("Kinetic pair segment requires every accepted graph edge")
            if pair.get("censoring_barrier") or pair.get("excluded_reason"):
                raise ValueError("Kinetic pair crosses a censoring/exclusion barrier")

    @property
    def artifact_ids(self) -> frozenset[str]:
        return frozenset(value for value in (
            self.partition_id, self.score_mapping_id, self.operator_id, self.flux_id,
            self.assessment_id, *self.window_ids, *self.pair_ids,
            *(dep.artifact_id for dep in self.dependencies),
        ) if value is not None)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KineticModelBinding:
        return cls(**dict(data))


def kinetic_cohort_digest(leaf_type_id: str, observation_cohort: Sequence[ProjectObservationKey],
                          window_ids: Sequence[str] = (), pair_ids: Sequence[str] = (),
                          timebase: Mapping[str, Any] | None = None) -> str:
    return canonical_json_digest({
        "leaf_type_id": leaf_type_id,
        "observations": [key.to_dict() for key in sorted(observation_cohort)],
        "window_ids": sorted(window_ids), "pair_ids": sorted(pair_ids),
        "timebase": dict(timebase or {}),
    })


def release_required_artifact_ids(release: Any) -> frozenset[str]:
    result = {release.type_taxonomy_id, *release.type_classifications.values(),
              *release.state_spaces, *release.state_classifications.values(), *release.representations}
    for membership in release.state_memberships:
        result.update((membership.taxonomy_id, membership.type_release_id, membership.type_classification_id))
        result.update(dep.artifact_id for dep in membership.source_dependencies)
    for binding in (*release.state_bindings, *release.kinetic_bindings):
        result.update(binding.artifact_ids)
    return frozenset(result)


def validate_release_v2_structure(release: Any) -> None:
    """Validate embedded references/coverage and declared dependency closure."""
    memberships = {item.membership_id: item for item in release.state_memberships}
    states = {item.binding_id: item for item in release.state_bindings}
    if len(memberships) != len(release.state_memberships) or len(states) != len(release.state_bindings):
        raise ValueError("Duplicate membership or State binding IDs")
    if len({item.binding_id for item in release.kinetic_bindings}) != len(release.kinetic_bindings):
        raise ValueError("Duplicate kinetic binding IDs")
    covered: dict[str, set[ProjectObservationKey]] = {}
    for membership in memberships.values():
        if membership.taxonomy_id != release.type_taxonomy_id:
            raise ValueError("State membership uses a different taxonomy revision")
        if release.type_classifications.get(membership.type_role) != membership.type_classification_id:
            raise ValueError("Membership Type classification/role is not selected in the release")
        if any(key.project_uuid != release.project_uuid for key in membership.resolved_observations):
            raise ValueError("State membership belongs to a different project")
    for binding in states.values():
        if binding.membership_id not in memberships:
            raise ValueError("State binding references a missing membership")
        membership = memberships[binding.membership_id]
        if binding.conditioning_digest != membership.conditioning_digest:
            raise ValueError("State binding conditioning digest differs from its membership")
        scope = set(membership.resolved_observations if binding.coverage_observations is None
                    else binding.coverage_observations)
        if not scope.issubset(membership.resolved_observations):
            raise ValueError("State output coverage escapes membership")
        previous = covered.setdefault(binding.channel, set())
        if previous.intersection(scope):
            raise ValueError(f"Overlapping authoritative outputs for State channel {binding.channel!r}")
        previous.update(scope)
    for binding in release.kinetic_bindings:
        if binding.state_binding_id not in states:
            raise ValueError("Kinetic binding references a missing State binding")
        membership = memberships[states[binding.state_binding_id].membership_id]
        if not set(binding.observation_cohort).issubset(membership.resolved_observations):
            raise ValueError("Kinetic cohort escapes State membership")
    # Historical v1 State IDs are not evidence of a validated v2 scope.
    if set(release.state_classifications.values()) - {item.assignment_id for item in states.values()}:
        raise ValueError("Legacy State classifications require explicit validated binding adoption")
    closure: dict[str, str] = {}
    for dependency in release.dependency_closure:
        previous = closure.setdefault(dependency.artifact_id, dependency.digest)
        if previous != dependency.digest:
            raise ValueError("Conflicting digests in release dependency closure")
    missing = release_required_artifact_ids(release) - closure.keys()
    if missing:
        raise ValueError(f"Release dependency closure is missing required artifacts: {sorted(missing)}")
    for owner in (*release.state_memberships, *release.state_bindings, *release.kinetic_bindings):
        for dependency in getattr(owner, "source_dependencies", getattr(owner, "dependencies", ())):
            if closure[dependency.artifact_id] != dependency.digest:
                raise ValueError("Consumed dependency digest differs from release closure")


def validate_release_dependency_closure(release: Any, artifact_digests: Mapping[str, str],
                                       artifact_dependencies: Mapping[str, Sequence[str]] | None = None) -> None:
    """Verify a v2 closure against independently loaded artifacts before use.

    ``artifact_dependencies`` must describe the recursively loaded artifacts'
    references; omitted graph data cannot prove transitive closure and is rejected.
    """
    if release.schema != BIOLOGY_RELEASE_V2_SCHEMA:
        raise ValueError("Legacy releases do not prove validated State dependency closure")
    validate_release_v2_structure(release)
    declared = {item.artifact_id: item.digest for item in release.dependency_closure}
    if artifact_dependencies is None:
        raise ValueError("Artifact dependency graph is required to verify transitive closure")
    for artifact_id, digest in declared.items():
        if artifact_id not in artifact_digests:
            raise ValueError(f"Required release artifact is unavailable: {artifact_id}")
        if artifact_digests[artifact_id] != digest:
            raise ValueError(f"Required release artifact digest mismatch: {artifact_id}")
        if artifact_id not in artifact_dependencies:
            raise ValueError(f"Required artifact dependency graph is unavailable: {artifact_id}")
        missing = set(artifact_dependencies[artifact_id]) - declared.keys()
        if missing:
            raise ValueError(f"Transitive release dependency closure is incomplete: {sorted(missing)}")


def validate_release_state_context(release: Any, taxonomy: TypeTaxonomy,
                                   exact_type_assignments: Mapping[ProjectObservationKey, str], *,
                                   windows: Mapping[str, Mapping[str, Any]] | None = None,
                                   pairs: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Validate saved membership decisions and exact kinetic leaves before use.

    Callers provide *only valid assigned hard calls* from the pinned Type role;
    unknown/ambiguous/incompatible observations must be absent from the mapping.
    Artifact availability/digests are checked separately by the closure validator.
    """
    if release.schema != BIOLOGY_RELEASE_V2_SCHEMA:
        raise ValueError("Legacy State scopes require explicit validated adoption")
    validate_release_v2_structure(release)
    if release.type_taxonomy_id != taxonomy.taxonomy_id:
        raise ValueError("Release taxonomy revision does not match")
    memberships = {item.membership_id: item for item in release.state_memberships}
    states = {item.binding_id: item for item in release.state_bindings}
    for membership in memberships.values():
        membership.expression.validate_taxonomy(taxonomy)
        if any(not membership.eligible_assignment(exact_type_assignments.get(key), "assigned", taxonomy)
               for key in membership.resolved_observations):
            raise ValueError("Frozen membership contains an ineligible or unresolved Type assignment")
    for binding in release.kinetic_bindings:
        binding.validate_context(taxonomy, memberships[states[binding.state_binding_id].membership_id],
                                 exact_type_assignments, windows=windows, pairs=pairs)
