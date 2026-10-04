"""Typed event/risk release outputs; neither is a current-State assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .interpretation import ProjectObservationKey, _json_safe, _uuid_text
from .state_contracts import StateDependency, _dependencies, _digest, _keys
from .state_supervision import SupervisionTask

BIOLOGY_RELEASE_V3_SCHEMA = "site.biology_release.v3"
_COMPATIBILITY = {"unchanged", "subset_reusable", "needs_application", "needs_refit",
                  "incompatible", "unverified", "unavailable"}


def _common(binding: Any, identifiers: tuple[str, ...]) -> None:
    for name in identifiers:
        if getattr(binding, name) is not None:
            object.__setattr__(binding, name, _uuid_text(getattr(binding, name), field_name=name))
    if not binding.channel.strip():
        raise ValueError("Output channel must not be empty")
    if binding.compatibility_status not in _COMPATIBILITY:
        raise ValueError("Unsupported output compatibility status")
    object.__setattr__(binding, "conditioning_digest", _digest(binding.conditioning_digest, "conditioning_digest"))
    object.__setattr__(binding, "dependencies", _dependencies(binding.dependencies))
    object.__setattr__(binding, "coverage_observations", tuple(sorted(_keys(binding.coverage_observations, "coverage_observations"))))


@dataclass(frozen=True)
class EventOutputBinding:
    binding_id: str
    channel: str
    membership_id: str
    definition_id: str
    event_set_id: str
    conditioning_digest: str
    review_revision: str
    coverage_observations: tuple[ProjectObservationKey, ...] = ()
    model_id: str | None = None
    proposal_id: str | None = None
    assessment_id: str | None = None
    dependencies: tuple[StateDependency, ...] = ()
    compatibility_status: str = "unchanged"
    schema: str = "site.event_output_binding.v1"

    def __post_init__(self):
        _common(self, ("binding_id", "membership_id", "definition_id", "event_set_id",
                       "model_id", "proposal_id", "assessment_id"))
        if self.schema != "site.event_output_binding.v1":
            raise ValueError("Unsupported event binding schema")
        if not self.review_revision:
            raise ValueError("Event binding requires a frozen review revision")

    @property
    def artifact_ids(self):
        return frozenset(x for x in (self.definition_id, self.event_set_id, self.model_id,
            self.proposal_id, self.assessment_id, *(d.artifact_id for d in self.dependencies)) if x)

    def to_dict(self):
        return _json_safe(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class PredictionOutputBinding:
    binding_id: str
    channel: str
    membership_id: str
    definition_id: str
    score_id: str
    conditioning_digest: str
    score_semantics: str
    task: dict[str, Any]
    model_id: str
    reference_id: str
    split_id: str
    assessment_id: str
    coverage_observations: tuple[ProjectObservationKey, ...] = ()
    at_risk_coverage: dict[str, Any] | None = None
    dependencies: tuple[StateDependency, ...] = ()
    compatibility_status: str = "unchanged"
    schema: str = "site.prediction_output_binding.v1"

    def __post_init__(self):
        _common(self, ("binding_id", "membership_id", "definition_id", "score_id", "model_id",
                       "reference_id", "split_id", "assessment_id"))
        if self.schema != "site.prediction_output_binding.v1":
            raise ValueError("Unsupported prediction binding schema")
        task = SupervisionTask.from_dict(self.task)
        expected = {"event_detection": "detection_score", "fixed_horizon_fate": "horizon_risk"}.get(task.family)
        if self.score_semantics != expected:
            raise ValueError("Prediction semantics must agree with an event/fate task; current State uses an assignment")
        if task.definition_id != self.definition_id:
            raise ValueError("Prediction task targets another definition")
        if self.score_semantics == "horizon_risk" and not self.at_risk_coverage:
            raise ValueError("Horizon risk requires explicit at-risk coverage")
        object.__setattr__(self, "task", task.to_dict())

    @property
    def artifact_ids(self):
        return frozenset((self.definition_id, self.score_id, self.model_id, self.reference_id,
                          self.split_id, self.assessment_id, *(d.artifact_id for d in self.dependencies)))

    def to_dict(self):
        return _json_safe(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class StateClassMapping:
    """Explicit, total stable-ID map into a broader StateSpace/partition."""
    mapping_id: str
    source_definition_id: str
    target_definition_id: str
    source_class_ids: tuple[str, ...]
    target_class_ids: tuple[str, ...]
    class_map: dict[str, str]
    schema: str = "site.state_class_mapping.v1"

    def __post_init__(self):
        for name in ("mapping_id", "source_definition_id", "target_definition_id"):
            object.__setattr__(self, name, _uuid_text(getattr(self, name), field_name=name))
        for name in ("source_class_ids", "target_class_ids"):
            values = tuple(_uuid_text(c, field_name=name) for c in getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError("State mapping class registries must be nonempty and unique")
            object.__setattr__(self, name, values)
        if self.schema != "site.state_class_mapping.v1":
            raise ValueError("Unsupported State mapping schema")
        if set(self.class_map) != set(self.source_class_ids) or set(self.class_map.values()) - set(self.target_class_ids):
            raise ValueError("State mapping must explicitly map every source class to a declared target class")

    def map_probabilities(self, probabilities):
        import numpy as np
        source = np.asarray(probabilities, dtype=float)
        if source.ndim != 2 or source.shape[1] != len(self.source_class_ids):
            raise ValueError("Probability columns do not match source State class order")
        result = np.zeros((len(source), len(self.target_class_ids)))
        for i, class_id in enumerate(self.source_class_ids):
            result[:, self.target_class_ids.index(self.class_map[class_id])] += source[:, i]
        result[~np.isfinite(source).all(axis=1)] = np.nan
        return result

    def to_dict(self):
        return _json_safe(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


def validate_typed_output_structure(release: Any) -> None:
    memberships = {m.membership_id: m for m in release.state_memberships}
    all_ids = [b.binding_id for b in (*release.state_bindings, *release.kinetic_bindings,
                                      *release.event_bindings, *release.prediction_bindings)]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate typed release binding IDs")
    covered = {}
    for kind, bindings in (("event", release.event_bindings), ("prediction", release.prediction_bindings)):
        for binding in bindings:
            if binding.membership_id not in memberships:
                raise ValueError("Typed output references a missing membership")
            membership = memberships[binding.membership_id]
            if binding.conditioning_digest != membership.conditioning_digest:
                raise ValueError("Typed output conditioning differs from membership")
            coverage = set(binding.coverage_observations)
            if not coverage.issubset(membership.resolved_observations):
                raise ValueError("Typed output coverage escapes membership")
            # Distinct horizons are separate semantic outputs on one channel.
            semantic = (binding.score_semantics, binding.task.get("horizon_s")) if kind == "prediction" else None
            key = (kind, binding.channel, binding.definition_id, semantic)
            prior = covered.setdefault(key, set())
            if prior & coverage:
                raise ValueError("Overlapping authoritative typed outputs for one channel/task")
            prior.update(coverage)
