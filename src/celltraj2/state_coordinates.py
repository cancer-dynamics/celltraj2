"""Versioned row-domain and coordinate contracts; no numerical fitting or GUI.

Static coordinates may share arbitrary validated State membership. A future
pair-trained representation has its own exact-leaf scope before partitioning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping
from .interpretation import _uuid_text, canonical_json_digest

SAMPLE_DOMAIN_SCHEMA = 'site.state_sample_domain.v1'
COORDINATE_SET_SCHEMA = 'site.state_coordinate_set.v1'
REPRESENTATION_SCOPE_SCHEMA = 'site.state_representation_scope.v1'


def _digest(value, name):
    if not re.fullmatch(r'[0-9a-f]{64}', str(value)):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return str(value)


@dataclass(frozen=True)
class SampleDomain:
    domain_id: str
    domain_kind: str
    row_count: int
    length: int
    snapshot_id: str
    snapshot_digest: str
    snapshot_row_order_digest: str
    dependency_digest: str
    row_order_digest: str
    schema: str = SAMPLE_DOMAIN_SCHEMA

    def __post_init__(self):
        if self.schema != SAMPLE_DOMAIN_SCHEMA:
            raise ValueError('Unsupported State sample-domain schema')
        if self.domain_kind not in ('observation', 'delay'):
            raise ValueError('Unsupported State row domain')
        if (isinstance(self.row_count, bool) or not isinstance(self.row_count, int)
                or self.row_count < 0 or isinstance(self.length, bool)
                or not isinstance(self.length, int) or self.length < 1):
            raise ValueError('Sample count and delay length must be nonnegative/positive integers')
        if (self.length == 1) != (self.domain_kind == 'observation'):
            raise ValueError('Delay 1 is the explicit observation domain')
        _uuid_text(self.snapshot_id, field_name='snapshot_id')
        for name in ('domain_id', 'snapshot_digest', 'snapshot_row_order_digest',
                     'dependency_digest', 'row_order_digest'):
            _digest(getattr(self, name), name)
        seed = {k: v for k, v in self.to_dict().items() if k != 'domain_id'}
        if self.domain_id != canonical_json_digest(seed):
            raise ValueError('Sample domain identity does not match its dependency/row manifest')

    def to_dict(self):
        return dict(vars(self))

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))


def validate_coordinate_set(data):
    """Validate a reusable ordered recipe without learning preprocessing."""
    result = dict(data)
    if result.get('schema') != COORDINATE_SET_SCHEMA:
        raise ValueError('Unsupported coordinate-set schema')
    _digest(result.get('domain_id'), 'domain_id')
    ids = list(result.get('coordinate_ids', ()))
    if not ids or any(not isinstance(item, str) or not item.strip() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError('An analysis set requires unique ordered coordinate IDs')
    if result.get('scaling') not in ('none', 'standardize'):
        raise ValueError('Unsupported coordinate scaling policy')
    if result.get('missing_policy') not in ('complete_rows', 'error'):
        raise ValueError('Unsupported coordinate missing-data policy')
    weights = list(result.get('weights', ()))
    if len(weights) != len(ids) or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v <= 0 for v in weights):
        raise ValueError('Coordinate weights must be finite positive values aligned with the columns')
    result.update(coordinate_ids=ids, weights=[float(v) for v in weights])
    return result


def validate_representation_fit_scope(data, *, taxonomy=None):
    """Validate shared static or exact-leaf dynamic provenance.

    A dynamic scope requires an actual taxonomy to establish that the selected
    UUID is a leaf. No dummy State component/partition binding is required.
    This is a contract seam only; PairSet construction and VAMP are later work.
    """
    if not isinstance(data, Mapping):
        raise ValueError('Representation scope must be a mapping')
    result = dict(data)
    if result.get('schema') != REPRESENTATION_SCOPE_SCHEMA:
        raise ValueError('Unsupported representation-scope schema')
    if result.get('kind') == 'static_shared':
        if any(result.get(k) is not None for k in ('exact_leaf_type_id', 'pair_set_id',
                'pair_set_digest', 'type_release_id', 'taxonomy_id')):
            raise ValueError('Static shared scope cannot carry a dynamic leaf/pair claim')
        return result
    if result.get('kind') != 'dynamic_leaf':
        raise ValueError('Unknown representation fit scope')
    for key in ('exact_leaf_type_id', 'pair_set_id', 'type_release_id', 'taxonomy_id'):
        result[key] = _uuid_text(result.get(key), field_name=key)
    _digest(result.get('pair_set_digest'), 'pair_set_digest')
    if taxonomy is None or taxonomy.taxonomy_id != result['taxonomy_id']:
        raise ValueError('Dynamic representation scope needs its frozen Type taxonomy')
    if result['exact_leaf_type_id'] not in {node.type_id for node in taxonomy.leaf_nodes}:
        raise ValueError('Dynamic representation scope must name one exact Type leaf')
    return result
