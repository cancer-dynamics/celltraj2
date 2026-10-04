"""Versioned contracts for exact-leaf trajectory pairs; no numerical fitting."""
from __future__ import annotations

import math
from .interpretation import _uuid_text
from .state_coordinates import SampleDomain, _digest

PAIR_RECIPE_SCHEMA = 'site.state_pair_recipe.v1'


def validate_pair_recipe(data, *, taxonomy=None):
    """A history length and a prediction lag are distinct, explicit quantities."""
    recipe = dict(data)
    if recipe.get('schema') != PAIR_RECIPE_SCHEMA:
        raise ValueError('Unsupported State pair recipe schema')
    domain = SampleDomain.from_dict(recipe['row_domain'])
    for name in ('exact_leaf_type_id', 'taxonomy_id', 'type_release_id', 'type_classification_id'):
        _uuid_text(recipe.get(name), field_name=name)
    _digest(recipe.get('type_assignments_digest'), 'type_assignments_digest')
    if recipe.get('split_digest') is not None:
        _digest(recipe['split_digest'], 'split_digest')
    if taxonomy is not None:
        if taxonomy.taxonomy_id != recipe['taxonomy_id']:
            raise ValueError('PairSet taxonomy identity differs from its Type scope')
        if recipe['exact_leaf_type_id'] not in {n.type_id for n in taxonomy.leaf_nodes}:
            raise ValueError('PairSets require an exact Type leaf, not a subtree or parent')
    lag = recipe.get('lag_frames')
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
        raise ValueError('Pair lag must be a positive integer number of graph links')
    if isinstance(recipe.get('history_length'), bool) or recipe.get('history_length') != domain.length:
        raise ValueError('Pair history length differs from its sample domain')
    mode = recipe.get('time_mode')
    if mode not in ('physical', 'frame'):
        raise ValueError('Pair time mode must be physical or frame')
    tolerance = recipe.get('tolerance_s')
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Pair timing tolerance must be finite and nonnegative')
    interval = recipe.get('interval_s')
    if mode == 'physical':
        if isinstance(interval, bool) or not isinstance(interval, (float, int)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError('Physical pairs require one explicit positive cadence')
        if recipe.get('physical_time_claim') is not True:
            raise ValueError('Physical pair timing claim is missing')
    elif interval is not None or recipe.get('physical_time_claim') is not False:
        raise ValueError('Frame pairs are descriptive and cannot claim a physical cadence')
    if recipe.get('anchor') != 'final' or recipe.get('support_policy') != 'complete_combined_path':
        raise ValueError('Pair anchors and full history/bridge support must be explicit')
    return recipe
