"""Agreement of integrated boundary motion with independently tracked centroids."""
from __future__ import annotations

import numpy as np
import hashlib


def registered_centroid_lookup(observations, scale, registration=None):
    """Join by observation identity, with the boundary's calibration/registration."""
    names = observations.dtype.names or ()
    if not all(f'centroid_{axis}' in names for axis in 'zyx'):
        return {}
    points = np.column_stack([observations[f'centroid_{axis}'] for axis in 'zyx']).astype(float)
    points *= np.asarray(scale, float)
    if registration is not None:
        points = registration.apply_zyx(points, observations['frame'])
    return {int(oid): point for oid, point in zip(observations['observation_id'], points)}


def add_centroid_motion_diagnostics(links, source_summary, centroids, track_links,
                                    *, min_matched_fraction=.5, min_centroid_displacement=.25,
                                    distance_unit='registered_coordinate_unit'):
    """Append per-link diagnostics; never change vectors, quality or acceptance.

    Surface averages use represented-point mass, not transported mass. Only
    finite, quality-valid points meeting the declared confidence are included.
    Ratios for small centroid motions and branch/merge links are unavailable.
    Surface and volume-centroid motion can differ under real deformation.
    """
    fields = [('centroid_displacement_zyx', '<f8', (3,)),
              ('boundary_mean_displacement_zyx', '<f8', (3,)),
              ('centroid_motion_error', '<f8'), ('centroid_motion_recovery', '<f8'),
              ('centroid_motion_cosine', '<f8'), ('centroid_motion_support', '<f8'),
              ('centroid_comparison_valid', 'u1')]
    output = np.empty(links.shape, dtype=links.dtype.descr + fields)
    for name in links.dtype.names:
        output[name] = links[name]
    for field in fields:
        output[field[0]] = 0 if field[0] == 'centroid_comparison_valid' else np.nan
    parents, pcounts = np.unique(track_links['parent_observation_id'], return_counts=True)
    children, ccounts = np.unique(track_links['child_observation_id'], return_counts=True)
    branches, merges = set(parents[pcounts > 1]), set(children[ccounts > 1])
    for index, link in enumerate(links):
        parent, child = int(link['source_observation_id']), int(link['target_observation_id'])
        delta = np.asarray(centroids.get(child, np.full(3, np.nan))) - np.asarray(centroids.get(parent, np.full(3, np.nan)))
        output['centroid_displacement_zyx'][index] = delta
        start, count = int(link['source_summary_start']), int(link['source_summary_count'])
        sl = slice(start, start+count)
        vectors = np.asarray(source_summary['barycentric_displacement_zyx'][sl])
        mass = np.asarray(source_summary['point_mass'][sl])
        fraction = np.asarray(source_summary['matched_fraction'][sl])
        flags = np.asarray(source_summary['quality_flags'][sl])
        represented = np.isfinite(mass) & (mass > 0)
        good = represented & np.all(np.isfinite(vectors), axis=1) & np.isfinite(fraction) & (fraction >= min_matched_fraction) & (flags == 0)
        total = mass[represented].sum()
        output['centroid_motion_support'][index] = mass[good].sum()/total if total > 0 else 0.
        if not good.any():
            continue
        mean = np.average(vectors[good], axis=0, weights=mass[good])
        output['boundary_mean_displacement_zyx'][index] = mean
        if not np.all(np.isfinite(delta)):
            continue
        output['centroid_motion_error'][index] = np.linalg.norm(mean-delta)
        magnitude = np.linalg.norm(delta)
        valid = (parent not in branches and child not in merges and bool(link['solver_converged'])
                 and not (int(link['quality_flags']) & 4) and magnitude >= min_centroid_displacement)
        if not valid:
            continue
        output['centroid_comparison_valid'][index] = 1
        output['centroid_motion_recovery'][index] = np.dot(mean, delta)/(magnitude*magnitude)
        if np.linalg.norm(mean) > 1e-12:
            output['centroid_motion_cosine'][index] = np.clip(np.dot(mean, delta)/(magnitude*np.linalg.norm(mean)), -1., 1.)
    valid = output['centroid_comparison_valid'] != 0
    def median(key, mask=valid):
        values = output[key][mask]
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None
    centroid_ids = np.array(sorted(centroids), dtype='<i8')
    centroid_values = np.asarray([centroids[int(oid)] for oid in centroid_ids], dtype='<f8').reshape(-1, 3)
    digest = hashlib.sha256(centroid_ids.tobytes()+centroid_values.tobytes()).hexdigest()
    summary = dict(event='boundary_motion_diagnostics', link_count=len(links), distance_unit=distance_unit,
        comparison_link_count=int(valid.sum()),
        median_centroid_recovery=median('centroid_motion_recovery'),
        median_centroid_cosine=median('centroid_motion_cosine'),
        median_centroid_error=median('centroid_motion_error'),
        median_boundary_support=median('centroid_motion_support', np.ones(len(links), bool)))
    schema = dict(schema='celltraj2.centroid_motion_diagnostics.v1',
        source='registered observation centroids', weighting='represented_point_mass',
        minimum_matched_fraction=float(min_matched_fraction),
        minimum_centroid_displacement=float(min_centroid_displacement),
        distance_unit=distance_unit, registered_centroids_sha256=digest,
        recovery_definition='dot(boundary_mean_displacement, centroid_displacement) / norm(centroid_displacement)^2',
        excluded_from_ratio='small/nonfinite centroid displacement, branch/merge, nonconverged solver, no eligible boundary vectors',
        interpretation='Agreement diagnostic, not material-motion accuracy; surface and volume centroids can differ under deformation.',
        summary={k: v for k, v in summary.items() if k != 'event'})
    return output, schema, summary
