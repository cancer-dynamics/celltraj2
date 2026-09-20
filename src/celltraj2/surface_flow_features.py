"""Local differential invariants of sampled surface flows and planar radial shape."""

import numpy as np
from scipy.spatial import cKDTree


def surface_flow_fields(positions, vectors, normals, *, spatial_ndim=3, neighbors=16, radius=None):
    """Fit weighted local tangent-plane derivatives without subtracting mean flow.

    Normals are outward. XYZ right-handed bases define signed normal vorticity.
    A 2D contour has divergence/strain but no intrinsic tangential vorticity.
    """
    p, v, n = (np.asarray(array, float)[:, ::-1] for array in (positions, vectors, normals))
    if p.shape != v.shape or n.shape != p.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Surface positions, vectors and normals must be matching N x 3 arrays.")
    if spatial_ndim not in (2, 3) or int(neighbors) < 3:
        raise ValueError("Surface derivatives require 2D/3D data and at least three neighbors.")
    if radius is not None and (not np.isfinite(radius) or radius <= 0):
        raise ValueError("Surface derivative radius must be finite and positive.")
    names = ("normal_gradient_squared", "surface_divergence", "tangential_vorticity", "tangential_strain_rate")
    result = {name: np.full(len(p), np.nan) for name in names}
    norms = np.linalg.norm(n, axis=1)
    valid = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(v), axis=1) & np.isfinite(norms) & (norms > 0)
    n = np.divide(n, norms[:, None], out=np.full_like(n, np.nan), where=norms[:, None] > 0)
    ids = np.flatnonzero(valid)
    if len(ids) < spatial_ndim + 1:
        return result
    tree = cKDTree(p[ids])
    vn = np.sum(v * n, axis=1)
    vt = v - vn[:, None] * n
    for i in ids:
        distances, local = tree.query(p[i], k=min(int(neighbors), len(ids)), distance_upper_bound=np.inf if radius is None else radius)
        keep = np.isfinite(distances)
        js, distances = ids[local[keep]], distances[keep]
        # Avoid fitting through an opposite-facing sheet in thin objects.
        same_sheet = n[js] @ n[i] > 0.25
        js, distances = js[same_sheet], distances[same_sheet]
        if len(js) < spatial_ndim + 1:
            continue
        if spatial_ndim == 2:
            tangent = np.array([-n[i, 1], n[i, 0], 0.0])
            if np.linalg.norm(tangent) < 1e-12:
                continue
            basis = (tangent / np.linalg.norm(tangent))[:, None]
        else:
            axis = np.eye(3)[int(np.argmin(np.abs(n[i])))]
            e1 = np.cross(axis, n[i])
            e1 /= np.linalg.norm(e1)
            basis = np.column_stack([e1, np.cross(n[i], e1)])
        q = (p[js] - p[i]) @ basis
        scale = float(np.max(distances))
        if scale <= 0:
            continue
        design = np.column_stack([np.ones(len(js)), q / scale])
        weights = np.exp(-2 * (distances / scale) ** 2) ** .5
        response = np.column_stack([vn[js], vt[js] @ basis])
        coeff, _, rank, singular = np.linalg.lstsq(design * weights[:, None], response * weights[:, None], rcond=1e-8)
        if rank != spatial_ndim or singular[-1] / singular[0] < 1e-6:
            continue
        gradient = coeff[1:, 0] / scale
        jacobian = coeff[1:, 1:].T / scale
        result["normal_gradient_squared"][i] = float(gradient @ gradient)
        result["surface_divergence"][i] = float(np.trace(jacobian))
        result["tangential_strain_rate"][i] = float(np.linalg.norm((jacobian + jacobian.T) / 2))
        if spatial_ndim == 3:
            result["tangential_vorticity"][i] = float(jacobian[1, 0] - jacobian[0, 1])
    return result


def radial_shape_magnitudes_2d(points_zyx, *, center_zyx=None, order=2, deviation=False, angular_samples=512):
    """Uniform-angle radial Fourier amplitudes about the object's area centroid.

    Multiple radii on exactly the same ray use the outermost boundary. This is a
    radial descriptor, not a complete representation of holes or overhangs.
    """
    points = np.asarray(points_zyx, float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.full(order + 1, np.nan)
    center = np.mean(points, axis=0) if center_zyx is None else np.asarray(center_zyx, float)
    xy = (points - center)[:, -2:]
    angles = np.mod(np.arctan2(xy[:, 0], xy[:, 1]), 2 * np.pi)
    radii = np.linalg.norm(xy, axis=1)
    sorting = np.argsort(angles)
    angles, radii = angles[sorting], radii[sorting]
    unique, starts = np.unique(angles, return_index=True)
    radii = np.maximum.reduceat(radii, starts)
    theta = np.linspace(0, 2 * np.pi, max(int(angular_samples), 32 * (order + 1)), endpoint=False)
    radial = np.interp(theta, unique, radii, period=2 * np.pi)
    if deviation:
        radial -= radial.mean()
    return np.array([abs(np.mean(radial * np.exp(-1j * ell * theta))) for ell in range(order + 1)])
