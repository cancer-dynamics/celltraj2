Directed boundary drift
=======================

The pure kernel projects forward moving-boundary displacement onto the unit
direction toward the nearest point on a different object's selected boundary.
Positive values mean motion toward the neighbor. From and To are ordered roles.
The caller supplies calibrated positions and vectors in the same physical basis.
Missing motion remains missing; real zero displacement remains valid.

Nearest queries are chunked and exclude the entire source object. Exact-distance
ties select the lowest target point ID. An optional normal filter is a minimum
outward-normal cosine; normals are not required when the filter is disabled.
Point support is descriptive, not the number of biological replicates.

SITE's ``boundary_drift_analysis`` adapter reads boundary-motion v2 source
summaries, pins dependencies and cohorts, and delegates to this module. It
does not recompute optimal transport inside Plot.

.. automodule:: celltraj2.boundary_drift
   :members:
