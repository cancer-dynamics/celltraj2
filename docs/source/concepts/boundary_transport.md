# Robust boundary transport

Boundary transport links sampled surface measures between two frames. It is
used in two places:

- boundary-informed object tracking compares candidate parent/child surfaces;
- surface motion maps points along links that an object tracker has already
  accepted.

The transport plan is a probabilistic correspondence, not proof that one
membrane material point became another. Shape-normal motion is generally more
identifiable than tangential membrane motion. Physics models should therefore
use the stored confidence and ambiguity fields, and should normally give the
normal component more weight unless texture, landmarks, or another material
cue constrains tangential identity.

## Processing order

The robust pipeline is:

```text
native boundary point ids and positions
  -> apply the selected registration in memory
  -> shared-density physical-grid sampling
  -> weighted robust/partial/balanced transport
  -> hard physical support gate
  -> mass-aware numerical sparsification
  -> barycentric source and target point summaries
  -> H5 motion set plus provenance and diagnostics
```

The boundary library itself is never registered or rewritten. The stored raw
edge displacement is always

```text
T_target(q_native) - T_source(p_native)
```

in registered physical Z/Y/X coordinates. If cost prealignment is enabled,
centering affects only the coordinates supplied to the transport cost. It does
not alter point identity or the stored full displacement.

## Sampling and mass

The boundary library's physical `point_spacing` is the primary resolution
control. If `max_boundary_points` imposes an additional per-link cap, source
and target are reduced with one shared physical voxel spacing. This replaces
the old row-index `linspace` cap, which could sample unrelated parts of two
surfaces and depended on point-table ordering.

Every sampled point is weighted by the number of canonical boundary points in
the represented voxel. Two mass models are available:

- `probability` normalizes each surface to total mass one. This is the default
  for object-link comparison because size alone does not scale the score.
- `surface_measure` keeps an approximate perimeter (2D) or area (3D) measure
  based on canonical point spacing. This is intended for physical growth and
  flux analyses; it requires an unmatched-aware method when source and target
  totals differ.

Equal source and target *point counts* are not required. Common physical
density plus explicit weights is the invariant that matters.

## Transport methods

### Robust unbalanced transport

`unbalanced` is the recommended mode. For source mass `a`, target mass `b`,
distance matrix `C`, and plan `pi`, it minimizes

```text
<C, pi>
  + epsilon KL(pi || a tensor b)
  + rho_source KL(pi 1 || a)
  + rho_target KL(pi^T 1 || b)
```

with a zero-support-aware generalized Sinkhorn iteration. The ordinary case
uses fast matrix scaling; if a supported kernel entry would underflow, the
solver automatically repeats the calculation in the log domain rather than
silently removing that edge. `epsilon` is the transport
blur (`sinkhorn_regularization`). `rho` is specified as `unbalanced_reach` and
controls how expensive it is to leave mass unmatched. A larger reach behaves
more like balanced transport; a smaller reach rejects unsupported matches more
readily.

`max_transport_distance` is a separate, exact support constraint in physical
units. No retained or summarized correspondence may cross it. Reach is a soft
match-versus-unmatched decision; the distance cap is a hard biological or
imaging plausibility limit. They should not be treated as interchangeable.

Unbalanced OT may create or destroy a small amount of raw plan mass because its
marginals are penalized rather than fixed. Coverage therefore uses overlap
mass, `sum(min(marginal, input_mass)) / sum(input_mass)`, and remains in
`[0, 1]`. Per-point `matched_fraction` uses the same interpretation and is
clipped to `[0, 1]`.

### Partial transport

`partial` transports a fixed fraction (`partial_mass`) of the smaller input
measure while respecting row/column capacity and the hard support gate. It is
useful for controlled comparisons when the expected overlap fraction is known.
It is not the default because a single fixed fraction is usually too rigid for
cells that protrude, retract, enter, divide, or are incompletely segmented.

### Balanced legacy modes

`emd` and `sinkhorn` retain both marginals exactly and remain available for
regression comparisons. They cannot leave outliers unmatched. A hard distance
gate is therefore not applied to these modes because it can make the balanced
problem infeasible. Lowering Sinkhorn blur can reduce diffuse numerical tails,
but it cannot fix forced mass conservation.

## Scores, coverage, and acceptance

For unbalanced tracking, `ot_cost` is the full regularized objective. Ranking
by transported distance alone would reward a candidate for deleting mass. For
partial and balanced modes, the comparison score is matched transport cost
divided by transported mass. Candidate boundary links must also satisfy the
configured minimum source and target coverage.

Centroid distance remains the first, inexpensive candidate gate. Registration
is applied before centroid gating and boundary cost evaluation. Optional
centroid cost prealignment compares residual shape after translation; when it
is used, centroid distance remains separately present in the object-link row.

For surface motion along an already accepted object link, low coverage does not
delete or change the object link. The motion-link `quality_flags` marks source
or target coverage below the configured warning threshold.

## Numerical sparsification

Solvers operate on the full supported plan. Stored raw edges are then filtered
using:

- `mass_tolerance`: absolute edge mass;
- `relative_mass_tolerance`: edge mass as a fraction of its source marginal;
- `retained_mass_fraction`: cumulative source-row mass retained from largest
  edges downward.

At least one edge is retained for every source row with positive plan mass.
`dropped_mass` and `raw_edge_count` make this reduction auditable. This stage is
for numerical tails and storage/display efficiency; it is deliberately *after*
unmatched-aware optimization. Pruning a balanced plan alone does not undo the
way forced distant matches changed the rest of that plan.

## Boundary-motion v2 storage

Motion sets use this layout:

```text
/boundaries/<boundary_set>/motion/<motion_set>/
  schema.json
  links
  transport/
    source_point_id
    target_point_id
    mass
    edge_cost
    registered_displacement_zyx
  source_summary/
    point_id
    point_mass
    matched_mass
    matched_fraction
    barycentric_displacement_zyx
    displacement_variance
    edge_distance_mean
    quality_flags
  target_summary/
    ...same columns...
```

Each `links` row contains half-open starts/counts for raw transport,
source-summary, and target-summary rows. It also stores:

- comparison score, raw transport cost, full objective, and matched mean cost;
- transported mass and source/target coverage;
- dropped numerical mass;
- mass-weighted edge-distance p50, p90, p99, and maximum;
- solver iteration count and convergence status;
- coverage quality flags.

For a source point `i`, the primary display/model vector is

```text
v_i = sum_j pi_ij (y_j - x_i) / sum_j pi_ij
```

and its confidence is `matched_fraction`. `displacement_variance` reports the
mass-weighted spread around this barycenter and is an ambiguity measure. Target
summaries use the same forward displacement direction while grouping by target
point; they support incoming-motion features without reconstructing the full
edge plan.

Raw sparse edges remain available for diagnostics and compatibility. Existing
v1 readers that use the original `links` and `transport` column names continue
to work; new readers should prefer the point summaries.

## Visualization and features

SITE displays one barycentric arrow per sampled source point, filters it by
minimum confidence, and applies the vector budget after confidence ranking.
The raw-edge display is an explicit debug option. Boundary coloring by mapped
displacement uses the barycentric vector magnitude rather than the mean of
individual edge lengths.

Object-level `mapped_fraction` is the fraction of transport-sampled
object-boundary points whose matched fraction reaches `mapped_mass_threshold`
(default 0.01), not the fraction touched by any floating-point Sinkhorn tail.
Using the sampled-point denominator keeps the metric from falling merely
because a point cap was active.

## Parameter selection

Tune parameters against representative good and bad links in the actual
registered data:

1. Set boundary-library `point_spacing` from the smallest meaningful surface
   detail.
2. Set `max_transport_distance` from the largest physically plausible
   frame-to-frame surface displacement after global registration.
3. Start transport blur near a fraction of the boundary spacing; increase it
   for noisier sampling and decrease it for sharper correspondences.
4. Sweep unmatched reach and inspect source/target coverage together with
   p90/p99 edge distance and barycentric variance.
5. Choose minimum tracking coverage after observing true links with real
   protrusion, retraction, and segmentation loss.
6. Check stability when boundary spacing and point caps change.

There is no universal micron threshold. The selected values and units are
stored in both track and motion schemas so parameter sweeps remain comparable.

## Known topology limit

At a division, current surface motion evaluates each parent-to-child link
independently. This reuses the parent's normalized measure for each child and
should not be interpreted as a conserved joint surface flux. A future division
mode should solve one parent against the union of its children (or use an
explicit branch allocation model). Current results preserve the individual
links and diagnostics so those cases can be detected and excluded or modeled
separately.
