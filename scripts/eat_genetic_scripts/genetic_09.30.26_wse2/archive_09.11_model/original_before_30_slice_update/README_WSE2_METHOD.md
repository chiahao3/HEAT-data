# Periodic WSe2 image-guided EAT genetic optimization

This directory adapts the Garnet optimizer to the centered twisted-bilayer
WSe2 reconstruction. FairChem remains periodic in x, y, and z. Because the
finite twisted sample crop is not commensurate in any direction, the measured
volume is centered between about 8 A of explicit vacuum on both sides of x, y,
and z.

## Geometry and image loss

The default centered 150x150 crop has sample-cell lengths
22.8846 x 22.8846 x 12 A. The padded FairChem/render cell has lengths
38.751256 x 38.751256 x 28 A and `pbc = [true, true, true]`. The requested
8-A lateral padding is snapped to 52 native pixels, or 7.933328 A per side,
because the FFT renderer requires whole pixels on a uniform grid. Z padding is
exactly 8 A. Initial sample coordinates and all experimental fields are shifted
by the realized padding in x, y, and z.

The Gaussian renderer operates on the complete 28x254x254 periodic grid. A
binary mask is one only on the centered ZYX ROI `[8:20, 52:202, 52:202]`, which
contains the original 12x150x150 experiment. The image loss is

```text
sum(mask * Huber(simulated - experimental)) / sum(mask)
```

and image correlation is computed from the same masked values. Vacuum on all
six faces therefore affects periodic physical separation and Gaussian tails,
but it cannot dilute the image objective by adding zero-loss voxels.

A second image term detects an overcomplete set of projected-column candidates
with a scale-normalized Laplacian of Gaussian. Detection runs on a configurable
halo of real source pixels around the requested crop and candidate centers are
then mapped back into the crop. A global nonnegative fit of the normalized
double-Gaussian kernels prunes unsupported candidates, refines positions, adds
positive-residual candidates, and merges close fitted kernels into composition
groups. Responsibility-deblended experimental intensity then defines an
adaptive ellipse for each group.

The fitted full-kernel amplitude is `I_tot,r`; it is not a raw sum truncated by
the displayed ellipse. One shared experimental scale is fitted while the W/Se
intensity ratio is held exactly to `element_double_gaussian_fits.csv`. During
EAT, smooth ellipse memberships `q_ri` give

```text
n_r = sum_i q_ri x_W,i
m_r = sum_i q_ri x_Se,i
L_ellipse = ellipse_composition_penalty * mean_r [
  (I_W n_r + I_Se m_r - I_tot,r) / I_Se
]^2
```

Memberships for each atom are normalized together with a background channel,
so overlapping ellipses divide ownership and never double-count an atom. The
background channel lets atoms outside all experimental regions remain
unassigned. Membership uses x/y only and is smooth in position.

The original voxelwise 3D Huber coefficient and the ellipse-composition square
coefficient are independently adjustable. Initialization diagnostics include
`experimental_log_joint_fit_adaptive_ellipses.png`, a CSV region key, JSON
metadata, and all fitted arrays in a compressed NPZ file.

## Sample-box confinement

A separate smooth confinement term prevents real W/Se occupancy from relaxing
into the vacuum. For every site it sums the squared softplus distances below
and above the original sample bounds in x, y, and z, weights that value by the
site's total W+Se occupancy, and divides by the total W+Se occupancy. The term
therefore has an atom-count-independent scale:

```text
L_box = box_penalty * sum_i[(W_i + Se_i) * violation_i^2]
                       / sum_i(W_i + Se_i)
```

The default coefficient is `0.01` per squared Angstrom with sharpness `8.0/A`.
For the first WSe2 result, the raw occupancy-normalized loss is about 15.05
squared Angstrom, so this coefficient would contribute about 0.15: the same
order as that structure's 0.088 guided score rather than scaling with its 1,069
atoms.
It is included in random-immigrant image prefit, continuous EAT/PBFGS,
fixed-identity position polishing, crossover-child relaxation, and guided GA
scoring. FairChem still receives `pbc = [true, true, true]`.

## Effective atoms and initialization

There is one permanent site class with three choices: W, Se, or implicit null.
The p50 feasible isosurface and cylinder initializer generate all proposal
positions. Candidate count is density times the unpadded feasible sample volume;
vacuum creates no proposal sites.

Every random immigrant first optimizes W/Se/null logits and positions using the
masked double-Gaussian image loss, a soft W:Se = 1:2 fractional-composition
penalty, feasible-region loss, pair repulsion, and an increasing entropy term.
The complete soft state, including every candidate site, is then handed to the
normal energy-inclusive EAT syntropization. No pre-syntropization rounding or
null-site compaction is performed. No formula-unit search, volume prior, exact
atom count, or exact composition projection is used.

## FairChem and genetic optimization

Random immigrants undergo the same FairChem-energy syntropization as other
optimized members. The guided objective is formation energy per atom plus masked
3D image, ellipse-composition, soft global composition, Se-vacancy bias, and
feasible-region terms. Syntropization performs the single W/Se/null rounding,
removes exact-null sites, and then runs the final fixed-species position
relaxation.

The conventional formation energy per current atom is retained unchanged. A
separate calibrated term reduces its opposition to Se vacancies:

```text
N_Se,reference = reference_atom_count * target_Se_fraction
L_Se-vacancy = -alpha * (N_Se,reference - sum_i x_Se,i)
                       / reference_atom_count
```

The configured reference count is fixed at 350 for the 150x150 crop, giving
`N_Se,reference = 233.333...` for the soft 1:2 W:Se target. The initial
`alpha = 1.8353 eV/vacancy` is the calibrated 75% neutralization value. The term
uses summed soft Se occupancy during syntropization and exact Se counts after
rounding. It is constant during the final fixed-species position relaxation,
but remains included in final guided scoring and diagnostics.

Later generations retain the Garnet workflow's elites, random immigrants,
superposition crossover, W/Se/null occupancy and species shocks, close-pair
pruning, PBFGS syntropization, fixed-identity final position polish, tournament
selection, checkpoints, local workers, and Slurm support. Parent A and parent B
are the lowest-score members of two independently sampled tournaments, with the
winning parent A excluded when tournament B is sampled. Subcell diversity does
not select or align crossover parents. All minimum-image distances remain
periodic in x, y, and z; disabling parent alignment does not disable z PBC.

## Preflight

Set `diagnostics.exit_after_initial_diagnostics: true` to stop after writing the
padded target, loss mask, feasible mask, NPZ fields, and
`geometry_and_mask_summary.json`. That JSON explicitly records the sample and
padded cells, requested and realized padding, all-periodic PBC, loss-mask voxel
count, and complete centered ZYX ROI. The same directory also contains the
halo-detection plot, joint reconstruction and adaptive-ellipse plot, smooth
membership-field plot, region CSV/JSON, and compressed fitted arrays.
