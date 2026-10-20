PTYCHOGRAPHY-GUIDED TWISTED WSe2 REFERENCE -- ORDINARY GA LOSS

FINAL RESULT
Loss:                         -0.356662093098
Formation energy per atom:    -0.386755815052 eV/atom
Weighted image term:          0.026938533410
Composition penalty:          0.000003862570
Sample-box confinement term:  0.003151325975
Voxel image correlation:      0.862673879
Atoms: 1517 (505 W, 1012 Se)
PBFGS: 12 iterations, termination gradient_tolerance,
gradient norm 0.002983318.
GPU: ws10, NVIDIA GeForce RTX 5090, job 198135.

THE EXACT SAME OBJECTIVE
Loss = (E_UMA - N_W*mu_W - N_Se*mu_Se)/N
     + 10 * mean_voxel_Huber(prediction, target; delta=0.2)
     + 10 * sum_species((N_species/N - target_fraction_species)^2)
     + 50 * mean_atom(sum_xyz(softplus boundary violation^2)).
Composition target W:Se = 1:2; boundary sharpness = 8 A^-1; boundary margin = 0.
Image renderer: the exact production double-XY-Gaussian + Gaussian-depth model.
Depth sigma = 4 A; extra lateral blur = 0.08 A; target transform linear;
background percentile 1; normalization percentile 99.5. No fitted intensity gain,
contrast renormalization or post-hoc image registration was introduced.
The 12 native slices and all 320 x 320 measured XY pixels enter the image loss.
Input elemental reference energies are copied from the ordinary GA run.

All atomic positions are free; species and atom count are fixed during PBFGS,
exactly as for a single ordinary-GA member. No GA mutations, insertions,
deletions, occupancy optimization or species transmutation were performed.
Cell is fixed at diag(64.687136,64.687136,28) A; xyz periodic. The measured region
is 48.82048 A square with padding (7.933328,7.933328,8) A per side.

PRODUCTION CODE AND HYPERPARAMETER VERIFICATION
source/ contains the ordinary-GA driver, image/geometry code and minimizer.
Its original source+input fingerprint matches the original run exactly:
e37450a00d10b8d710640f77dee4dfc370f64cb1be04dc1dba121d962f717325
The bundled default YAML has its embedded API credential removed; the saved
run config and copied references supply all actual settings without credentials.
Final optimization calls ordinary.relax_candidate unchanged, with:
  max_iterations=500; gradient_tolerance=0.003; bfgs_history=8;
  max_line_search=10; cost_plateau_tolerance=0.0003;
  cost_plateau_patience=10; cost_plateau_min_iterations=50.
Only file/output paths and runtime GPU selection change from the original config.
The old ordinary winner was independently re-evaluated: loss differs from its
recorded value by -7.059e-08.

EXPERIMENTAL ALIGNMENT BEFORE PBFGS
Started from the Materials Project bulk-derived Se-W-Se monolayer basis, not
from a HEAT or ordinary-GA optimized structure. The earlier energy-only relaxed
flake is included and scored for comparison, but is not used to initialize.
Fourier peaks in the original experimental TIFF seed the absolute orientation.
Eight alternatives cover both twist signs and the two 60-degree sublattice
orientations of each monolayer. Per-layer XY translations, z heights, small
angle corrections and bounded symmetric in-plane affine lattice adjustments
are fitted directly to the experimental image with its exact renderer/Huber loss.
Diagonal lattice factors are bounded to [0.97,1.03]; shear to [-0.02,0.02].
The angle seed is +/-1.5 degrees about the Fourier orientation; each layer may
adjust by +/-4 degrees to register to the data. Each rigid/affine fit is limited
to 140 L-BFGS-B iterations; several hit this limit. This preliminary fit is
initialization only and has separate numerical settings from final PBFGS.
The image initialization includes a 4 A halo to avoid missing edge intensity;
atoms are then cropped into the exact original measured XY region before the
full production loss and PBFGS. The best cropped image fit is selected.
The registered lattice therefore need not retain the exact bulk metric.

Selected alignment trial: 7.
Registration parameter angles: [111.78427058273432, 108.95489973308379] degrees.
Registration parameter twist modulo 60: -2.829371 degrees.
Final W-neighbor relative orientation modulo 60: -2.866785 degrees.
The neighbor-based angle is affected by strain; it is a global estimate.
Aligned-start image correlation: 0.854269087.
Final image correlation: 0.862673879.
Aligned-start total loss: -0.289711456826.

INTERPRETATION
This is a reference obtained by fitting the experimental data under the exact
ordinary-GA objective. It is not independent experimental ground truth or a
proven global minimum. Its composition is selected by the geometrical crop;
fixed-position PBFGS cannot repair vacancies or alter species/counts.
The final total loss, not energy alone, is the number to compare with GA loss.
See objective_comparison.csv for a decomposition including the previous
energy-only relaxation and the ordinary winner under this same objective.
attempt_rigid_only/ retains the less successful first rigid registration and
its result; final_* at this folder's top level are the improved main result.

MAIN FILES
final_guided_relaxed.xsf / .extxyz / .cif: final reference structure.
final_result.json: full score decomposition, coordinates, and optimizer metrics.
aligned_ideal_initial.*: registered starting bilayer before final PBFGS.
ptycho_comparison_320x320.png / .pdf: experimental, initial and final z-summed images
using the same display scale, without registration/interpolation.
*_slices.tif: exact measured-region 12 x 320 x 320 predictions/target/residual.
*_zsum_320x320.tif: corresponding z-summed arrays in normalized model units.
objective_evaluations.csv: all final PBFGS objective evaluations.
alignment_trial_*.json: all eight preliminary fits, including iteration-limit flags.
inputs/: original TIFF, bulk basis, fits, references and source run parameters.
source/: snapshotted production code. No API key is needed to use the saved inputs.

REPRODUCING AFTER DOWNLOAD
The folder is self-contained for structure, microscopy inputs and project code.
UMA model weights and scientific Python dependencies are external requirements,
not included in the archive. See environment_versions.json.
With a CUDA-compatible fairchem/ASE/PyTorch environment and cached uma-s-1p1:
  python run_baseline.py
  python summarize.py
For a fresh optimization, remove pbfgs_progress.json first: production PBFGS
otherwise resumes from that checkpoint. Submit.sh documents the cluster command;
adapt its local Python path and Slurm node/partition for another machine.
All input paths are resolved relative to this folder at runtime.

FAIRCHEM DEPENDENCY SOURCE
The local FairChem checkout has modifications. Its exact commit is recorded
in source/fairchem_revision.txt and its tracked core changes are included in
source/fairchem_local_changes.patch. For matching dependency code, check out
that commit and apply this patch before installing FairChem. Model weights
remain external. No model files or API credentials are bundled.
