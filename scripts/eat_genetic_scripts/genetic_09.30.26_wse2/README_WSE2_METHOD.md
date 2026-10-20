# WSe2 Gaussian-depth model — 09/30/26

The active driver and adjacent configuration are `genetic_09.30.26_wse2.py`
and `genetic_09.30.26_wse2.yaml`. Image formation is implemented in
`gaussian_image_model.py`. No fitted depth-response or envelope file is loaded.

## Image model

Each atom contributes a species-weighted double Gaussian in XY multiplied by
an analytic Gaussian in Z:

```
R(z - z_atom) = dz / (sqrt(2*pi)*sigma_z)
               * exp(-0.5*((z - z_atom)/sigma_z)**2)
peak_W = max(G_xy_W) * dz/(sqrt(2*pi)*sigma_z)
I(x,y,z) = sum_atoms[occupancy * G_xy * R] / peak_W
```

`dz` is the measured slice spacing (1 A). The Z Gaussian has unit continuous
integral before sampling, with a shared default sigma of **4 A** for W and Se.
It is evaluated directly at every slice, without a tabulated response,
interpolation knots, multiplicative envelope, registration shift, or periodic
wrapping along Z. Tails outside the measured window are lost and are **not**
renormalized; small depth-dependent changes in observed total brightness can
therefore remain. The Gaussian extends past the atom-center confinement box.

XY shapes, species intensity ratios, bilinear deposition, and periodic XY FFT
convolution are unchanged. They use `element_double_gaussian_fits.csv` plus
0.08 A additional lateral blur. Effective core/tail sigma values are about
0.131/0.230 A for W and 0.132/0.236 A for Se; Se has 0.582453 times W's
integrated intensity before finite-window losses.

The historical 08/23 brightness convention is restored. An isolated,
grid-aligned W atom's **combined core + tail peak is 1**. The scale is
calculated once from the current kernels, independently of experiment or
the candidate structure. All species receive the same multiplier; their
relative intensities are preserved. Overlapping atoms may exceed 1 and
off-grid atoms can have lower sampled peaks. There is no fitted `raw_gain`.

The experiment is normalized separately: subtract the **1st percentile of
the whole measured 3D crop**, clip negatives to zero, then divide by the
**99.5th percentile of that background-subtracted crop**. Vacuum padding is
excluded. Values above 1 are retained. Neither slices nor candidate structures
are individually normalized. This equates the experimental percentile with
the isolated W reference peak by convention, not by an experimental gain fit.

The relevant YAML settings are:

```yaml
experimental:
  target_background_percentile: 1.0
  target_normalization_percentile: 99.5
imaging:
  shared_sigma_z_angstrom: 4.0
  normalize_relative_intensity_reference_peak_to_one: true
  clip_experimental_to_unit_interval: false
```

This restores the old normalization convention with the current analytic,
nonperiodic Z renderer. It does not reinstate the older periodic-Z FFT renderer
or its numerical clipping/integral-restoration step.

## Geometry and objective

The current YAML uses the centered **320 x 320** crop of the older **12-slice**
TIFF: `/data2/jt577/2026/08.2026/blob_EAT/08.24/objp_zstack_iter0200.tif`
(source shape 12 x 578 x 612). The source crop is y=129:449, x=146:466.
The configured voxel spacing remains (z,y,x) = (1,0.152564,0.152564) A,
as in the older workflow; the TIFF does not supply an ImageJ spacing record.
XY vacuum is 7.933328 A per side after snapping the requested 8 A to pixels.
Z vacuum is **8 A on each side**, giving a **28 A** cell depth and a rendered
shape of 28 x 424 x 424. The measured ROI is z=8:20, y=52:372, x=52:372;
only those 12 x 320 x 320 voxels contribute to image loss or normalization.
FairChem remains periodic in XYZ, independently of the imaging model.

The soft atom-center box covers the measured XY extent and
**z = 0–12 A** in TIFF coordinates (**8–20 A** after adding cell padding),
with weight **50**, sharpness **8 inverse A**,
and zero inward margin. The experimental level-set penalty remains disabled.
There is no explicit planarity or curvature penalty. Formation energy,
image Huber loss (weight 40, delta 0.20), composition penalty (weight 10), and
all other configured search/relaxation settings are retained.

The slice-count validation now requires `target_z_slices` to equal the actual
TIFF depth rather than hardcoding 30. Depth resampling remains disabled.
The 30-slice dataset is still supported by selecting that TIFF, setting
`target_z_slices: 30`, and restoring its desired box/padding (previously
7–23 A and 8,8,0). A shorter measured window loses more Gaussian tails;
extra vacuum does not add measured information or renormalize those tails.

Prefit, main optimization, and saved image comparisons call the same Gaussian
renderer. `image_model_settings.json` records sigma, calculated kernel scale,
experimental background/percentiles, normalization convention, and
lateral-coefficient hash. Saved NPZ image fields identify the analytic depth
model and sigma, rather than storing fitted tables or envelope samples.
Checkpoint fingerprints include these settings and the new renderer; old
model checkpoints cannot resume this objective. Old `depth_calibration_file`
and `raw_gain` configuration keys are rejected rather than silently ignored.

## Running and verification

### Mutation through a current-generation random immigrant

The YAML now configures:

```yaml
genetic:
  elite_fraction: 0.10
  random_fraction: 0.25
  crossover_fraction: 0.45
  mutation_fraction: 0.20
```

At population size 50, integer allocation produces **5 elites, 13 random
immigrants, 22 crossover children, and 10 mutation children**. Fractions use
the existing proportional/largest-remainder allocation, preserving at least
one slot per positive operator. A positive mutation fraction requires a
positive random fraction. Set `mutation_fraction: 0` to disable mutations;
omitting it in older configurations also defaults to zero.

Generation 0 remains the best imported records plus new random immigrants.
For subsequent generations, complete the random immigrants' usual image
prefit, EAT optimization, and final position relaxation first. Each mutation
selects a parent from the previous population's usual parent pool using
`parent_a_tournament_size`, then samples a partner uniformly from the
**current generation's completed random immigrants**. Donors can be reused;
there is no deduplication. A donor's optimized state is used, not its initial
random coordinates or an earlier generation's random member.

The two structures go through the same `crossover_state` superposition and
pruning as ordinary crossover, followed by the same EAT/final relaxation.
Mutation children are tagged `mutated_child` and retain both parent references.
Ordinary crossover still uses the two existing parent tournaments.

The donor completion barrier applies to serial, local parallel, and Slurm
execution. Slurm donor and offspring batches have separate artifact directories
under `slurm/random_donors/` and `slurm/offspring/`. Sorting donors by member
index before per-member seeded selection makes the choice independent of
worker completion order. Resuming a partially completed generation reuses
completed donors and children without duplicating their evaluations.

Live plots show **purple mutation dots, green crossover dots, red random
dots**, and dark gray elite/imported dots. Operator labels and fractions also
appear in logs and saved run parameters/summary. No new GA run was launched
as part of this change. Use a fresh output directory for a mutation run;
unrecognized old-code checkpoints fail the compatibility check. The immediately
preceding production driver is supported by the extension procedure below. Imports from an
unchanged scientific objective can still seed generation 0 as described below.

All 16 tests passed across `test_ga_mutations.py`, `test_ga_import_seeding.py`,
and `test_gaussian_09_30.py`; see `test_ga_mutations.log`. Mutation tests execute
the actual generation loop with lightweight worker/relaxation stand-ins,
checking donor states, ordering, backend-equivalent parent selection,
mid-generation resume, allocation, Slurm directory separation, and actual plot
colors. The Gaussian tests exercise real TIFF preprocessing and image losses.
These are CPU checks; they do not constitute a full FairChem optimization run.

### Importing initial seeds

To seed a fresh run with the best `n` previous records, set
`paths.import_structures_json` to that run's `structures.json` and
`genetic.import_parent_count` to a positive `n`. For example:

```yaml
paths:
  output_dir: "/data2/jt577/2026/09.2026/blob_EAT/09.30/runs_12_slice_gaussian_320_restart"
  import_structures_json: "/data2/jt577/2026/09.2026/blob_EAT/09.30/runs_12_slice_gaussian_320/structures.json"
genetic:
  population_size: 50
  import_parent_count: 3
```

These are edits to the corresponding sections of the existing YAML. Generation
0 contains the three lowest-score valid imported records and 47 new random
immigrants. Image prefit remains enabled for random immigrants. Imported seeds
are copied with their stored states/scores; no deduplication or new relaxation
is performed when importing. Use the same scientific objective and geometry.
Generation 1 onward uses only the preceding population for parents and elites,
with the existing YAML operator fractions. Imports are not re-added to the
parent pool each generation. Population size is configurable; imports are
capped at that size and missing valid records are replaced by random members.
An empty import path gives all-random generation 0. With a nonempty path,
`import_parent_count: 0` retains the legacy automatic count (up to the larger
of population/parent-pool size, capped to population when seeding); use an
explicit positive count for a mostly-random restart. Choose a new output
directory and seed for a fresh restart, not an existing checkpoint directory.

`test_ga_import_seeding.py` checks generation construction with and without
image prefit, variable populations, duplicate imports, and generation-0 resume.

```bash
/data2/jt577/2026/05.2026/ws09_ghost_venv/bin/python genetic_09.30.26_wse2.py
```

The YAML output directory is
`/data2/jt577/2026/09.2026/blob_EAT/09.30/runs_12_slice_gaussian_320`.
Use this as a fresh run; the prior 30-slice checkpoints are incompatible.
No full GA run was launched as part of this code update.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /data2/jt577/2026/05.2026/ws09_ghost_venv/bin/python -m unittest -v test_gaussian_09_30.py
```

Tests cover Gaussian symmetry and mass, absent envelope and Z wrapping,
finite-window tail loss, species intensity and additivity, XYZ/occupancy
finite-difference gradients, extra-Z-padding invariance, invalid sigma,
actual configured-crop production construction and box penalties, image-prefit
steps, final image serialization, and changed/old checkpoint rejection.
Normalization checks verify a W peak of 1 across several depths and widths,
overlap peaks above 1, independence from the experimental brightness scale,
and exact agreement with the percentile/background recipe on the real TIFF.
The 12-slice integration test verifies the padded cell, translated box,
exact measured loss mask, and exclusion of vacuum from the loss. Additional
checks reject a mismatched slice count and prepare the actual 30-slice TIFF.
They run on CPU without evaluating FairChem. See `test_gaussian_09_30.log`.

`archive_09.11_model/` preserves the copied old driver, config, fitted renderer,
calibration, documentation, and historical tests/reports. Those reports refer
to the old model and are not validation of this version. The separate
`genetic_09.11.26_wse2` directory was not modified.

## Extending a completed run

Keep the same `paths.output_dir` and other run settings, increase
`genetic.generations` to the desired **total**, and rerun the normal command:

```bash
python genetic_09.30.26_wse2.py --config genetic_09.30.26_wse2.yaml
```

There is no `--resume` flag: checkpoint detection is automatic. Increasing 100
to 200 restores the final population from generation index 99 and runs indices
100–199. Existing completed generations and their scores are retained. A
partially completed generation also retains its completed members.

The checkpoint reader recognizes the immediately preceding production script
(SHA-256 `0f94627187c7e0ad7c34e4461829b3bf7b9c0c5985725f76509cc72dacf30320`).
It verifies the original settings and scientific assets with that known code
hash and the original generation limit. It then preserves the original
fingerprint in every generation/member file. The checkpoint's
`fingerprint_generations_requested` field remembers the original limit even
after `generations_requested` is increased, so repeated restarts and further
extensions remain compatible. Scientific-setting changes, changed assets,
unrecognized code versions, and decreased generation limits are rejected.
Do not manually edit checkpoint fingerprints or delete old generation files.

`test_ga_generation_extension.py` covers old/new checkpoint compatibility,
100→200→300 extensions, partial populations, immutable completed files,
Slurm worker fingerprints, and rejection of changed settings/assets or corrupt
saved generations. These tests do not evaluate FAIRChem.
