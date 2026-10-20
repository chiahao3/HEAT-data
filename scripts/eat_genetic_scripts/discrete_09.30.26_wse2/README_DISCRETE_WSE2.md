# Discrete WSe₂ genetic algorithm — 09/30/26

The entry point is `discrete_09.30.26_wse2.py`, configured by
`discrete_09.30.26_wse2.yaml`. It contains ordinary W and Se atoms with continuous
positions and discrete identities/counts. The copied fractional-species search,
soft image prefit, entropy stages, superposition crossover, and rounding machinery
have been removed. There is no import of the original genetic driver.

## Objective and local relaxation

The score is

```
(E_FAIRChem - N_W*E_ref_W - N_Se*E_ref_Se) / N
+ 40 * mean_measured_voxels(Huber(predicted - experimental, delta=0.20))
+ 10 * sum_species((N_species/N - target_fraction_species)**2)
+ 50 * mean_atoms(sum_xyz(softplus_box_violations**2))
```

All four weights/settings are configurable. The elemental reference structures
are selected by the same Materials Project rule as the prior workflow and
computed with ordinary FAIRChem (`uma-s-1p1`, task `omat`). Reference energies are
cached with model/task metadata in `reference_energies.json`. Set
`model.mp_api_key: "YOUR_MATERIALS_PROJECT_KEY"` in the YAML for the first run.
A blank key falls back to the `MP_API_KEY` environment variable. Alternatively,
set `model.reference_energies_json` to a saved compatible file. The key is
omitted from saved run parameters and restart fingerprints.
For a controlled comparison, verify the same elemental references in both runs.

Position-only projected BFGS uses the full objective gradient: `-forces/N` plus
the image and box gradients. Species and atom counts are fixed during each
local relaxation. Composition is therefore constant during each positional
relaxation but affects GA selection. Best evaluated positions are retained.
The fixed cell remains periodic in xyz, including explicit vacuum. The sample
box is a soft penalty, not a rigid planar lattice constraint.

The image model is unchanged: native 12-slice, centered 320×320 TIFF crop;
background percentile 1 and normalization percentile 99.5 over the entire
measured crop; double-Gaussian XY kernels plus 0.08 Å blur; Gaussian depth
response with sigma 4 Å, no envelope or depth wrapping. An isolated grid-aligned
W peak is normalized to 1. The voxel loss excludes synthetic padding. The
feasible-region mask is used only to initialize and relocate atoms.

## Genetic operators

- **Random initialization:** use the same intensity-prioritized cylindrical
  candidate pool, then independently assign occupied/empty and W/Se labels.
  Species probabilities follow the soft composition target (1:2 by default).
  Feasible positional jitter and close-contact repair follow. There is no
  species optimization during initialization.
- **Atom count:** `initialization.mean_atom_count` overrides the initial mean.
  When null, `occupied_probability: 0.30` times the candidate-pool size sets it.
  These are discrete Bernoulli draws, not fractional atom weights. For the
  supplied TIFF, the pool has 2,860 sites and the default pre-repair mean is 858
  atoms. Repair can lower this count; a minimum of one atom prevents empty
  structures. `hard_atoms.maximum_count: null` uses the candidate-pool size as
  the count ceiling. Neither repair nor relaxation imposes exact WSe₂ counts.
- **Selection:** independent score tournaments, sizes 4 and 2, excluding parent A
  from parent B's tournament. Elites are copied unchanged. No deduplication.
- **Crossover:** choose a random plane in the measured atomic box; take one side
  from A and the other side from B, keeping each atom's species. Parents remain
  in their common experimental coordinate frame; there is no alignment or
  translation. Remove close contacts in random order and enforce count bounds.
- **Mutation:** each crossover child independently has a **30% chance** of one
  mutation. Conditional weights are 35% rattle, 25% relocate, 20% insert, and
  20% delete, matching the dislocation baseline. Rattle affects 20% of atoms
  with sigma 0.25 Å. Insert/delete changes one atom; either species can change.
  Optional W↔Se transmutation is available with default weight zero. An
  inapplicable move is recorded as such. Random immigrants and elites are not
  passed through this mutation step.
- **Population:** size 50, 100 generations, fractions 10% elite, 25% random,
  65% crossover. Largest-remainder rounding gives **5 elite + 13 random + 32
  crossover** slots; mutated children occupy their existing crossover slots.
  Generation zero is random except for optional imported best-n structures.
  Imports are selected by their saved score, without deduplication, and then
  rescored/relaxed under the current objective. Cells must match.

## Running

From this directory:

```bash
./run_discrete_09.30.26_wse2.sh --validate-only --output-dir /tmp/wse2-discrete-check
./run_discrete_09.30.26_wse2.sh
./run_discrete_09.30.26_wse2.sh --resume
```

`DISCRETE_PYTHON` overrides the launcher's Python executable. `--config` and
`--output-dir` override the input YAML and output path. Relative YAML paths are
resolved relative to that YAML. The default output is
`../09.30/runs_12_slice_discrete_320`, separate from existing searches.

For a single GPU allocation, submit `slurm_discrete_09.30.26_wse2.sh` from this
directory, supplying the partition/account required by the cluster. It runs
one driver inside the allocation, not one Slurm job per member. For multiple
local workers set `runtime.parallel_workers` and comma-separated visible GPU
indices in `runtime.gpu_ids`; request matching resources. Selection waits for
all members in the preceding generation. Each worker owns its own calculator.

## Outputs and restart

- `structures.json`, `members/*.json`: objective components, atom counts,
  structures, ancestry, mutations, convergence, and compute metrics.
- `best_structure.xsf`, `best_structure.json`, and `xsf_members/*.xsf`.
- `discrete_optimization_live_scores_by_generation.png`: **red random, green
  crossover, purple mutated**, gray elite, orange imported; black best-so-far.
- Experimental, best-predicted, and residual slice TIFFs at run completion.
- `run_parameters.json`, image-model and geometry metadata, reference energies.
- `plans/`, `ga_checkpoint.json`, and member progress checkpoints for restart.

If setup fails before generation plans or members exist, rerun the original
command after fixing the configuration; `--resume` is not required.

`--resume` reuses completed members and generation plans. An unfinished member
restarts positional BFGS from its saved best positions with a fresh BFGS
history. Recorded compute totals include work through the latest progress
checkpoint; work after that checkpoint before an abrupt crash cannot be
recovered. Per-member calculator counts exclude reference setup and elites.
Summary wall time is explicitly for the current invocation; summed member
wall time is not overall elapsed time when workers run concurrently. Compare
multiple seeds and use both model evaluations and elapsed time for a paper.

Changing science parameters, source code, TIFF, element table, or imported
input files invalidates restart. Increasing the generation count and changing
worker throughput are supported. A directory lock prevents simultaneous
drivers from writing the same run. Candidates with no interatomic edges that
FAIRChem cannot evaluate are marked invalid and excluded from best-structure
outputs; their selection score is 1e30. Other calculator errors stop the run.
If every member is invalid, the driver stops with a diagnostic.

## Checks

```bash
/data2/jt577/2026/05.2026/ws09_ghost_venv/bin/python -m unittest test_discrete_wse2 -v
```

Tests use a small synthetic image and an explicitly test-only analytic
calculator to check total-objective finite differences, GA operators, import,
local relaxation, output generation, and resume. `--validate-only` separately
checks the actual TIFF, candidate counts, and image/box gradients without
loading FAIRChem or fetching references.
