# HEAT: code and data

Supporting code, inputs, recorded search results, and plotting scripts for
*Hereditary Effective Atom Theory for Large-Scale Atomic Structure Prediction*.

The main examples are twisted-bilayer WSe2 reconstruction from ptychography,
reconstruction of the 240-atom H sublattice in Mg(BH4)2, and reconstruction of
an Fe edge-dislocation active region. These are individual stochastic runs,
not a multi-seed scaling study.

## Contents

| Directory | Contents |
| --- | --- |
| `inputs/MgB2H8/` | Original experimental CIF, `cm803019e_si_003.cif` |
| `inputs/NIST_dump_files/` | Original Fe dislocation and corresponding base structures in LAMMPS dump format |
| `inputs/ptycho_data/` | Reconstructed experimental phase TIFF used by the WSe2 searches |
| `parameters/` | Recorded run parameters, with embedded credentials removed |
| `configs/` | Portable configurations derived from those recorded parameters |
| `scripts/eat_genetic_scripts/` | HEAT code and WSe2 conventional-GA code |
| `scripts/ordinary_genetic_scripts/` | Conventional H-sublattice and dislocation GAs |
| `scripts/misc/baseline_ptycho_guided_320/` | WSe2 reference construction, source snapshot, inputs, relaxed structures, and scores |
| `images/` | Convergence scripts, logs/metrics, and existing figure images |
| `images/wse2/gaussian_splats/` | Additional trial structure, cropped XSF, simulated phase images, and rendering metadata |

`eat`/`GEAT` in historical filenames denotes HEAT; `ordinary` denotes the
conventional GA. Blender scenes and backup files are excluded. Some historical
rendering scripts and existing scientific figure images are retained; the shared
Blender utilities needed to rerun those scripts are not included.

## Updated WSe2 comparison

The conventional-GA metrics now extend through generation **446**, while the
HEAT comparison ends at generation **128**. The total-loss plot shows each run
only through its own endpoint, marked by a filled circle. HEAT used 543,767
interatomic-model evaluations through generation 128; the conventional GA used
682,069 through generation 446, approximately 25% more. These are evaluation
counts, not elapsed-time ratios.

The final best losses are approximately -0.3513 for HEAT and 0.01993 for the
conventional GA. The conventional projection in `images/wse2/figure_280x280/`
has been updated to generation 446; its rendering metadata is stored alongside
it in `ordinary_generation446_render_metadata.json`.

The previous `xsfs/` directory and structure-conversion helper have been removed
from this version. The recorded convergence metrics, input structures, guided
WSe2 reference, and additional trial XSF in `gaussian_splats/` remain included.
Scripts that require the removed per-generation XSF exports or export manifests
cannot currently reproduce those structural images from this checkout alone.

## Reproduce convergence plots

From the repository root, in a Python environment:

```sh
python3 -m pip install -r requirements-plotting.txt
python3 images/H_sublatt/plot_h_sublattice_convergence.py
python3 images/edge/plot_dislocation_convergence.py
python3 images/wse2/plot_wse2_convergence.py
```

These commands read the bundled logs/metrics and overwrite the corresponding
plots in `images/`. They do not run structure searches or require model weights.
The H plot converts the logged energy difference to energy per H atom. The edge
plot uses its specified core normalization; these display conversions do not
change the objectives used in the original searches.

The WSe2 script is needed to reproduce the loss, formation-energy, and image-
penalty curves. The references are read from included results:

- The total-loss plot uses `baseline_ptycho_guided_320/run_summary.json`: the
  reference relaxed against the same energy, image, composition, and box objective.
- The formation-energy plot uses the `previous_energy_only_relaxed` row of
  `baseline_ptycho_guided_320/objective_comparison.csv`: a recorded re-evaluation
  of the earlier energy-only structure (-0.4223027245 eV/atom). Its coordinates
  are included as `inputs/previous_energy_only_relaxed.extxyz` in that folder.
  This replaces a missing historical results-file dependency. The ideal-loss
  helper uses this energy with hypothetical zero penalties, not the measured
  image penalty.

To regenerate only the updated total-loss plot, use
`python3 images/wse2/plot_wse2_convergence.py --loss-only`.

`images/wse2/plot_wse2_pearson_convergence.py` and the structural-image rendering
helpers require the removed `xsfs/` exports and manifests, in addition to PyTorch
and Pillow. They are retained as workflow records, not standalone reproduction
commands for this version. Some also retain historical input paths, including
`ptycho_data/`; the bundled experimental TIFF is under `inputs/ptycho_data/`.
`evaluate_reference_image.py` additionally refers to a historical energy-only
reference folder that is not bundled. The double-Gaussian fit table is
`scripts/eat_genetic_scripts/genetic_09.30.26_wse2/element_double_gaussian_fits.csv`.

## Run new searches

Full searches require ASE, PyTorch, NumPy, SciPy, Matplotlib, Pillow, PyYAML,
`mp-api`, and the **EAT-modified FAIRChem implementation**. A stock FAIRChem
installation does not provide all effective-atom operations. Model weights
are not bundled; the runs use `uma-s-1p1` with task `omat`.

The WSe2 guided-reference folder records its environment in
`environment_versions.json`, its FAIRChem revision in
`source/fairchem_revision.txt`, and local FAIRChem modifications in
`source/fairchem_local_changes.patch`. These are dependency provenance for that
run, not a tested universal installation lockfile for every historical search.
Use an environment with the required EAT implementation before launching.

Supply your own Materials Project credential through the environment when
elemental reference structures are fetched:

```sh
export MP_API_KEY="your-own-key"
```

Do not place credentials in committed configuration files. Model access may
also require authentication in your local environment.

First validate paths and prepare a configuration without loading FAIRChem:

```sh
python3 run_search.py heat h --prepare-only
python3 run_search.py conventional edge --prepare-only
python3 run_search.py heat wse2 --prepare-only
```

Remove `--prepare-only` to launch. Supported methods are `heat` and
`conventional`; cases are `h`, `edge`, and `wse2`. For example:

```sh
python3 run_search.py heat edge --output-dir runs/heat/edge-new
```

The launcher resolves inputs relative to this checkout and writes an absolute-
path runtime YAML in the output directory. It uses the current Python interpreter.
Choose a fresh output directory for each independent run. Edit the relevant
`configs/METHOD_CASE.json` to change scientific or hardware settings. Recorded
H/dislocation settings use two workers and GPU IDs `0,1`; WSe2 uses one worker.
Adjust these to your hardware. Changes can affect timing and stochastic results.

The active script-side YAML files also contain recorded settings with portable
paths. The launcher is preferred because it is independent of the working
directory. Slurm templates require your own partition/account and an activated
environment; their original workstation names are retained only in filenames.

## Inputs and provenance

- **Mg(BH4)2:** original supplementary CIF from Filinchuk et al.,
  [Chemistry of Materials (2009), DOI: 10.1021/cm803019e](https://doi.org/10.1021/cm803019e).
  The reference is prepared by relaxing the complete structure before withholding
  H atoms, as described in the manuscript and search code.
- **Fe dislocations:** LAMMPS dumps from the
  [NIST Interatomic Potentials Repository, Mendelev Fe-2 implementation](https://www.ctcms.nist.gov/potentials/entry/2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2/2003--Mendelev-M-I--Fe-2--LAMMPS--ipr1.html).
  These are computational structures, not experimental CIFs. The study further
  relaxes its selected active region with FAIRChem.
- **WSe2:** the TIFF is a reconstructed experimental phase volume, not raw 4D-STEM
  diffraction data. Cite the experimental study identified in the manuscript
  (Nguyen et al., [DOI: 10.1126/science.adl2029](https://doi.org/10.1126/science.adl2029)).
  The guided-reference README documents layer construction, alignment, and
  relaxation. Its loss is approximately -0.3566620931; it is a constructed
  comparison structure, not proof of the global minimum.

## Archive and reproducibility notes

Historical logs, parameter records, source snapshots, and development files can
contain original absolute paths as provenance. The launch configurations above
replace those paths for current use. `archive_09.11_model/` contains older WSe2
development work and is not the production 12-slice workflow; its old tests may
refer to external inputs that are not included. Original source fingerprints
refer to original files, before credential sanitization and portability edits.

This archive supports replotting recorded convergence results and includes the
main search drivers and inputs. Full GPU searches have not been rerun as part
of archive preparation. It does not include every evaluated candidate, raw
microscope diffraction data, or pretrained model weights.

## Remaining release work

- Select a code license and clarify redistribution terms for experimental inputs.
- Provide and verify installation instructions for the EAT-modified FAIRChem
  dependency, including the appropriate revision/patch for the H and edge runs.
  The recorded environment currently covers the guided WSe2 reference.
- Validate full searches in a clean GPU environment. Input-path checks and
  plot reproduction do not establish that every search dependency is installed.
- Restore or separately archive the per-generation structure exports and their
  manifests if structural-image and Pearson-curve reproduction is required.
- Archive a specific version of this public repository with a persistent
  identifier, then use that version in the manuscript availability statements.

No repository-wide redistribution license has been selected yet. Third-party
data, model weights, and dependency code retain their respective terms and
attribution requirements.
