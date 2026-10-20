# Calibrated 30-slice WSe2 genetic optimization

The driver remains `genetic_09.01.26_wse2.py` and loads the adjacent YAML by default. The original source, YAML and README are preserved in `original_before_30_slice_update/`. Existing analysis results and the previous CPU prototype are unchanged.

## Default experiment and geometry

The input is `../chia-hao_data/objp_zstack_iter0001_30_slices.tif`, using the centered **150×150 crop, all 30 slices**, without resampling. The source crop is y=214:364, x=231:381. Voxel spacing is (z,y,x) = (1, 0.152564, 0.152564) Å, retaining the calibration's assumed 1 Å depth spacing.

Requested extra padding is (x,y,z) = (8,8,0) Å per side. Whole-pixel snapping makes XY padding 7.933328 Å, producing a 30×254×254 render grid and a 38.751256×38.751256×30 Å FairChem cell. FairChem remains periodic in all axes; the image renderer is periodic only in XY. The image objective and slice-comparison metrics use only the measured 30×150×150 ROI, including its measured vacuum slices.

The allowed atomic box is independent of that image ROI. It spans the measured XY crop and **7≤z≤23 Å in TIFF coordinates**. `sample_box.z_min_angstrom` and `z_max_angstrom` set these bounds. The existing occupancy-normalized smooth box penalty remains at weight 50, sharpness 8 Å⁻¹. Candidate seeding excludes depths outside these bounds. The separate intensity-level-set penalty remains optional and defaults to zero.

Extra z padding remains available: change `system.vacuum_padding_angstrom` to e.g. `"8,8,8"`. The renderer translates its experimental origin and atomic bounds automatically; then the allowed z interval in cell coordinates is 15–31 Å. Measured-image predictions and box costs are unchanged for correspondingly shifted atoms. `sample_box.confinement_penalty: 0` explicitly disables the box penalty, but allows weakly visible vacuum atoms.

## Blob model

For atom i and species s, the raw prediction is

`I(x,y,z) = g E(z) Σᵢ Σₛ oᵢₛ aₛ Lₛ(x−xᵢ,y−yᵢ) K(z−zᵢ)`.

- `oᵢW`, `oᵢSe` and implicit null occupancy are the existing EAT variables. `aW=1`, `aSe=0.5824533131834269`.
- Each unit-integral lateral profile `Ls` is the fitted core+tail double Gaussian in `element_double_gaussian_fits.csv`, with fixed extra isotropic XY Gaussian blur **σ=0.08 Å** added in quadrature. Effective W core/tail σ are 0.131169/0.230443 Å; Se are 0.132432/0.235900 Å. Core integral fractions are 0.234326 and 0.214940, respectively.
- `K` is the accepted shared, nonnegative **31-bin flexible depth response**, with linear interpolation and zero outside its support. It is not a Gaussian. Its tabulated RMS width is 3.894058 Å; central FWHM is about 6.83235 Å. The full-domain mass is one. The centered lag table is stored in `calibration/accepted_30_slice_model.json`.
- `E(z) = exp[-(max(9−z,0)/1.326847883729309)² − (max(z−20,0)/1.219451739561209)²]`. This fixed experimental-coordinate envelope is one from 9–20 Å and suppresses the **summed simulated intensity field** outside. The envelope plateau and atomic box bounds are separate parameters.
- The calibrated raw gain is **g=12.298994422022743**. Both experiment and model are divided by one fixed target percentile scale (99.5th percentile, 0.09016454655677088 for this crop). No per-candidate, per-slice, unit-peak, clipped-field or observed-window renormalization is performed.
- The fitted additive background is zero for every slice. The requested suppression is `E`, not an additional adjustable background subtraction. The renderer preserves the calibration's tiny Fourier ringing.

Positions are physical cell coordinates. Subtract padding to obtain TIFF coordinates. The renderer uses the centered response directly and adds **no hidden registration offset**. To replay the old calibration XSF specifically, add its recorded XY registration and change `z_old` to `z_old−8+8.810930895711342` Å before adding any new z padding. Arbitrary old GA imports must be transformed into the new cell/frame before use; old checkpoints cannot resume this objective.

## Optimization and outputs

The blob shapes, envelope and gain remain fixed at the accepted calibration. **Atomic XYZ coordinates and W/Se/null occupancies remain optimizable**, including by autograd and PBFGS. Bilinear XY deposition and linear depth interpolation are differentiable within bins, with interpolation kinks at bin boundaries.

The guided score contains formation energy, voxel Huber loss (weight 100, δ=0.20 in normalized intensity), the existing quadratic global 1:2 composition penalty (weight 10), box confinement, and optional level-set/vacancy-bias terms. The Se-vacancy bias stays disabled. The depth-sum ellipse penalty is removed from both prefits, the main objective and ranking. No ellipse detection runs. Some old serialized ellipse fields remain as zero/empty compatibility entries.

The actual image prefit, main optimization and final image outputs all call the same calibrated renderer. Optional member slice comparisons (`diagnostics.member_slice_comparisons_enable`) show experiment, calibrated model and residual over the measured ROI. The new default output directory is `runs_30_slice_calibrated_150/`. Population/generation settings remain as configured previously (50/100); this update does not launch that run.

Each constructed problem writes `image_model_calibration.json` with the depth table, envelope, normalization and provenance. Checkpoint identity includes the TIFF, calibration, lateral CSV, renderer and minimizer contents as well as parameters and driver code. Old-objective checkpoints and changed calibration assets are rejected.

## CPU verification

Run the actual-production integration tests with:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data2/jt577/2026/05.2026/ws09_ghost_venv/bin/python \
  cpu_integration_tests/run_tests.py
```

These tests construct the real GA problem through its lightweight coordinator path, avoiding model/reference downloads. They exercise production rendering against the accepted saved TIFF, full position/occupancy gradients including box forces, padding invariance, absence of axial wrapping, vacuum hiding, actual image-prefit steps, diagnostics/serialization, and checkpoint rejection. See `cpu_integration_tests/results.json` and `run.log`. They do **not** evaluate FairChem energy or run a GA population.

Launch a full run separately in the FairChem environment with:

```bash
/data2/jt577/2026/05.2026/ws09_ghost_venv/bin/python genetic_09.01.26_wse2.py
```
