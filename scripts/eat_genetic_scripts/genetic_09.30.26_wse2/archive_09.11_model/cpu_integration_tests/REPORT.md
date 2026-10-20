# 30-slice GA update: CPU verification

All production integration checks passed on CPU. No FairChem energy evaluation or full GA population was run.

- Saved accepted fit replay: maximum raw-voxel difference 1.05e-06 in production float32; 7.42e-09 in float64 (saved reference is float32).
- Production float32 position/occupancy gradients are finite. All 10 float64 finite-difference checks passed, including confinement forces; maximum absolute difference 3.76e-09.
- Adding 8 Å z padding on both sides preserves measured-image predictions and confinement costs after translating positions and bounds.
- No axial periodic wrap. All 30 measured slices remain in the loss.
- The retired ellipse term contributes exactly zero even if its internal compatibility weight is changed.
- All 24 tested vacuum placements cost more than a true Se deletion after including the explicit 7–23 Å confinement, even against the nominal stoichiometric deletion cost.
- Four actual image-prefit steps reduced the tested objective from 0.534840 to 0.406213.
- Final image serialization and preparation diagnostics passed with no ellipse regions.
- All 30 slice comparisons were generated using optional extra z padding; their measured-ROI loss agrees with the optimizer. These plots replay the reference calibration structure, not a new GA reconstruction.
- Changed-calibration and old-objective checkpoints are rejected.

See [full numerical results](results.json), [test log](run.log), and [example slice comparison](extra_z_padding/slice_comparisons/generation_000_member_000_calibration_replay/slice_z015_comparison.png). Reproduction instructions and the exact model are in [the method README](../README_WSE2_METHOD.md).
