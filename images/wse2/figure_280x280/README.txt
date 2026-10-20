WSe2 figure assets — central 280 x 280 measured pixels

Run these commands from the paper_genetic_EAT_fixed_scaffold_2026 directory:
  python3 images/wse2/render_ordinary_generation126.py
  python3 images/wse2/prepare_figure_280.py
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python scripts/blender/wse2/render_figure.py
  python3 images/wse2/finish_figure_280.py

Open figure_panel_preview.png to review the seven proposed panels together.
The existing convergence figure is reused unchanged. No manuscript or existing
assembled figure is modified.

PROJECTIONS
  experimental_zsum_280x280_axes.png / .pdf / .svg
  heat_reconstructed_zsum_280x280_axes.png / .pdf / .svg
  ordinary_reconstructed_zsum_280x280_axes.png / .pdf / .svg

All three are plotted from the numerical TIFFs, with the same saved intensity
range (-0.0004069160786, 1.961752415).
No independent contrast adjustment is applied. TIFF/PNG orientation is retained:
x increases rightward; image y increases downward. Samples are at i*0.152564 A;
pixel boundaries are half a pixel beyond the sample coordinates. Labels are
18 pt and tick labels 14 pt before figure scaling. PNGs are 400 dpi; PDF/SVG
keep the text and axes as vector elements.

The ordinary projection uses generation 126, member 30 (1298 full atoms),
regenerated with the run's exact GaussianImageModel. Its image penalty was
checked against the member CSV. Historical best-export TIFFs from generation
144 are preserved separately. HEAT's displayed best at generation 126 was
first found at generation 115. See ordinary_generation126_render_metadata.json.

STRUCTURES
  heat_generation_0000_depth.png / .blend
  heat_final_depth.png / .blend
  generation_0000_best_loss_cropped_280x280.xsf

Generation 0: 898 atoms ({'Se': 605, 'W': 293}).
Final: 1150 atoms ({'Se': 761, 'W': 389}), generation
115, member 31.
The generation-0 crop applies the exact final-crop bounds and coordinate shift.
Original atom z coordinates and vacuum spacing are retained in XSF. The final
crop was independently reproduced from its full-generation XSF and verified.
W is blue; Se is magenta. Both views use the same orthographic camera and sphere
radii as the reference style. Camera looks along +y; x right, z up. Camera span
is 48 A by 18 A. No depth exaggeration. Raw Blender PNGs are 3200 x 1200.

EXPERIMENTAL FIELD
  experimental_field_isosurface_depth.png / .blend (fully opaque)
  experimental_field_isosurface_depth_fade.png / .blend (foreground emphasized)

This is one global isosurface extracted from the actual 12 x 280 x 280 measured
TIFF volume, not fitted Gaussian blobs. The crop is raw TIFF [0:12,149:429,166:446].
Spacing is (z,y,x) = (1.0,0.152564,0.152564) A. Its z sum was checked to equal the
provided experimental projection exactly. There is no smoothing, component
filtering, or depth exaggeration. Smooth shading only changes surface lighting.

Threshold: normalized intensity 0.2, raw intensity
0.0247253611684. Normalization is the run's saved convention:
I_norm = (raw - 0.0) /
         0.123626805842.
This uses the saved 99.5-percentile scale; it is not a crop-specific min/max
normalization and values can exceed one. Change FIELD_ISOVALUE near the top of
prepare_figure_280.py and rerun to choose another contour.

The field reaches the measured depth boundaries. Surfaces are closed against
zero outside the sampled volume for display; the flat boundary caps are not
evidence of physical surfaces or measured decay outside the 12 slices.
The opacity-faded version retains the whole mesh and uses the older figure's
viewing-depth convention: nearest 1.5 A opaque, then alpha=max(0.01,(1+d/3)^-2).
This opacity encodes distance along y, not z, species, phase, or uncertainty.
Blue is a display color for the measured field, not a W assignment.

Optional *_depth_axes.png / .pdf variants have calibrated x and z axes.
figure_metadata.json and field_mesh_metadata.json record inputs and geometry.
Saved Blender scenes contain the actual mesh and are editable without the TIFF.

ORIGINAL SCRIPTS FOUND
  07.2026/blender_scripts_for_paper/ptycho_wse2/render_ptycho_eat_structure.py
  07.2026/blender_scripts_for_paper/ptycho_wse2/render_ptycho_phase_field.py
  07.2026/blender_scripts_for_paper/ptycho_wse2/render_ptycho_birdseye_orthographic.py
These paths are relative to the 2026 workspace. The current renderer uses a local
copy of the structure style in scripts/blender/wse2/structure_style.py. Original
scripts and historical images are preserved.
