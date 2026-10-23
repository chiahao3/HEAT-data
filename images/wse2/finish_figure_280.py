"""Add optional physical axes to the Blender views and make a panel preview.

Run after prepare_figure_280.py and scripts/blender/wse2/render_figure.py.
The existing loss plot is reused unchanged. This does not edit the manuscript
or replace the manuscript's current assembled figure.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / 'figure_280x280'


def add_depth_axes(stem, config):
    camera = config['camera']
    width = camera['horizontal_span_A']
    height = camera['vertical_span_A']
    cx, _, cz = camera['center_experimental_xyz_A']
    extent = [cx-width/2, cx+width/2, cz-height/2, cz+height/2]
    array = np.asarray(Image.open(OUT / f'{stem}.png'))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.imshow(array, extent=extent, origin='upper', interpolation='nearest')
    ax.set(xlabel=r'$x$ (Å)', ylabel=r'$z$ (Å)',
           xticks=[0, 10, 20, 30, 40], yticks=[0, 5, 10])
    ax.tick_params(labelsize=18, width=1.2, length=5)
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=.12, bottom=.22, right=.99, top=.98)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / f'{stem}_axes.{suffix}', dpi=400, facecolor='white')
    plt.close(fig)


def make_preview(config):
    fig = plt.figure(figsize=(13.4, 11.5), facecolor='white')
    # Three rows preserve the proposed reading order while giving cross-sections
    # sufficient lateral space. This is a review sheet, not a manuscript overwrite.
    boxes = [
        ('a', 'Experimental', 'experimental_zsum_280x280_axes.png', [.025,.67,.29,.29]),
        ('b', 'Measured-field isosurface', 'experimental_field_isosurface_depth_fade_axes.png', [.34,.68,.64,.26]),
        ('c', 'HEAT · generation 0', 'heat_generation_0000_depth_axes.png', [.025,.385,.47,.25]),
        ('d', f"HEAT · best at generation {config.get('heat_display_generation', config['final_generation'])}", 'heat_final_depth_axes.png', [.52,.385,.47,.25]),
        ('e', 'Search convergence', '../wse2_total_loss_convergence.png', [.025,.04,.29,.30]),
        ('f', 'HEAT reconstruction', 'heat_reconstructed_zsum_280x280_axes.png', [.35,.04,.29,.30]),
        ('g', 'Ordinary-GA reconstruction', 'ordinary_reconstructed_zsum_280x280_axes.png', [.675,.04,.29,.30]),
    ]
    for letter, title, name, box in boxes:
        ax = fig.add_axes(box)
        ax.imshow(Image.open(OUT / name))
        ax.axis('off')
        fig.text(box[0], box[1]+box[3]+.01,
                 r'$\mathbf{' + letter + '}$  ' + title, fontsize=16)
    fig.text(.355,.67, rf'$I_{{\mathrm{{norm}}}}={config["normalized_isovalue"]:.2f}$; complete surface, view-depth opacity fading', fontsize=13)
    fig.text(.52,.375, 'W: blue    Se: magenta', fontsize=13)
    fig.text(.027,.012, '280 × 280 pixels · 42.71792 Å × 42.71792 Å · shared projection intensity scale', fontsize=13)
    fig.savefig(OUT / 'figure_panel_preview.png', dpi=220, facecolor='white')
    fig.savefig(OUT / 'figure_panel_preview.pdf', dpi=300, facecolor='white')
    plt.close(fig)


def write_readme(config):
    text = f'''WSe2 figure assets — central 280 x 280 measured pixels

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
range ({config['intensity_scale'][0]:.10g}, {config['intensity_scale'][1]:.10g}).
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

Generation 0: {config['gen0_atoms']} atoms ({config['gen0_species']}).
Final: {config['final_atoms']} atoms ({config['final_species']}), generation
{config['final_generation']}, member {config['final_member']}.
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

Threshold: normalized intensity {config['normalized_isovalue']}, raw intensity
{config['raw_isovalue']:.12g}. Normalization is the run's saved convention:
I_norm = (raw - {config['normalization']['target_background']}) /
         {config['normalization']['target_normalization']:.12g}.
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
'''
    (OUT / 'README.txt').write_text(text)


def main():
    config = json.loads((OUT / 'figure_metadata.json').read_text())
    for stem in ['heat_generation_0000_depth', 'heat_final_depth',
                 'experimental_field_isosurface_depth', 'experimental_field_isosurface_depth_fade']:
        add_depth_axes(stem, config)
    make_preview(config)
    write_readme(config)
    print('Wrote physical-axis variants, panel preview, and README.txt to', OUT)


if __name__ == '__main__':
    main()
