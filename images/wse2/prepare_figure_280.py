"""Prepare calibrated WSe2 panels and inputs for render_figure.py.

Run with the usual Python (NumPy, Pillow, Matplotlib). No ASE/TIFF package is
needed. Original TIFFs, PNGs and XSFs are read only. Then run the Blender driver
printed at the end. All paths resolve relative to this paper directory.
"""
from pathlib import Path
import json
import hashlib
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# USER SETTINGS
OUTPUT_DIR = Path(__file__).resolve().parent / 'figure_280x280'
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 14
OUTPUT_DPI = 400
FIELD_ISOVALUE = 0.20  # (raw phase - saved background) / saved normalization

PAPER = Path(__file__).resolve().parents[2]
HEAT = PAPER / 'xsfs/wse2/xsfs_wse2_EAT'
ORDINARY = PAPER / 'xsfs/wse2/xsfs_wse2_ordinary'
TIFF = PAPER / 'ptycho_data/objp_zstack_iter0200.tif'


def read_tiff(path):
    with Image.open(path) as im:
        frames = []
        for index in range(im.n_frames):
            im.seek(index)
            frames.append(np.array(im, dtype=np.float32))
    return frames[0] if len(frames) == 1 else np.stack(frames)


def read_xsf(path):
    lines = path.read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.strip() == 'PRIMVEC')
    cell = np.array([[float(v) for v in line.split()[:3]] for line in lines[i+1:i+4]])
    i = next(i for i, line in enumerate(lines) if line.strip() == 'PRIMCOORD')
    count = int(lines[i+1].split()[0])
    rows = [line.split() for line in lines[i+2:i+2+count]]
    symbols = np.array([{'34': 'Se', '74': 'W'}.get(r[0], r[0]) for r in rows])
    positions = np.array([[float(v) for v in row[1:4]] for row in rows])
    assert len(positions) == count
    return cell, symbols, positions


def write_xsf(path, cell, symbols, positions):
    lines = ['# Central 280x280 measured-pixel crop; original z and vacuum retained.',
             'CRYSTAL', 'PRIMVEC']
    lines += [' '.join(f'{v:.14f}' for v in row) for row in cell]
    lines += ['PRIMCOORD', f'{len(symbols)} 1']
    lines += [f'{s} ' + ' '.join(f'{v:.14f}' for v in row)
              for s, row in zip(symbols, positions)]
    path.write_text('\n'.join(lines) + '\n')


def crop_structure(source, crop):
    cell, symbols, positions = read_xsf(source)
    lower, upper = np.array(crop['original_cell_xy_bounds_A'])
    keep = np.all((positions[:, :2] >= lower) & (positions[:, :2] < upper), axis=1)
    positions = positions[keep] + np.array(crop['coordinate_shift_xyz_A'])
    return np.diag(crop['cell_lengths_A']), symbols[keep], positions


def save_projection(array, stem, spacing, scale):
    # Preserve the original PNG orientation: x rightward, image y downward.
    # Samples are at i*d, matching the optimizer; bounds are half a pixel beyond.
    ny, nx = array.shape
    extent = [-spacing/2, (nx-.5)*spacing, (ny-.5)*spacing, -spacing/2]
    fig, ax = plt.subplots(figsize=(4.7, 4.7))
    ax.imshow(array, origin='upper', extent=extent, cmap='gray',
              vmin=scale[0], vmax=scale[1], interpolation='nearest')
    ax.set(xlabel=r'$x$ (Å)', ylabel=r'$y$ (Å)',
           xticks=[0, 10, 20, 30, 40], yticks=[0, 10, 20, 30, 40])
    ax.tick_params(labelsize=TICK_LABEL_SIZE, width=1.25, length=5)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.subplots_adjust(left=.21, bottom=.19, right=.98, top=.98)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT_DIR / f'{stem}_axes.{suffix}', dpi=OUTPUT_DPI,
                    facecolor='white')
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HEAT / 'export_manifest.json').read_text())
    ordinary_manifest = json.loads((ORDINARY / 'export_manifest.json').read_text())
    crop = manifest['cropped_structures']['cropped_280x280']
    spacing = manifest['normalization']['voxel_size_zyx_A']
    scale = manifest['display_scale_input_tiff_units']
    assert scale == ordinary_manifest['display_scale_input_tiff_units']
    roi = manifest['cropped_280_original_tiff_roi_zyx']
    raw_volume = read_tiff(TIFF)[tuple(slice(*bounds) for bounds in roi)]
    assert raw_volume.shape == (12, 280, 280)
    normalization = manifest['normalization']
    field = (raw_volume - normalization['target_background']) / normalization['target_normalization']
    assert np.isfinite(field).all()
    np.save(OUTPUT_DIR / 'experimental_field_normalized_280x280.npy', field)

    projections = {}
    for label, path in {
        'experimental': HEAT / 'images/experimental_zsum_cropped_280x280.tif',
        'heat_reconstructed': HEAT / 'images/reconstructed_zsum_cropped_280x280.tif',
        'ordinary_reconstructed': ORDINARY / 'images/generation_0126_reconstructed_zsum_cropped_280x280.tif',
    }.items():
        array = read_tiff(path)
        assert array.shape == (280, 280) and np.isfinite(array).all()
        if label == 'experimental':
            assert np.array_equal(array, raw_volume.sum(axis=0))
            assert np.array_equal(array, read_tiff(ORDINARY / 'images/experimental_zsum_cropped_280x280.tif'))
        save_projection(array, f'{label}_zsum_280x280', spacing[2], scale)
        projections[label] = dict(source=str(path.relative_to(PAPER)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 shape=list(array.shape))

    gen0_source = HEAT / 'generation_0000_best_loss.xsf'
    cell, symbols, positions = crop_structure(gen0_source, crop)
    gen0_path = OUTPUT_DIR / 'generation_0000_best_loss_cropped_280x280.xsf'
    write_xsf(gen0_path, cell, symbols, positions)
    # Independently reproduce the already supplied final crop to verify geometry.
    final_source = HEAT / f"generation_{manifest['best_generation']:04d}_best_loss.xsf"
    expected_cell, expected_symbols, expected_positions = crop_structure(final_source, crop)
    final_path = HEAT / 'best_cropped_280x280.xsf'
    final_cell, final_symbols, final_positions = read_xsf(final_path)
    assert np.allclose(expected_cell, final_cell, atol=1e-8)
    assert np.array_equal(expected_symbols, final_symbols)
    assert np.allclose(expected_positions, final_positions, atol=1e-8)
    assert len(final_symbols) == crop['retained_atoms']

    config = dict(
        paper_dir=str(PAPER), output_dir=str(OUTPUT_DIR),
        sources=projections, experimental_tiff=str(TIFF.relative_to(PAPER)),
        original_tiff_roi_zyx=roi, voxel_size_zyx_A=spacing,
        lateral_size_A=crop['xy_material_lengths_A'], measured_z_size_A=12.0,
        normalized_isovalue=FIELD_ISOVALUE,
        raw_isovalue=float(FIELD_ISOVALUE * normalization['target_normalization'] + normalization['target_background']),
        normalization=normalization,
        intensity_scale=scale,
        image_orientation='Original TIFF/PNG: x right, y down. Pixel samples at i*spacing.',
        field_policy='Direct measured voxels, one global threshold, no Gaussian fitting, no spatial smoothing, no component filtering, no depth exaggeration.',
        field_boundary_policy='Outside measured support is zero for meshing; surfaces are clipped at the measured sample bounds. Boundary caps are display closures, not inferred experimental depth structure.',
        gen0_xsf=str(gen0_path.relative_to(PAPER)),
        final_xsf=str(final_path.relative_to(PAPER)),
        gen0_atoms=len(symbols), gen0_species=dict(Counter(symbols.tolist())),
        final_atoms=len(final_symbols), final_species=dict(Counter(final_symbols.tolist())),
        final_generation=manifest['best_generation'], final_member=manifest['best_member_index'],
        heat_display_generation=126, ordinary_display_generation=126,
        crop_geometry=crop,
        camera=dict(horizontal_span_A=48.0, vertical_span_A=18.0,
                    center_experimental_xyz_A=[crop['xy_material_lengths_A'][0]/2,
                                               crop['xy_material_lengths_A'][1]/2, 6.0],
                    direction='+y', image_size=[3200,1200]),
    )
    (OUTPUT_DIR / 'figure_metadata.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Prepared {OUTPUT_DIR}')
    print(f'Generation 0 crop: {len(symbols)} atoms; final crop: {len(final_symbols)} atoms.')
    print('Verified TIFF projection, shared display scale, and final XSF crop.')
    print('Next: Blender --background --factory-startup --python scripts/blender/wse2/render_figure.py')


if __name__ == '__main__':
    main()
