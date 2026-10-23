"""Export generation-128 WSe2 panels for the manuscript SVG.

Uses the original run's image model, normalization, and full structures before
cropping. Run with Python, NumPy, Torch, Pillow, and Matplotlib installed.
Outputs are isolated from the earlier generation-126 figure assets.
"""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize


BASE = Path(__file__).resolve().parent
PAPER = BASE.parents[1]
ASSETS = BASE / 'figure_280x280'
OUT = ASSETS / 'generation_0128'
GENERATION = 128
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 14 * AXIS_LABEL_SIZE / 18


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    helpers = load_module('wse2_figure_helpers', BASE / 'prepare_figure_280.py')
    helpers.OUTPUT_DIR = OUT
    helpers.AXIS_LABEL_SIZE = AXIS_LABEL_SIZE
    helpers.TICK_LABEL_SIZE = TICK_LABEL_SIZE
    model_path = PAPER / 'scripts/eat_genetic_scripts/genetic_09.30.26_wse2/gaussian_image_model.py'
    renderer = load_module('wse2_generation128_image_model', model_path)
    fit_path = model_path.with_name('element_double_gaussian_fits.csv')
    fits = {}
    with fit_path.open() as f:
        for row in csv.DictReader(f):
            if row['species'] in ('W', 'Se'):
                fits[row['species']] = {
                    'core_sigma_xy_A': float(row['core_sigma_xy_angstrom']),
                    'tail_sigma_xy_A': float(row['tail_sigma_xy_angstrom']),
                    'core_integrated_weight': float(row['core_integrated_weight_VAngstrom3']),
                    'tail_integrated_weight': float(row['tail_integrated_weight_VAngstrom3']),
                    'total_integrated_weight': float(row['total_integrated_weight_VAngstrom3']),
                }
    torch.set_num_threads(4)
    experimental = helpers.read_tiff(PAPER / 'ptycho_data/objp_zstack_iter0200.tif')[:12, 129:449, 146:466]
    metadata = {'display_generation': GENERATION, 'renderer': str(model_path.relative_to(PAPER)), 'methods': {}}
    shared_norm = shared_scale = None

    for name, directory, metrics in [
        ('heat', 'xsfs_wse2_EAT', 'eat'),
        ('ordinary', 'xsfs_wse2_ordinary', 'ordinary'),
    ]:
        source_dir = PAPER / 'xsfs/wse2' / directory
        source = source_dir / f'generation_{GENERATION:04d}_best_loss.xsf'
        manifest = json.loads((source_dir / 'export_manifest.json').read_text())
        norm = manifest['normalization']
        scale = manifest['display_scale_input_tiff_units']
        if shared_norm is None:
            shared_norm, shared_scale = norm, scale
        assert norm == shared_norm and scale == shared_scale
        assert sha256(fit_path) == norm['element_fit_sha256']
        full = manifest['cropped_structures']['cropped_320x320']
        crop = manifest['cropped_structures']['cropped_280x280']
        voxel = np.array(norm['voxel_size_zyx_A'])
        padding = np.array(full['vacuum_each_side_xyz_A'])
        shape = tuple(np.rint(np.array(full['cell_lengths_A'])[::-1] / voxel).astype(int))
        model = renderer.GaussianImageModel(
            norm['sigma_z_A'], norm['additional_lateral_blur_A'], fits,
            ['W', 'Se'], shape, voxel, padding, torch.device('cpu'), torch.float32)
        assert np.isclose(model.gain, norm['kernel_peak_scale'], rtol=1e-5)
        start = np.rint(padding[::-1] / voxel).astype(int)
        roi = tuple(slice(int(a), int(a + b)) for a, b in zip(start, [12, 320, 320]))
        cell, symbols, positions = helpers.read_xsf(source)
        assert np.allclose(np.diag(cell), full['cell_lengths_A'])
        with (BASE / metrics / 'member_metrics.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if int(r['generation']) <= GENERATION and r['score']]
        best = min(rows, key=lambda r: float(r['score']))
        best_source = source_dir / f"generation_{int(best['generation']):04d}_best_loss.xsf"
        for actual, expected in zip((cell, symbols, positions), helpers.read_xsf(best_source)):
            np.testing.assert_array_equal(actual, expected)
        assert len(symbols) == int(best['atom_count'])
        print(f'Rendering {name}: best at generation {GENERATION}, first found at {best["generation"]}, member {best["member_index"]}.', flush=True)
        occupancy = np.stack([symbols == 'W', symbols == 'Se'], axis=1).astype(np.float32)
        with torch.no_grad():
            volume, _ = model.render(torch.tensor(occupancy), torch.tensor(positions, dtype=torch.float32))
        predicted = volume.numpy()[roi]
        target = np.maximum(experimental - norm['target_background'], 0) / norm['target_normalization']
        residual = np.abs(predicted - target)
        huber = np.where(residual <= .2, .5 * residual**2, .2 * (residual - .1))
        image_term = float(huber.mean() * 10)
        assert np.isclose(image_term, float(best['image_term']), rtol=1e-4)
        raw = predicted.sum(axis=0)[20:-20, 20:-20] * norm['target_normalization']
        assert raw.shape == (280, 280) and np.isfinite(raw).all()
        stem = f'{name}_generation_0128_reconstructed_zsum_280x280'
        Image.fromarray(raw.astype(np.float32)).save(OUT / f'{stem}.tif')
        Image.fromarray(colormaps['gray'](Normalize(*scale)(raw), bytes=True)).save(OUT / f'{stem}.png')
        helpers.save_projection(raw, stem, voxel[2], scale)
        crop_cell, crop_symbols, crop_positions = helpers.crop_structure(source, crop)
        helpers.write_xsf(OUT / f'{name}_generation_0128_cropped_280x280.xsf', crop_cell, crop_symbols, crop_positions)
        record = dict(
            source_xsf=str(source.relative_to(PAPER)), source_xsf_sha256=sha256(source),
            first_found_generation=int(best['generation']), member_index=int(best['member_index']),
            score=float(best['score']), atom_count=len(symbols),
            cropped_atom_count=len(crop_symbols), cropped_species=dict(Counter(crop_symbols.tolist())),
            saved_weighted_image_penalty=float(best['image_term']), rerendered_weighted_image_penalty=image_term,
            cropped_zsum_correlation=float(np.corrcoef(raw.ravel(), experimental.sum(axis=0)[20:-20, 20:-20].ravel())[0, 1]),
            display_scale=scale, axis_label_size=helpers.AXIS_LABEL_SIZE, tick_label_size=helpers.TICK_LABEL_SIZE,
        )
        if name == 'heat':
            for actual, expected in zip((crop_cell, crop_symbols, crop_positions), helpers.read_xsf(source_dir / 'best_cropped_280x280.xsf')):
                if actual.dtype.kind in 'US':
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
            saved = helpers.read_tiff(source_dir / 'images/reconstructed_zsum_cropped_280x280.tif')
            np.testing.assert_allclose(raw, saved, rtol=2e-4, atol=2e-6)
            record['existing_projection_max_abs_error'] = float(np.abs(raw - saved).max())
            # The existing Blender image is valid: its cropped atomic input is
            # verified above to be the same structure retained at generation 128.
            for suffix in ['.png', '_axes.png', '_axes.pdf']:
                shutil.copy2(ASSETS / f'heat_final_depth{suffix}', OUT / f'heat_generation_0128_depth{suffix}')
            record['depth_render'] = 'Reused existing Blender render after verifying identical cropped coordinates and species.'
        metadata['methods'][name] = record
        print(f'{name}: weighted image penalty {image_term:.9f}; cropped correlation {record["cropped_zsum_correlation"]:.9f}', flush=True)

    metadata['files_sha256'] = {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.suffix in {'.png', '.pdf', '.svg', '.tif', '.xsf'}}
    (OUT / 'render_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (OUT / 'README.txt').write_text('''Generation-128 WSe2 figure assets

Use these PNGs in the assembled figure:
  heat_generation_0128_reconstructed_zsum_280x280_axes.png
  ordinary_generation_0128_reconstructed_zsum_280x280_axes.png
  heat_generation_0128_depth.png

Projection SVG/PDF variants are also supplied, together with unadorned PNGs
and numerical TIFFs. The depth view also has optional axes PNG/PDF variants.
All projections retain the existing 280x280 crop, common intensity scale,
24 pt axis labels, and 18.67 pt tick labels. They use the full structures before
cropping and the same 12 measured depth slices as experiment.

HEAT's best structure at generation 128 was first found at generation 115,
member 31. Its depth image is reused after checking identical coordinates and
species. Ordinary uses generation 128, member 33. Both weighted image losses
are checked against the run metrics. See render_metadata.json for provenance.

The experimental and generation-0 panels remain valid. Label the final
structure and both reconstruction panels as Generation 128 in the SVG.

Regenerate from the paper directory:
  python3 images/wse2/render_generation128.py
''')
    print(f'Assets saved to {OUT}', flush=True)


if __name__ == '__main__':
    main()
