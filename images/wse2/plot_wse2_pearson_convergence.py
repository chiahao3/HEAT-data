"""Pearson correlation of the best-total-loss candidate's projected image.

Render full structures with the original Gaussian model, sum the 12 measured
slices, then compare the central 280x280 pixels with the experimental image.
No image registration or independent contrast fitting is used. Selection is by
total loss, so correlation need not increase monotonically. Also refresh the
existing image-penalty plot with the Conventional legend.
"""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json

import numpy as np
import torch

import plot_wse2_convergence as plotting
import prepare_figure_280 as helpers
from render_generation128 import load_module

BASE = Path(__file__).resolve().parent
PAPER = BASE.parents[1]


def main():
    selected = {
        'HEAT': plotting.best_candidates_so_far(plotting.load_results(plotting.HEAT_CSV)),
        'Conventional': plotting.best_candidates_so_far(plotting.load_results(plotting.ORDINARY_CSV)),
    }
    horizon = min(items[-1].generation for items in selected.values())
    selected = {name: [c for c in items if c.generation <= horizon]
                for name, items in selected.items()}
    model_path = PAPER / 'scripts/eat_genetic_scripts/genetic_09.30.26_wse2/gaussian_image_model.py'
    renderer = load_module('pearson_gaussian_model', model_path)
    fit_path = model_path.with_name('element_double_gaussian_fits.csv')
    fits = {}
    with fit_path.open() as handle:
        for row in csv.DictReader(handle):
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
    target_projection = experimental.sum(axis=0)[20:-20, 20:-20]
    curves, records = {}, []
    for name, directory in [('HEAT', 'xsfs_wse2_EAT'), ('Conventional', 'xsfs_wse2_ordinary')]:
        source_dir = PAPER / 'xsfs/wse2' / directory
        manifest = json.loads((source_dir / 'export_manifest.json').read_text())
        norm = manifest['normalization']
        assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == norm['element_fit_sha256']
        full = manifest['cropped_structures']['cropped_320x320']
        voxel = np.array(norm['voxel_size_zyx_A'])
        padding = np.array(full['vacuum_each_side_xyz_A'])
        shape = tuple(np.rint(np.array(full['cell_lengths_A'])[::-1] / voxel).astype(int))
        model = renderer.GaussianImageModel(
            norm['sigma_z_A'], norm['additional_lateral_blur_A'], fits,
            ['W', 'Se'], shape, voxel, padding, torch.device('cpu'), torch.float32)
        assert np.isclose(model.gain, norm['kernel_peak_scale'], rtol=1e-5)
        start = np.rint(padding[::-1] / voxel).astype(int)
        roi = tuple(slice(int(a), int(a + b)) for a, b in zip(start, [12, 320, 320]))
        target = np.maximum(experimental - norm['target_background'], 0) / norm['target_normalization']
        previous_score, source, correlation, image_term = None, None, None, None
        curves[name] = []
        for candidate in selected[name]:
            if previous_score is None or candidate.score < previous_score:
                source = source_dir / f'generation_{candidate.generation:04d}_best_loss.xsf'
                cell, symbols, positions = helpers.read_xsf(source)
                assert np.allclose(np.diag(cell), full['cell_lengths_A'])
                occupancy = np.stack([symbols == 'W', symbols == 'Se'], axis=1).astype(np.float32)
                with torch.no_grad():
                    volume, _ = model.render(torch.tensor(occupancy), torch.tensor(positions, dtype=torch.float32))
                predicted = volume.numpy()[roi]
                residual = np.abs(predicted - target)
                image_term = float(np.where(residual <= .2, .5 * residual**2, .2 * (residual - .1)).mean() * 10)
                if not np.isclose(image_term, candidate.image_penalty, rtol=2e-4, atol=1e-7):
                    raise ValueError(f'{name} generation {candidate.generation}: rendered image penalty does not match logged candidate')
                projection = predicted.sum(axis=0)[20:-20, 20:-20] * norm['target_normalization']
                correlation = float(np.corrcoef(projection.ravel(), target_projection.ravel())[0, 1])
                assert np.isfinite(correlation) and -1 <= correlation <= 1
                previous_score = candidate.score
                print(f'{name} generation {candidate.generation}: Pearson r={correlation:.6f}', flush=True)
            curves[name].append(SimpleNamespace(generation=candidate.generation, correlation=correlation))
            records.append(dict(method=name, generation=candidate.generation,
                best_total_loss=candidate.score, pearson_correlation=correlation,
                weighted_image_penalty=image_term, source_xsf=str(source.relative_to(PAPER))))
    with (BASE / 'wse2_pearson_correlation_convergence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plotting.plot_metric(curves['HEAT'], curves['Conventional'], 'correlation',
                         'Pearson correlation', (0, 1), BASE / 'wse2_pearson_correlation_convergence.png')
    plotting.plot_metric(selected['HEAT'], selected['Conventional'], 'image_penalty',
                         'Ptycho image penalty', (plotting.IMAGE_Y_MIN, plotting.IMAGE_Y_MAX),
                         BASE / 'wse2_image_penalty_convergence.png')


if __name__ == '__main__':
    main()
