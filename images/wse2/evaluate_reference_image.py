"""Score the relaxed ideal-bilayer reference with the original image model.

Uses saved coordinates without registration, rotation, gain fitting, or further
relaxation. Validates the renderer against the logged HEAT image penalty.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import prepare_figure_280 as helpers
from render_generation128 import load_module

BASE = Path(__file__).resolve().parent
PAPER = BASE.parents[1]
REFERENCE = PAPER / 'scripts/misc/baseline_ideal_twisted_bilayer_3deg_320/relaxed_free_edges'


def main():
    model_path = PAPER / 'scripts/eat_genetic_scripts/genetic_09.30.26_wse2/gaussian_image_model.py'
    renderer = load_module('reference_gaussian_model', model_path)
    fit_path = model_path.with_name('element_double_gaussian_fits.csv')
    source_dir = PAPER / 'xsfs/wse2/xsfs_wse2_EAT'
    manifest = json.loads((source_dir / 'export_manifest.json').read_text())
    norm = manifest['normalization']
    assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == norm['element_fit_sha256']
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
    full = manifest['cropped_structures']['cropped_320x320']
    voxel = np.array(norm['voxel_size_zyx_A'])
    padding = np.array(full['vacuum_each_side_xyz_A'])
    shape = tuple(np.rint(np.array(full['cell_lengths_A'])[::-1] / voxel).astype(int))
    torch.set_num_threads(4)
    model = renderer.GaussianImageModel(norm['sigma_z_A'], norm['additional_lateral_blur_A'],
        fits, ['W', 'Se'], shape, voxel, padding, torch.device('cpu'), torch.float32)
    assert np.isclose(model.gain, norm['kernel_peak_scale'], rtol=1e-5)
    start = np.rint(padding[::-1] / voxel).astype(int)
    roi = tuple(slice(int(a), int(a+b)) for a, b in zip(start, [12, 320, 320]))
    experimental_path = PAPER / 'ptycho_data/objp_zstack_iter0200.tif'
    experimental = helpers.read_tiff(experimental_path)[:12, 129:449, 146:466]
    target = np.maximum(experimental - norm['target_background'], 0) / norm['target_normalization']
    parameters = json.loads((PAPER / 'parameters/eat/wse2/run_parameters.json').read_text())['resolved_parameters']

    def evaluate(path):
        cell, symbols, positions = helpers.read_xsf(path)
        np.testing.assert_allclose(cell, np.diag(full['cell_lengths_A']), atol=1e-8)
        occupancy = np.stack([symbols == 'W', symbols == 'Se'], axis=1).astype(np.float32)
        assert np.all(occupancy.sum(axis=1) == 1)
        with torch.no_grad():
            volume, _ = model.render(torch.tensor(occupancy), torch.tensor(positions, dtype=torch.float32))
        predicted = volume.numpy()[roi]
        residual = np.abs(predicted - target)
        raw_loss = float(np.where(residual <= .2, .5*residual**2, .2*(residual-.1)).mean())
        projection = predicted.sum(axis=0)[20:-20, 20:-20] * norm['target_normalization']
        pos = torch.tensor(positions, dtype=torch.float32)
        pos = torch.remainder(pos, torch.tensor(np.diag(cell).copy(), dtype=torch.float32))
        lower = padding.copy()
        upper = padding + np.array([320*voxel[2], 320*voxel[1], 12*voxel[0]])
        lower[2] += parameters['sample_box_z_min_angstrom']
        upper[2] = padding[2] + parameters['sample_box_z_max_angstrom']
        margin = parameters['sample_box_confinement_margin_angstrom']
        k = parameters['sample_box_confinement_sharpness']
        lo = torch.nn.functional.softplus(k*(torch.tensor(lower+margin, dtype=torch.float32)-pos))/k
        hi = torch.nn.functional.softplus(k*(pos-torch.tensor(upper-margin, dtype=torch.float32)))/k
        box = float((lo**2+hi**2).sum(dim=1).mean()) * parameters['sample_box_confinement_penalty']
        fractions = occupancy.astype(np.float64).mean(axis=0)
        composition = float(np.sum((fractions-np.array([1/3, 2/3]))**2)) * parameters['composition_penalty']
        return dict(atom_count=len(symbols), raw_huber_image_loss=raw_loss,
            weighted_image_penalty=10*raw_loss,
            weighted_composition_penalty=composition, weighted_sample_box_penalty=box,
            volume_pearson_correlation=float(np.corrcoef(predicted.ravel(), target.ravel())[0, 1]),
            cropped_projection_pearson_correlation=float(np.corrcoef(
                projection.ravel(), experimental.sum(axis=0)[20:-20,20:-20].ravel())[0, 1])), predicted

    check, _ = evaluate(source_dir / 'generation_0128_best_loss.xsf')
    with (BASE / 'eat/member_metrics.csv').open() as handle:
        best = min((r for r in csv.DictReader(handle) if int(r['generation']) <= 128
                    and r['valid'].lower() == 'true'), key=lambda r: float(r['score']))
    np.testing.assert_allclose(check['weighted_image_penalty'], float(best['image_term']), rtol=1e-4)
    np.testing.assert_allclose(check['weighted_sample_box_penalty'], float(best['sample_box_term']), rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(check['weighted_composition_penalty'], float(best['composition_penalty']), rtol=1e-4, atol=1e-7)
    source = REFERENCE / 'final_relaxed.xsf'
    result, predicted = evaluate(source)
    result['formation_energy_per_atom_eV'] = json.loads((REFERENCE / 'relaxation_results.json').read_text())['final']['formation_energy_per_atom_eV']
    result['total_loss'] = sum(result[k] for k in ['formation_energy_per_atom_eV',
        'weighted_image_penalty', 'weighted_composition_penalty', 'weighted_sample_box_penalty'])
    result.update(source_xsf=str(source.relative_to(PAPER)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        experimental_tiff_sha256=hashlib.sha256(experimental_path.read_bytes()).hexdigest(),
        normalization=norm, huber_delta=.2, image_weight=10,
        comparison_shape_zyx=[12,320,320], registration='none; saved coordinates used directly',
        validation_heat_image_term=check['weighted_image_penalty'],
        validation_heat_logged_image_term=float(best['image_term']))
    (REFERENCE / 'image_penalty_results.json').write_text(json.dumps(result, indent=2)+'\n')
    frames = [Image.fromarray(layer.astype(np.float32)) for layer in predicted]
    frames[0].save(REFERENCE / 'predicted_normalized_phase_320x320.tif', save_all=True, append_images=frames[1:])
    print(json.dumps({k: result[k] for k in ['atom_count', 'raw_huber_image_loss',
        'weighted_image_penalty', 'volume_pearson_correlation', 'cropped_projection_pearson_correlation',
        'validation_heat_image_term', 'validation_heat_logged_image_term',
        'weighted_composition_penalty', 'weighted_sample_box_penalty', 'total_loss']}, indent=2))


if __name__ == '__main__':
    main()
