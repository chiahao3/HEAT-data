"""Regenerate the conventional-GA generation-446 projection with the run renderer.

Uses the original image normalization, full structure, and central 280-pixel crop.
Updates only the individual projection assets and their metadata.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import runpy
import numpy as np
from PIL import Image
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize

BASE = Path(__file__).resolve().parent
PAPER = BASE.parents[1]
ORDINARY = PAPER / 'xsfs/wse2/xsfs_wse2_ordinary'
OUT = BASE / 'figure_280x280'
SOURCE = ORDINARY / 'g0446_m0000_elite.xsf'
TARGET = ORDINARY / 'images/generation_0446_reconstructed_zsum_cropped_280x280.tif'


def main():
    helpers = runpy.run_path(str(BASE / 'prepare_figure_280.py'))
    helpers['save_projection'].__globals__.update(AXIS_LABEL_SIZE=21, TICK_LABEL_SIZE=17)
    manifest = json.loads((PAPER / 'xsfs/wse2/xsfs_wse2_EAT/export_manifest.json').read_text())
    norm = manifest['normalization']
    module_path = PAPER / 'scripts/eat_genetic_scripts/genetic_09.30.26_wse2/gaussian_image_model.py'
    spec = importlib.util.spec_from_file_location('wse2_image_model', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fits_path = module_path.with_name('element_double_gaussian_fits.csv')
    assert hashlib.sha256(fits_path.read_bytes()).hexdigest() == norm['element_fit_sha256']
    fits = {}
    with fits_path.open() as f:
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
    full = manifest['cropped_structures']['cropped_320x320']
    voxel = np.array(norm['voxel_size_zyx_A'])
    padding = np.array(full['vacuum_each_side_xyz_A'])
    shape = tuple(np.rint(np.array(full['cell_lengths_A'])[::-1]/voxel).astype(int))
    model = module.GaussianImageModel(norm['sigma_z_A'], norm['additional_lateral_blur_A'],
                                     fits, ['W', 'Se'], shape, voxel, padding,
                                     torch.device('cpu'), torch.float32)
    assert np.isclose(model.gain, norm['kernel_peak_scale'], rtol=1e-5)
    start = np.rint(padding[::-1]/voxel).astype(int)
    roi = tuple(slice(int(a), int(a+b)) for a,b in zip(start, [12,320,320]))

    def render(path):
        cell, symbols, positions = helpers['read_xsf'](path)
        assert np.allclose(np.diag(cell), full['cell_lengths_A'])
        occ = np.stack([symbols=='W',symbols=='Se'], axis=1).astype(np.float32)
        with torch.no_grad():
            predicted, _ = model.render(torch.tensor(occ), torch.tensor(positions,dtype=torch.float32))
        return predicted.numpy()[roi], symbols, positions

    # Verify the shared renderer against the retained HEAT projection.
    heat_dir = PAPER / 'xsfs/wse2/xsfs_wse2_EAT'
    reference, _, _ = render(heat_dir / 'generation_0128_best_loss.xsf')
    reference_sum = reference.sum(axis=0)[20:-20,20:-20] * norm['target_normalization']
    saved = helpers['read_tiff'](heat_dir/'images/reconstructed_zsum_cropped_280x280.tif')
    np.testing.assert_allclose(reference_sum, saved, rtol=2e-4, atol=2e-6)

    predicted, symbols, positions = render(SOURCE)
    raw = predicted.sum(axis=0)[20:-20,20:-20] * norm['target_normalization']
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raw.astype(np.float32)).save(TARGET)
    scale = manifest['display_scale_input_tiff_units']
    rgba = colormaps['gray'](Normalize(*scale)(raw), bytes=True)
    Image.fromarray(rgba).save(TARGET.with_suffix('.png'))
    helpers['save_projection'](raw, 'ordinary_reconstructed_zsum_280x280', voxel[2], scale)

    exp = helpers['read_tiff'](PAPER/'ptycho_data/objp_zstack_iter0200.tif')
    exp = exp[:12,129:449,146:466]
    exp_norm = np.maximum(exp-norm['target_background'],0)/norm['target_normalization']
    residual = np.abs(predicted-exp_norm)
    huber = np.where(residual<=.2,.5*residual**2,.2*(residual-.1))
    with (BASE/'ordinary/member_metrics.csv').open() as f:
        row = next(r for r in csv.DictReader(f) if r['generation']=='446' and r['member_index']=='0')
    assert len(symbols)==int(row['atom_count'])
    assert np.isclose(float(huber.mean()*10),float(row['image_term']),rtol=1e-4)
    cropped_huber = float(huber[:,20:-20,20:-20].mean())
    exp_sum = exp.sum(axis=0)[20:-20,20:-20]
    crop = manifest['cropped_structures']['cropped_280x280']
    _, crop_species, _ = helpers['crop_structure'](SOURCE,crop)
    metadata = dict(generation=446, member_index=0, source_xsf=str(SOURCE.relative_to(PAPER)),
                    source_xsf_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    renderer=str(module_path.relative_to(PAPER)),
                    original_export_reproduction_max_abs_error=float(np.abs(reference_sum-saved).max()),
                    atom_count=len(symbols), cropped_atom_count=len(crop_species),
                    score=float(row['score']), formation_energy_per_atom=float(row['formation_energy_per_atom']),
                    saved_weighted_image_penalty=float(row['image_term']),
                    rerendered_weighted_image_penalty=float(huber.mean()*10),
                    cropped_image_huber_mean=cropped_huber,
                    cropped_zsum_correlation=float(np.corrcoef(raw.ravel(),exp_sum.ravel())[0,1]),
                    output_tiff=str(TARGET.relative_to(PAPER)), display_scale=scale)
    config_path = OUT/'figure_metadata.json'
    config = json.loads(config_path.read_text())
    config['sources']['ordinary_reconstructed'] = dict(source=str(TARGET.relative_to(PAPER)),
        sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),shape=[280,280],generation=446,member_index=0)
    config['ordinary_display_generation']=446
    config_path.write_text(json.dumps(config,indent=2)+'\n')

    (OUT/'ordinary_generation446_render_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
