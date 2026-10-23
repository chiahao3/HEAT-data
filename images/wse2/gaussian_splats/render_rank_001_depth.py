"""Crop the trial to the paper's 280-pixel footprint and render its depth view.
Run with Blender --background --factory-startup --python THIS_FILE.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
from collections import Counter
import numpy as np

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[2]
SOURCE = OUT / 'rank_001_trial_0009_r0.944340.xsf'
STEM = SOURCE.stem + '_cropped_280x280'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

manifest = json.loads((PAPER / 'xsfs/wse2/xsfs_wse2_EAT/export_manifest.json').read_text())
crop = manifest['cropped_structures']['cropped_280x280']
lines = SOURCE.read_text().splitlines()
i = lines.index('PRIMVEC')
cell = np.array([[float(x) for x in row.split()] for row in lines[i+1:i+4]])
np.testing.assert_allclose(np.diag(cell), manifest['cropped_structures']['cropped_320x320']['cell_lengths_A'])
i = lines.index('PRIMCOORD')
n = int(lines[i+1].split()[0])
rows = [line.split() for line in lines[i+2:i+2+n]]
symbols = np.array([{'74':'W','34':'Se'}.get(row[0], row[0]) for row in rows])
positions = np.array([[float(v) for v in row[1:4]] for row in rows])
lo, hi = np.array(crop['original_cell_xy_bounds_A'])
keep = np.all((positions[:,:2] >= lo) & (positions[:,:2] < hi),axis=1)
positions = positions[keep] + np.array(crop['coordinate_shift_xyz_A'])
symbols = symbols[keep]
newcell = np.diag(crop['cell_lengths_A'])
cropped = OUT / (STEM + '.xsf')
text = ['# Central 280x280 measured-pixel crop; all atom heights retained.','CRYSTAL','PRIMVEC']
text += [' '.join(f'{v:.14f}' for v in row) for row in newcell]
text += ['PRIMCOORD',f'{len(symbols)} 1']
text += [s+' '+' '.join(f'{v:.14f}' for v in row) for s,row in zip(symbols,positions)]
cropped.write_text('\n'.join(text)+'\n')
renderer = load('paper_renderer', PAPER / 'scripts/blender/wse2/render_figure.py')
config = json.loads((PAPER / 'images/wse2/figure_280x280/figure_metadata.json').read_text())
style = renderer.load_style()
renderer.set_style(style, config, SimpleNamespace(quick=False,no_save=True,no_render=False))
style.INPUT_XSF = cropped
style.SIDE_OUTPUT_RENDER = OUT / (STEM + '_depth.png')
import bpy
bpy.context.preferences.filepaths.save_version = 0
(OUT / (STEM + '_metadata.json')).write_text(json.dumps({
    'source':SOURCE.name,'original_atom_count':n,'cropped_atom_count':len(symbols),
    'species_counts':dict(Counter(symbols.tolist())), 'crop':crop,
    'camera':config['camera'],'render_samples':style.RENDER_SAMPLES,
    'output_image':style.SIDE_OUTPUT_RENDER.name},indent=2)+'\n')
print('Crop:',n,'->',len(symbols),dict(Counter(symbols.tolist())),flush=True)
style.main()
