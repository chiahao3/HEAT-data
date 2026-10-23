"""Render the full trial with the production image model, then crop its projection."""
from pathlib import Path
import csv, json, hashlib, importlib.util
import numpy as np
import torch
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import Normalize

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[2]
SOURCE = OUT / 'rank_001_trial_0009_r0.944340.xsf'
STEM = SOURCE.stem + '_ptycho_280x280'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helpers=load('projection_helpers',PAPER/'images/wse2/prepare_figure_280.py')
helpers.OUTPUT_DIR=OUT
model_path=PAPER/'scripts/eat_genetic_scripts/genetic_09.30.26_wse2/gaussian_image_model.py'
renderer=load('trial_gaussian_model',model_path)
fit_path=model_path.with_name('element_double_gaussian_fits.csv')
manifest=json.loads((PAPER/'xsfs/wse2/xsfs_wse2_EAT/export_manifest.json').read_text())
norm=manifest['normalization'];full=manifest['cropped_structures']['cropped_320x320']
assert hashlib.sha256(fit_path.read_bytes()).hexdigest()==norm['element_fit_sha256']
fits={}
with fit_path.open() as handle:
    for row in csv.DictReader(handle):
        if row['species'] in ('W','Se'):
            fits[row['species']]={key:float(row[value]) for key,value in {
                'core_sigma_xy_A':'core_sigma_xy_angstrom','tail_sigma_xy_A':'tail_sigma_xy_angstrom',
                'core_integrated_weight':'core_integrated_weight_VAngstrom3',
                'tail_integrated_weight':'tail_integrated_weight_VAngstrom3',
                'total_integrated_weight':'total_integrated_weight_VAngstrom3'}.items()}
voxel=np.array(norm['voxel_size_zyx_A']);padding=np.array(full['vacuum_each_side_xyz_A'])
shape=tuple(np.rint(np.array(full['cell_lengths_A'])[::-1]/voxel).astype(int))
torch.set_num_threads(4)
model=renderer.GaussianImageModel(norm['sigma_z_A'],norm['additional_lateral_blur_A'],fits,
    ['W','Se'],shape,voxel,padding,torch.device('cpu'),torch.float32)
assert np.isclose(model.gain,norm['kernel_peak_scale'],rtol=1e-5)
cell,symbols,positions=helpers.read_xsf(SOURCE)
np.testing.assert_allclose(np.diag(cell),full['cell_lengths_A'])
occupancy=np.stack([symbols=='W',symbols=='Se'],axis=1).astype(np.float32)
assert np.all(occupancy.sum(axis=1)==1)
with torch.no_grad():
    volume,_=model.render(torch.tensor(occupancy),torch.tensor(positions,dtype=torch.float32))
start=np.rint(padding[::-1]/voxel).astype(int)
roi=tuple(slice(int(a),int(a+b)) for a,b in zip(start,[12,320,320]))
predicted=volume.numpy()[roi]
raw=predicted.sum(axis=0)[20:-20,20:-20]*norm['target_normalization']
assert raw.shape==(280,280) and np.isfinite(raw).all()
scale=manifest['display_scale_input_tiff_units']
Image.fromarray(raw.astype(np.float32)).save(OUT/(STEM+'.tif'))
Image.fromarray(colormaps['gray'](Normalize(*scale)(raw),bytes=True)).save(OUT/(STEM+'.png'))
helpers.save_projection(raw,STEM,voxel[2],scale)
frames=[Image.fromarray(layer.astype(np.float32)) for layer in predicted[:,20:-20,20:-20]]
frames[0].save(OUT/(STEM+'_normalized_slices.tif'),save_all=True,append_images=frames[1:])
metadata={'source_xsf':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'renderer_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'normalization':norm,
    'display_scale':scale,'policy':'Render all original atoms; extract 12 measured slices; sum depth; crop 20 pixels from each XY edge. No registration or contrast fitting.',
    'full_atom_count':len(symbols),'output_shape':list(raw.shape)}
tiff=PAPER/'ptycho_data/objp_zstack_iter0200.tif'
if tiff.exists():
    exp=helpers.read_tiff(tiff)[:12,129:449,146:466]
    target=np.maximum(exp-norm['target_background'],0)/norm['target_normalization']
    residual=np.abs(predicted-target)
    metadata['weighted_full_image_penalty']=float(np.where(residual<=.2,.5*residual**2,.2*(residual-.1)).mean()*10)
    metadata['cropped_projection_pearson_r']=float(np.corrcoef(raw.ravel(),exp.sum(axis=0)[20:-20,20:-20].ravel())[0,1])
(OUT/(STEM+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved:',OUT/(STEM+'.png'))
print('Pearson r:',metadata.get('cropped_projection_pearson_r'))
