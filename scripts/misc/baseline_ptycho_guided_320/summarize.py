#!/usr/bin/env python3
"""Create a portable result report and directly comparable microscopy figures."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
import tifffile

BASE=Path(__file__).resolve().parent
initial=json.loads((BASE/'initial_objective.json').read_text())
result=json.loads((BASE/'final_result.json').read_text())
alignment=json.loads((BASE/'alignment_summary.json').read_text())
summary=json.loads((BASE/'run_summary.json').read_text())
comparison=json.loads((BASE/'objective_comparison.json').read_text())
final=read(BASE/'final_guided_relaxed.extxyz')
start=read(BASE/'aligned_ideal_initial.extxyz')
assert np.array_equal(start.numbers,final.numbers)
assert np.allclose(start.cell.array,final.cell.array)
assert result['score']<=initial['score']
assert abs(sum(result[k] for k in ('formation_energy_per_atom','image_term','composition_penalty','sample_box_term'))-result['score'])<1e-10
assert json.loads((BASE/'source_fingerprint_verification.json').read_text())['matches']
assert all(json.loads((BASE/'hyperparameter_verification.json').read_text()).values())

# Fit local W-neighbor orientations, separately from the affine registration parameters.
orientations=[]
for layer in (0,1):
    atoms=final[(final.arrays['layer']==layer)&(final.numbers==74)]
    i,j,d,D=neighbor_list('ijdD',atoms,3.6)
    D=D[d>2.8]
    theta=np.arctan2(D[:,1],D[:,0])
    orientation=float(np.rad2deg(np.angle(np.mean(np.exp(6j*theta)))/6))
    orientations.append(orientation)
save_geometry=dict(final_W_sixfold_orientation_modulo_60_degrees=orientations,
                   final_W_relative_twist_modulo_60_degrees=(orientations[1]-orientations[0]+30)%60-30,
                   same_cell=True,same_composition=True,atoms=len(final))
(BASE/'final_geometry_checks.json').write_text(json.dumps(save_geometry,indent=2)+'\n')

with (BASE/'objective_comparison.csv').open('w') as handle:
    columns=['structure','score','formation_energy_per_atom','image_term','composition_penalty','sample_box_term','image_correlation','atom_count']
    writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
    for name,row in comparison.items():
        writer.writerow(dict(structure=name,**{k:row[k] for k in columns[1:]}))

# Render the aligned starting flake with precisely the copied renderer, no model loading.
sys.path.insert(0,str(BASE/'source'))
import importlib.util
spec=importlib.util.spec_from_file_location('ordinary_ga',BASE/'source/discrete_09.30.26_wse2.py')
ordinary=importlib.util.module_from_spec(spec);spec.loader.exec_module(ordinary)
import torch
torch.set_num_threads(4)
config=json.loads((BASE/'effective_config.json').read_text())
# Resolve only paths for portability after downloading the entire folder.
config['paths']['output_dir']=str(BASE/'postprocess_diagnostics')
config['paths']['element_fit_csv']=str(BASE/'inputs/element_double_gaussian_fits.csv')
config['experimental']['tiff_file']=str(BASE/'inputs/experimental.tif')
problem=ordinary.DiscreteProblem(config,device='cpu',diagnostics=False)
ids=torch.tensor([ordinary.SPECIES.index(s) for s in start.get_chemical_symbols()])
labels=torch.nn.functional.one_hot(ids,2).to(torch.float32)
roi=tuple(slice(int(a),int(b)) for a,b in zip(problem.experimental['sample_roi_start_zyx'],problem.experimental['sample_roi_stop_zyx']))
with torch.no_grad():
    image,_=problem.gaussian_renderer.render(labels,torch.tensor(start.positions,dtype=torch.float32))
    image=image.cpu().numpy()[roi]
tifffile.imwrite(BASE/'aligned_initial_slices.tif',image,metadata={'axes':'ZYX'})
tifffile.imwrite(BASE/'aligned_initial_zsum_320x320.tif',image.sum(0))
experimental=tifffile.imread(BASE/'experimental_slices.tif').sum(0)
predicted=tifffile.imread(BASE/'best_predicted_slices.tif').sum(0)
vmax=float(np.quantile(experimental,.995))
fig,axes=plt.subplots(1,3,figsize=(13,4.4),constrained_layout=True)
for ax,data,title in zip(axes,[experimental,image.sum(0),predicted],
                         ['Experimental, 320 × 320','Aligned ideal bilayer','After guided PBFGS']):
    ax.imshow(data,origin='lower',vmin=0,vmax=vmax,cmap='gray')
    ax.set_title(title);ax.set_axis_off()
fig.savefig(BASE/'ptycho_comparison_320x320.png',dpi=200)
fig.savefig(BASE/'ptycho_comparison_320x320.pdf')
plt.close(fig)

dependencies={}
for name in ('numpy','torch','ase','fairchem-core','scipy','PyYAML','tifffile','matplotlib'):
    try:dependencies[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:dependencies[name]='not recorded'
(BASE/'environment_versions.json').write_text(json.dumps(dependencies,indent=2)+'\n')
readme=f'''PTYCHOGRAPHY-GUIDED TWISTED WSe2 REFERENCE -- ORDINARY GA LOSS

FINAL RESULT
Loss:                         {result['score']:.12f}
Formation energy per atom:    {result['formation_energy_per_atom']:.12f} eV/atom
Weighted image term:          {result['image_term']:.12f}
Composition penalty:          {result['composition_penalty']:.12f}
Sample-box confinement term:  {result['sample_box_term']:.12f}
Voxel image correlation:      {result['image_correlation']:.9f}
Atoms: {len(final)} ({result['species_counts']['W']} W, {result['species_counts']['Se']} Se)
PBFGS: {result['relaxation']['iterations']} iterations, termination {result['relaxation']['termination_reason']},
gradient norm {result['relaxation']['gradient_norm']:.9f}.
GPU: {summary['host']}, {summary['gpu']}, job {summary['slurm_job_id']}.

THE EXACT SAME OBJECTIVE
Loss = (E_UMA - N_W*mu_W - N_Se*mu_Se)/N
     + 10 * mean_voxel_Huber(prediction, target; delta=0.2)
     + 10 * sum_species((N_species/N - target_fraction_species)^2)
     + 50 * mean_atom(sum_xyz(softplus boundary violation^2)).
Composition target W:Se = 1:2; boundary sharpness = 8 A^-1; boundary margin = 0.
Image renderer: the exact production double-XY-Gaussian + Gaussian-depth model.
Depth sigma = 4 A; extra lateral blur = 0.08 A; target transform linear;
background percentile 1; normalization percentile 99.5. No fitted intensity gain,
contrast renormalization or post-hoc image registration was introduced.
The 12 native slices and all 320 x 320 measured XY pixels enter the image loss.
Input elemental reference energies are copied from the ordinary GA run.

All atomic positions are free; species and atom count are fixed during PBFGS,
exactly as for a single ordinary-GA member. No GA mutations, insertions,
deletions, occupancy optimization or species transmutation were performed.
Cell is fixed at diag(64.687136,64.687136,28) A; xyz periodic. The measured region
is 48.82048 A square with padding (7.933328,7.933328,8) A per side.

PRODUCTION CODE AND HYPERPARAMETER VERIFICATION
source/ contains the ordinary-GA driver, image/geometry code and minimizer.
Its original source+input fingerprint matches the original run exactly:
{json.loads((BASE/'source_fingerprint_verification.json').read_text())['recorded_fingerprint']}
The bundled default YAML has its embedded API credential removed; the saved
run config and copied references supply all actual settings without credentials.
Final optimization calls ordinary.relax_candidate unchanged, with:
  max_iterations=500; gradient_tolerance=0.003; bfgs_history=8;
  max_line_search=10; cost_plateau_tolerance=0.0003;
  cost_plateau_patience=10; cost_plateau_min_iterations=50.
Only file/output paths and runtime GPU selection change from the original config.
The old ordinary winner was independently re-evaluated: loss differs from its
recorded value by {json.loads((BASE/'ordinary_score_verification.json').read_text())['difference']:.3e}.

EXPERIMENTAL ALIGNMENT BEFORE PBFGS
Started from the Materials Project bulk-derived Se-W-Se monolayer basis, not
from a HEAT or ordinary-GA optimized structure. The earlier energy-only relaxed
flake is included and scored for comparison, but is not used to initialize.
Fourier peaks in the original experimental TIFF seed the absolute orientation.
Eight alternatives cover both twist signs and the two 60-degree sublattice
orientations of each monolayer. Per-layer XY translations, z heights, small
angle corrections and bounded symmetric in-plane affine lattice adjustments
are fitted directly to the experimental image with its exact renderer/Huber loss.
Diagonal lattice factors are bounded to [0.97,1.03]; shear to [-0.02,0.02].
The angle seed is +/-1.5 degrees about the Fourier orientation; each layer may
adjust by +/-4 degrees to register to the data. Each rigid/affine fit is limited
to 140 L-BFGS-B iterations; several hit this limit. This preliminary fit is
initialization only and has separate numerical settings from final PBFGS.
The image initialization includes a 4 A halo to avoid missing edge intensity;
atoms are then cropped into the exact original measured XY region before the
full production loss and PBFGS. The best cropped image fit is selected.
The registered lattice therefore need not retain the exact bulk metric.

Selected alignment trial: {alignment['branch']}.
Registration parameter angles: {alignment['angles_degrees']} degrees.
Registration parameter twist modulo 60: {alignment['lattice_twist_modulo_60_degrees']:.6f} degrees.
Final W-neighbor relative orientation modulo 60: {save_geometry['final_W_relative_twist_modulo_60_degrees']:.6f} degrees.
The neighbor-based angle is affected by strain; it is a global estimate.
Aligned-start image correlation: {initial['image_correlation']:.9f}.
Final image correlation: {result['image_correlation']:.9f}.
Aligned-start total loss: {initial['score']:.12f}.

INTERPRETATION
This is a reference obtained by fitting the experimental data under the exact
ordinary-GA objective. It is not independent experimental ground truth or a
proven global minimum. Its composition is selected by the geometrical crop;
fixed-position PBFGS cannot repair vacancies or alter species/counts.
The final total loss, not energy alone, is the number to compare with GA loss.
See objective_comparison.csv for a decomposition including the previous
energy-only relaxation and the ordinary winner under this same objective.
attempt_rigid_only/ retains the less successful first rigid registration and
its result; final_* at this folder's top level are the improved main result.

MAIN FILES
final_guided_relaxed.xsf / .extxyz / .cif: final reference structure.
final_result.json: full score decomposition, coordinates, and optimizer metrics.
aligned_ideal_initial.*: registered starting bilayer before final PBFGS.
ptycho_comparison_320x320.png / .pdf: experimental, initial and final z-summed images
using the same display scale, without registration/interpolation.
*_slices.tif: exact measured-region 12 x 320 x 320 predictions/target/residual.
*_zsum_320x320.tif: corresponding z-summed arrays in normalized model units.
objective_evaluations.csv: all final PBFGS objective evaluations.
alignment_trial_*.json: all eight preliminary fits, including iteration-limit flags.
inputs/: original TIFF, bulk basis, fits, references and source run parameters.
source/: snapshotted production code. No API key is needed to use the saved inputs.

REPRODUCING AFTER DOWNLOAD
The folder is self-contained for structure, microscopy inputs and project code.
UMA model weights and scientific Python dependencies are external requirements,
not included in the archive. See environment_versions.json.
With a CUDA-compatible fairchem/ASE/PyTorch environment and cached uma-s-1p1:
  python run_baseline.py
  python summarize.py
For a fresh optimization, remove pbfgs_progress.json first: production PBFGS
otherwise resumes from that checkpoint. Submit.sh documents the cluster command;
adapt its local Python path and Slurm node/partition for another machine.
All input paths are resolved relative to this folder at runtime.
'''
(BASE/'README.txt').write_text(readme)
manifest={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in BASE.rglob('*') if p.is_file() and p.name!='checksums_sha256.json' and '__pycache__' not in str(p)}
(BASE/'checksums_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Final loss',result['score'],'image correlation',result['image_correlation'])
print('W-neighbor twist',save_geometry['final_W_relative_twist_modulo_60_degrees'])
print('Wrote report, comparison CSV, images, environment and checksums.')
