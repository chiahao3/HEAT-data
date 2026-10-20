#!/usr/bin/env python3
"""Register bulk-derived WSe2 layers to microscopy, then use ordinary-GA PBFGS.

The source/ directory is a snapshot of the actual ordinary-GA implementation.
No surrogate objective or energy-only ASE relaxation is used for the final fit.
"""
from pathlib import Path
import copy
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
import tifffile

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'source'))
spec = importlib.util.spec_from_file_location('ordinary_ga', BASE/'source/discrete_09.30.26_wse2.py')
ordinary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ordinary)


def save(name, value):
    ordinary.atomic_json(BASE/name, value)


def write_atoms(stem, atoms):
    for extension in ('extxyz', 'xsf', 'cif'):
        write(BASE/f'{stem}.{extension}', atoms)


def build_basis():
    doc=json.loads((BASE/'inputs/materials_project_selected_record.json').read_text())
    st=doc['structure']; lattice=np.array(st['lattice']['matrix'])
    symbols=np.array([s['species'][0]['element'] for s in st['sites']])
    xyz=np.array([s['xyz'] for s in st['sites']]); c=lattice[2,2]
    w=np.flatnonzero(symbols=='W'); origin=xyz[w[np.argmin(xyz[w,2])]]
    dz=(xyz[:,2]-origin[2]+c/2)%c-c/2; keep=abs(dz)<c/4
    basis=xyz[keep]-origin; basis[:,2]=dz[keep]
    return lattice,basis,symbols[keep]


def fft_orientation():
    # Use experimental data alone for global orientation; never seed from GA atoms.
    image=tifffile.imread(BASE/'inputs/experimental.tif').sum(axis=0)
    ny,nx=image.shape
    field=(image-image.mean())*np.outer(np.hanning(ny),np.hanning(nx))
    fourier=abs(np.fft.fftshift(np.fft.fft2(field,s=(2048,2048))))
    f=np.fft.fftshift(np.fft.fftfreq(2048,d=.152564))
    yy,xx=np.meshgrid(f,f,indexing='ij'); radius=np.hypot(xx,yy)
    mask=(radius>.32)&(radius<.38)&(yy>0)
    field=np.where(mask,fourier,0)
    ids=np.flatnonzero((field==maximum_filter(field,size=11))&mask)
    ids=ids[np.argsort(field.ravel()[ids])[-3:]]
    angles=np.rad2deg(np.arctan2(yy.ravel()[ids],xx.ravel()[ids]))
    phase=np.angle(np.sum(field.ravel()[ids]*np.exp(6j*np.deg2rad(angles))))/6
    # Triangular real-space a1 direction is 30 degrees from first-shell reciprocal rays.
    phi=float((np.rad2deg(phase)-30)%60)
    save('fft_orientation.json',dict(reciprocal_peak_angles_degrees=angles,
         reciprocal_peak_radii_inverse_A=radius.ravel()[ids],
         real_space_orientation_modulo_60_degrees=phi,
         source='Full original experimental TIFF only; no GA structure used.'))
    return phi


def main():
    start=time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required.')
    torch.set_num_threads(4)
    c=json.loads((BASE/'inputs/ordinary_run_parameters.json').read_text())['config']
    original=copy.deepcopy(c)
    c['paths']['output_dir']=str(BASE/'diagnostics')
    c['paths']['element_fit_csv']=str(BASE/'inputs/element_double_gaussian_fits.csv')
    c['model']['reference_energies_json']=str(BASE/'inputs/reference_energies.json')
    c['experimental']['tiff_file']=str(BASE/'inputs/experimental.tif')
    c['runtime']['device']='cuda'; c['runtime']['gpu_ids']='0'
    ordinary.validate_config(c)
    physics=('system','experimental','sample_box','imaging','composition','relaxation')
    equality={k:c[k]==original[k] for k in physics if k!='experimental'}
    equality['experimental_except_file_path']={k:v for k,v in c['experimental'].items() if k!='tiff_file'}=={k:v for k,v in original['experimental'].items() if k!='tiff_file'}
    assert all(equality.values())
    save('effective_config.json',c)
    save('hyperparameter_verification.json',equality)
    problem=ordinary.DiscreteProblem(c,device='cuda',diagnostics=True)
    problem.ensure_references()
    center=(problem.box_lower+problem.box_upper)/2
    lattice,basis,basis_symbols=build_basis()
    phi=fft_orientation()
    ij=np.array([(i,j) for i in range(-20,21) for j in range(-20,21)])
    primitive=(ij@lattice[:2])[:,None,:]+basis[None,:,:]
    primitive=primitive.reshape(-1,3)
    primitive_symbols=np.tile(basis_symbols,len(ij))
    device=problem.device
    roi=tuple(slice(int(a),int(b)) for a,b in zip(problem.experimental['sample_roi_start_zyx'],problem.experimental['sample_roi_stop_zyx']))
    target=problem.target
    target_sum=torch.zeros_like(target[0]); target_sum[roi[1:]]=target[roi].sum(0)
    translation_pixels=torch.arange(-21,22,device=device)
    yy,xx=torch.meshgrid(translation_pixels,translation_pixels,indexing='ij')
    trials=[]

    for branch in range(4):
        base_angles=np.array([phi-1.5+(60 if branch&1 else 0),phi+1.5+(60 if branch&2 else 0)])
        positions=[]; symbols=[]; layers=[]
        for layer,angle in enumerate(base_angles):
            t=np.deg2rad(angle)
            rot=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
            p=primitive@rot.T
            guard=4.
            keep=np.all((p[:,:2]>=problem.box_lower[:2]-center[:2]-guard)&
                        (p[:,:2]<problem.box_upper[:2]-center[:2]+guard),axis=1)
            positions.append(p[keep]); symbols.extend(primitive_symbols[keep]); layers.extend([layer]*int(keep.sum()))
        local=torch.as_tensor(np.concatenate(positions),device=device,dtype=problem.dtype)
        layer_ids=torch.as_tensor(layers,device=device)
        ids=torch.as_tensor([ordinary.SPECIES.index(s) for s in symbols],device=device)
        labels=torch.nn.functional.one_hot(ids,2).to(problem.dtype)
        cent=torch.as_tensor(center,device=device,dtype=problem.dtype)
        params=np.array([0.,0.,0.,0.,0.,0.,10.2,17.8])

        def coordinates(parameters):
            angles=parameters[:2][layer_ids]*np.pi/180
            cs,sn=torch.cos(angles),torch.sin(angles)
            x=cs*local[:,0]-sn*local[:,1]+cent[0]+parameters[2:6].reshape(2,2)[layer_ids,0]
            y=sn*local[:,0]+cs*local[:,1]+cent[1]+parameters[2:6].reshape(2,2)[layer_ids,1]
            z=local[:,2]+parameters[6:8][layer_ids]
            return torch.stack((x,y,z),dim=1)

        # Coarse per-layer phase alignment by full-image cross correlation.
        with torch.no_grad():
            p=coordinates(torch.as_tensor(params,device=device,dtype=problem.dtype))
            for layer in (0,1):
                which=layer_ids==layer
                pred,_=problem.gaussian_renderer.render(labels[which],p[which])
                corr=torch.fft.irfft2(torch.fft.rfft2(target_sum)*torch.conj(torch.fft.rfft2(pred[roi[0]].sum(0))),s=target_sum.shape)
                small=corr[yy.remainder(corr.shape[0]),xx.remainder(corr.shape[1])]
                index=int(torch.argmax(small)); dy=float(yy.reshape(-1)[index]); dx=float(xx.reshape(-1)[index])
                params[2+layer*2:4+layer*2]=[dx*.152564,dy*.152564]
        calls=[0]
        def loss_gradient(value):
            x=torch.tensor(value,device=device,dtype=problem.dtype,requires_grad=True)
            p=coordinates(x)
            prediction,_=problem.gaussian_renderer.render(labels,p)
            loss=c['imaging']['penalty']*torch.nn.functional.huber_loss(prediction[problem.mask],target[problem.mask],delta=c['imaging']['huber_delta'])
            grad=torch.autograd.grad(loss,x)[0]
            calls[0]+=1
            return float(loss.detach()),grad.detach().cpu().numpy().astype(float)
        initial=loss_gradient(params)[0]
        fitted=minimize(loss_gradient,params,jac=True,method='L-BFGS-B',
                        bounds=[(-4,4),(-4,4),(-4,4),(-4,4),(-4,4),(-4,4),(9.75,10.65),(17.35,18.25)],
                        options=dict(maxiter=90,maxls=15,ftol=1e-10,gtol=1e-6))
        with torch.no_grad():
            p=coordinates(torch.as_tensor(fitted.x,device=device,dtype=problem.dtype)).cpu().numpy()
        keep=np.all((p[:,:2]>=problem.box_lower[:2])&(p[:,:2]<problem.box_upper[:2]),axis=1)
        sy=np.array(symbols)[keep].tolist(); p=p[keep]
        terms,_=problem.image_and_box(sy,p,gradient=False)
        angles=base_angles+fitted.x[:2]
        trial=dict(branch=branch,initial_halo_image_term=initial,final_halo_image_term=float(fitted.fun),
                   cropped_image_terms=terms,angles_degrees=angles,
                   lattice_twist_modulo_60_degrees=float((angles[1]-angles[0]+30)%60-30),
                   parameters=fitted.x,success=bool(fitted.success),message=str(fitted.message),
                   alignment_evaluations=calls[0],atoms=len(sy),
                   stacking_branches=[bool(branch&1),bool(branch&2)])
        save(f'alignment_trial_{branch}.json',trial)
        a=Atoms(sy,positions=p,cell=problem.cell,pbc=True)
        a.arrays['layer']=np.array(layers)[keep]
        write_atoms(f'alignment_trial_{branch}',a)
        trials.append((terms['image_term'],a,trial))
        print('ALIGNMENT',json.dumps(ordinary.jsonable(trial)),flush=True)
    _,atoms,alignment=min(trials,key=lambda x:x[0])
    write_atoms('aligned_ideal_initial',atoms)
    save('alignment_summary.json',alignment)

    # Verify the existing ordinary winner under the exact same objective.
    old=json.loads((BASE/'inputs/ordinary_best_structure.json').read_text())
    score,_,terms=problem.objective(old['structure']['symbols'],np.array(old['structure']['positions']))
    assert abs(score-old['score'])<1e-5,(score,old['score'])
    save('ordinary_score_verification.json',dict(recorded_score=old['score'],recomputed=terms,difference=score-old['score']))
    comparisons={'ordinary_winner':terms}
    previous=read(BASE/'inputs/previous_energy_only_relaxed.extxyz')
    comparisons['previous_energy_only_relaxed']=problem.objective(previous.get_chemical_symbols(),previous.positions)[2]
    initial=problem.objective(atoms.get_chemical_symbols(),atoms.positions)[2]
    comparisons['aligned_ideal_initial']=initial
    save('initial_objective.json',initial)
    print('INITIAL_OBJECTIVE',json.dumps(initial),flush=True)

    raw_objective=problem.objective
    history_file=(BASE/'objective_evaluations.csv').open('w',buffering=1)
    fields=['evaluation','score','formation_energy_per_atom','image_loss_raw','image_term','sample_box_term','composition_penalty','image_correlation']
    history=csv.DictWriter(history_file,fieldnames=fields); history.writeheader(); calls=[0]
    def logged_objective(symbols,positions):
        value,gradient,t=raw_objective(symbols,positions)
        calls[0]+=1
        history.writerow(dict(evaluation=calls[0],**{k:t[k] for k in fields[1:]}))
        return value,gradient,t
    problem.objective=logged_objective
    # Exact production optimizer and all saved stopping hyperparameters, unmodified.
    result=ordinary.relax_candidate(problem,ordinary.candidate(atoms.get_chemical_symbols(),atoms.positions),
                                    progress_path=BASE/'pbfgs_progress.json')
    history_file.close()
    save('final_result.json',result)
    final=Atoms(**result['structure']); final.arrays['layer']=atoms.arrays['layer'].copy()
    write_atoms('final_guided_relaxed',final)
    assert np.array_equal(final.numbers,atoms.numbers)
    assert np.allclose(final.cell.array,atoms.cell.array)
    assert abs(result['score']-sum(result[k] for k in ('formation_energy_per_atom','image_term','sample_box_term','composition_penalty')))<1e-10
    comparisons['final_guided_relaxed']={k:v for k,v in result.items() if k not in ('structure','compute_metrics','relaxation')}
    save('objective_comparison.json',comparisons)
    # Save the exact crop in the objective's normalized units, no image registration.
    ordinary.save_best_images(problem,[dict(result,valid=True)])
    for path in (BASE/'diagnostics').glob('*slices.tif'):
        array=tifffile.imread(path)
        tifffile.imwrite(BASE/path.name,array,metadata={'axes':'ZYX'})
        tifffile.imwrite(BASE/path.name.replace('_slices','_zsum_320x320'),array.sum(0))
    save('run_summary.json',dict(status='complete',started_from='MP bulk-derived ideal layers registered directly to experimental TIFF',
          host=platform.node(),gpu=torch.cuda.get_device_name(0),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
          wall_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat(),
          same_objective_and_relaxation_hyperparameters=True,atom_count=len(final),
          initial_score=initial['score'],final_score=result['score'],relaxation=result['relaxation'],
          caveat='A data-guided local reference, not independently established experimental ground truth. Fixed composition during PBFGS; atom counts follow the aligned hard crop.'))
    print('FINAL_RESULT',json.dumps({k:v for k,v in result.items() if k!='structure'}),flush=True)


if __name__=='__main__':
    main()
