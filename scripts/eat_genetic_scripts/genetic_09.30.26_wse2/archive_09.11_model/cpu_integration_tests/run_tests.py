"""CPU regression tests against the actual GA module (no FairChem evaluation)."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/wse2_integration_matplotlib'
from pathlib import Path
import sys, importlib.util, json, copy, time, hashlib
import numpy as np
import torch
import tifffile
from ase.io import read
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('wse2_ga',ROOT/'genetic_09.01.26_wse2.py')
ga=importlib.util.module_from_spec(spec);spec.loader.exec_module(ga)
from calibrated_image_model import initialize_calibrated_renderer
ANALYSIS=ROOT.parent/'chia-hao_data/depth_calibration_2026-09-10/30_slices'
torch.set_num_threads(1)

def log(text):print(text,flush=True)
def make_problem(padding='8,8,0',label='native'):
    a=ga.get_settings()
    # Supported coordinator construction skips calculator/reference downloads;
    # all real preparation, rendering, losses, prefit, serializers are exercised.
    a.ga_execution_backend='slurm';a.ga_resuming_from_checkpoint=True
    a.output_dir=str(HERE/label);a.vacuum_padding_angstrom=padding
    return ga.WSe2VolumeProblem(a)

def double_precision(p):
    p.dtype=torch.float64
    for k,v in vars(p).items():
        if torch.is_tensor(v) and v.is_floating_point():setattr(p,k,v.double())
    initialize_calibrated_renderer(p)

def main():
    start=time.time();results={};assert not torch.cuda.is_available()
    log('Constructing production GA on CPU and replaying accepted calibration')
    p=make_problem();assert str(p.device)=='cpu'
    atoms=read(ROOT.parent/'09.09/rank_0001_score_m0p05229673_gen_010_member_023_crossover_child.xsf')
    cal=p.calibrated_renderer.calibration
    pos=atoms.positions.copy();pos[:,:2]+=cal['xy_reference_registration_A']
    obs=np.load(ANALYSIS/'results/observations.npz')
    pos[:,2]=obs['atom_depths']+cal['fit']['offset']
    occ=np.column_stack([np.array(atoms.get_chemical_symbols())==s for s in p.species]).astype(float)
    p.set_site_layout(np.zeros(len(pos),int))
    pt=torch.tensor(pos,dtype=p.dtype);ot=torch.tensor(occ,dtype=p.dtype)
    mask=p.image_loss_mask_t
    def roi(x):return x[mask].reshape(30,150,150)
    saved=tifffile.imread(ANALYSIS/'flexible_depth_with_envelope/flexible_joint_0.001_ZYX.tif')
    with torch.no_grad():pred=p.render_differentiable_gaussian_image(ot,pt)[0];raw=roi(pred).numpy()*p.experimental['target_normalization']
    err32=float(abs(raw-saved).max());assert err32<2e-6,err32
    results['production_float32_raw_max_error']=err32
    float32_aux=p.auxiliary_imaging_and_boundary_terms(ot,pt,True)
    assert torch.isfinite(float32_aux['grad_occ']).all() and torch.isfinite(float32_aux['grad_z']).all()
    results['production_float32_gradients_finite']=True
    assert p.target_shape_zyx==(30,254,254) and p.image_loss_voxel_count==30*150*150
    assert np.allclose(p.sample_box_lower_A,[7.933328,7.933328,7])
    assert np.allclose(p.sample_box_upper_A,[30.817928,30.817928,23])
    assert not p.experimental['feasible_mask'][:7].any() and not p.experimental['feasible_mask'][24:].any()
    results['geometry']={'render_shape_zyx':list(p.target_shape_zyx),'measured_shape_zyx':[30,150,150],
        'atomic_box_lower_xyz_A':p.sample_box_lower_A.tolist(),'atomic_box_upper_xyz_A':p.sample_box_upper_A.tolist(),
        'extra_padding_xyz_A':p.vacuum_padding.tolist(),'normalization':p.experimental['target_normalization']}
    double_precision(p);pt=pt.double();pt[:]=torch.tensor(pos);ot=ot.double()
    with torch.no_grad():raw64=roi(p.render_differentiable_gaussian_image(ot,pt)[0]).numpy()*p.experimental['target_normalization']
    err64=float(abs(raw64-saved).max());assert err64<2e-8,err64
    results['production_float64_raw_max_error']=err64
    log('Optional z padding and nonperiodic depth response')
    q=make_problem('8,8,8','extra_z_padding');double_precision(q);q.set_site_layout(np.zeros(len(pos),int))
    shifted=pt+torch.tensor([0.,0.,8.],dtype=torch.float64)
    with torch.no_grad():padded=q.render_differentiable_gaussian_image(ot,shifted)[0][q.image_loss_mask_t].reshape(30,150,150)
    assert np.max(abs(padded.numpy()*q.experimental['target_normalization']-raw64))<1e-12
    assert np.allclose(q.sample_box_lower_A[2],15) and np.allclose(q.sample_box_upper_A[2],31)
    box0=p.sample_box_confinement_terms(ot,pt)[0];box1=q.sample_box_confinement_terms(ot,shifted)[0]
    assert float(abs(box0-box1))<1e-12
    invisible=pt[:1].clone();invisible[:,2]=-35
    with torch.no_grad():assert float(p.render_differentiable_gaussian_image(ot[:1],invisible)[0].abs().max())==0
    results['optional_z_padding_image_and_confinement_invariant']=True
    results['no_axial_wrap']=True
    log('Production auxiliary gradients, including box forces and soft occupancies')
    p.set_site_layout(np.zeros(2,int));testp=pt[[1,4]].clone()+torch.tensor([.023,.017,.043]);testp[0,2]=5.4;testp[1,2]=24.6
    testo=torch.tensor([[.61,.19],[.12,.71]],dtype=torch.float64)
    aux=p.auxiliary_imaging_and_boundary_terms(testo,testp,True)
    assert aux['image']['ellipse_composition_objective_term']==0 and aux['image']['ellipse_composition_loss_raw']==0
    assert aux['grad_z'][0,2]<0 and aux['grad_z'][1,2]>0
    checks=[];h=1e-5
    for name,v,grad in [('positions',testp,aux['grad_z']),('occupancies',testo,aux['grad_occ'])]:
        for i in range(2):
            for j in range(v.shape[1]):
                plus=v.clone();minus=v.clone();plus[i,j]+=h;minus[i,j]-=h
                if name=='positions':
                    fa=p.auxiliary_imaging_and_boundary_terms(testo,plus,False)['objective_term'];fb=p.auxiliary_imaging_and_boundary_terms(testo,minus,False)['objective_term']
                else:
                    fa=p.auxiliary_imaging_and_boundary_terms(plus,testp,False)['objective_term'];fb=p.auxiliary_imaging_and_boundary_terms(minus,testp,False)['objective_term']
                fd=(fa-fb)/(2*h);error=abs(fd-float(grad[i,j]));assert error<2e-6+abs(fd)*1e-5,(name,i,j,error)
                checks.append({'variable':name,'atom':i,'component':j,'autograd':float(grad[i,j]),'finite_difference':fd,'absolute_error':error})
    results['gradient_checks']=checks
    p.ellipse_composition_penalty=1e9
    assert p.auxiliary_imaging_and_boundary_terms(testo,testp,False)['objective_term']==aux['objective_term']
    p.ellipse_composition_penalty=0
    results['ellipse_penalty_removed_from_actual_objective']=True
    zeros=torch.zeros_like(testo);assert float(p.sample_box_confinement_terms(zeros,testp)[0])==0
    log('Vacuum hiding versus a true Se deletion, with the new slab penalty')
    p.set_site_layout(np.zeros(len(pos),int))
    se=np.flatnonzero(occ[:,p.species.index('Se')]);probes=se[np.linspace(10,len(se)-11,6).astype(int)]
    hiding=[]
    for idx in probes:
        deleted=ot.clone();deleted[idx]=0
        with torch.no_grad():target=p.render_differentiable_gaussian_image(deleted,pt)[0]
        delete_box=float(p.sample_box_confinement_terms(deleted,pt)[0])
        # Nominal 1:2 reference is deliberately the tougher case for deletion.
        comp_delete=10*sum((np.array([116,231])/347-np.array([1/3,2/3]))**2)
        for z in (0.,.5,29.,29.5):
            moved=pt.clone();moved[idx,2]=z
            with torch.no_grad():field=p.render_differentiable_gaussian_image(ot,moved)[0]
            image=float(100*torch.nn.functional.huber_loss(field[mask],target[mask],delta=.2))
            box=float(p.sample_box_confinement_terms(ot,moved)[0])
            assert image+box>delete_box+comp_delete
            hiding.append({'atom':int(idx),'z_A':z,'image_plus_box':image+box,'deletion_box_plus_nominal_composition':delete_box+comp_delete})
    results['vacuum_hiding_cases_rejected']=len(hiding)
    (HERE/'vacuum_checks.json').write_text(json.dumps(hiding,indent=2)+'\n')
    log('Actual image-prefit steps and final image serialization')
    a=p.args;a.random_immigrant_image_prefit_soft_steps=4;a.random_immigrant_image_prefit_entropy_final_weight=0.
    a.random_immigrant_image_prefit_initial_occupancy_mode='current_state';a.random_immigrant_image_prefit_initial_logit_noise=0.
    # Move one atom outside the slab to exercise prefit's box term.
    noisy=pt.clone();noisy[0,2]=5.4
    state=p.pack_y(.85*ot,noisy)
    fitted,summary=ga.random_immigrant_image_prefit_state(p,state,a,seed=17,member_index=0)
    assert torch.isfinite(fitted).all()
    assert summary['soft_handoff_metrics']['objective']<summary['initial_metrics']['objective']
    assert summary['soft_handoff_metrics']['ellipse_composition_objective']==0
    results['prefit_initial_objective']=summary['initial_metrics']['objective']
    results['prefit_final_objective']=summary['soft_handoff_metrics']['objective']
    p.write_final_image_fields(p.pack_y(ot,pt),p.pack_y(ot,pt))
    assert (p.output_dir/'final_gaussian_image_comparison.npz').exists()
    p.write_lightweight_final_ellipse_comparison(state,state)
    assert not (p.output_dir/'final_ellipse_composition_comparison.csv').exists()
    # Exercise the non-resuming preparation diagnostics too.
    ga.prepare_experimental_volume(a,HERE/'preparation_diagnostics',write_diagnostics=True)
    log('Measured-ROI slice comparisons with optional extra z padding')
    q.args.member_slice_comparisons_enable=True
    image_score=p.auxiliary_imaging_and_boundary_terms(ot,pt,False)['image']['objective_term']
    record={'generation':0,'member_index':0,'kind':'calibration_replay','seed':17,
        'score':image_score,'y':q.pack_y(ot,shifted),'assigned_site_region_indices':[0]*len(pos)}
    marker=q.output_dir/'slice_comparisons/generation_000_member_000_calibration_replay/slice_comparison_complete.json'
    marker.unlink(missing_ok=True)
    gallery=Path(q.write_member_slice_comparisons(record))
    assert len(list(gallery.glob('slice_z*_comparison.png')))==30
    summaries=list(gallery.glob('*.json'))
    summary_path=next(path for path in summaries if 'slice_metrics' in json.loads(path.read_text()))
    plotted=json.loads(summary_path.read_text())
    assert plotted['target_shape_zyx']==[30,150,150]
    mean_slice_loss=np.mean([row['huber_loss'] for row in plotted['slice_metrics']])
    assert abs(mean_slice_loss-image_score/100)<1e-8
    results['slice_comparison_count']=30
    results['slice_comparison_roi_matches_objective']=True
    log('Checkpoint calibration changes and old checkpoints must be rejected')
    checkpoint_args=copy.deepcopy(a);checkpoint_args.output_dir=str(HERE/'checkpoint_probe')
    Path(checkpoint_args.output_dir).mkdir(exist_ok=True)
    cf=HERE/'checkpoint_probe/calibration.json';text=Path(a.depth_calibration_file).read_text();cf.write_text(text)
    checkpoint_args.depth_calibration_file=str(cf)
    fp=ga.ga_checkpoint_fingerprint(checkpoint_args)
    cp=Path(checkpoint_args.output_dir)/'ga_checkpoint.json'
    cp.write_text(json.dumps({'schema_version':ga.GA_CHECKPOINT_SCHEMA_VERSION,'config_fingerprint':fp}))
    assert ga.read_ga_checkpoint_for_args(checkpoint_args) is not None
    changed=json.loads(text);changed['fit']['gain']*=1.01;cf.write_text(json.dumps(changed))
    assert fp!=ga.ga_checkpoint_fingerprint(checkpoint_args)
    try:ga.read_ga_checkpoint_for_args(checkpoint_args)
    except RuntimeError:pass
    else:raise AssertionError('Changed calibration checkpoint accepted')
    cf.write_text(text)
    cp.write_text(json.dumps({'schema_version':ga.GA_CHECKPOINT_SCHEMA_VERSION,'config_fingerprint':'old-objective'}))
    try:ga.read_ga_checkpoint_for_args(checkpoint_args)
    except RuntimeError:pass
    else:raise AssertionError('Old objective checkpoint accepted')
    results['checkpoint_changes_rejected']=True
    results['all_passed']=True;results['elapsed_seconds']=time.time()-start
    results['device']='cpu';results['fairchem_evaluated']=False
    results['scope']='Production rendering, losses, prefit and serialization; no energy calculation or GA population run.'
    results['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'genetic_09.01.26_wse2.py',ROOT/'calibrated_image_model.py',ROOT/'calibration/accepted_30_slice_model.json',ROOT/'genetic_09.01.26_wse2.yaml']}
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    log('PASS: '+json.dumps({k:v for k,v in results.items() if k not in ['gradient_checks','source_hashes','geometry']}))
if __name__=='__main__':main()
