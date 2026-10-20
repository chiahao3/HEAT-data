"""CPU-only transfer tests for the accepted 30-slice renderer; no GA mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']='/tmp/wse2_cpu_checks_matplotlib'
from pathlib import Path
import sys,json,csv,hashlib,time,importlib.util
import numpy as np
import torch
from scipy.optimize import lsq_linear
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ANALYSIS=ROOT/'chia-hao_data/depth_calibration_2026-09-10/30_slices'
sys.path.insert(0,str(ANALYSIS))
from depth_model import read_xsf,lateral_basis
import run_analysis as original

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def dump(name,x):(HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 with (HERE/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(x):print(time.strftime('%H:%M:%S'),x,flush=True)

class CPURenderer:
 def __init__(self,coeff,fit,voxel=.152564,padded=254,roi_start=52,size=150):
  self.voxel=voxel;self.padded=padded;self.roi_start=roi_start;self.size=size
  self.mass=torch.tensor(fit['mass'],device='cpu');self.lags=torch.tensor(fit['centered_lags'],device='cpu');self.gain=fit['gain'];self.widths=fit['widths']
  rows={r['species']:r for r in csv.DictReader(coeff.open()) if r['species'] in ['W','Se']}
  reference=sum(float(rows['W'][a+'_integrated_weight_VAngstrom3']) for a in ['core','tail'])
  fy=torch.fft.fftfreq(padded);fx=torch.fft.rfftfreq(padded);freq=fy[:,None]**2+fx[None,:]**2
  kernels=[]
  for sp in ['W','Se']:
   kernel=0
   for part in ['core','tail']:
    sig=np.sqrt(float(rows[sp][part+'_sigma_xy_angstrom'])**2+.08**2)/voxel
    weight=float(rows[sp][part+'_integrated_weight_VAngstrom3'])/reference
    kernel=kernel+weight*torch.exp(-2*np.pi**2*sig**2*freq)
   kernels.append(kernel)
  self.fourier=torch.stack(kernels)
  self.spatial=torch.fft.irfft2(self.fourier,s=(padded,padded))
 def axial(self,zpos,zgrid):
  k=torch.clamp(1-torch.abs(zgrid[None,:,None]-zpos[:,None,None]-self.lags),min=0)@self.mass
  env=torch.exp(-(torch.clamp(9-zgrid,min=0)/self.widths[0])**2-(torch.clamp(zgrid-20,min=0)/self.widths[1])**2)
  return k,env
 def render(self,positions,occ,zgrid=None):
  # Species/depth images are bilinearly splatted in XY, convolved only in XY.
  # Z has explicit nonperiodic fractional-bin response evaluation.
  if zgrid is None:zgrid=torch.arange(30,device='cpu')
  N=self.padded;nz=len(zgrid);k,e=self.axial(positions[:,2],zgrid)
  pix=positions[:,:2]/self.voxel;base=torch.floor(pix).long();frac=pix-base
  density=torch.zeros(2*nz*N*N,device='cpu')
  sz=torch.arange(2*nz).reshape(2,nz,1)*N*N
  for dy in [0,1]:
   for dx in [0,1]:
    w=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
    ids=(base[:,1]+dy).remainder(N)*N+(base[:,0]+dx).remainder(N)
    value=occ.T[:,None,:]*k.T[None,:,:]*w[None,None,:]
    density=density.index_add(0,(sz+ids).reshape(-1),value.reshape(-1))
  density=density.reshape(2,nz,N,N)
  field=torch.fft.irfft2(torch.fft.rfft2(density)*self.fourier[:,None,:,:],s=(N,N)).sum(0)
  start=self.roi_start;crop=field[:,start:start+self.size,start:start+self.size]
  return self.gain*crop*e[:,None,None]
 def small_render(self,positions,occ,zgrid=None):
  # Equivalent direct shifted kernels, economical for finite-difference probes.
  if zgrid is None:zgrid=torch.arange(30,device='cpu')
  k,e=self.axial(positions[:,2],zgrid);start=self.roi_start;result=0
  for i in range(len(positions)):
   pix=positions[i,:2]/self.voxel;base=torch.floor(pix).long();frac=pix-base
   field=0
   for dy in [0,1]:
    for dx in [0,1]:
     w=(frac[0] if dx else 1-frac[0])*(frac[1] if dy else 1-frac[1])
     shifted=torch.roll(self.spatial,(int(base[1])+dy,int(base[0])+dx),dims=(-2,-1))[:,start:start+self.size,start:start+self.size]
     field=field+w*(occ[i,:,None,None]*shifted).sum(0)
   result=result+self.gain*k[i,:,None,None]*field
  return result*e[:,None,None]

def main():
 began=time.time();assert not torch.cuda.is_available()
 fitfile=ANALYSIS/'flexible_depth_with_envelope/summary.json';summary=json.loads(fitfile.read_text());f=summary['all_fit'][summary['best_flexible']]
 prov=json.loads((ANALYSIS/'results/provenance.json').read_text());o=np.load(ANALYSIS/'results/observations.npz');history=json.loads((original.RUN/'blob_optimization_history.json').read_text())
 sy,oldpos=read_xsf(original.XSF);pos=oldpos.copy();pos[:,:2]+=np.array(prov['xy_registration']['shift_xy_angstrom']);pos[:,2]=o['atom_depths']+f['offset']
 occ=np.column_stack([sy=='W',sy=='Se']).astype(float)
 R=CPURenderer(original.FIT,f,prov['voxel_xy_angstrom'],int(history['target_shape_zyx'][1]),int(history['sample_roi_start_zyx'][1]))
 pt=torch.tensor(pos);ot=torch.tensor(occ)
 log('CPU renderer replay, zero added z vacuum, optional padding invariance')
 with torch.no_grad():
  model=R.render(pt,ot).numpy();padded=R.render(pt,ot,torch.arange(-8.,38.)).numpy()[8:38]
 saved=tifffile.imread(ANALYSIS/'flexible_depth_with_envelope/flexible_joint_0.001_ZYX.tif')
 err=float(np.max(abs(model-saved)));assert err<2e-8,err
 paderror=float(np.max(abs(model-padded)));assert paderror<1e-12
 wrap_a=R.small_render(torch.tensor([[pos[0,0],pos[0,1],-35.]]),torch.tensor([[0.,1.]]));assert float(wrap_a.abs().max())==0
 gap=float(30-np.ptp(pos[:,2]));geometry=dict(sample_shape_zyx=[30,150,150],render_shape_zyx=[30,R.padded,R.padded],extra_padding_xyz_A=[52*R.voxel,52*R.voxel,0.],cell_xyz_A=[R.padded*R.voxel,R.padded*R.voxel,30.],atom_z_min_A=float(pos[:,2].min()),atom_z_max_A=float(pos[:,2].max()),periodic_z_gap_between_outermost_atom_centers_A=gap,note='Geometric separation only; no FairChem energy convergence test.')
 log('Autograd versus central differences for positions and soft W/Se occupancies')
 # Two atoms, away from interpolation knots, against an actual residual field.
 selected=np.array([int(np.flatnonzero(sy=='W')[20]),int(np.flatnonzero(sy=='Se')[40])])
 pp=pt[selected].clone()+torch.tensor([.021,.013,.027]);oo=torch.tensor([[.67,.13],[.11,.72]])
 with torch.no_grad():fixed=torch.tensor(model)-R.small_render(pt[selected],ot[selected])
 target=torch.tensor(o['raw_crop_zyx']);normalization=float(np.percentile(o['raw_crop_zyx'],99.5));delta=.2
 def objective(p,q):return 100*torch.nn.functional.huber_loss((fixed+R.small_render(p,q))/normalization,target/normalization,delta=delta,reduction='mean')
 pp.requires_grad_();oo.requires_grad_();loss=objective(pp,oo);gp,go=torch.autograd.grad(loss,(pp,oo));gradient=[];step=1e-5
 for name,arr,grad in [('position',pp,gp),('occupancy',oo,go)]:
  for i in range(arr.shape[0]):
   for j in range(arr.shape[1]):
    a=arr.detach().clone();b=a.clone();a[i,j]+=step;b[i,j]-=step
    with torch.no_grad():fd=float((objective(a,oo.detach())-objective(b,oo.detach()))/(2*step)) if name=='position' else float((objective(pp.detach(),a)-objective(pp.detach(),b))/(2*step))
    analytic=float(grad[i,j]);error=abs(fd-analytic)
    assert error<2e-7+1e-4*abs(fd),(name,i,j,error,fd,analytic)
    gradient.append(dict(variable=name,atom_index=int(selected[i]),component=j,analytic=analytic,finite_difference=fd,absolute_error=error))
 table('gradient_checks.csv',gradient)
 # Make sure both computational implementations agree, including soft species.
 with torch.no_grad():smallerror=float((R.render(pp.detach(),oo.detach())-R.small_render(pp.detach(),oo.detach())).abs().max())
 assert smallerror<1e-12
 log('Synthetic missing-Se identification and soft-occupancy recovery')
 basis,_=lateral_basis(sy,oldpos,original.FIT,R.voxel,R.roi_start,150,R.padded,prov['xy_registration']['shift_xy_angstrom'],.08)
 with torch.no_grad():K,E=R.axial(pt[:,2],torch.arange(30.));K=(K*E).numpy()
 se=np.flatnonzero(sy=='Se');B=basis[se].reshape(len(se),-1);kern=K[se];gram=f['gain']**2*(B@B.T)*(kern@kern.T)
 lower=R.roi_start*R.voxel;upper=lower+150*R.voxel
 interior=np.flatnonzero(np.all((pos[se,:2]>lower+2)&(pos[se,:2]<upper-2),axis=1))
 probes=interior[np.linspace(0,len(interior)-1,6).astype(int)]
 totalW=int(sum(sy=='W'));totalSe=int(sum(sy=='Se'))
 def comp(w,se):
  frac=np.array([w,se],float)/(w+se);return float(10*np.sum((frac-np.array([1/3,2/3]))**2))
 comp0=comp(totalW,totalSe);comp_removed=comp(totalW,totalSe-1)-comp0
 nominal_removed=comp(116,231)-comp(116,232)
 def huber(a):
  x=np.abs(a);return np.where(x<=delta,.5*x*x,delta*(x-.5*delta))
 deletion=[]
 for j in probes:
  changes=np.diag(gram)-2*gram[:,j]
  winner=int(np.argmin(changes));assert winner==j
  contribution=f['gain']*K[se[j],:,None,None]*basis[se[j]][None,:,:]
  image_penalty=float(100*np.mean(huber(contribution/normalization)))
  deletion.append(dict(deleted_atom_index=int(se[j]),best_detected_atom_index=int(se[winner]),correct=True,intact_candidate_image_penalty=image_penalty,correct_vacancy_composition_penalty_increment=comp_removed,correct_vacancy_beats_intact_with_image_plus_composition=bool(comp_removed<image_penalty)))
 G=gram[np.ix_(probes,probes)];L=np.linalg.cholesky(G).T;truth=np.array([0.,1.,0.,1.,1.,0.]);soft=lsq_linear(L,L@truth,bounds=(0,1),tol=1e-12,lsmr_tol=1e-12,max_iter=1000)
 occerr=float(np.max(abs(soft.x-truth)));assert soft.success and occerr<1e-4
 table('synthetic_se_vacancies.csv',deletion)
 log('Testing whether cutoff-suppressed vacuum atoms are excluded by image loss')
 # Missing atom target: removing the atom is exact; moving it into vacuum
 # leaves a suppressed residual but retains the atom count/composition.
 vacuum=[];zvalues=np.r_[0.,.5,1.,2.,4.,6.,9.,12.,15.,18.,20.,23.,25.,27.,28.,29.,29.5]
 for j in probes:
  for zz in zvalues:
   with torch.no_grad():kk,ee=R.axial(torch.tensor([zz]),torch.arange(30.));visible=(kk*ee).numpy()[0]
   ghost=f['gain']*visible[:,None,None]*basis[se[j]][None,:,:]
   img=float(100*np.mean(huber(ghost/normalization)))
   vacuum.append(dict(atom_index=int(se[j]),trial_z_A=float(zz),visible_response_mass=float(visible.sum()),image_penalty_against_exact_vacancy=img,extra_composition_penalty_if_moved=0.,extra_composition_penalty_if_deleted=comp_removed,moved_atom_beats_true_deletion_without_energy=bool(img<comp_removed),nominal_stoichiometric_deletion_composition_cost=nominal_removed,moved_atom_beats_deletion_at_nominal_stoichiometry=bool(img<nominal_removed)))
 table('vacuum_atom_scan.csv',vacuum)
 # Real-data single-deletion diagnostics, not validated vacancy calls.
 residual=(model-o['raw_crop_zyx'])/normalization;base_loss=float(100*np.mean(huber(residual)));real=[]
 for j in probes:
  contribution=f['gain']*K[se[j],:,None,None]*basis[se[j]][None,:,:]
  changed=float(100*np.mean(huber(residual-contribution/normalization)))
  real.append(dict(atom_index=int(se[j]),image_objective_change_on_deletion=changed-base_loss,composition_objective_change=comp_removed,image_plus_composition_change=changed-base_loss+comp_removed,note='No relaxation or energy evaluation; not a defect assignment'))
 table('real_data_se_deletion_checks.csv',real)
 # Summary plot shows suppression and the composition-vs-hiding tradeoff.
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for j in probes:
  rows=[v for v in vacuum if v['atom_index']==se[j]]
  axes[0].plot([v['trial_z_A'] for v in rows],[v['visible_response_mass'] for v in rows],label=f'Se {se[j]}')
  axes[1].semilogy([v['trial_z_A'] for v in rows],[max(v['image_penalty_against_exact_vacancy'],1e-20) for v in rows])
 axes[0].set(xlabel='Trial atom depth (Å)',ylabel='Visible unit-response mass',title='Envelope attenuates vacuum atoms too')
 axes[1].axhline(nominal_removed,color='k',ls='--',label='Deletion cost at nominal W116 Se232')
 axes[1].set(xlabel='Trial atom depth (Å)',ylabel='Image penalty for a displaced atom',title='Exact missing-Se synthetic target');axes[1].legend(fontsize=8)
 fig.savefig(HERE/'vacuum_visibility.png',dpi=160);plt.close(fig)
 edge=[v for v in vacuum if v['trial_z_A'] in [0.,.5,29.,29.5]]
 result=dict(device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],torch_version=torch.__version__,geometry=geometry,renderer_max_error_vs_saved_float32=err,added_z_padding_roi_max_error=paderror,no_axial_wrap_test=True,direct_vs_fft_renderer_error=smallerror,gradient_checks_passed=len(gradient),max_gradient_absolute_error=max(v['absolute_error'] for v in gradient),synthetic_vacancies_correct=len(deletion),synthetic_vacancies_tested=len(deletion),synthetic_soft_occupancy_max_error=occerr,reference_counts={'W':totalW,'Se':totalSe},nominal_stoichiometric_deletion_cost=nominal_removed,nominal_stoichiometric_vacuum_hiding_cases=sum(v['moved_atom_beats_deletion_at_nominal_stoichiometry'] for v in edge),composition_penalty_weight=10.,image_penalty_weight=100.,huber_delta_normalized=.2,normalization_percentile_99p5=normalization,composition_cost_for_one_Se_deletion=comp_removed,synthetic_deletions_beating_intact_with_composition=sum(v['correct_vacancy_beats_intact_with_image_plus_composition'] for v in deletion),vacuum_edge_visible_mass_range=[min(v['visible_response_mass'] for v in edge),max(v['visible_response_mass'] for v in edge)],vacuum_hiding_beats_true_deletion_cases=sum(v['moved_atom_beats_true_deletion_without_energy'] for v in edge),vacuum_edge_cases=len(edge),elapsed_seconds=time.time()-began,energy_evaluated=False,energy_scope_note='This CPU environment lacks FairChem and ASE; image/gradient/geometry/composition tests only. No full GA or energy validation.',input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fitfile,original.XSF,original.FIT,ANALYSIS/'results/observations.npz']},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 dump('results.json',result)
 report=f'''# CPU checks for the 30-slice GA renderer

All checks ran on CPU with CUDA hidden. The production GA code/configuration were not changed in this test. The prototype uses the accepted fixed-0.08 Å XY kernels, flexible depth response, calibrated gain and fixed cutoff. Lateral padding is {52*R.voxel:.6f} Å per side; added z padding is zero. The physical cell would be {R.padded*R.voxel:.6f} × {R.padded*R.voxel:.6f} × 30 Å.

- Saved accepted-renderer replay: maximum voxel difference {err:.3g} (saved TIFF is float32).
- {len(gradient)} position/occupancy finite-difference checks passed; largest absolute error {result['max_gradient_absolute_error']:.3g}.
- Adding optional 8 Å z padding changes no measured voxel: maximum difference {paderror:.3g}. No axial wrapping is used in imaging.
- The aligned reference structure spans z={pos[:,2].min():.3f}–{pos[:,2].max():.3f} Å, leaving {gap:.3f} Å between outermost atom centers across the periodic z boundary. This is a geometric check, not energy convergence.
- All {len(deletion)} synthetic single-Se vacancies were identified among all Se deletion candidates at fixed coordinates; selected soft occupancies were recovered with maximum error {occerr:.3g}.
- The supplied XSF contains {totalW} W and {totalSe} Se; it is Se-rich relative to 1:2. The current quadratic composition weight 10 gives an incremental objective change of {comp_removed:.6g} for removing one Se. Correct synthetic deletion beats the intact candidate with image+composition in {result['synthetic_deletions_beating_intact_with_composition']}/{len(deletion)} probes, using image weight 100 and normalized Huber delta 0.2.

The vacuum test exposes a separate issue: an atom moved into the vacuum is itself strongly attenuated by the fitted envelope. For the supplied Se-rich reference, removing one Se reduces the composition penalty, and moving it into vacuum wins in {result['vacuum_hiding_beats_true_deletion_cases']}/{len(edge)} probes. Near nominal W116 Se232 stoichiometry, however, the one-Se deletion cost is only {nominal_removed:.6g}, yet the residual image cost of a vacuum-displaced Se is smaller in {result['nominal_stoichiometric_vacuum_hiding_cases']}/{len(edge)} probes. The latter comparison uses the same isolated vacancy residual and a nominal stoichiometric count reference; it is a diagnostic of image/composition competition, not an energy evaluation of a new structure. This does not include the energy cost of an isolated atom; FairChem could oppose it. It does show that the image term alone is not a reliable vacuum exclusion mechanism. Optional z confinement remains useful as a diagnostic safeguard and should be judged in the eventual full-energy test.

The 30-slice window can supply the geometric z vacuum without additional padding. The cutoff envelope stays fixed in experimental coordinates. Atomic positions/occupancies remain differentiable (piecewise at interpolation knots). The apparent z thickness of a blurred blob does not determine the physical vacuum separation of the atomic structure.

![Vacuum-atom visibility and penalty](vacuum_visibility.png)

Detailed gradient, synthetic-vacancy, real-data deletion and vacuum-scan CSVs are saved alongside `results.json`. Real-data deletion changes are screening diagnostics, not vacancy identifications. No physical energy, atomic relaxation, or complete GA was evaluated; FairChem/ASE are absent from this test environment.

Reproduce: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /data2/jt577/ws11_venv/bin/python run_cpu_tests.py`.
'''
 (HERE/'REPORT.md').write_text(report)
 log('CPU tests complete');print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','script_sha256']},indent=2))
if __name__=='__main__':main()
