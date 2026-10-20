"""Scientific and scheduler checks; surrogate calculator is restricted to tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
import torch
from ase.calculators.calculator import Calculator, all_changes
from gaussian_image_model import GaussianImageModel

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('discrete_wse2',HERE/'discrete_09.30.26_wse2.py')
ga=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)

class Harmonic(Calculator):
    implemented_properties=['energy','forces']
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        delta=atoms.positions-np.array([6.,6.,4.])
        self.results=dict(energy=float(.03*np.sum(delta**2)),forces=-.06*delta)

class DiscreteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        z,y,x=np.mgrid[:6,:24,:24]
        volume=(.01+.8*np.exp(-((x-8)**2+(y-9)**2)/10-(z-2)**2/7)+.7*np.exp(-((x-16)**2+(y-16)**2)/12-(z-3)**2/6)).astype(np.float32)
        path=cls.root/'sample.tif';tifffile.imwrite(path,volume,metadata={'axes':'ZYX'})
        c=ga.load_config(ga.DEFAULT_CONFIG)
        c['paths']['output_dir']=str(cls.root/'base')
        c['experimental'].update(tiff_file=str(path),crop_xy_size_source_pixels=24,target_xy_pixels=24,target_z_slices=6,source_voxel_size_angstrom='1, .5, .5')
        c['system']['vacuum_padding_angstrom']='2, 2, 2'
        c['sample_box']['z_max_angstrom']=6
        c['feasible_region'].update(candidate_site_density_per_angstrom3=.04,cylinder_lateral_radius_angstrom=.75,cylinder_z_radius_angstrom=1)
        c['hard_atoms']['minimum_distance_angstrom']=.65
        c['runtime'].update(device='cpu',cpu_threads=2)
        c['relaxation'].update(max_iterations=4,cost_plateau_tolerance=0)
        c['initialization']['seed']=123
        cls.config=c
        cls.problem=ga.DiscreteProblem(c,device='cpu',calculator=Harmonic(),references=[-.4,-.2])
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_random_is_hard_and_reproducible(self):
        a=ga.random_candidate(self.problem,np.random.default_rng(31));b=ga.random_candidate(self.problem,np.random.default_rng(31))
        self.assertEqual(a['symbols'],b['symbols']);np.testing.assert_array_equal(a['positions'],b['positions'])
        self.assertTrue(set(a['symbols']) <= {'W','Se'})
        for i,pos in enumerate(a['positions']):self.assertTrue(ga.separation_ok(pos,a['positions'][:i],self.problem.lengths,.65))
        self.assertFalse(np.array_equal(a['positions'],ga.random_candidate(self.problem,np.random.default_rng(32))['positions']))
    def test_preprocessing_roi_and_normalization(self):
        ex=self.problem.experimental;self.assertEqual(int(ex['image_loss_mask'].sum()),6*24*24)
        linear=np.maximum(ex['source_crop']-np.percentile(ex['source_crop'],1),0)
        np.testing.assert_allclose(ex['sample_target_image'],linear/np.percentile(linear,99.5),atol=1e-6)
        self.assertEqual(ex['target_shape_zyx'],(10,32,32));self.assertTrue((ex['feasible_mask']<=ex['image_loss_mask']).all())
    def test_crossover_sides_and_species(self):
        a=ga.candidate(['W']*3,[[3,4,4],[6,4,4],[10,4,4]])
        b=ga.candidate(['Se']*3,[[3,9,4],[6,9,4],[10,9,4]])
        child=ga.crossover(self.problem,a,b,np.random.default_rng(13));info=child['crossover_plane']
        for s,pos in zip(child['symbols'],child['positions']):
            source=a if s=='W' else b
            self.assertTrue(any(np.allclose(pos,p) for p in source['positions']))
            sign=(pos-info['origin_xyz_A'])@np.array(info['normal_xyz'])
            self.assertTrue(sign>=0 if s=='W' else sign<0)
    def test_mutation_operators_and_probability(self):
        p=copy.copy(self.problem);p.config=copy.deepcopy(self.config)
        base=ga.candidate(['W','Se','Se'],[[4,4,4],[7,4,4],[10,4,4]])
        for operation in ('rattle','relocate','insert','delete','transmute'):
            m=p.config['mutation'];m['probability']=1
            for name in ('rattle','relocate','insert','delete','transmute'):m[name+'_weight']=int(name==operation)
            child=ga.mutate(p,base,np.random.default_rng(3));self.assertEqual(child['mutation'],operation)
            self.assertEqual(len(child['symbols']),4 if operation=='insert' else 2 if operation=='delete' else 3)
        p.config['mutation']['probability']=.3;rng=np.random.default_rng(7)
        attempts=sum(ga.mutate(p,base,rng)['mutation_attempted'] for _ in range(1000));self.assertTrue(250<attempts<350,attempts)
        self.assertEqual(base['symbols'],['W','Se','Se']);np.testing.assert_array_equal(base['positions'],[[4,4,4],[7,4,4],[10,4,4]])
    def test_inapplicable_mutation_recorded(self):
        p=copy.copy(self.problem);p.config=copy.deepcopy(self.config)
        p.config['mutation'].update(probability=1,rattle_weight=0,relocate_weight=0,insert_weight=0,delete_weight=1,transmute_weight=0)
        child=ga.mutate(p,ga.candidate(['W'],[[4,4,4]]),np.random.default_rng(2))
        self.assertEqual(child['mutation'],'none_inapplicable');self.assertTrue(child['mutation_attempted'])
    def test_objective_and_finite_difference_gradient(self):
        p=copy.copy(self.problem);p.dtype=torch.float64;p.target=p.target.to(p.dtype)
        p.gaussian_renderer=GaussianImageModel(4,.08,p.gaussian_element_fits,p.species,p.target_shape_zyx,p.target_voxel_size_A,p.vacuum_position_shift,p.device,p.dtype)
        pos=np.array([[3.17,4.23,2.13],[8.31,8.14,5.29],[7.11,8.26,6.19]]);symbols=['W','Se','Se']
        value,grad,terms=p.objective(symbols,pos)
        self.assertAlmostEqual(terms['formation_energy_per_atom'],(.03*np.sum((pos-[6,6,4])**2)+.8)/3)
        self.assertAlmostEqual(terms['composition_penalty'],0)
        self.assertAlmostEqual(value,terms['formation_energy_per_atom']+terms['image_term']+terms['sample_box_term'])
        for atom,axis in [(0,0),(0,2),(1,1),(2,2)]:
            h=1e-5;plus,minus=pos.copy(),pos.copy();plus[atom,axis]+=h;minus[atom,axis]-=h
            numeric=(p.objective(symbols,plus)[0]-p.objective(symbols,minus)[0])/(2*h)
            self.assertAlmostEqual(numeric,grad[atom,axis],places=6)
    def test_box_gradient_sign(self):
        terms,grad=self.problem.image_and_box(['W','Se'],np.array([[1.17,4.23,1.31],[8.31,8.14,8.29]]))
        self.assertGreater(terms['sample_box_term'],0);self.assertLess(grad[0,0],0);self.assertGreater(grad[1,2],0)
    def test_relaxation_preserves_atoms_and_improves(self):
        child=ga.candidate(['W','Se','Se'],[[3.17,4.23,3.13],[8.31,8.14,5.29],[7.11,8.26,6.19]])
        initial=self.problem.objective(child['symbols'],child['positions'])[0]
        result=ga.relax_candidate(self.problem,child)
        self.assertLessEqual(result['score'],initial+1e-6);self.assertEqual(result['structure']['symbols'],child['symbols'])
        self.assertGreater(result['compute_metrics']['model_energy_evaluations'],0)
    def test_generations_selection_and_elites(self):
        p=copy.copy(self.problem);p.config=copy.deepcopy(self.config)
        p.config['genetic'].update(population_size=10,elite_fraction=.2,random_fraction=.3,crossover_fraction=.5)
        p.config['mutation']['probability']=1
        tasks=ga.plan_generation(p,[],0,17);self.assertEqual(len(tasks),10)
        self.assertTrue(all(t['kind']=='random_immigrant' for t in tasks))
        records=[dict(score=float(i),generation=0,member_index=i,structure=t['candidate']) for i,t in enumerate(tasks)]
        tasks=ga.plan_generation(p,records,1,17);self.assertEqual(len(tasks),10)
        self.assertEqual(sum(t['kind']=='elite' for t in tasks),2);self.assertEqual(sum(t['kind']=='random_immigrant' for t in tasks),3)
        for t in tasks[5:]:
            self.assertEqual(len(t['parent_refs']),2);self.assertNotEqual(t['parent_refs'][0]['member_index'],t['parent_refs'][1]['member_index'])
            self.assertTrue(t['candidate']['mutation_attempted'])
        self.assertEqual(ga.generation_counts(ga.load_config(ga.DEFAULT_CONFIG)),(5,13,32))
    def test_import_top_n_no_dedup(self):
        p=copy.copy(self.problem);p.config=copy.deepcopy(self.config);path=self.root/'import.json'
        s=dict(symbols=['W','Se'],cart_positions_angstrom=[[4,4,4],[7,7,4]],cell_angstrom=p.cell.tolist())
        ga.atomic_json(path,dict(structures=[dict(score=v,structure=s) for v in [3,1,1,2]]))
        p.config['paths']['import_structures_json']=str(path);p.config['genetic']['import_parent_count']=2
        imports=ga.import_candidates(p);self.assertEqual([r[1]['source_score'] for r in imports],[1,1])
        tasks=ga.plan_generation(p,[],0,19,imports);self.assertEqual(sum(t['kind']=='imported' for t in tasks),2);self.assertEqual(len(tasks),50)
    def test_end_to_end_and_resume(self):
        c=copy.deepcopy(self.config);c['paths']['output_dir']=str(self.root/'run')
        c['genetic'].update(population_size=4,generations=2,elite_fraction=.25,random_fraction=.25,crossover_fraction=.5)
        c['relaxation']['max_iterations']=2
        refs=self.root/'refs.json';ga.atomic_json(refs,dict(model_name=c['model']['model_name'],task_name=c['model']['task_name'],energy_per_atom_eV={'W':-.4,'Se':-.2}))
        c['model']['reference_energies_json']=str(refs)
        def install(p):
            if p.calc is None:p.install_calculator(Harmonic())
        with patch.object(ga.DiscreteProblem,'ensure_calculator',install):
            first=ga.run(c);again=ga.run(c,resume=True);self.assertEqual(first,again)
            c['genetic']['generations']=3;extended=ga.run(c,resume=True)
        self.assertEqual(len(first),8);self.assertEqual(len(extended),12)
        out=Path(c['paths']['output_dir'])
        for name in ['best_structure.xsf','structures.json','best_predicted_slices.tif','discrete_optimization_live_scores_by_generation.png']:self.assertTrue((out/name).is_file(),name)
        self.assertEqual(json.loads((out/'ga_checkpoint.json').read_text())['status'],'complete')
        self.assertEqual([ga.COLORS[k] for k in ['mutated_child','crossover_child','random_immigrant']],['purple','green','red'])
    def test_config_validation(self):
        import yaml
        path=self.root/'bad.yaml';path.write_text(yaml.safe_dump({'syntropization':{'beta0':1}}))
        with self.assertRaisesRegex(ValueError,'Unknown configuration'):ga.load_config(path)
        c=copy.deepcopy(self.config);c['genetic']['crossover_fraction']=.1
        with self.assertRaisesRegex(ValueError,'sum to one'):ga.validate_config(c)

    def test_yaml_api_key_and_setup_retry(self):
        c=copy.deepcopy(self.config)
        c['paths']['output_dir']=str(self.root/'setup_retry')
        c['model']['mp_api_key']='test-placeholder-key'
        with patch.dict(ga.os.environ,{'MP_API_KEY':'environment-placeholder'}):
            self.assertEqual(ga.reference_api_key(c),'test-placeholder-key')
            blank=copy.deepcopy(c);blank['model']['mp_api_key']=''
            self.assertEqual(ga.reference_api_key(blank),'environment-placeholder')
        self.assertNotIn('mp_api_key',ga.public_config(c)['model'])
        signature=ga.fingerprint(c)
        c['model']['mp_api_key']='changed-placeholder-key'
        self.assertEqual(ga.fingerprint(c),signature)
        # Both calls reach setup, even without --resume and after a code change.
        with patch.object(ga.DiscreteProblem,'ensure_references',side_effect=RuntimeError('reference setup test')):
            with self.assertRaisesRegex(RuntimeError,'reference setup test'):ga.run(c)
            out=Path(c['paths']['output_dir'])
            saved=json.loads((out/'ga_checkpoint.json').read_text());saved['fingerprint']='previous-code'
            ga.atomic_json(out/'ga_checkpoint.json',saved)
            with self.assertRaisesRegex(RuntimeError,'reference setup test'):ga.run(c)
        text=(out/'run_parameters.json').read_text()
        self.assertNotIn('placeholder-key',text)
        self.assertNotIn('mp_api_key',text)

    def test_invalid_disconnected_member_does_not_abort(self):
        task=dict(candidate=ga.candidate(['W'],[[4,4,4]]),kind='random_immigrant',
                  parent_refs=[],generation=99,member_index=0,seed=12)
        with patch.object(ga,'relax_candidate',side_effect=ValueError('No edges found in input system')):
            result=ga.evaluate_task(task,self.problem)
        self.assertFalse(result['valid'])
        self.assertEqual(result['score'],1e30)
        task['member_index']=1
        with patch.object(ga,'relax_candidate',side_effect=ValueError('unexpected calculator bug')):
            with self.assertRaisesRegex(ValueError,'unexpected calculator bug'):
                ga.evaluate_task(task,self.problem)

    def test_member_progress_restarts_with_saved_compute(self):
        child=ga.candidate(['W','Se'],[[4.17,4.23,3.13],[8.31,8.14,5.29]])
        path=self.root/'progress.json'
        ga.atomic_json(path,dict(symbols=child['symbols'],positions=child['positions'],iterations=4,
            compute_metrics=dict(model_energy_evaluations=10,model_evaluation_seconds=1.5,
                                 objective_evaluations=12,member_wall_seconds=2.0)))
        result=ga.relax_candidate(self.problem,child,path)
        self.assertGreaterEqual(result['compute_metrics']['model_energy_evaluations'],10)
        self.assertGreaterEqual(result['compute_metrics']['objective_evaluations'],14)
        self.assertEqual(result['relaxation']['iterations'],4)

if __name__=='__main__':unittest.main(verbosity=2)
