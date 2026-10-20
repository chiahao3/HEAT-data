"""Exercise the real GA generation loop and import loader without FairChem."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from test_gaussian_09_30 import ga


class ImportSeedingTests(unittest.TestCase):
    def run_case(self, population=6, n=2, prefit=True, resume=False, imports=True):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args = ga.get_settings()
            args.output_dir = str(root)
            args.ga_population_size = population
            args.ga_generations = 2
            args.ga_parent_pool_size = population
            args.ga_import_parent_count = n
            args.ga_execution_backend = 'local'
            args.ga_parallel_workers = 1
            args.ga_live_plot_enable = False
            args.ga_mutation_fraction = 0.0
            args.random_immigrant_image_prefit_enable = prefit
            source = root/'imports.json'
            # Unsorted scores and identical states: select by score, keep duplicates.
            source.write_text(json.dumps({'structures': [
                dict(score=score, continuous_state_vector=[1,0,1,2,3],
                     assigned_site_region_indices=[0]) for score in [-5,-10,-10]
            ]}))
            args.ga_import_structures_json = str(source) if imports else ''
            problem = MagicMock()
            problem.output_dir = root
            problem.device = 'cpu'
            problem.dtype = torch.float64
            problem.n_species = 2
            problem.image_loss_voxel_count = 12
            problem.initial_site_layout.return_value = np.array([0])
            problem.rounded_state.side_effect = lambda y: y.clone()
            state = torch.tensor([1,0,1,2,3], dtype=torch.float64)
            checkpoint = None
            if resume:
                imported = ga.load_imported_parent_results(problem,args)
                saved = ga.copy_imported_parent_to_generation(imported[0],0,0)
                checkpoint = dict(seed=123,current_generation=0,all_generations=[],
                                  current_generation_results=[saved])
            evaluations, pools, completed = [], [], []

            def evaluate(p,y,layout,a,seed,generation,index,kind,**kwargs):
                evaluations.append((generation,index,kind))
                return dict(generation=generation, member_index=index, kind=kind,
                            score=10+generation+index, y=y.clone(), y_rounded=y.clone(),
                            assigned_blob_indices=list(layout), history=[], stage_summaries=[])

            def record(p,a,seed,db,log,all_generations,generation_results,result,**kwargs):
                generation_results.append(result)

            def choose(pool,*args):
                pools.append(list(pool))
                return pool[0],pool[-1]

            replacements = {
                'restore_ga_checkpoint': MagicMock(return_value=checkpoint),
                'ga_log_path': MagicMock(return_value=root/'ga.log'),
                'structures_db_path': MagicMock(return_value=root/'structures.json'),
                'running_log_path_from_args': MagicMock(return_value=root/'running.log'),
                'random_immigrant_image_prefit_site_layout': MagicMock(return_value=np.array([0])),
                'make_initial_state': MagicMock(return_value=state),
                'evaluate_candidate_member': evaluate,
                'record_completed_ga_result': record,
                'choose_tournament_pair': choose,
                'crossover_state': MagicMock(return_value=(state,np.array([0]))),
                'write_ga_summary': lambda p,a,log,generations,best: completed.extend(generations),
            }
            for name in ('append_running_log','append_ga_log','reset_ga_log','terminal_print',
                         'write_ga_checkpoint','cleanup_completed_member_checkpoints','ga_member_log_record'):
                replacements[name] = MagicMock()
            for name,value in replacements.items():
                stack.enter_context(patch.object(ga,name,value))
            with redirect_stdout(io.StringIO()):
                ga.run_genetic_algorithm(problem,args,123)
            expected_imports = min(n,3,population) if imports else 0
            self.assertEqual(Counter(r['kind'] for r in completed[0]),
                             Counter(dict(imported_seed=expected_imports,
                                          random_immigrant=population-expected_imports)))
            imported_scores = sorted(r['score'] for r in completed[0] if r['kind']=='imported_seed')
            self.assertEqual(imported_scores, sorted([-5,-10,-10])[:expected_imports])
            self.assertEqual(len([e for e in evaluations if e[0]==0]),population-expected_imports)
            counts = ga.allocate_generation_counts(population,args)
            self.assertEqual(Counter(r['kind'] for r in completed[1]),
                             Counter(dict(elite_copy=counts['elite'],random_immigrant=counts['random'],
                                          crossover_child=counts['crossover'])))
            for pool in pools:
                self.assertEqual(len(pool),population)
                self.assertTrue(all(r['generation']==0 for r in pool))
            for generation in completed:
                self.assertEqual(sorted(r['member_index'] for r in generation),list(range(population)))
            expected_random_calls = population-expected_imports+counts['random']
            self.assertEqual(replacements['random_immigrant_image_prefit_site_layout'].call_count,
                             expected_random_calls if prefit else 0)

    def test_imports_seed_first_generation_with_either_constructor(self):
        for prefit in (True,False):
            with self.subTest(prefit=prefit):
                self.run_case(prefit=prefit)

    def test_configurable_population_import_limits_and_empty_path(self):
        self.run_case(population=7,n=1)
        self.run_case(population=3,n=5)
        self.run_case(population=6,imports=False)

    def test_resume_partial_generation_zero_preserves_import_slots(self):
        self.run_case(resume=True)


if __name__ == '__main__':
    unittest.main()
