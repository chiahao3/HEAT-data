"""Test real GA orchestration with lightweight stand-ins for expensive workers."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from test_gaussian_09_30 import ga


class MutationTests(unittest.TestCase):
    def run_ga(self, backend='local', parallel=1, checkpoint=None):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args = ga.get_settings()
            args.output_dir = str(root)
            args.ga_population_size = args.ga_parent_pool_size = 8
            args.ga_generations = 2
            args.ga_elite_fraction = args.ga_random_fraction = .25
            args.ga_crossover_fraction = args.ga_mutation_fraction = .25
            args.ga_import_structures_json = ''
            args.ga_execution_backend = backend
            args.ga_parallel_workers = parallel
            args.ga_live_plot_enable = False
            problem = MagicMock()
            problem.output_dir = root
            problem.device = 'cpu'
            problem.dtype = torch.float64
            problem.n_species = 2
            problem.image_loss_voxel_count = 12
            problem._slurm_task_phase = ''
            problem.initial_site_layout.return_value = np.array([0])
            state = torch.tensor([1.,0.,1.,2.,3.],dtype=torch.float64)
            pending = queue.Queue()
            evaluations, batches, checkpoints, final, creations = [], [], [], [], []
            recorded = list(checkpoint['current_generation_results']) if checkpoint else []

            def evaluate(p,y,layout,a,seed,g,index,kind,**kwargs):
                evaluations.append((g,index,kind))
                # Optimized state differs from initial input; mutation must use this.
                relaxed = y + .001*(1+g*100+index)
                return dict(generation=g,member_index=index,kind=kind,score=g+index,
                            y=relaxed,y_rounded=relaxed.clone(),assigned_blob_indices=list(layout),
                            history=[],stage_summaries=[],parent_refs=kwargs.get('parent_refs',[]))

            def record(p,a,seed,db,log,all_g,current,result,**kwargs):
                current.append(result)
                recorded.append(result)
                checkpoints.append(copy.deepcopy(dict(seed=123,current_generation=result['generation'],
                                                      all_generations=all_g,current_generation_results=current)))

            def crossover(p,ay,al,by,bl,a,**kwargs):
                creations.append((ay.clone(),by.clone(),list(recorded)))
                return ay+by,np.array([0])

            def complete_tasks(p,a,seed,db,log,all_g,current,tasks):
                batches.append((tasks[0]['generation'],p._slurm_task_phase,[t['kind'] for t in tasks]))
                # Deliberately reverse worker completion order.
                for task in reversed(tasks):
                    r=evaluate(p,torch.tensor(task['y0'],dtype=p.dtype),task['assigned_blob_indices'],
                               a,task['seed'],task['generation'],task['member_index'],task['kind'],
                               parent_refs=task['parent_refs'])
                    record(p,a,seed,db,log,all_g,current,r)

            def parallel_collect(p,a,seed,db,log,all_g,current,rq,workers,count):
                self.assertEqual(pending.qsize(),count)
                complete_tasks(p,a,seed,db,log,all_g,current,[pending.get() for _ in range(count)])

            def slurm_collect(p,a,seed,db,log,all_g,current,g,tasks,job):
                complete_tasks(p,a,seed,db,log,all_g,current,tasks)

            replacements = dict(
                restore_ga_checkpoint=MagicMock(return_value=checkpoint),
                ga_log_path=MagicMock(return_value=root/'ga.log'),
                structures_db_path=MagicMock(return_value=root/'structures.json'),
                running_log_path_from_args=MagicMock(return_value=root/'running.log'),
                random_immigrant_image_prefit_site_layout=MagicMock(return_value=np.array([0])),
                make_initial_state=MagicMock(return_value=state),
                evaluate_candidate_member=evaluate,record_completed_ga_result=record,
                crossover_state=crossover,
                start_ga_workers=MagicMock(return_value=(pending,queue.Queue(),[MagicMock()])),
                collect_parallel_ga_tasks=parallel_collect,
                collect_slurm_ga_tasks=slurm_collect,
                submit_slurm_generation=MagicMock(return_value=('fake-job',None)),
                write_ga_summary=lambda p,a,log,generations,best: final.extend(generations))
            for name in ('append_running_log','append_ga_log','reset_ga_log','terminal_print',
                         'write_ga_checkpoint','cleanup_completed_member_checkpoints','ga_member_log_record',
                         'move_problem_tensor_state_to_device','stop_ga_workers','write_slurm_task_files'):
                replacements[name]=MagicMock()
            for name,value in replacements.items():
                stack.enter_context(patch.object(ga,name,value))
            with redirect_stdout(io.StringIO()):
                ga.run_genetic_algorithm(problem,args,123)
            self.assertEqual(problem._slurm_task_phase,'')
            self.assertEqual(Counter(r['kind'] for r in final[0]),{'random_immigrant':8})
            self.assertEqual(Counter(r['kind'] for r in final[1]),
                             {'elite_copy':2,'random_immigrant':2,'crossover_child':2,'mutated_child':2})
            for g in final:
                self.assertEqual(sorted(r['member_index'] for r in g),list(range(8)))
            by_ref={ga.parent_ref(r):r for g in final for r in g}
            for child in final[1]:
                if child['kind'] not in ('crossover_child','mutated_child'):
                    continue
                pa,pb=[by_ref[ref] for ref in child['parent_refs']]
                self.assertEqual(pa['generation'],0)
                self.assertEqual(pb['generation'],1 if child['kind']=='mutated_child' else 0)
                if child['kind']=='mutated_child':
                    self.assertEqual(pb['kind'],'random_immigrant')
                expected = pa['y']+pb['y']+.001*(101+child['member_index'])
                torch.testing.assert_close(child['y'],expected,rtol=0,atol=0)
            # Every offspring construction follows BOTH current donors' completion.
            for _,_,available in creations:
                self.assertEqual({r['member_index'] for r in available
                                  if r['generation']==1 and r['kind']=='random_immigrant'},{2,3})
            if not checkpoint and (backend=='slurm' or parallel>1):
                self.assertEqual([len(b[2]) for b in batches],[8,2,4])
                if backend=='slurm':
                    self.assertEqual([b[1] for b in batches],['','random_donors','offspring'])
            return final,checkpoints,evaluations

    def test_serial_parallel_and_slurm_have_identical_mutation_parents(self):
        serial,_,_=self.run_ga()
        for kwargs in ({'parallel':2},{'backend':'slurm'}):
            other,_,_=self.run_ga(**kwargs)
            for a,b in zip(serial[1],other[1]):
                self.assertEqual(a['parent_refs'],b['parent_refs'])
                torch.testing.assert_close(a['y'],b['y'],rtol=0,atol=0)

    def test_resume_during_donors_and_after_one_mutation(self):
        original,checkpoints,_=self.run_ga()
        for index in (2,6):
            cp=next(c for c in checkpoints if c['current_generation']==1
                    and c['current_generation_results'][-1]['member_index']==index)
            for kwargs in ({},{'parallel':2},{'backend':'slurm'}):
                resumed,_,evaluated=self.run_ga(checkpoint=copy.deepcopy(cp),**kwargs)
                saved={r['member_index'] for r in cp['current_generation_results']}
                self.assertTrue(all(g==1 and i not in saved for g,i,k in evaluated))
                for a,b in zip(original[1],resumed[1]):
                    self.assertEqual(a['parent_refs'],b['parent_refs'])
                    torch.testing.assert_close(a['y'],b['y'],rtol=0,atol=0)

    def test_allocation_and_missing_donors(self):
        args=ga.get_settings()
        # Test a fixed allocation, independently of the user's active run YAML.
        args.ga_elite_fraction=.10
        args.ga_random_fraction=.25
        args.ga_crossover_fraction=.45
        args.ga_mutation_fraction=.20
        counts=ga.allocate_generation_counts(50,args)
        self.assertEqual(counts,{'elite':5,'random':13,'crossover':22,'mutation':10})
        args.ga_random_fraction=0
        with self.assertRaisesRegex(ValueError,'requires a positive random_fraction'):
            ga.allocate_generation_counts(50,args)
        args.ga_random_fraction=.25
        with self.assertRaisesRegex(ValueError,'too small'):
            ga.allocate_generation_counts(3,args)
        args.ga_mutation_fraction=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            ga.allocate_generation_counts(50,args)
        with self.assertRaisesRegex(RuntimeError,'must finish'):
            ga.current_mutation_donors([],1,2,4)

    def test_slurm_phases_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            p=SimpleNamespace(output_dir=Path(directory),_slurm_task_phase='random_donors')
            first=ga.write_slurm_task_files(p,1,[{'member_index':2}])
            p._slurm_task_phase='offspring'
            second=ga.write_slurm_task_files(p,1,[{'member_index':6}])
            self.assertNotEqual(first['tasks'],second['tasks'])
            self.assertTrue((first['tasks']/'task_00001.json').exists())
            self.assertTrue((second['tasks']/'task_00001.json').exists())

    def test_live_plot_colors(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.axes import Axes
        colors={}
        scatter=Axes.scatter
        def capture(ax,*args,**kwargs):
            colors[kwargs.get('label')]=kwargs.get('c')
            return scatter(ax,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            args=ga.get_settings();args.ga_live_plot_enable=True;args.ga_live_plot_file=''
            problem=SimpleNamespace(output_dir=Path(directory))
            members=[dict(generation=1,member_index=i,kind=k,score=-i)
                     for i,k in enumerate(('random_immigrant','crossover_child','mutated_child'))]
            with patch.object(Axes,'scatter',capture):
                ga.update_live_ga_plot(problem,args,[members])
            self.assertEqual(colors['random immigrant'],'red')
            self.assertEqual(colors['crossover'],'green')
            self.assertEqual(colors['mutation'],'purple')
            self.assertTrue(ga.live_plot_path(problem,args).exists())


if __name__=='__main__':
    unittest.main()
