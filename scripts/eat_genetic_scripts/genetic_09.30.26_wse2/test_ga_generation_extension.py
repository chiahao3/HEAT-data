"""Checkpoint compatibility and generation-limit extension; no potential calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_gaussian_09_30 import ga

OLD_SCRIPT = '0f94627187c7e0ad7c34e4461829b3bf7b9c0c5985725f76509cc72dacf30320'


def old_fingerprint(args):
    # Independent reproduction of the previous production fingerprint format.
    excluded = {'slurm_worker_mode','slurm_worker_task','slurm_worker_result',
                'slurm_worker_resource_log','ga_resuming_from_checkpoint',
                '_ga_checkpoint_fingerprint_override','worker_health_check_interval_seconds'}
    payload = dict(resolved_parameters={k:ga.json_safe_value(v) for k,v in sorted(vars(args).items()) if k not in excluded},
        scientific_assets_sha256={str(p):ga.file_sha256(p) for p in (
            Path(args.gaussian_element_fit_csv),ga.SCRIPT_DIR/'gaussian_image_model.py',
            ga.SCRIPT_DIR/'minimizer.py',ga.resolve_script_relative_path(args.experimental_tiff))},
        optimizer_script_sha256=OLD_SCRIPT)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for name in ('gaussian_image_model.py','minimizer.py','fits.csv','sample.tif'):
            (root/name).write_text('unchanged fixture '+name)
        self.patch = patch.object(ga,'SCRIPT_DIR',root)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.args = SimpleNamespace(output_dir=str(root/'run'),config_path=str(root/'input.yaml'),
            ga_generations=100,ga_population_size=2,gaussian_element_fit_csv=str(root/'fits.csv'),
            experimental_tiff=str(root/'sample.tif'),ga_import_structures_json='',image_penalty=10.,
            slurm_worker_mode=False)
        self.problem = SimpleNamespace(output_dir=Path(self.args.output_dir),device='cpu',dtype=torch.float64)
        self.generations = [[dict(generation=g,member_index=m,score=-g-m/10,
            y=torch.tensor([1.,0.,g,m,3.]),y_rounded=torch.tensor([1.,0.,g,m,3.])) for m in range(2)] for g in range(100)]

    def checkpoint(self, legacy=True, current=None):
        args = copy.copy(self.args)
        if legacy:
            args._ga_checkpoint_fingerprint_override = old_fingerprint(self.args)
        ga.write_ga_checkpoint(self.problem,args,123,self.generations,100,current or [],status='complete' if not current else 'running')
        path=ga.ga_checkpoint_path(self.problem)
        payload=json.loads(path.read_text())
        if legacy:
            payload.pop('fingerprint_generations_requested')
            ga.write_json_atomic(path,payload)
        return payload

    def test_old_checkpoint_extend_and_restore_all_completed(self):
        old=self.checkpoint()
        before=ga.ga_generation_checkpoint_path(self.problem,99).read_bytes()
        self.args.ga_generations=200
        restored=ga.restore_ga_checkpoint(self.problem,self.args)
        self.assertEqual(restored['current_generation'],100)
        self.assertEqual(len(restored['all_generations']),100)
        self.assertEqual(restored['all_generations'][-1][1]['score'],-99.1)
        self.assertEqual(list(range(restored['current_generation'],self.args.ga_generations)),list(range(100,200)))
        self.assertEqual(ga.ga_checkpoint_fingerprint(self.args),old['config_fingerprint'])
        ga.write_ga_checkpoint(self.problem,self.args,123,restored['all_generations'],100,[],status='running')
        head=ga.read_json_file(ga.ga_checkpoint_path(self.problem))
        self.assertEqual(head['generations_requested'],200)
        self.assertEqual(head['fingerprint_generations_requested'],100)
        self.assertEqual(before,ga.ga_generation_checkpoint_path(self.problem,99).read_bytes())
        # Restart after parameters/checkpoint were already updated to 200.
        restarted=copy.copy(self.args)
        del restarted._ga_checkpoint_fingerprint_override
        del restarted._ga_checkpoint_generation_origin
        self.assertEqual(ga.checkpoint_seed_for_args(restarted),123)
        restarted.ga_generations=300
        self.assertEqual(ga.read_ga_checkpoint_for_args(restarted)['config_fingerprint'],old['config_fingerprint'])

    def test_new_code_run_can_be_extended(self):
        old=self.checkpoint(legacy=False)
        self.args.ga_generations=200
        ga.read_ga_checkpoint_for_args(self.args)
        self.assertEqual(ga.ga_checkpoint_fingerprint(self.args),old['config_fingerprint'])

    def test_unchanged_limit_and_partial_generation(self):
        # A partial generation must be below the previously requested limit.
        self.args.ga_generations=101
        partial=[dict(generation=100,member_index=0,score=-100.,y=torch.ones(5),y_rounded=torch.ones(5))]
        self.checkpoint(current=partial)
        self.args.ga_generations=200
        result=ga.restore_ga_checkpoint(self.problem,self.args)
        self.assertEqual(len(result['current_generation_results']),1)
        self.assertEqual(result['current_generation_results'][0]['member_index'],0)

    def test_old_completed_run_still_accepts_100(self):
        self.checkpoint()
        self.assertEqual(ga.read_ga_checkpoint_for_args(self.args)['completed_generation_count'],100)

    def test_reject_changed_science_even_after_prior_acceptance(self):
        self.checkpoint()
        self.args.ga_generations=200
        ga.read_ga_checkpoint_for_args(self.args)
        self.args.image_penalty=20.
        with self.assertRaisesRegex(RuntimeError,'configuration/code mismatch'):
            ga.read_ga_checkpoint_for_args(self.args)

    def test_reject_changed_assets_and_lower_limits(self):
        self.checkpoint()
        self.args.ga_generations=99
        with self.assertRaisesRegex(RuntimeError,'Cannot decrease'):
            ga.read_ga_checkpoint_for_args(self.args)
        self.args.ga_generations=200
        Path(self.args.experimental_tiff).write_text('changed TIFF')
        with self.assertRaisesRegex(RuntimeError,'configuration/code mismatch'):
            ga.read_ga_checkpoint_for_args(self.args)

    def test_reject_unknown_code_or_corrupted_generation(self):
        self.checkpoint()
        self.args.ga_generations=200
        with patch.object(ga,'CHECKPOINT_COMPATIBLE_SCRIPT_SHA256',()):
            with self.assertRaisesRegex(RuntimeError,'configuration/code mismatch'):
                ga.read_ga_checkpoint_for_args(self.args)
        path=ga.ga_generation_checkpoint_path(self.problem,99)
        record=ga.read_json_file(path);record['results'][0]['score']=12345
        ga.write_json_atomic(path,record)
        with self.assertRaisesRegex(RuntimeError,'integrity check'):
            ga.restore_ga_checkpoint(self.problem,self.args)

    def test_slurm_worker_fingerprint_matches_controller(self):
        old=self.checkpoint()
        worker=copy.copy(self.args);worker.ga_generations=200;worker.slurm_worker_mode=True
        ga.read_ga_checkpoint_for_args(worker)
        self.assertEqual(ga.ga_checkpoint_fingerprint(worker),old['config_fingerprint'])

if __name__=='__main__':
    unittest.main(verbosity=2)
