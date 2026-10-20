"""Gaussian depth and production integration checks; no FairChem evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/wse2_gaussian_tests')
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import tifffile
from gaussian_image_model import GaussianImageModel

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wse2_gaussian_ga', ROOT/'genetic_09.30.26_wse2.py')
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)
torch.set_num_threads(2)
FITS = ga.load_double_gaussian_element_fits(ROOT/'element_double_gaussian_fits.csv')[1]


def model(nz=30, padding=0, sigma=4):
    return GaussianImageModel(sigma, .08, FITS, ['W', 'Se'],
                              (nz, 48, 48), (1, .152564, .152564), (0, 0, padding),
                              'cpu', torch.float64)


class GaussianTests(unittest.TestCase):
    def test_isolated_w_peak_and_overlapping_atoms(self):
        for sigma in (2., 4., 6.):
            m = model(sigma=sigma)
            for z in (7., 15., 23.):
                pos = torch.tensor([[10*.152564, 12*.152564, z]], dtype=torch.float64)
                occ = torch.tensor([[1., 0.]], dtype=torch.float64)
                field = m.render(occ,pos)[0]
                self.assertAlmostEqual(float(field.max()), 1., places=12)
                doubled = m.render(occ.repeat(2,1),pos.repeat(2,1))[0]
                self.assertAlmostEqual(float(doubled.max()), 2., places=12)

    def test_obsolete_gain_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'old.yaml'
            config.write_text('imaging:\n  raw_gain: 12.3\n')
            with self.assertRaisesRegex(ValueError, 'raw_gain is no longer used'):
                ga.get_settings(config)

    def test_gaussian_symmetry_mass_and_no_envelope(self):
        m = model(nz=161, padding=80)
        z = torch.tensor([80., 86.3], dtype=torch.float64)
        r = m.axial(z)
        np.testing.assert_allclose(r.sum(1), 1, atol=1e-13, rtol=0)
        np.testing.assert_allclose(r[0], r[0].flip(0), atol=1e-15, rtol=0)
        # Peak is identical inside and outside the former envelope plateau.
        m = model()
        r = m.axial(torch.tensor([7., 15., 23.], dtype=torch.float64))
        np.testing.assert_allclose(r[np.arange(3), [7, 15, 23]], 1/(np.sqrt(2*np.pi)*4))
        # A 1-A shift shifts the profile exactly, without an absolute-z envelope.
        r = m.axial(torch.tensor([7., 8.], dtype=torch.float64))
        np.testing.assert_allclose(r[0, :-1], r[1, 1:], atol=1e-15)
        self.assertLess(float(r[0].sum()), 1.)  # Lost tails are not renormalized.
        self.assertLess(float(m.axial(torch.tensor([0.]))[0, -1]), 1e-10)

    def test_species_mass_additivity_and_xyz_occupancy_gradients(self):
        m = model()
        pos = torch.tensor([[2.13, 2.29, 14.37], [3.46, 3.79, 17.21]], dtype=torch.float64, requires_grad=True)
        occ = torch.tensor([[.8, .1], [.05, .7]], dtype=torch.float64, requires_grad=True)
        field, _ = m.render(occ, pos)
        separate = sum(m.render(occ[i:i+1], pos[i:i+1])[0] for i in range(2))
        torch.testing.assert_close(field, separate)
        ratio = FITS['Se']['total_integrated_weight']/FITS['W']['total_integrated_weight']
        expected = m.gain * ((occ[:, 0]+ratio*occ[:, 1])[:, None]*m.axial(pos[:, 2])).sum(0)
        torch.testing.assert_close(field.sum((1,2)), expected)
        # A spatially weighted scalar checks both lateral deposition and depth gradients.
        weights = torch.linspace(.1, 1, field.numel(), dtype=torch.float64).reshape(field.shape)
        self.assertTrue(torch.autograd.gradcheck(lambda o,p: (m.render(o,p)[0]**2*weights).sum(),
                                                (occ,pos), eps=1e-5, atol=1e-5, rtol=1e-4))

    def test_extra_z_padding_preserves_measured_field(self):
        a, b = model(), model(nz=40, padding=5)
        pos = torch.tensor([[2.13, 3.17, 14.3]], dtype=torch.float64)
        occ = torch.tensor([[1., 0.]], dtype=torch.float64)
        shifted = pos + torch.tensor([0., 0., 5.])
        torch.testing.assert_close(a.render(occ,pos)[0], b.render(occ,shifted)[0][5:35])

    def test_invalid_sigma(self):
        for sigma in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                model(sigma=sigma)

    def test_slice_count_validation_and_30_slice_support(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ga.get_settings()
            args.target_xy_pixels = args.crop_xy_size_source_pixels = 32
            args.target_z_slices = 30
            with self.assertRaisesRegex(ValueError, "must match the TIFF's 12 measured slices"):
                ga.prepare_experimental_volume(args, Path(directory), write_diagnostics=False)
            args.experimental_tiff = str(ROOT.parent/'chia-hao_data/objp_zstack_iter0001_30_slices.tif')
            args.vacuum_padding_angstrom = '8,8,0'
            args.sample_box_z_min_angstrom = 7
            args.sample_box_z_max_angstrom = 23
            volume = ga.prepare_experimental_volume(args, Path(directory), write_diagnostics=False)
            self.assertEqual(volume['sample_target_image'].shape, (30,32,32))
            self.assertEqual(volume['image_loss_voxel_count'], 30*32*32)

    def test_production_config_box_prefit_serialization_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ga.get_settings()
            self.assertTrue(args.config_path.endswith('genetic_09.30.26_wse2.yaml'))
            self.assertEqual(args.gaussian_shared_sigma_z_angstrom, 4.)
            self.assertFalse(hasattr(args, 'depth_calibration_file'))
            self.assertFalse(hasattr(args, 'gaussian_raw_gain'))
            self.assertTrue(args.gaussian_normalize_relative_intensity_reference_peak_to_one)
            self.assertEqual(args.target_z_slices, 12)
            n = args.target_xy_pixels
            args.ga_execution_backend = 'slurm'
            args.ga_resuming_from_checkpoint = True  # Coordinator: no model downloads.
            args.output_dir = directory
            p = ga.WSe2VolumeProblem(args)
            self.assertEqual(p.target_shape_zyx, (28, n+104, n+104))
            np.testing.assert_allclose(p.sample_box_lower_A, [7.933328, 7.933328, 8])
            np.testing.assert_allclose(p.sample_box_upper_A, [7.933328+n*.152564, 7.933328+n*.152564, 20])
            self.assertAlmostEqual(p.cell[2,2], 28.)
            np.testing.assert_array_equal(p.experimental['sample_roi_start_zyx'], [8,52,52])
            np.testing.assert_array_equal(p.experimental['sample_roi_stop_zyx'], [20,52+n,52+n])
            mask = p.experimental['image_loss_mask']
            self.assertEqual(int(mask.sum()), 12*n*n)
            self.assertFalse(mask[:8].any())
            self.assertFalse(mask[20:].any())
            self.assertTrue(mask[8:20,52:52+n,52:52+n].all())
            # Reconstruct the old experimental convention directly from the TIFF.
            source = tifffile.imread(args.experimental_tiff).astype(np.float32)
            y0, x0 = (source.shape[1]-n)//2, (source.shape[2]-n)//2
            crop = source[:,y0:y0+n,x0:x0+n]
            background = float(np.percentile(crop, 1.))
            shifted = np.maximum(crop-background, 0.)
            normalization = float(np.percentile(shifted, 99.5))
            np.testing.assert_allclose(p.experimental['sample_target_linear_normalized'], shifted/normalization, rtol=1e-6)
            self.assertEqual(p.experimental['target_background'], background)
            self.assertEqual(p.experimental['target_normalization'], normalization)
            self.assertGreater(float(p.experimental['sample_target_linear_normalized'].max()), 1.)
            self.assertAlmostEqual(p.gaussian_rendered_single_atom_peak_amplitudes[p.species.index('W')], 1., places=6)
            self.assertEqual(p.sample_box_confinement_penalty, 50)
            self.assertEqual(p.sample_box_confinement_sharpness, 8)
            p.set_site_layout(np.zeros(2, dtype=int))
            pos = torch.tensor([[20.13, 20.29, 15.37], [21.46, 23.79, 16.21]], dtype=p.dtype)
            occ = torch.tensor([[1., 0.], [0., 1.]], dtype=p.dtype)
            normal = p.sample_box_confinement_terms(occ, pos)[0]
            outside = pos.clone(); outside[0, 2] = 25
            penalty = p.sample_box_confinement_terms(occ, outside)[0]
            self.assertGreater(float(penalty), float(normal)+99.)
            aux = p.auxiliary_imaging_and_boundary_terms(occ, pos, True)
            prediction = p.render_differentiable_gaussian_image(occ,pos)[0]
            direct_loss = torch.nn.functional.huber_loss(
                prediction[8:20,52:52+n,52:52+n],
                p.target_image_t[8:20,52:52+n,52:52+n], delta=.2)
            self.assertAlmostEqual(float(direct_loss), aux['image']['loss_raw'], places=7)
            self.assertTrue(torch.isfinite(aux['grad_z']).all())
            self.assertTrue(torch.isfinite(aux['grad_occ']).all())
            args.random_immigrant_image_prefit_soft_steps = 2
            args.random_immigrant_image_prefit_entropy_final_weight = 0
            args.random_immigrant_image_prefit_initial_occupancy_mode = 'current_state'
            args.random_immigrant_image_prefit_initial_logit_noise = 0
            state = p.pack_y(.8*occ, outside)
            fitted, summary = ga.random_immigrant_image_prefit_state(p,state,args,seed=17,member_index=0)
            self.assertTrue(torch.isfinite(fitted).all())
            self.assertLess(summary['soft_handoff_metrics']['objective'], summary['initial_metrics']['objective'])
            p.write_final_image_fields(p.pack_y(occ,pos), p.pack_y(occ,pos))
            fields = np.load(Path(directory)/'final_gaussian_image_comparison.npz')
            self.assertEqual(float(fields['gaussian_shared_sigma_z_angstrom']), 4.)
            self.assertNotIn('depth_response_mass', fields)
            self.assertNotIn('depth_envelope', fields)
            self.assertNotIn('gaussian_raw_gain', fields)
            self.assertIn('gaussian_kernel_peak_scale', fields)
            metadata = json.loads((Path(directory)/'image_model_settings.json').read_text())
            self.assertFalse(metadata['depth_envelope'])
            self.assertEqual(metadata['intensity_normalization'], 'unit_isolated_grid_aligned_W_peak')
            # Experimental brightness scale affects the target only, never kernels.
            from gaussian_image_model import initialize_gaussian_renderer
            before = p.gaussian_renderer.render(occ,pos)[0]
            p.experimental['target_normalization'] *= 3
            initialize_gaussian_renderer(p)
            torch.testing.assert_close(before, p.gaussian_renderer.render(occ,pos)[0], rtol=0, atol=0)
            fingerprint = ga.ga_checkpoint_fingerprint(args)
            changed = copy.deepcopy(args); changed.gaussian_shared_sigma_z_angstrom = 5
            self.assertNotEqual(fingerprint, ga.ga_checkpoint_fingerprint(changed))
            changed = copy.deepcopy(args); changed.target_normalization_percentile = 99.
            self.assertNotEqual(fingerprint, ga.ga_checkpoint_fingerprint(changed))
            checkpoint = Path(directory)/'ga_checkpoint.json'
            checkpoint.write_text(json.dumps({'schema_version':ga.GA_CHECKPOINT_SCHEMA_VERSION,
                                              'config_fingerprint':'old-envelope-objective'}))
            with self.assertRaises(RuntimeError):
                ga.read_ga_checkpoint_for_args(args)


if __name__ == '__main__':
    unittest.main()
