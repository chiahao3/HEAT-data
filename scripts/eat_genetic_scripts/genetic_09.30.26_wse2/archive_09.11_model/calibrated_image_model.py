"""Calibrated W/Se image formation, independent of FairChem and GA machinery."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import torch


def load_calibration(path, coefficient_path=None):
    data = json.loads(Path(path).read_text())
    fit = data['fit']
    mass = np.asarray(fit['mass'], float)
    lags = np.asarray(fit['centered_lags'], float)
    if (data['schema_version'] != 1 or mass.shape != (31,) or lags.shape != (31,)
            or not np.isfinite(mass).all() or not np.isfinite(lags).all()
            or np.any(mass < 0) or not np.isclose(mass.sum(), 1., atol=1e-10)
            or not np.allclose(np.diff(lags), 1.)
            or not np.isfinite(fit['gain']) or fit['gain'] <= 0
            or not np.isfinite(fit['widths']).all() or np.any(np.asarray(fit['widths']) <= 0)):
        raise ValueError('Invalid calibrated depth response.')
    if coefficient_path is not None:
        digest = hashlib.sha256(Path(coefficient_path).read_bytes()).hexdigest()
        if digest != data['element_fit_sha256']:
            raise ValueError('Lateral coefficient CSV differs from the accepted calibration.')
    return data


def atomic_box_bounds(args, sample_lengths_xyz, padding_xyz):
    """Bounds specified in experimental coordinates, then shifted into the cell."""
    lower = np.array(padding_xyz, dtype=float, copy=True)
    upper = lower + np.asarray(sample_lengths_xyz, dtype=float)
    zmin = float(args.sample_box_z_min_angstrom)
    zmax = float(args.sample_box_z_max_angstrom)
    if not (np.isfinite([zmin, zmax]).all() and 0 <= zmin < zmax <= sample_lengths_xyz[2]):
        raise ValueError('sample_box z_min/z_max must satisfy 0 <= min < max <= measured depth.')
    lower[2] += zmin
    upper[2] = padding_xyz[2] + zmax
    if 2 * float(args.sample_box_confinement_margin_angstrom) >= np.min(upper - lower):
        raise ValueError('Confinement margin leaves an empty atomic box.')
    return lower, upper


class CalibratedImageModel:
    """XY FFT convolution and nonperiodic, piecewise linear axial response.

    Occupancies and physical positions remain differentiable. Calibration is
    fixed across candidates. The envelope attenuates the summed field in TIFF
    coordinates. Neither lost tails nor attenuation are normalized away.
    """
    def __init__(self, calibration, element_fits, species, shape_zyx, voxel_zyx,
                 padding_xyz, normalization, device, dtype):
        self.calibration = calibration
        self.shape = tuple(shape_zyx)
        self.device, self.dtype = device, dtype
        self.voxel = torch.as_tensor(voxel_zyx, device=device, dtype=dtype)
        self.padding_z = float(padding_xyz[2])
        self.normalization = float(normalization)
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError('Target normalization must be positive and finite.')
        if not np.allclose(voxel_zyx, calibration['voxel_size_zyx_A'], rtol=0, atol=1e-8):
            raise ValueError('Accepted calibration requires native TIFF voxel spacing.')
        f = calibration['fit']
        self.mass = torch.tensor(f['mass'], device=device, dtype=dtype)
        self.lags = torch.tensor(f['centered_lags'], device=device, dtype=dtype)
        self.zgrid = torch.arange(self.shape[0], device=device, dtype=dtype) * self.voxel[0] - self.padding_z
        lo, hi = calibration['envelope_plateau_A']
        self.envelope = torch.exp(-((lo-self.zgrid).clamp_min(0)/f['widths'][0])**2
                                  -((self.zgrid-hi).clamp_min(0)/f['widths'][1])**2)
        fy = torch.fft.fftfreq(self.shape[1], device=device, dtype=dtype) / self.voxel[1]
        fx = torch.fft.rfftfreq(self.shape[2], device=device, dtype=dtype) / self.voxel[2]
        freq2 = fy[:, None]**2 + fx[None, :]**2
        reference = element_fits['W']['total_integrated_weight']
        kernels = []
        for symbol in species:
            row = element_fits[symbol]
            kernel = torch.zeros_like(freq2)
            for part in ('core', 'tail'):
                sigma2 = row[part+'_sigma_xy_A']**2 + calibration['additional_lateral_blur_A']**2
                kernel = kernel + row[part+'_integrated_weight']/reference * torch.exp(-2*np.pi**2*sigma2*freq2)
            kernels.append(kernel)
        self.fourier = torch.stack(kernels)
        self.gain = f['gain'] / normalization

    def axial(self, positions_z):
        z = positions_z - self.padding_z
        triangles = (1 - torch.abs(self.zgrid[None, :, None] - z[:, None, None] - self.lags)).clamp_min(0)
        return triangles @ self.mass

    def render(self, occ, positions):
        nz, ny, nx = self.shape
        ns = occ.shape[1]
        response = self.axial(positions[:, 2])
        pixels = positions[:, :2] / self.voxel[[2, 1]]
        base = torch.floor(pixels).long()
        fraction = pixels - base.to(positions.dtype)
        offsets = torch.arange(ns*nz, device=occ.device).reshape(ns, nz, 1)*ny*nx
        density = torch.zeros(ns*nz*ny*nx, device=occ.device, dtype=occ.dtype)
        for dy in (0, 1):
            for dx in (0, 1):
                weight = (fraction[:, 0] if dx else 1-fraction[:, 0]) * (fraction[:, 1] if dy else 1-fraction[:, 1])
                indices = (base[:, 1]+dy).remainder(ny)*nx + (base[:, 0]+dx).remainder(nx)
                values = occ.T[:, None, :] * response.T[None, :, :] * weight[None, None, :]
                density.index_add_(0, (offsets+indices).reshape(-1), values.reshape(-1))
        density = density.reshape(ns, nz, ny, nx)
        components = torch.fft.irfft2(torch.fft.rfft2(density) * self.fourier[:, None], s=(ny, nx))
        components = components * (self.gain*self.envelope)[None, :, None, None]
        # Retain the calibration's tiny Fourier ringing; no clipping/rescaling.
        return components.sum(dim=0), components


def initialize_calibrated_renderer(problem):
    """Install the renderer and descriptive fields used by existing serializers."""
    p = problem
    cal = load_calibration(p.args.depth_calibration_file, p.gaussian_element_fit_csv)
    p.calibrated_renderer = CalibratedImageModel(
        cal, p.calibrated_element_fits, p.species, p.target_shape_zyx,
        p.target_voxel_size_A, p.vacuum_position_shift,
        p.experimental['target_normalization'], p.device, p.dtype)
    p.image_model_metadata = dict(cal, target_normalization=p.experimental['target_normalization'],
                                 calibration_file=str(p.args.depth_calibration_file),
                                 normalized_gain=p.calibrated_renderer.gain)
    p.gaussian_shared_sigma_z_A = cal['fit']['std']  # Legacy attribute: response RMS, not Gaussian sigma.
    p.gaussian_phase_table_to_image_amplitude_scale = p.calibrated_renderer.gain / p.calibrated_element_fits['W']['total_integrated_weight']
    p.gaussian_intensity_scale_source = 'fixed_calibrated_gain_divided_by_target_normalization'
    p.gaussian_component_amplitudes = np.array([
        [p.calibrated_element_fits[s][part+'_integrated_weight'] for part in ('core','tail')]
        for s in p.species]) * p.gaussian_phase_table_to_image_amplitude_scale
    p.gaussian_amplitudes = p.gaussian_component_amplitudes.sum(axis=1)
    p.gaussian_reference_integrated_amplitude = float(p.gaussian_amplitudes[p.species.index('W')])
    # Informational nominal peak at a grid-aligned XY atom and response knot,
    # before envelope attenuation; actual peaks depend on subvoxel position/depth.
    p.gaussian_rendered_single_atom_peak_amplitudes = (
        torch.fft.irfft2(p.calibrated_renderer.fourier, s=p.target_shape_zyx[1:]).amax(dim=(-2,-1))
        * p.calibrated_renderer.mass.max() * p.calibrated_renderer.gain).cpu().numpy()
    p.gaussian_component_amplitudes_t = torch.as_tensor(p.gaussian_component_amplitudes, device=p.device, dtype=p.dtype)
    # Multiple local workers may construct the same problem simultaneously.
    with tempfile.NamedTemporaryFile(mode='w', dir=p.output_dir, delete=False) as handle:
        handle.write(json.dumps(p.image_model_metadata, indent=2)+'\n')
        temporary = handle.name
    os.replace(temporary, p.output_dir/'image_model_calibration.json')
