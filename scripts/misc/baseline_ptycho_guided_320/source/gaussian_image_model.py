"""Double-Gaussian XY and analytic Gaussian Z image formation, 09/30/26."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

import numpy as np
import torch


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


class GaussianImageModel:
    """XY FFT convolution and nonperiodic Gaussian Z; no fitted depth assets.

    R(dz) = dz_pixel/(sqrt(2*pi)*sigma) * exp(-dz**2/(2*sigma**2)).
    The continuous response has unit mass. Finite-window tails are not
    renormalized, and no envelope, wrapping, or registration is applied in Z.
    """
    def __init__(self, sigma_z, lateral_blur, element_fits, species,
                 shape_zyx, voxel_zyx, padding_xyz, device, dtype):
        self.sigma_z = float(sigma_z)
        if not np.isfinite(self.sigma_z) or self.sigma_z <= 0:
            raise ValueError('Gaussian sigma must be finite and positive.')
        if not np.isfinite(lateral_blur) or lateral_blur < 0:
            raise ValueError('Lateral blur must be finite and nonnegative.')
        self.shape = tuple(shape_zyx)
        self.device, self.dtype = device, dtype
        self.voxel = torch.as_tensor(voxel_zyx, device=device, dtype=dtype)
        self.padding_z = float(padding_xyz[2])
        if not np.isfinite(voxel_zyx).all() or np.any(np.asarray(voxel_zyx) <= 0):
            raise ValueError('Voxel spacing must be finite and positive.')
        self.zgrid = torch.arange(self.shape[0], device=device, dtype=dtype) * self.voxel[0] - self.padding_z
        fy = torch.fft.fftfreq(self.shape[1], device=device, dtype=dtype) / self.voxel[1]
        fx = torch.fft.rfftfreq(self.shape[2], device=device, dtype=dtype) / self.voxel[2]
        freq2 = fy[:, None]**2 + fx[None, :]**2
        reference = element_fits['W']['total_integrated_weight']
        kernels = []
        for symbol in species:
            row = element_fits[symbol]
            kernel = torch.zeros_like(freq2)
            for part in ('core', 'tail'):
                sigma2 = row[part+'_sigma_xy_A']**2 + lateral_blur**2
                kernel = kernel + row[part+'_integrated_weight']/reference * torch.exp(-2*np.pi**2*sigma2*freq2)
            kernels.append(kernel)
        self.fourier = torch.stack(kernels)
        # Scale the combined W core+tail kernel ONCE, independently of the
        # experimental percentile and of the candidate structure. The reference
        # atom is centered on an XY grid point and exactly on a Z slice.
        w_kernel = torch.fft.irfft2(self.fourier[species.index('W')], s=self.shape[1:])
        axial_peak = self.voxel[0] / (np.sqrt(2*np.pi)*self.sigma_z)
        self.reference_peak_before_scaling = float((w_kernel.max()*axial_peak).cpu())
        if not np.isfinite(self.reference_peak_before_scaling) or self.reference_peak_before_scaling <= 0:
            raise ValueError('The isolated W reference peak must be finite and positive.')
        self.gain = 1.0 / self.reference_peak_before_scaling

    def axial(self, positions_z):
        z = positions_z - self.padding_z
        distance = self.zgrid[None, :] - z[:, None]
        return (self.voxel[0] / (np.sqrt(2*np.pi)*self.sigma_z)
                * torch.exp(-0.5*(distance/self.sigma_z)**2))

    def render(self, species_labels, positions):
        nz, ny, nx = self.shape
        # Constant one-hot W/Se labels; only positions are differentiated.
        ns = species_labels.shape[1]
        response = self.axial(positions[:, 2])
        pixels = positions[:, :2] / self.voxel[[2, 1]]
        base = torch.floor(pixels).long()
        fraction = pixels - base.to(positions.dtype)
        offsets = torch.arange(ns*nz, device=species_labels.device).reshape(ns, nz, 1)*ny*nx
        density = torch.zeros(ns*nz*ny*nx, device=species_labels.device, dtype=species_labels.dtype)
        for dy in (0, 1):
            for dx in (0, 1):
                weight = (fraction[:, 0] if dx else 1-fraction[:, 0]) * (fraction[:, 1] if dy else 1-fraction[:, 1])
                indices = (base[:, 1]+dy).remainder(ny)*nx + (base[:, 0]+dx).remainder(nx)
                values = species_labels.T[:, None, :] * response.T[None, :, :] * weight[None, None, :]
                density.index_add_(0, (offsets+indices).reshape(-1), values.reshape(-1))
        density = density.reshape(ns, nz, ny, nx)
        components = torch.fft.irfft2(torch.fft.rfft2(density) * self.fourier[:, None], s=(ny, nx))
        components = components * self.gain
        # Preserve the lateral FFT result; no clipping/rescaling.
        return components.sum(dim=0), components


def initialize_gaussian_renderer(problem):
    """Install the shared renderer and record explicit, reproducible settings."""
    p = problem
    p.gaussian_renderer = GaussianImageModel(
        p.args.gaussian_shared_sigma_z_angstrom,
        p.args.gaussian_additional_lateral_blur_angstrom,
        p.gaussian_element_fits, p.species, p.target_shape_zyx,
        p.target_voxel_size_A, p.vacuum_position_shift, p.device, p.dtype)
    p.image_model_metadata = dict(
        schema_version=3, model='double_xy_gaussian_gaussian_depth_no_envelope',
        sigma_z_A=p.gaussian_renderer.sigma_z,
        intensity_normalization='unit_isolated_grid_aligned_W_peak',
        reference_peak_before_scaling=p.gaussian_renderer.reference_peak_before_scaling,
        additional_lateral_blur_A=float(p.args.gaussian_additional_lateral_blur_angstrom),
        target_normalization=float(p.experimental['target_normalization']),
        kernel_peak_scale=p.gaussian_renderer.gain, depth_envelope=False,
        target_background=float(p.experimental['target_background']),
        target_background_percentile=float(p.args.target_background_percentile),
        target_normalization_percentile=float(p.args.target_normalization_percentile),
        depth_response='dz_pixel/(sqrt(2*pi)*sigma_z)*exp(-0.5*((z-z_atom)/sigma_z)**2)',
        depth_window_renormalization=False, voxel_size_zyx_A=np.asarray(p.target_voxel_size_A).tolist(),
        element_fit_sha256=hashlib.sha256(Path(p.gaussian_element_fit_csv).read_bytes()).hexdigest(),
        position_convention='Cell coordinates including padding; no registration offset.')
    p.gaussian_shared_sigma_z_A = p.gaussian_renderer.sigma_z
    p.gaussian_phase_table_to_image_amplitude_scale = p.gaussian_renderer.gain / p.gaussian_element_fits['W']['total_integrated_weight']
    p.gaussian_intensity_scale_source = 'unit_relative_rendered_csv_peak'
    p.gaussian_component_amplitudes = np.array([
        [p.gaussian_element_fits[s][part+'_integrated_weight'] for part in ('core','tail')]
        for s in p.species]) * p.gaussian_phase_table_to_image_amplitude_scale
    p.gaussian_amplitudes = p.gaussian_component_amplitudes.sum(axis=1)
    p.gaussian_reference_integrated_amplitude = float(p.gaussian_amplitudes[p.species.index('W')])
    # Nominal grid-aligned XY peak with the atom centered exactly on a Z plane.
    p.gaussian_rendered_single_atom_peak_amplitudes = (
        torch.fft.irfft2(p.gaussian_renderer.fourier, s=p.target_shape_zyx[1:]).amax(dim=(-2,-1))
        * (p.gaussian_renderer.voxel[0]/(np.sqrt(2*np.pi)*p.gaussian_shared_sigma_z_A))
        * p.gaussian_renderer.gain).cpu().numpy()
    p.gaussian_component_amplitudes_t = torch.as_tensor(p.gaussian_component_amplitudes, device=p.device, dtype=p.dtype)
    with tempfile.NamedTemporaryFile(mode='w', dir=p.output_dir, delete=False) as handle:
        handle.write(json.dumps(p.image_model_metadata, indent=2)+'\n')
        temporary = handle.name
    os.replace(temporary, p.output_dir/'image_model_settings.json')
