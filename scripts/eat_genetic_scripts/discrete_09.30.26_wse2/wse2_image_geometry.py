"""Standalone experimental geometry, Gaussian data, and feasible-position sampling.

These numerical routines preserve the 09/30 microscopy conventions. They do
not construct or optimize candidate species; the driver supplies real atoms.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from ase import Atoms
from gaussian_image_model import atomic_box_bounds

SCRIPT_DIR = Path(__file__).resolve().parent

def resolve_script_relative_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (SCRIPT_DIR / p).resolve()

def load_double_gaussian_element_fits(path):
    """Load and validate element-resolved two-Gaussian lateral fit parameters."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Double-Gaussian element fit CSV not found: {path}")
    required = {
        "species",
        "core_sigma_xy_angstrom",
        "tail_sigma_xy_angstrom",
        "core_integrated_weight_VAngstrom3",
        "tail_integrated_weight_VAngstrom3",
        "total_integrated_weight_VAngstrom3",
        "fit_status",
    }
    fits = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Double-Gaussian CSV {path} is missing columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            symbol = str(row["species"]).strip()
            if not symbol:
                raise ValueError(f"{path}:{line_number}: empty species symbol.")
            if symbol in fits:
                raise ValueError(f"{path}:{line_number}: duplicate species {symbol!r}.")
            if str(row["fit_status"]).strip().lower() != "ok":
                raise ValueError(
                    f"{path}:{line_number}: fit_status for {symbol} is not 'ok'."
                )
            core_sigma = float(row["core_sigma_xy_angstrom"])
            tail_sigma = float(row["tail_sigma_xy_angstrom"])
            core_weight = float(row["core_integrated_weight_VAngstrom3"])
            tail_weight = float(row["tail_integrated_weight_VAngstrom3"])
            total_weight = float(row["total_integrated_weight_VAngstrom3"])
            values = np.asarray(
                [core_sigma, tail_sigma, core_weight, tail_weight, total_weight],
                dtype=float,
            )
            if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(
                    f"{path}:{line_number}: sigmas and integrated weights for "
                    f"{symbol} must be finite and positive."
                )
            if tail_sigma <= core_sigma:
                raise ValueError(
                    f"{path}:{line_number}: tail sigma must exceed core sigma for {symbol}."
                )
            if not np.isclose(total_weight, core_weight + tail_weight, rtol=1e-8, atol=1e-10):
                raise ValueError(
                    f"{path}:{line_number}: component weights do not sum to the total for {symbol}."
                )
            fits[symbol] = {
                "core_sigma_xy_A": core_sigma,
                "tail_sigma_xy_A": tail_sigma,
                "core_integrated_weight": core_weight,
                "tail_integrated_weight": tail_weight,
                "total_integrated_weight": total_weight,
            }
    if not fits:
        raise ValueError(f"Double-Gaussian element fit CSV is empty: {path}")
    return path, fits

def parse_vacuum_padding(text):
    values = np.asarray([float(x.strip()) for x in str(text).split(",") if x.strip()], dtype=float)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.shape != (3,):
        raise ValueError("--vacuum-padding-angstrom must be one value or three comma-separated values.")
    if np.any(values < 0.0):
        raise ValueError("--vacuum-padding-angstrom must be non-negative.")
    return values

def parse_pbc(text):
    if isinstance(text, bool):
        return np.repeat(bool(text), 3)
    parts = [part.strip().lower() for part in str(text).split(",") if part.strip()]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise ValueError("system.pbc must be one boolean or three comma-separated booleans.")
    truthy = {"true", "t", "yes", "y", "1", "on"}
    falsy = {"false", "f", "no", "n", "0", "off"}
    values = []
    for part in parts:
        if part in truthy:
            values.append(True)
        elif part in falsy:
            values.append(False)
        else:
            raise ValueError(f"Could not parse PBC value {part!r}. Use true/false.")
    return np.asarray(values, dtype=bool)

def vacuum_padded_cell_and_shift(cell, padding):
    cell = np.asarray(cell, dtype=float)
    padding = np.asarray(padding, dtype=float)
    lengths = np.linalg.norm(cell, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Cell vectors must have positive length.")
    directions = cell / lengths[:, None]
    padded_cell = cell * ((lengths + 2.0 * padding) / lengths)[:, None]
    position_shift = np.sum(directions * padding[:, None], axis=0)
    return padded_cell, position_shift

def doc_value(doc, name, default=None):
    return doc.get(name, default) if isinstance(doc, dict) else getattr(doc, name, default)

def big_if_none(value):
    return 1.0e9 if value is None else float(value)

def structure_to_ase_atoms(structure):
    symbols = [site.specie.symbol for site in structure]
    positions = [[float(x) for x in site.coords] for site in structure]
    cell = [[float(x) for x in row] for row in structure.lattice.matrix]
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    return atoms

def fetch_elemental_reference_atoms(symbol, api_key):
    if not api_key:
        raise RuntimeError("Set MP_API_KEY to fetch elemental references from Materials Project.")
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("Install mp-api to fetch elemental references from Materials Project.") from exc

    fields = ["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom", "structure"]
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(chemsys=symbol, fields=fields))
    docs = [
        doc for doc in docs
        if set(doc_value(doc, "structure").composition.get_el_amt_dict()) == {symbol}
    ]
    if not docs:
        raise RuntimeError(f"No pure Materials Project reference found for element {symbol!r}.")
    docs.sort(key=lambda d: (big_if_none(doc_value(d, "energy_above_hull")), big_if_none(doc_value(d, "formation_energy_per_atom"))))
    doc = docs[0]
    return {
        "symbol": symbol,
        "material_id": str(doc_value(doc, "material_id")),
        "formula_pretty": doc_value(doc, "formula_pretty"),
        "energy_above_hull": doc_value(doc, "energy_above_hull"),
        "formation_energy_per_atom": doc_value(doc, "formation_energy_per_atom"),
        "atoms": structure_to_ase_atoms(doc_value(doc, "structure")),
    }

def parse_float_triplet(value, name):
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = list(value)
    else:
        pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values.")
    result = np.asarray([float(piece) for piece in pieces], dtype=float)
    if np.any(result < 0.0):
        raise ValueError(f"{name} values must be nonnegative.")
    return result

def normalize_image_transform_name(value):
    name = str(value).strip().lower().replace("_", "")
    aliases = {
        "linear": "linear",
        "identity": "linear",
        "sqrt": "sqrt",
        "squareroot": "sqrt",
        "cuberoot": "cuberoot",
        "cbrt": "cuberoot",
    }
    if name not in aliases:
        raise ValueError("imaging.target_transform must be linear, sqrt, or cuberoot.")
    return aliases[name]

def numpy_image_transform(linear_field, transform, epsilon):
    field = np.maximum(np.asarray(linear_field), 0.0)
    transform = normalize_image_transform_name(transform)
    epsilon = float(epsilon)
    if transform == "linear":
        return field
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("imaging.transform_epsilon must be finite and positive.")
    if transform == "sqrt":
        return np.maximum(np.sqrt(field + epsilon) - math.sqrt(epsilon), 0.0)
    return np.maximum(np.cbrt(field + epsilon) - np.cbrt(epsilon), 0.0)

def torch_image_transform(linear_field, transform, epsilon):
    field = torch.clamp(linear_field, min=0.0)
    if transform == "linear":
        return field
    epsilon_t = torch.as_tensor(epsilon, device=field.device, dtype=field.dtype)
    if transform == "sqrt":
        return torch.clamp_min(
            torch.sqrt(field + epsilon_t) - torch.sqrt(epsilon_t), 0.0
        )
    return torch.clamp_min(
        torch.pow(field + epsilon_t, 1.0 / 3.0)
        - torch.pow(epsilon_t, 1.0 / 3.0),
        0.0,
    )

def periodic_central_gradient_zyx(field_zyx, voxel_size_zyx_A):
    """Return the periodic physical-coordinate gradient of a ZYX field."""
    field = np.asarray(field_zyx, dtype=np.float64)
    return tuple(
        (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis))
        / (2.0 * float(spacing))
        for axis, spacing in enumerate(np.asarray(voxel_size_zyx_A, dtype=float))
    )

def distance_scaled_level_set(
    smoothed_target,
    threshold,
    voxel_size_zyx_A,
    periodic=True,
):
    """Build a sign-exact, approximately Angstrom-scaled intensity level set."""
    scalar = np.asarray(smoothed_target, dtype=np.float64) - float(threshold)
    mask = scalar >= 0.0
    interface = np.zeros_like(mask, dtype=bool)
    if periodic:
        for axis in range(3):
            interface |= mask != np.roll(mask, 1, axis=axis)
            interface |= mask != np.roll(mask, -1, axis=axis)
        gradients = periodic_central_gradient_zyx(scalar, voxel_size_zyx_A)
        boundary_mode = "periodic"
    else:
        padded_mask = np.pad(mask, 1, mode="constant", constant_values=False)
        for axis in range(3):
            before = [slice(1, -1) for _ in range(3)]
            after = [slice(1, -1) for _ in range(3)]
            before[axis] = slice(0, -2)
            after[axis] = slice(2, None)
            interface |= mask != padded_mask[tuple(before)]
            interface |= mask != padded_mask[tuple(after)]
        gradients = tuple(
            np.gradient(
                scalar,
                *np.asarray(voxel_size_zyx_A, dtype=float),
                edge_order=1,
            )
        )
        boundary_mode = "nonperiodic_sample_embedded_in_vacuum"
    gradient_norm = np.sqrt(sum(component * component for component in gradients))
    valid = interface & np.isfinite(gradient_norm) & (gradient_norm > 1e-12)
    if not np.any(valid):
        raise ValueError("Could not estimate a nonzero level-set slope at the mask interface.")
    slope = float(np.median(gradient_norm[valid]))
    level_set_A = (scalar / slope).astype(np.float32)
    return level_set_A, {
        "normalization": "median_physical_gradient_on_mask_interface",
        "boundary_mode": boundary_mode,
        "boundary_gradient_scale_intensity_per_A": slope,
        "interface_voxel_count": int(np.count_nonzero(interface)),
        "positive_fraction": float(np.mean(level_set_A >= 0.0)),
        "mask_sign_agreement_fraction": float(np.mean((level_set_A >= 0.0) == mask)),
        "boundary_gradient_norm_median_after_scaling": float(
            np.median(gradient_norm[valid] / slope)
        ),
    }

def cylinder_pruned_feasible_sites_zyx(
    feasible_mask,
    voxel_zyx_A,
    lateral_radius_A,
    z_radius_A,
    intensity=None,
):
    """Select a deterministic, intensity-prioritized cylindrical packing.

    Every accepted feasible voxel removes periodically neighboring candidates
    whose lateral separation is at most ``lateral_radius_A`` and whose absolute
    z separation is at most ``z_radius_A``.  A high-intensity representative of
    every connected feasible component is attempted before the global
    intensity-ordered pass.
    """
    from scipy.ndimage import generate_binary_structure, label

    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    feasible_zyx = np.argwhere(feasible_mask).astype(np.int32)
    if len(feasible_zyx) == 0:
        raise ValueError("Cylinder initialization received an empty feasible mask.")
    voxel_zyx_A = np.asarray(voxel_zyx_A, dtype=float)
    lateral_radius_A = float(lateral_radius_A)
    z_radius_A = float(z_radius_A)
    if lateral_radius_A < 0.0 or z_radius_A < 0.0:
        raise ValueError("Cylinder initialization radii must be nonnegative.")

    labels, component_count = label(
        feasible_mask,
        structure=generate_binary_structure(3, 3),
    )
    component_ids = labels[
        feasible_zyx[:, 0], feasible_zyx[:, 1], feasible_zyx[:, 2]
    ]
    component_sizes = np.bincount(labels.reshape(-1))
    if intensity is None:
        intensity_values = np.zeros(len(feasible_zyx), dtype=float)
    else:
        intensity = np.asarray(intensity, dtype=float)
        intensity_values = intensity[
            feasible_zyx[:, 0], feasible_zyx[:, 1], feasible_zyx[:, 2]
        ]

    limits = np.ceil(
        np.asarray([z_radius_A, lateral_radius_A, lateral_radius_A], dtype=float)
        / voxel_zyx_A
    ).astype(int)
    oz, oy, ox = np.meshgrid(
        np.arange(-limits[0], limits[0] + 1),
        np.arange(-limits[1], limits[1] + 1),
        np.arange(-limits[2], limits[2] + 1),
        indexing="ij",
    )
    offsets = np.column_stack((oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)))
    physical_offsets = offsets * voxel_zyx_A[None, :]
    keep = (
        (np.abs(physical_offsets[:, 0]) <= z_radius_A)
        & (
            physical_offsets[:, 1] ** 2 + physical_offsets[:, 2] ** 2
            <= lateral_radius_A**2
        )
    )
    offsets = offsets[keep].astype(np.int32, copy=False)
    shape = np.asarray(feasible_mask.shape, dtype=np.int32)
    available = feasible_mask.copy()
    selected = []

    def accept(site):
        site = np.asarray(site, dtype=np.int32)
        if not available[tuple(site)]:
            return False
        selected.append(site.copy())
        neighbors = np.mod(offsets + site[None, :], shape[None, :])
        available[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]] = False
        return True

    for component_id in np.argsort(component_sizes[1:])[::-1] + 1:
        members = np.where(component_ids == int(component_id))[0]
        if len(members):
            accept(feasible_zyx[members[int(np.argmax(intensity_values[members]))]])
    for index in np.argsort(intensity_values, kind="stable")[::-1]:
        accept(feasible_zyx[int(index)])

    selected = np.asarray(selected, dtype=np.int32)
    selected_components = labels[
        selected[:, 0], selected[:, 1], selected[:, 2]
    ]
    represented_components = len(
        {int(value) for value in selected_components if int(value) > 0}
    )
    return selected, {
        "scheme": "intensity_prioritized_cylinder_with_angstrom_jitter",
        "lateral_radius_A": lateral_radius_A,
        "z_radius_A": z_radius_A,
        "feasible_voxels": int(len(feasible_zyx)),
        "selected_before_count_limit": int(len(selected)),
        "component_count": int(component_count),
        "represented_component_count": int(represented_components),
        "unrepresented_component_count": int(component_count - represented_components),
        "coverage_complete": bool(not np.any(available)),
    }

def periodic_cubic_bspline_coefficients(field_zyx):
    """Periodic cubic B-spline coefficients interpolating the grid samples."""
    from scipy.ndimage import spline_filter

    return spline_filter(
        np.asarray(field_zyx, dtype=np.float64),
        order=3,
        mode="grid-wrap",
        output=np.float64,
    )

def periodic_cubic_bspline_sample_zyx(coefficients_zyx, positions_xyz_A, voxel_size_zyx_A):
    """Sample a periodic C2 cubic B-spline at Cartesian xyz positions."""
    dz_A, dy_A, dx_A = [float(value) for value in voxel_size_zyx_A]
    nz, ny, nx = coefficients_zyx.shape
    coords = torch.stack(
        (
            positions_xyz_A[:, 2] / dz_A,
            positions_xyz_A[:, 1] / dy_A,
            positions_xyz_A[:, 0] / dx_A,
        ),
        dim=1,
    )
    base_float = torch.floor(coords)
    frac = coords - base_float
    base = base_float.to(torch.long)

    def cubic_weights(t):
        return torch.stack(
            (
                (1.0 - t) ** 3 / 6.0,
                (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
                (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
                t**3 / 6.0,
            ),
            dim=1,
        )

    wz, wy, wx = (cubic_weights(frac[:, axis]) for axis in range(3))
    result = torch.zeros(
        len(positions_xyz_A),
        device=positions_xyz_A.device,
        dtype=positions_xyz_A.dtype,
    )
    shape = (nz, ny, nx)
    for az, oz in enumerate((-1, 0, 1, 2)):
        iz = torch.remainder(base[:, 0] + oz, shape[0])
        for ay, oy in enumerate((-1, 0, 1, 2)):
            iy = torch.remainder(base[:, 1] + oy, shape[1])
            for ax, ox in enumerate((-1, 0, 1, 2)):
                ix = torch.remainder(base[:, 2] + ox, shape[2])
                result = result + (
                    wz[:, az] * wy[:, ay] * wx[:, ax] * coefficients_zyx[iz, iy, ix]
                )
    return result

def prepare_experimental_volume(args, output_dir, write_diagnostics=True):
    """Keep all native TIFF slices and add vacuum; slab bounds are independent."""
    import tifffile
    from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

    tiff_path = resolve_script_relative_path(args.experimental_tiff)
    if not tiff_path.is_file():
        raise FileNotFoundError(f"Experimental TIFF not found: {tiff_path}")
    source = np.asarray(tifffile.imread(tiff_path), dtype=np.float32)
    if source.ndim != 3:
        raise ValueError(f"Experimental TIFF must be a ZYX stack; got {source.shape}.")

    source_voxel_A = np.asarray(
        parse_float_triplet(
            args.source_voxel_size_angstrom,
            "experimental.source_voxel_size_angstrom",
        ),
        dtype=float,
    )
    if np.any(source_voxel_A <= 0.0):
        raise ValueError("Experimental voxel sizes must be positive.")

    crop_size = int(args.crop_xy_size_source_pixels)
    target_nxy = int(args.target_xy_pixels)
    target_nz = int(args.target_z_slices)
    if crop_size <= 1 or crop_size > min(source.shape[1:]):
        raise ValueError(f"Bad centered crop size {crop_size} for TIFF {source.shape}.")
    if target_nxy <= 1 or target_nz <= 1:
        raise ValueError("Target dimensions must exceed one voxel per axis.")
    y0 = (source.shape[1] - crop_size) // 2
    x0 = (source.shape[2] - crop_size) // 2
    source_crop = source[:, y0 : y0 + crop_size, x0 : x0 + crop_size]
    if target_nz != source.shape[0]:
        raise ValueError(
            f"experimental.target_z_slices={target_nz} must match the TIFF's "
            f"{source.shape[0]} measured slices; depth resampling is disabled."
        )
    if target_nxy != crop_size:
        raise ValueError("This workflow requires native XY resolution.")
    factors = (
        target_nz / source_crop.shape[0],
        target_nxy / crop_size,
        target_nxy / crop_size,
    )
    sample_linear = zoom(
        source_crop,
        factors,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32)

    sample_cell_lengths_A = np.asarray(
        [
            crop_size * source_voxel_A[2],
            crop_size * source_voxel_A[1],
            source.shape[0] * source_voxel_A[0],
        ],
        dtype=float,
    )
    sample_voxel_A = np.asarray(
        [
            sample_cell_lengths_A[2] / target_nz,
            sample_cell_lengths_A[1] / target_nxy,
            sample_cell_lengths_A[0] / target_nxy,
        ],
        dtype=float,
    )

    # Historical normalization: one background and percentile over the entire
    # measured crop, before adding vacuum. Never normalize slices separately.
    background = float(np.percentile(sample_linear, float(args.target_background_percentile)))
    sample_linear = np.maximum(sample_linear - background, 0.0)
    normalization = float(
        np.percentile(sample_linear, float(args.target_normalization_percentile))
    )
    if normalization <= 0.0:
        raise ValueError("Experimental crop has zero robust intensity scale.")
    sample_linear_normalized = (sample_linear / normalization).astype(np.float32)
    if bool(args.gaussian_clip_experimental_to_unit_interval):
        sample_linear_normalized = np.clip(
            sample_linear_normalized, 0.0, 1.0
        ).astype(np.float32)
    image_transform = normalize_image_transform_name(args.image_target_transform)
    sample_target_image = numpy_image_transform(
        sample_linear_normalized,
        image_transform,
        float(args.gaussian_transform_epsilon),
    ).astype(np.float32)
    sample_target_sqrt = np.sqrt(sample_linear_normalized).astype(np.float32)
    mask_sigma = parse_float_triplet(
        args.mask_smoothing_sigma_pixels,
        "feasible_region.mask_smoothing_sigma_pixels",
    )
    sample_smoothed = gaussian_filter(
        sample_target_sqrt,
        sigma=mask_sigma,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    threshold_mode = str(args.mask_threshold_mode).strip().lower()
    if threshold_mode == "percentile":
        threshold_value = float(
            np.percentile(sample_smoothed, float(args.mask_threshold))
        )
    elif threshold_mode in ("absolute", "value"):
        threshold_value = float(args.mask_threshold)
    elif threshold_mode in ("fraction_of_max", "max_fraction"):
        threshold_value = float(args.mask_threshold) * float(sample_smoothed.max())
    else:
        raise ValueError(
            "feasible_region.threshold_mode must be percentile, absolute, "
            "or fraction_of_max."
        )
    sample_feasible_mask = sample_smoothed >= threshold_value
    # Restrict candidate seeding to the measured atomic slab.
    atomic_lower, atomic_upper = atomic_box_bounds(args, sample_cell_lengths_A, np.zeros(3))
    z_coordinates = np.arange(target_nz) * sample_voxel_A[0]
    sample_feasible_mask &= ((z_coordinates >= atomic_lower[2]) & (z_coordinates <= atomic_upper[2]))[:, None, None]
    if not np.any(sample_feasible_mask):
        raise ValueError("The selected threshold produced an empty W/Se feasible mask.")
    sample_level_set_A, level_set_metadata = distance_scaled_level_set(
        sample_smoothed,
        threshold_value,
        sample_voxel_A,
        periodic=False,
    )

    requested_pbc = parse_pbc(args.pbc)
    if not np.all(requested_pbc):
        raise ValueError(
            "The WSe2 FairChem workflow requires pbc=true,true,true; vacuum is "
            "represented explicitly inside the periodic z cell."
        )
    requested_vacuum_padding_A = parse_vacuum_padding(
        args.vacuum_padding_angstrom
    )
    # The FFT image renderer requires a single uniform Cartesian grid, so each
    # per-side padding must contain a whole number of voxels. Snap the requested
    # physical padding to the nearest representable value and use that realized
    # value consistently for both the render grid and the FairChem cell.
    sample_voxel_xyz_A = sample_voxel_A[::-1].copy()
    padding_slices_xyz = np.rint(
        requested_vacuum_padding_A / sample_voxel_xyz_A
    ).astype(int)
    padding_slices_xyz = np.where(
        requested_vacuum_padding_A > 0.0,
        np.maximum(padding_slices_xyz, 1),
        0,
    )
    vacuum_padding_A = padding_slices_xyz * sample_voxel_xyz_A
    padding_slices_zyx = padding_slices_xyz[::-1].copy()
    sample_shape_zyx = np.asarray(sample_target_image.shape, dtype=int)
    full_shape_array = sample_shape_zyx + 2 * padding_slices_zyx
    full_shape = tuple(int(value) for value in full_shape_array)
    roi_start_zyx = padding_slices_zyx
    roi_stop_zyx = padding_slices_zyx + sample_shape_zyx
    roi_zyx = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(roi_start_zyx, roi_stop_zyx)
    )
    target_image = np.zeros(full_shape, dtype=np.float32)
    target_linear_normalized = np.zeros(full_shape, dtype=np.float32)
    target_sqrt = np.zeros(full_shape, dtype=np.float32)
    smoothed_target = np.zeros(full_shape, dtype=np.float32)
    feasible_mask = np.zeros(full_shape, dtype=bool)
    image_loss_mask = np.zeros(full_shape, dtype=bool)
    target_image[roi_zyx] = sample_target_image
    target_linear_normalized[roi_zyx] = sample_linear_normalized
    target_sqrt[roi_zyx] = sample_target_sqrt
    smoothed_target[roi_zyx] = sample_smoothed
    feasible_mask[roi_zyx] = sample_feasible_mask
    image_loss_mask[roi_zyx] = True

    # The periodic spline sees a negative level set throughout all six vacuum
    # regions. Outside the measured ROI, its magnitude is the physical distance
    # to the nearest measured voxel; the experimental intensity level set is
    # retained exactly inside the ROI.
    outside_distance_A = distance_transform_edt(
        ~image_loss_mask,
        sampling=sample_voxel_A,
    )
    level_set_A = (-outside_distance_A).astype(np.float32)
    level_set_A[roi_zyx] = sample_level_set_A

    original_cell = np.diag(sample_cell_lengths_A).astype(float)
    padded_cell, position_shift = vacuum_padded_cell_and_shift(
        original_cell,
        vacuum_padding_A,
    )
    expected_cell_lengths_xyz_A = full_shape_array[::-1] * sample_voxel_xyz_A
    if not np.allclose(
        np.diag(padded_cell),
        expected_cell_lengths_xyz_A,
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise RuntimeError("Padded xyz cell and padded render grid are inconsistent.")

    payload = {
        "tiff_path": tiff_path,
        "source_shape_zyx": tuple(int(v) for v in source.shape),
        "source_voxel_size_A": source_voxel_A,
        "source_crop": source_crop,
        "crop_origin": np.asarray([0.0, float(y0), float(x0)]),
        "crop_origin_yx": np.asarray([y0, x0], dtype=int),
        "sample_shape_zyx": tuple(int(v) for v in sample_shape_zyx),
        "target_shape_zyx": tuple(int(v) for v in full_shape),
        "target_voxel_size_A": sample_voxel_A,
        "sample_cell": original_cell,
        "cell": padded_cell,
        "cell_length_A": float(sample_cell_lengths_A[0]),
        "sample_cell_lengths_A": sample_cell_lengths_A,
        "requested_vacuum_padding_A": requested_vacuum_padding_A,
        "vacuum_padding_A": vacuum_padding_A,
        "vacuum_padding_slices_xyz": padding_slices_xyz,
        "vacuum_padding_slices_zyx": padding_slices_zyx,
        "vacuum_position_shift_A": position_shift,
        "sample_roi_start_zyx": roi_start_zyx,
        "sample_roi_stop_zyx": roi_stop_zyx,
        "sample_roi_z_start": int(roi_start_zyx[0]),
        "sample_roi_z_stop": int(roi_stop_zyx[0]),
        "image_loss_mask": image_loss_mask,
        "image_loss_voxel_count": int(np.count_nonzero(image_loss_mask)),
        "target_background": background,
        "target_normalization": normalization,
        "target_linear_normalized": target_linear_normalized,
        "target_sqrt": target_sqrt,
        "target_image": target_image,
        "sample_target_linear_normalized": sample_linear_normalized,
        "sample_target_sqrt": sample_target_sqrt,
        "sample_target_image": sample_target_image,
        "image_transform": image_transform,
        "image_transform_epsilon": float(args.gaussian_transform_epsilon),
        "smoothed_target_sqrt": smoothed_target,
        "sample_smoothed_target_sqrt": sample_smoothed,
        "threshold_mode": threshold_mode,
        "threshold_config": float(args.mask_threshold),
        "threshold_value": threshold_value,
        "heavy_allowed_mask": feasible_mask,
        "heavy_allowed_fraction": float(sample_feasible_mask.mean()),
        "heavy_level_set_A": level_set_A,
        "feasible_mask": feasible_mask,
        "sample_feasible_mask": sample_feasible_mask,
        "feasible_fraction": float(sample_feasible_mask.mean()),
        "level_set_A": level_set_A,
        "sample_level_set_A": sample_level_set_A,
        "level_set_metadata": level_set_metadata,
        "pbc": requested_pbc,
    }

    if write_diagnostics:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir = output_dir / "initial_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        image_vmax = max(float(np.percentile(sample_target_image, 99.5)), 1e-12)
        image_u16 = np.round(
            np.clip(target_image / image_vmax, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
        tifffile.imwrite(
            diagnostics_dir / f"experimental_target_{image_transform}_padded_ZYX.tif",
            image_u16,
            imagej=True,
            metadata={"axes": "ZYX", "unit": "angstrom", "spacing": float(sample_voxel_A[0])},
        )
        tifffile.imwrite(
            diagnostics_dir / "experimental_image_loss_mask_padded_ZYX.tif",
            image_loss_mask.astype(np.uint8) * 255,
            imagej=True,
            metadata={"axes": "ZYX", "unit": "angstrom", "spacing": float(sample_voxel_A[0])},
        )
        tifffile.imwrite(
            diagnostics_dir / "wse2_feasible_mask_padded_ZYX.tif",
            feasible_mask.astype(np.uint8) * 255,
            imagej=True,
            metadata={"axes": "ZYX", "unit": "angstrom", "spacing": float(sample_voxel_A[0])},
        )
        np.savez_compressed(
            diagnostics_dir / "experimental_target_and_wse2_region.npz",
            target_image=target_image,
            target_linear_normalized=target_linear_normalized,
            image_loss_mask=image_loss_mask,
            feasible_mask=feasible_mask,
            level_set_A=level_set_A,
            target_voxel_size_A=sample_voxel_A,
            sample_cell=original_cell,
            padded_cell=padded_cell,
            requested_vacuum_padding_A=requested_vacuum_padding_A,
            vacuum_padding_A=vacuum_padding_A,
            vacuum_padding_slices_xyz=padding_slices_xyz,
            vacuum_padding_slices_zyx=padding_slices_zyx,
            vacuum_position_shift_A=position_shift,
            sample_roi_start_zyx=roi_start_zyx,
            sample_roi_stop_zyx=roi_stop_zyx,
            crop_origin_yx=np.asarray([y0, x0], dtype=int),
            source_tiff=np.asarray(str(tiff_path)),
        )
        geometry_summary = {
            "atomic_z_bounds_experimental_A": [float(atomic_lower[2]), float(atomic_upper[2])],
            "experimental_tiff": str(tiff_path),
            "source_shape_zyx": list(source.shape),
            "center_crop_origin_yx": [int(y0), int(x0)],
            "center_crop_size_source_pixels": int(crop_size),
            "sample_shape_zyx": list(sample_target_image.shape),
            "padded_render_shape_zyx": list(full_shape),
            "target_voxel_size_zyx_angstrom": sample_voxel_A.tolist(),
            "sample_cell_lengths_xyz_angstrom": sample_cell_lengths_A.tolist(),
            "requested_vacuum_padding_each_side_xyz_angstrom": (
                requested_vacuum_padding_A.tolist()
            ),
            "vacuum_padding_each_side_xyz_angstrom": vacuum_padding_A.tolist(),
            "vacuum_padding_slices_xyz": padding_slices_xyz.tolist(),
            "vacuum_padding_slices_zyx": padding_slices_zyx.tolist(),
            "padded_cell_lengths_xyz_angstrom": np.diag(padded_cell).tolist(),
            "pbc_xyz": requested_pbc.astype(bool).tolist(),
            "sample_roi_start_zyx": roi_start_zyx.tolist(),
            "sample_roi_stop_zyx": roi_stop_zyx.tolist(),
            "image_loss_voxel_count": int(np.count_nonzero(image_loss_mask)),
            "expected_image_loss_voxel_count": int(np.prod(sample_target_image.shape)),
            "image_loss_normalization": "sum_masked_huber_divided_by_mask_voxel_count",
            "candidate_volume_excludes_vacuum": True,
        }
        (diagnostics_dir / "geometry_and_mask_summary.json").write_text(
            json.dumps(geometry_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload
