#!/usr/bin/env python3
"""Reconstruct a hidden H sublattice with EAT and a genetic algorithm.

The reference structure can come from the Materials Project (the legacy
default) or from an experimental CIF/XSF supplied in the YAML file.  In both
cases the complete unit cell is relaxed with FAIRChem first.  The requested
variable species (normally H) are then removed and EAT searches only their
sublattice while keeping the relaxed heavy-atom scaffold and cell fixed.
Random members can use either legacy fractional EAT initialization or the
ordinary baseline's fixed-count hard initialization. Each syntropized member
is hard-rounded, FIRE-relaxed with fixed scaffold/cell, and ranked by that
relaxed hard score. Parent A and B come from independent best-of tournaments.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from minimizer import minimizer


MAX_NUM_ELEMENTS = 100
SYM_DUMMY = "H"
SCRIPT_DIR = Path(__file__).resolve().parent


OUTPUT_DIR = str(SCRIPT_DIR / "out_Mg2FeH6_H_sublattice_GA")
MP_API_KEY = os.environ.get("MP_API_KEY", "")
MODEL_NAME = "uma-s-1p1"
TASK_NAME = "omat"
# This benchmark is intended to run on the GPU. The optional Triton bmm
# override is disabled below, matching the working dislocation script; UMA
# then uses standard ATen bmm kernels without compiling Triton's cuda_utils.c.
MODEL_DEVICE = "cuda"
MP_MATERIAL_ID = ""
MP_FORMULA = "Mg2FeH6"
SPECIES = "Mg, Fe, H"
VARIABLE_SPECIES = "H"
SUPERCELL = "1, 1, 1"
FEASIBLE_GRID = "16, 16, 16"
FEASIBLE_MIN_DISTANCE_ANGSTROM = 1.0
INITIAL_H_H_MIN_DISTANCE_ANGSTROM = 1.0
RELAX_FMAX = 0.05
RELAX_STEPS = 200
RELAX_MAXSTEP = 0.10
RELAX_CELL = True
RELAX_SUPERCELL = False
PBC = "true, true, true"

NULL_DENSITY_PER_ANGSTROM3 = 0.15
SUBCELL_GRID = "4, 4, 4"
INIT_OCC_TOTAL = 1.0
INIT_OCC_NOISE = 0.0
INIT_SPECIES_DIRICHLET_ALPHA = 5.0
INIT_POSITION_MODE = "subcell_grid"
INIT_POSITION_JITTER_ANGSTROM = 0.15
KEEP_THRESH = 1e-3
SEED = None
RANDOM_MEMBER_INITIALIZATION = "eat"
DISCRETE_COUNT_MIN = 1
DISCRETE_COUNT_MAX_FACTOR = 1.0
DISCRETE_REPAIR_ATTEMPTS = 2000

FINAL_RELAX_ENABLE = True
FINAL_RELAX_FMAX = 0.05
FINAL_RELAX_STEPS = 500
FINAL_RELAX_MAXSTEP_ANGSTROM = 0.10
FINAL_RELAX_UNCONVERGED_PENALTY = 0.0

COMPOSITION_PENALTY = 20.0
HOLE_BOUNDARY_PENALTY = 20.0
HOLE_BOUNDARY_MARGIN_ANGSTROM = 0.0
OCCUPANCY_LOWER_BOUND_ENABLE = False
OCCUPANCY_LOWER_BOUND_GAMMA = 0.7
OCCUPANCY_LOWER_BOUND_PENALTY = 0.05
OCCUPANCY_LOWER_BOUND_SOFTPLUS_SHARPNESS = 1.0
TSALLIS_Q = 2.0
BETA0 = 1e-3
BETA_MULT = 1.6
BETA_MAX_STAGES = 0
ENTROPY_STOP_AVG = 1e-4
BINARY_STOP_MAXMIN = 1e-3
ALSO_REQUIRE_BINARY_STOP = False

MAXIT = 100
TOL = 5e-3
BFGS_HIST = 10
MAX_LINE_SEARCH = 3
DISABLE_TRAJECTORY_HISTORY = True
HISTORY_EVERY = 1

GA_ENABLE = True
GA_POPULATION_SIZE = 20
GA_GENERATIONS = 20
GA_PARENT_POOL_SIZE = 19
GA_ELITE_FRACTION = 0.10
GA_CROSSOVER_FRACTION = 0.80
GA_RANDOM_FRACTION = 0.10
GA_PARENT_A_TOURNAMENT_SIZE = 4
GA_SUPERPOSITION_OCC_SCALE = 0.3
GA_SUPERPOSITION_PRUNE_DISTANCE_ANGSTROM = 0.5
GA_SUPERPOSITION_PRUNE_OCC_THRESH = 1e-3
GA_CROSSOVER_POS_SIGMA = 0.12
GA_CROSSOVER_OCC_SIGMA = 0.12
GA_CROSSOVER_SPECIES_SIGMA = 0.12
GA_IMPORT_STRUCTURES_JSON = ""
GA_IMPORT_PARENT_COUNT = 0
GA_LIVE_PLOT_ENABLE = True
GA_LIVE_PLOT_FILE = ""
GA_LIVE_PLOT_METRIC = "score"
GA_LIVE_PLOT_ALPHA = 0.45
GA_LIVE_PLOT_JITTER = 0.16
GA_LOG_FILE = ""
GA_PARALLEL_WORKERS = 1
GA_PARALLEL_GPU_IDS = "0,1"
GA_EXECUTION_BACKEND = "local"

TERMINAL_VERBOSE = False
HEARTBEAT_INTERVAL_SECONDS = 600.0
RUNNING_LOG_FILE = ""

_BMM_OVERRIDE_ATTEMPTED = False


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_suffix(".yaml")


DEFAULT_CONFIG = {
    "paths": {
        "output_dir": OUTPUT_DIR,
        "import_structures_json": GA_IMPORT_STRUCTURES_JSON,
    },
    "model": {
        "mp_api_key": "",
        "model_name": MODEL_NAME,
        "task_name": TASK_NAME,
        "device": MODEL_DEVICE,
    },
    "crystal": {
        "material_id": MP_MATERIAL_ID,
        "material_formula": MP_FORMULA,
        "ground_truth_file": "",
        "species": SPECIES,
        "variable_species": VARIABLE_SPECIES,
        "supercell": SUPERCELL,
        "feasible_grid": FEASIBLE_GRID,
        "feasible_min_distance_angstrom": FEASIBLE_MIN_DISTANCE_ANGSTROM,
        "initial_h_h_min_distance_angstrom": INITIAL_H_H_MIN_DISTANCE_ANGSTROM,
        "pbc": PBC,
    },
    "ground_truth_relaxation": {
        "fmax": RELAX_FMAX,
        "steps": RELAX_STEPS,
        "maxstep": RELAX_MAXSTEP,
        "relax_cell": RELAX_CELL,
        "relax_supercell": RELAX_SUPERCELL,
    },
    "initialization": {
        "null_density_per_angstrom3": NULL_DENSITY_PER_ANGSTROM3,
        "subcell_grid": SUBCELL_GRID,
        "init_occ_total": INIT_OCC_TOTAL,
        "init_occ_noise": INIT_OCC_NOISE,
        "init_species_dirichlet_alpha": INIT_SPECIES_DIRICHLET_ALPHA,
        "init_position_mode": INIT_POSITION_MODE,
        "init_position_jitter_angstrom": INIT_POSITION_JITTER_ANGSTROM,
        "keep_thresh": KEEP_THRESH,
        "seed": SEED,
        "random_member_initialization": RANDOM_MEMBER_INITIALIZATION,
        "discrete_count_min": DISCRETE_COUNT_MIN,
        "discrete_count_max_factor": DISCRETE_COUNT_MAX_FACTOR,
        "discrete_repair_attempts": DISCRETE_REPAIR_ATTEMPTS,
    },
    "candidate_relaxation": {
        "enable": FINAL_RELAX_ENABLE,
        "fmax": FINAL_RELAX_FMAX,
        "steps": FINAL_RELAX_STEPS,
        "maxstep_angstrom": FINAL_RELAX_MAXSTEP_ANGSTROM,
        "unconverged_penalty": FINAL_RELAX_UNCONVERGED_PENALTY,
    },
    "objective": {
        "composition_penalty": COMPOSITION_PENALTY,
        "hole_boundary_penalty": HOLE_BOUNDARY_PENALTY,
        "hole_boundary_margin_angstrom": HOLE_BOUNDARY_MARGIN_ANGSTROM,
        "occupancy_lower_bound_enable": OCCUPANCY_LOWER_BOUND_ENABLE,
        "occupancy_lower_bound_gamma": OCCUPANCY_LOWER_BOUND_GAMMA,
        "occupancy_lower_bound_penalty": OCCUPANCY_LOWER_BOUND_PENALTY,
        "occupancy_lower_bound_softplus_sharpness": OCCUPANCY_LOWER_BOUND_SOFTPLUS_SHARPNESS,
    },
    "syntropization": {
        "tsallis_q": TSALLIS_Q,
        "beta0": BETA0,
        "beta_mult": BETA_MULT,
        "beta_max_stages": BETA_MAX_STAGES,
        "entropy_stop_avg": ENTROPY_STOP_AVG,
        "binary_stop_maxmin": BINARY_STOP_MAXMIN,
        "also_require_binary_stop": ALSO_REQUIRE_BINARY_STOP,
    },
    "minimizer": {
        "maxit": MAXIT,
        "tol": TOL,
        "bfgs_hist": BFGS_HIST,
        "max_line_search": MAX_LINE_SEARCH,
    },
    "history": {
        "disable_trajectory_history": DISABLE_TRAJECTORY_HISTORY,
        "history_every": HISTORY_EVERY,
    },
    "genetic": {
        "enable": GA_ENABLE,
        "population_size": GA_POPULATION_SIZE,
        "generations": GA_GENERATIONS,
        "elite_fraction": GA_ELITE_FRACTION,
        "crossover_fraction": GA_CROSSOVER_FRACTION,
        "random_fraction": GA_RANDOM_FRACTION,
        "parent_pool_size": GA_PARENT_POOL_SIZE,
        "import_parent_count": GA_IMPORT_PARENT_COUNT,
        "parent_a_tournament_size": GA_PARENT_A_TOURNAMENT_SIZE,
        "superposition_occ_scale": GA_SUPERPOSITION_OCC_SCALE,
        "superposition_prune_distance_angstrom": GA_SUPERPOSITION_PRUNE_DISTANCE_ANGSTROM,
        "superposition_prune_occ_thresh": GA_SUPERPOSITION_PRUNE_OCC_THRESH,
        "crossover_pos_sigma": GA_CROSSOVER_POS_SIGMA,
        "crossover_occ_sigma": GA_CROSSOVER_OCC_SIGMA,
        "crossover_species_sigma": GA_CROSSOVER_SPECIES_SIGMA,
        "live_plot_enable": GA_LIVE_PLOT_ENABLE,
        "live_plot_file": GA_LIVE_PLOT_FILE,
        "live_plot_metric": GA_LIVE_PLOT_METRIC,
        "live_plot_alpha": GA_LIVE_PLOT_ALPHA,
        "live_plot_jitter": GA_LIVE_PLOT_JITTER,
        "parallel_workers": GA_PARALLEL_WORKERS,
        "parallel_gpu_ids": GA_PARALLEL_GPU_IDS,
        "execution_backend": GA_EXECUTION_BACKEND,
        "log_file": GA_LOG_FILE,
    },
    "runtime": {
        "terminal_verbose": TERMINAL_VERBOSE,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "running_log_file": RUNNING_LOG_FILE,
    },
}


CONFIG_KEY_MAP = {
    "paths": {
        "output_dir": "output_dir",
        "import_structures_json": "ga_import_structures_json",
    },
    "model": {
        "mp_api_key": "mp_api_key",
        "model_name": "model_name",
        "task_name": "task_name",
        "device": "device",
    },
    "crystal": {
        "material_id": "material_id",
        "material_formula": "material_formula",
        "ground_truth_file": "ground_truth_file",
        "species": "species",
        "variable_species": "variable_species",
        "supercell": "supercell",
        "feasible_grid": "feasible_grid",
        "feasible_min_distance_angstrom": "feasible_min_distance_angstrom",
        "initial_h_h_min_distance_angstrom": "initial_h_h_min_distance_angstrom",
        "pbc": "pbc",
    },
    "ground_truth_relaxation": {
        "fmax": "relax_fmax",
        "steps": "relax_steps",
        "maxstep": "relax_maxstep",
        "relax_cell": "relax_cell",
        "relax_supercell": "relax_supercell",
    },
    "initialization": {
        "null_density_per_angstrom3": "null_density_per_angstrom3",
        "subcell_grid": "subcell_grid",
        "init_occ_total": "init_occ_total",
        "init_occ_noise": "init_occ_noise",
        "init_species_dirichlet_alpha": "init_species_dirichlet_alpha",
        "init_position_mode": "init_position_mode",
        "init_position_jitter_angstrom": "init_position_jitter_angstrom",
        "keep_thresh": "keep_thresh",
        "seed": "seed",
        "random_member_initialization": "random_member_initialization",
        "discrete_count_min": "discrete_count_min",
        "discrete_count_max_factor": "discrete_count_max_factor",
        "discrete_repair_attempts": "discrete_repair_attempts",
    },
    "candidate_relaxation": {
        "enable": "final_relax_enable",
        "fmax": "final_relax_fmax",
        "steps": "final_relax_steps",
        "maxstep_angstrom": "final_relax_maxstep_angstrom",
        "unconverged_penalty": "final_relax_unconverged_penalty",
    },
    "objective": {
        "composition_penalty": "composition_penalty",
        "hole_boundary_penalty": "hole_boundary_penalty",
        "hole_boundary_margin_angstrom": "hole_boundary_margin_angstrom",
        "occupancy_lower_bound_enable": "occupancy_lower_bound_enable",
        "occupancy_lower_bound_gamma": "occupancy_lower_bound_gamma",
        "occupancy_lower_bound_penalty": "occupancy_lower_bound_penalty",
        "occupancy_lower_bound_softplus_sharpness": "occupancy_lower_bound_softplus_sharpness",
    },
    "syntropization": {
        "tsallis_q": "tsallis_q",
        "beta0": "beta0",
        "beta_mult": "beta_mult",
        "beta_max_stages": "beta_max_stages",
        "entropy_stop_avg": "entropy_stop_avg",
        "binary_stop_maxmin": "binary_stop_maxmin",
        "also_require_binary_stop": "also_require_binary_stop",
    },
    "minimizer": {
        "maxit": "maxit",
        "tol": "tol",
        "bfgs_hist": "bfgs_hist",
        "max_line_search": "max_line_search",
    },
    "history": {
        "disable_trajectory_history": "disable_trajectory_history",
        "history_every": "history_every",
    },
    "genetic": {
        "enable": "ga_enable",
        "population_size": "ga_population_size",
        "generations": "ga_generations",
        "elite_fraction": "ga_elite_fraction",
        "crossover_fraction": "ga_crossover_fraction",
        "random_fraction": "ga_random_fraction",
        "parent_pool_size": "ga_parent_pool_size",
        "import_parent_count": "ga_import_parent_count",
        "parent_a_tournament_size": "ga_parent_a_tournament_size",
        "superposition_occ_scale": "ga_superposition_occ_scale",
        "superposition_prune_distance_angstrom": "ga_superposition_prune_distance_angstrom",
        "superposition_prune_occ_thresh": "ga_superposition_prune_occ_thresh",
        "crossover_pos_sigma": "ga_crossover_pos_sigma",
        "crossover_occ_sigma": "ga_crossover_occ_sigma",
        "crossover_species_sigma": "ga_crossover_species_sigma",
        "live_plot_enable": "ga_live_plot_enable",
        "live_plot_file": "ga_live_plot_file",
        "live_plot_metric": "ga_live_plot_metric",
        "live_plot_alpha": "ga_live_plot_alpha",
        "live_plot_jitter": "ga_live_plot_jitter",
        "parallel_workers": "ga_parallel_workers",
        "parallel_gpu_ids": "ga_parallel_gpu_ids",
        "execution_backend": "ga_execution_backend",
        "log_file": "ga_log_file",
    },
    "runtime": {
        "terminal_verbose": "terminal_verbose",
        "heartbeat_interval_seconds": "heartbeat_interval_seconds",
        "running_log_file": "running_log_file",
    },
}


def strip_yaml_comment(line):
    in_quote = None
    for i, ch in enumerate(line):
        if ch in ("'", '"'):
            in_quote = None if in_quote == ch else (ch if in_quote is None else in_quote)
        elif ch == "#" and in_quote is None:
            return line[:i]
    return line


def parse_yaml_scalar(text):
    value = text.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_simple_yaml(path):
    data = {}
    current_section = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = strip_yaml_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            if not stripped.endswith(":"):
                raise ValueError(f"Top-level YAML entries must be sections: {raw_line!r}")
            current_section = stripped[:-1].strip()
            data.setdefault(current_section, {})
            continue
        if current_section is None or ":" not in stripped:
            raise ValueError(f"Malformed YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        data[current_section][key.strip()] = parse_yaml_scalar(value)
    return data


def deep_update_config(base, override):
    out = copy.deepcopy(base)
    for section, values in override.items():
        if section not in out or not isinstance(values, dict):
            out[section] = values
            continue
        out[section].update(values)
    return out


def flatten_config(config):
    flat = {}
    for section, mapping in CONFIG_KEY_MAP.items():
        for key, attr in mapping.items():
            if section in config and key in config[section]:
                flat[attr] = config[section][key]
    return SimpleNamespace(**flat)


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Reconstruct a hidden hydrogen sublattice with EAT/GA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    return parser.parse_args()


def get_settings(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    if path.exists():
        config = deep_update_config(config, read_simple_yaml(path))
    args = flatten_config(config)
    args.config_path = str(path)
    args.mp_api_key = args.mp_api_key or os.environ.get("MP_API_KEY", "")
    return args


def validate_new_ga_controls(args):
    mode = str(args.random_member_initialization).strip().lower()
    if mode not in {"eat", "legacy_eat", "fractional", "discrete", "ordinary_discrete", "poisson_discrete"}:
        raise ValueError("initialization.random_member_initialization must be 'eat' or 'discrete'.")
    if int(args.discrete_count_min) < 1:
        raise ValueError("initialization.discrete_count_min must be at least 1.")
    if float(args.discrete_count_max_factor) < 1.0:
        raise ValueError("initialization.discrete_count_max_factor must be at least 1.0.")
    if int(args.discrete_repair_attempts) < 1:
        raise ValueError("initialization.discrete_repair_attempts must be positive.")
    if int(args.ga_parent_a_tournament_size) < 1:
        raise ValueError("genetic.parent_a_tournament_size must be positive.")
    if bool(args.final_relax_enable):
        if float(args.final_relax_fmax) <= 0.0:
            raise ValueError("candidate_relaxation.fmax must be positive.")
        if int(args.final_relax_steps) < 1:
            raise ValueError("candidate_relaxation.steps must be positive.")
        if float(args.final_relax_maxstep_angstrom) <= 0.0:
            raise ValueError("candidate_relaxation.maxstep_angstrom must be positive.")
    if float(args.final_relax_unconverged_penalty) < 0.0:
        raise ValueError("candidate_relaxation.unconverged_penalty cannot be negative.")


def json_safe_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, SimpleNamespace):
        return {str(k): json_safe_value(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    return value


def write_run_parameters(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(getattr(args, "config_path", "")).expanduser()
    payload = {
        "script": str(Path(__file__).resolve()),
        "config_path": str(config_path),
        "resolved_parameters": {
            str(key): json_safe_value(value)
            for key, value in sorted(vars(args).items())
        },
    }
    if config_path.exists():
        payload["config_file_text"] = config_path.read_text(encoding="utf-8")
    path = output_dir / "run_parameters.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def create_new_output_directory_or_fail(args):
    """Atomically reserve a new run directory without overwriting prior work."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output directory exists: {output_dir}") from exc
    return output_dir


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")


def timestamp_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def terminal_print(args, *values, **kwargs):
    if bool(getattr(args, "terminal_verbose", False)):
        print(*values, **kwargs)


def running_log_path_from_args(args):
    path = str(getattr(args, "running_log_file", "") or "").strip()
    if path:
        return Path(path).expanduser()
    return Path(args.output_dir).expanduser().resolve() / "running_log.txt"


def append_running_log(args, message):
    path = running_log_path_from_args(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp_now()} | {message}\n")


def parse_int_triplet(text, name):
    if isinstance(text, (list, tuple)):
        vals = [int(x) for x in text]
    else:
        vals = [int(float(x.strip())) for x in str(text).replace("x", ",").split(",") if x.strip()]
    if len(vals) != 3 or any(v <= 0 for v in vals):
        raise ValueError(f"{name} must contain three positive integers.")
    return np.asarray(vals, dtype=int)


def parse_pbc(text):
    if isinstance(text, bool):
        return np.asarray([text, text, text], dtype=bool)
    if isinstance(text, (list, tuple)):
        parts = list(text)
    else:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise ValueError("pbc must be one boolean or three comma-separated booleans.")
    return np.asarray([str(p).strip().lower() in ("true", "1", "yes", "on") for p in parts], dtype=bool)


def parse_species_arg(text, minimum=2):
    species = [s.strip() for s in str(text).replace(";", ",").split(",") if s.strip()]
    if len(species) < int(minimum):
        raise ValueError(f"Species list must contain at least {int(minimum)} real species.")
    for symbol in species:
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown chemical symbol {symbol!r}.")
    return species


def doc_value(doc, name, default=None):
    if doc is None:
        return default
    if isinstance(doc, dict):
        return doc.get(name, default)
    return getattr(doc, name, default)


def structure_to_ase_atoms(structure):
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    if not getattr(structure, "is_ordered", True):
        raise ValueError("Disordered Materials Project structures are not supported.")
    species = [str(getattr(site.specie, "symbol", site.specie)) for site in structure.sites]
    frac_positions = np.asarray(structure.frac_coords, dtype=float)
    return Atoms(symbols=species, scaled_positions=frac_positions, cell=lattice, pbc=True)


def resolve_path_from_config(path_text, config_path):
    """Resolve a ground-truth path relative to the YAML file when needed."""
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        config_parent = Path(str(config_path or DEFAULT_CONFIG_PATH)).expanduser().resolve().parent
        path = config_parent / path
    return path.resolve()


def read_ground_truth_file(path):
    """Read one periodic CIF/XSF structure, including CIF symmetry expansion."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth structure file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".cif", ".xsf"):
        raise ValueError(
            f"Unsupported ground_truth_file format {suffix!r}; use a CIF or XSF file."
        )

    # ASE's CIF reader applies the space-group symmetry operations and returns
    # the full conventional cell, rather than only the asymmetric unit.  XSF
    # is read directly as a periodic cell.  index=0 keeps the behavior
    # deterministic for files containing multiple frames.
    atoms = read(path, index=0, format=suffix[1:])
    if isinstance(atoms, (list, tuple)):
        if not atoms:
            raise ValueError(f"No structures were found in {path}")
        atoms = atoms[0]
    if not isinstance(atoms, Atoms) or len(atoms) == 0:
        raise ValueError(f"No atoms were found in ground-truth structure {path}")
    atoms = atoms.copy()
    if int(getattr(atoms.cell, "rank", 0)) < 3:
        raise ValueError(
            f"Ground-truth structure {path} does not contain a full 3D cell; "
            "a periodic CIF/XSF is required."
        )
    atoms.set_pbc((True, True, True))
    atoms.wrap()
    counts = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return atoms, {
        "material_id": "",
        "formula_pretty": atoms.get_chemical_formula(),
        "energy_above_hull": None,
        "source_type": "file",
        "source_file": str(path),
        "source_format": suffix[1:],
        "source_composition_counts": counts,
    }


def fetch_material_project_atoms(material_id, formula, api_key):
    """Fetch one ordered MP structure by ID, or the lowest-hull formula match."""
    api_key = str(api_key or MP_API_KEY or os.environ.get("MP_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("Set model.mp_api_key or MP_API_KEY to query the Materials Project.")
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        if str(material_id or "").strip():
            docs = mpr.materials.summary.search(
                material_ids=[str(material_id).strip()],
                fields=["material_id", "formula_pretty", "energy_above_hull", "structure"],
            )
        else:
            if not str(formula or "").strip():
                raise ValueError("Provide crystal.material_id or crystal.material_formula.")
            docs = mpr.materials.summary.search(
                formula=str(formula).strip(),
                fields=["material_id", "formula_pretty", "energy_above_hull", "structure"],
            )
    if not docs:
        query = str(material_id or formula)
        raise RuntimeError(f"No Materials Project structure found for {query!r}.")
    docs = sorted(
        docs,
        key=lambda d: (
            float(doc_value(d, "energy_above_hull", 1e9) or 1e9),
            str(doc_value(d, "material_id", "")),
        ),
    )
    doc = docs[0]
    structure = doc_value(doc, "structure")
    if structure is None:
        raise RuntimeError("Materials Project result did not contain a structure.")
    return structure_to_ase_atoms(structure), {
        "material_id": str(doc_value(doc, "material_id", "")),
        "formula_pretty": str(doc_value(doc, "formula_pretty", "")),
        "energy_above_hull": float(doc_value(doc, "energy_above_hull", 0.0) or 0.0),
        "source_type": "materials_project",
        "source_file": "",
        "source_format": "",
    }


def relax_with_fairchem(atoms, calc, args, label):
    """Relax a real-atom structure before H removal and EAT."""
    relaxed = atoms.copy()
    relaxed.calc = calc
    if bool(args.relax_cell):
        optimizer = FIRE(FrechetCellFilter(relaxed), logfile=None, maxstep=float(args.relax_maxstep))
    else:
        optimizer = FIRE(relaxed, logfile=None, maxstep=float(args.relax_maxstep))
    converged = optimizer.run(fmax=float(args.relax_fmax), steps=int(args.relax_steps))
    relaxed.wrap()
    forces = np.asarray(relaxed.get_forces(), dtype=float)
    energy = float(relaxed.get_potential_energy())
    info = {
        "label": str(label),
        "converged": bool(converged),
        "steps": int(getattr(optimizer, "nsteps", 0)),
        "total_energy_eV": energy,
        "energy_per_atom_eV": energy / max(1, len(relaxed)),
        "max_force_eV_per_angstrom": float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0,
        "relax_cell": bool(args.relax_cell),
    }
    return relaxed, info


def fetch_elemental_reference_atoms(symbol, api_key):
    if not api_key:
        raise ValueError("An MP API key is required for elemental references. Set model.mp_api_key or MP_API_KEY.")
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=symbol,
            fields=["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom", "structure"],
        )
    if not docs:
        raise RuntimeError(f"No Materials Project elemental reference found for {symbol}.")
    docs = sorted(docs, key=lambda d: (float(doc_value(d, "energy_above_hull", 1e9) or 1e9), str(doc_value(d, "material_id", ""))))
    doc = docs[0]
    return {
        "symbol": symbol,
        "material_id": str(doc_value(doc, "material_id")),
        "formula_pretty": str(doc_value(doc, "formula_pretty")),
        "energy_above_hull": float(doc_value(doc, "energy_above_hull", 0.0) or 0.0),
        "formation_energy_per_atom": float(doc_value(doc, "formation_energy_per_atom", 0.0) or 0.0),
        "atoms": structure_to_ase_atoms(doc_value(doc, "structure")),
    }


def project_simplex_rows(v):
    if v.numel() == 0:
        return v
    u, _ = torch.sort(v, dim=1, descending=True)
    cssv = torch.cumsum(u, dim=1) - 1.0
    ind = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype).reshape(1, -1)
    cond = u - cssv / ind > 0.0
    rho = cond.sum(dim=1) - 1
    theta = cssv[torch.arange(v.shape[0], device=v.device), rho] / (rho.to(v.dtype) + 1.0)
    return torch.clamp(v - theta[:, None], min=0.0)


def project_capped_simplex(occ):
    y = torch.clamp(occ, min=0.0)
    sums = y.sum(dim=1)
    mask = sums > 1.0
    if bool(mask.any().item()):
        y = y.clone()
        y[mask] = project_simplex_rows(y[mask])
    return y


def tsallis_entropy_term(occ, q):
    null = torch.clamp(1.0 - occ.sum(dim=1), min=0.0)
    return torch.sum((1.0 - torch.sum(occ**q, dim=1) - null**q) / (q - 1.0))


def tsallis_entropy_grad(occ, q):
    null = torch.clamp(1.0 - occ.sum(dim=1), min=0.0)
    return (q / (q - 1.0)) * (null[:, None] ** (q - 1.0) - occ ** (q - 1.0))


def binarity_metrics(occ, q):
    ent_avg = float(tsallis_entropy_term(occ, q).item()) / max(1, occ.shape[0])
    null = torch.clamp(1.0 - occ.sum(dim=1), min=0.0)
    probs = torch.cat([occ, null[:, None]], dim=1)
    max_nonbinary = float((1.0 - probs.max(dim=1).values).max().item()) if probs.numel() else 0.0
    return ent_avg, max_nonbinary


def set_enable_eat_grad(atoms, flag):
    if flag:
        atoms.info["enable_eat_grad"] = True
    else:
        atoms.info.pop("enable_eat_grad", None)


def set_multi_species_eat_weights(atoms, weights_np, atomic_numbers_for_species):
    weights_np = np.asarray(weights_np, dtype=np.float32).reshape(len(atoms), len(atomic_numbers_for_species))
    ew = np.zeros((len(atoms), MAX_NUM_ELEMENTS), dtype=np.float32)
    for species_idx, atomic_number in enumerate(atomic_numbers_for_species):
        ew[:, int(atomic_number)] = weights_np[:, species_idx]
    if "eat_weights" in atoms.arrays:
        atoms.arrays["eat_weights"][:] = ew
    else:
        atoms.new_array("eat_weights", ew)


def get_energy_gradients_and_forces(atoms, calc, atomic_numbers_for_species, want_grad=True):
    a = atoms.copy()
    set_enable_eat_grad(a, want_grad)
    a.calc = calc
    energy = float(a.get_potential_energy())
    forces = np.asarray(a.get_forces(), dtype=float)
    if not want_grad:
        return energy, None, forces
    eat_grad = np.asarray(a.calc.results["eat_grad"], dtype=float)
    return energy, eat_grad[:, [int(z) for z in atomic_numbers_for_species]], forces


def split_occupancy_and_species_probs(occ, fallback_species_probs=None):
    occupancy = torch.clamp(occ.sum(dim=1, keepdim=True), min=0.0, max=1.0)
    if fallback_species_probs is None:
        species_probs = torch.full_like(occ, 1.0 / occ.shape[1])
    else:
        species_probs = fallback_species_probs.to(device=occ.device, dtype=occ.dtype)
    active = occupancy > 1e-12
    species_from_occ = occ / occupancy.clamp_min(1e-12)
    species_probs = torch.where(active, species_from_occ, species_probs)
    return occupancy, project_simplex_rows(species_probs)


def combine_occupancy_and_species_probs(occupancy, species_probs):
    occupancy = torch.clamp(occupancy, min=0.0, max=1.0)
    species_probs = project_simplex_rows(species_probs)
    return project_capped_simplex(occupancy * species_probs)


def periodic_delta(from_position, to_position, cell, pbc=(True, True, True)):
    cell = np.asarray(cell, dtype=float)
    from_frac = np.linalg.solve(cell.T, np.asarray(from_position, dtype=float))
    to_frac = np.linalg.solve(cell.T, np.asarray(to_position, dtype=float))
    dfrac = to_frac - from_frac
    pbc = np.asarray(pbc, dtype=bool).reshape(3)
    dfrac[pbc] = dfrac[pbc] - np.round(dfrac[pbc])
    return dfrac @ cell


def periodic_pair_distance2(positions, cell, pbc=(True, True, True)):
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return np.zeros((0, 0), dtype=float)
    frac = np.linalg.solve(np.asarray(cell, dtype=float).T, positions.T).T
    ds = frac[:, None, :] - frac[None, :, :]
    pbc = np.asarray(pbc, dtype=bool).reshape(3)
    ds[:, :, pbc] = ds[:, :, pbc] - np.round(ds[:, :, pbc])
    dr = ds @ np.asarray(cell, dtype=float)
    return np.sum(dr * dr, axis=-1)


def srtiO3_basis():
    return [
        ("Sr", np.asarray([0.0, 0.0, 0.0], dtype=float)),
        ("Ti", np.asarray([0.5, 0.5, 0.5], dtype=float)),
        ("O", np.asarray([0.5, 0.5, 0.0], dtype=float)),
        ("O", np.asarray([0.5, 0.0, 0.5], dtype=float)),
        ("O", np.asarray([0.0, 0.5, 0.5], dtype=float)),
    ]


def centered_hole_start(supercell, hole_size, text):
    text = str(text).strip().lower()
    if text in ("center", "centred", "centered", ""):
        start = (supercell - hole_size) // 2
    else:
        start = parse_int_triplet(text, "crystal.hole_start")
    if np.any(start < 0) or np.any(start + hole_size > supercell):
        raise ValueError("crystal.hole_start + crystal.hole_size must fit inside crystal.supercell.")
    return start.astype(int)


def disable_optional_torch_native_bmm_override():
    """Use standard GPU ATen bmm instead of the optional Triton override.

    The UMA model's Wigner rotation path can route bmm through Triton. On
    ws09 that route attempts to compile ``cuda_utils.c`` and fails because the
    Python development header is unavailable. The dislocation benchmark
    already uses this workaround successfully while retaining CUDA execution.
    """
    global _BMM_OVERRIDE_ATTEMPTED
    if _BMM_OVERRIDE_ATTEMPTED:
        return
    _BMM_OVERRIDE_ATTEMPTED = True
    try:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_op_symbols="bmm")
        print(
            "[fairchem] disabled optional torch._native Triton bmm override; "
            "using standard ATen bmm on GPU",
            flush=True,
        )
    except (ImportError, AttributeError):
        pass


class ModelEvaluationTracker:
    """Count and time actual calculator executions, grouped by GA phase.

    One evaluation means one call to the FAIR-Chem calculator's ``calculate``
    method.  A single call may jointly produce energy, forces, and EAT
    gradients; subsequent ASE property reads served from the calculator cache
    are deliberately not counted again.
    """

    def __init__(self):
        self.phase = "setup"
        self.total_count = 0
        self.total_seconds = 0.0
        self.by_phase = {}

    def record(self, phase, elapsed_seconds):
        phase = str(phase)
        entry = self.by_phase.setdefault(phase, {"count": 0, "seconds": 0.0})
        entry["count"] += 1
        entry["seconds"] += float(elapsed_seconds)
        self.total_count += 1
        self.total_seconds += float(elapsed_seconds)

    def snapshot(self):
        return {
            "count": int(self.total_count),
            "seconds": float(self.total_seconds),
            "by_phase": {
                key: {"count": int(value["count"]), "seconds": float(value["seconds"])}
                for key, value in self.by_phase.items()
            },
        }

    @staticmethod
    def difference(after, before):
        phases = set(after["by_phase"]) | set(before["by_phase"])
        return {
            "model_energy_evaluations": int(after["count"] - before["count"]),
            "model_evaluation_seconds": float(after["seconds"] - before["seconds"]),
            "by_phase": {
                phase: {
                    "count": int(after["by_phase"].get(phase, {}).get("count", 0) - before["by_phase"].get(phase, {}).get("count", 0)),
                    "seconds": float(after["by_phase"].get(phase, {}).get("seconds", 0.0) - before["by_phase"].get(phase, {}).get("seconds", 0.0)),
                }
                for phase in sorted(phases)
            },
        }


def instrument_calculator(calc, tracker):
    """Instrument an ASE calculator without changing its caching semantics."""
    original_calculate = calc.calculate

    def counted_calculate(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_calculate(*args, **kwargs)
        finally:
            tracker.record(tracker.phase, time.perf_counter() - start)

    calc.calculate = counted_calculate
    return calc


class InpaintingProblem:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        requested_device = str(getattr(args, "device", "auto") or "auto").strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU execution was requested, but torch.cuda.is_available() is false. "
                "Check the CUDA-enabled PyTorch environment on ws09."
            )
        self.device = torch.device(requested_device)
        self.dtype = torch.float32

        self.all_species = parse_species_arg(args.species)
        self.all_species_index = {s: i for i, s in enumerate(self.all_species)}
        self.variable_species = parse_species_arg(args.variable_species, minimum=1)
        if any(s not in self.all_species_index for s in self.variable_species):
            raise ValueError("crystal.variable_species must be a subset of crystal.species.")
        self.species = list(self.variable_species)
        self.n_species = len(self.species)
        self.species_index = {s: i for i, s in enumerate(self.species)}
        self.atomic_numbers = [atomic_numbers[s] for s in self.species]
        self.all_atomic_numbers = [atomic_numbers[s] for s in self.all_species]
        if self.species != ["H"]:
            raise ValueError("This H-sublattice benchmark currently expects crystal.variable_species: H.")

        self.geometry_mode = "h_sublattice"
        self.supercell = parse_int_triplet(args.supercell, "crystal.supercell")
        self.pbc = parse_pbc(args.pbc)
        if not np.all(self.pbc):
            raise ValueError("The periodic H-sublattice benchmark requires pbc=true,true,true.")

        disable_optional_torch_native_bmm_override()
        # Pass the device explicitly so both FAIRChem and EAT use CUDA.
        predictor = pretrained_mlip.get_predict_unit(args.model_name, device=str(self.device))
        self.calc = FAIRChemCalculator(predictor, task_name=args.task_name)
        self.model_evaluation_tracker = ModelEvaluationTracker()
        instrument_calculator(self.calc, self.model_evaluation_tracker)
        self.objective_evaluation_count = 0
        ground_truth_file = str(getattr(args, "ground_truth_file", "") or "").strip()
        if ground_truth_file:
            source_path = resolve_path_from_config(ground_truth_file, args.config_path)
            self.mp_atoms, self.mp_metadata = read_ground_truth_file(source_path)
            self.ground_truth_source_type = "file"
            self.ground_truth_source_file = str(source_path)
        else:
            self.mp_atoms, self.mp_metadata = fetch_material_project_atoms(
                args.material_id, args.material_formula, args.mp_api_key
            )
            self.ground_truth_source_type = "materials_project"
            self.ground_truth_source_file = ""
        source_symbols = set(self.mp_atoms.get_chemical_symbols())
        unknown_symbols = sorted(source_symbols.difference(self.all_species_index))
        if unknown_symbols:
            raise ValueError(
                "The ground-truth structure contains elements not listed in "
                f"crystal.species: {unknown_symbols}. Set crystal.species to include every element."
            )
        self.mp_metadata["source_type"] = self.ground_truth_source_type
        self.mp_metadata["source_file"] = self.ground_truth_source_file
        self.mp_unit_cell = self.mp_atoms.copy()
        self.mp_unit_cell.calc = self.calc
        write(self.output_dir / "ground_truth_unit_cell_initial.xsf", self.mp_unit_cell)
        # Preserve the old MP-specific filename for existing MP workflows.
        if self.ground_truth_source_type == "materials_project":
            write(self.output_dir / "ground_truth_mp_unit_cell_initial.xsf", self.mp_unit_cell)
        unit_relax_label = (
            "relaxed_mp_unit_cell"
            if self.ground_truth_source_type == "materials_project"
            else "relaxed_file_unit_cell"
        )
        relaxed_unit, self.unit_relaxation_info = relax_with_fairchem(
            self.mp_unit_cell, self.calc, args, unit_relax_label
        )
        write(self.output_dir / "ground_truth_unit_cell_relaxed.xsf", relaxed_unit)
        if self.ground_truth_source_type == "materials_project":
            write(self.output_dir / "ground_truth_mp_unit_cell_relaxed.xsf", relaxed_unit)

        relaxed_supercell = relaxed_unit.repeat(tuple(self.supercell.tolist()))
        relaxed_supercell.wrap()
        if bool(args.relax_supercell):
            supercell_relax_label = (
                "relaxed_mp_supercell"
                if self.ground_truth_source_type == "materials_project"
                else "relaxed_file_supercell"
            )
            relaxed_supercell, self.supercell_relaxation_info = relax_with_fairchem(
                relaxed_supercell, self.calc, args, supercell_relax_label
            )
        else:
            self.supercell_relaxation_info = {
                "label": (
                    "relaxed_mp_supercell"
                    if self.ground_truth_source_type == "materials_project"
                    else "relaxed_file_supercell"
                ),
                "converged": True,
                "steps": 0,
                "relax_cell": False,
            }
        self.ground_truth_atoms = relaxed_supercell
        self.cell = np.asarray(self.ground_truth_atoms.cell.array, dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)
        self.base_cell = np.asarray(relaxed_unit.cell.array, dtype=float)
        self.base_fixed_positions = np.asarray(
            [p for s, p in zip(relaxed_unit.get_chemical_symbols(), relaxed_unit.get_positions()) if s not in self.variable_species],
            dtype=float,
        )
        self.base_fixed_symbols = [
            s for s in relaxed_unit.get_chemical_symbols() if s not in self.variable_species
        ]

        (
            self.full_symbols,
            self.full_positions,
            self.full_cell_indices,
            self.fixed_symbols,
            self.fixed_positions,
            self.fixed_cell_indices,
            self.missing_symbols,
            self.missing_positions,
            self.missing_cell_indices,
        ) = self.build_hydride_supercell()
        self.fixed_weights = self.full_one_hot_weights(self.fixed_symbols)
        self.fixed_counts = self.full_one_hot_weights(self.fixed_symbols).sum(axis=0)
        self.missing_counts = self.full_one_hot_weights(self.missing_symbols).sum(axis=0)
        self.full_counts = self.fixed_counts + self.missing_counts
        self.target_composition = self.full_counts / float(self.full_counts.sum())
        self.target_missing_count = int(len(self.missing_symbols))
        if self.target_missing_count <= 0:
            raise ValueError("The fetched structure contains no H atoms to reconstruct.")

        self.hole_start = np.zeros(3, dtype=int)
        self.hole_size = self.supercell.copy()
        self.hole_stop = self.supercell.copy()
        self.hole_min_frac = np.zeros(3, dtype=float)
        self.hole_max_frac = np.ones(3, dtype=float)
        self.slab_size = None
        self.vacuum_size_z = 0
        self.hole_volume_angstrom3 = float(abs(np.linalg.det(self.cell)))
        # Retain the configured seed density for diagnostics and run metadata.
        # The known missing-H count sets only the minimum candidate-site
        # reservoir.  Crossover may retain more active sites than this floor;
        # energy, soft composition matching, and syntropization determine how
        # many H atoms are ultimately realized.
        self.null_density_per_angstrom3 = float(args.null_density_per_angstrom3)
        if self.null_density_per_angstrom3 <= 0.0:
            raise ValueError("initialization.null_density_per_angstrom3 must be positive.")
        self.baseline_n_sites = self.target_missing_count
        self.n_sites = self.baseline_n_sites
        self.material_atom_density_per_angstrom3 = len(self.full_symbols) / max(self.hole_volume_angstrom3, 1e-12)
        self.estimated_search_region_atoms = float(self.target_missing_count)
        # Retain the stratified initialization grid used by the generic EAT
        # machinery and live-plot diagnostics. The H benchmark normally seeds
        # from the feasible physical grid, but legacy subcell seeding remains
        # supported through this stored value.
        self.subcell_grid = parse_int_triplet(args.subcell_grid, "initialization.subcell_grid")
        self.feasible_grid = parse_int_triplet(args.feasible_grid, "crystal.feasible_grid")
        self.feasible_min_distance_angstrom = float(args.feasible_min_distance_angstrom)
        self.initial_h_h_min_distance_angstrom = float(args.initial_h_h_min_distance_angstrom)
        if self.feasible_min_distance_angstrom < 0.0 or self.initial_h_h_min_distance_angstrom < 0.0:
            raise ValueError("Grid and H-H distance thresholds must be non-negative.")
        self.feasible_grid_base_frac = self.build_feasible_base_grid()
        self.feasible_grid_frac = self.tile_feasible_grid_to_supercell(self.feasible_grid_base_frac)
        self.feasible_grid_positions = self.feasible_grid_frac @ self.cell
        if len(self.feasible_grid_positions) < self.target_missing_count:
            raise ValueError(
                f"Only {len(self.feasible_grid_positions)} feasible grid points were found, "
                f"but {self.target_missing_count} H atoms are required. Increase crystal.feasible_grid "
                "or lower the distance thresholds."
            )

        self.tsallis_q = float(args.tsallis_q)
        if self.tsallis_q <= 1.0:
            raise ValueError("syntropization.tsallis_q must be > 1.")
        self.composition_penalty = float(args.composition_penalty)
        self.hole_boundary_penalty = 0.0
        self.occupancy_lower_bound_enable = bool(args.occupancy_lower_bound_enable)
        self.occupancy_lower_bound_gamma = float(args.occupancy_lower_bound_gamma)
        self.occupancy_lower_bound_penalty = float(args.occupancy_lower_bound_penalty)
        self.occupancy_lower_bound_softplus_sharpness = float(args.occupancy_lower_bound_softplus_sharpness)
        self.occupancy_lower_bound = self.occupancy_lower_bound_gamma * self.estimated_search_region_atoms
        for name, value in (
            ("composition_penalty", self.composition_penalty),
            ("hole_boundary_penalty", self.hole_boundary_penalty),
            ("occupancy_lower_bound_gamma", self.occupancy_lower_bound_gamma),
            ("occupancy_lower_bound_penalty", self.occupancy_lower_bound_penalty),
            ("occupancy_lower_bound_softplus_sharpness", self.occupancy_lower_bound_softplus_sharpness),
        ):
            if value < 0.0:
                raise ValueError(f"objective.{name} must be non-negative.")
        if self.occupancy_lower_bound_softplus_sharpness <= 0.0:
            raise ValueError("objective.occupancy_lower_bound_softplus_sharpness must be positive.")

        self.element_reference_metadata = []
        element_ref_energies = []
        api_key = str(args.mp_api_key or MP_API_KEY or "")
        for symbol in self.all_species:
            ref = fetch_elemental_reference_atoms(symbol, api_key)
            ref["atoms"].calc = self.calc
            energy_total = float(ref["atoms"].get_potential_energy())
            energy_per_atom = energy_total / len(ref["atoms"])
            element_ref_energies.append(energy_per_atom)
            self.element_reference_metadata.append({
                "symbol": symbol,
                "material_id": ref["material_id"],
                "formula_pretty": ref["formula_pretty"],
                "energy_above_hull": ref["energy_above_hull"],
                "formation_energy_per_atom": ref["formation_energy_per_atom"],
                "computed_energy_total_eV": energy_total,
                "computed_energy_per_atom_eV": energy_per_atom,
                "num_atoms": int(len(ref["atoms"])),
            })
        self.element_ref_energies = np.asarray(element_ref_energies, dtype=float)
        self.variable_ref_energies = np.asarray(
            [self.element_ref_energies[self.all_species_index[s]] for s in self.variable_species], dtype=float
        )

        full_atoms = Atoms(symbols=self.full_symbols, positions=self.full_positions, cell=self.cell, pbc=self.pbc)
        full_atoms.calc = self.calc
        self.full_reference_energy_total = float(full_atoms.get_potential_energy())
        self.full_reference_formation_energy_total = (
            self.full_reference_energy_total - float(np.dot(self.full_counts, self.element_ref_energies))
        )
        self.full_reference_formation_energy_per_atom = self.full_reference_formation_energy_total / float(self.full_counts.sum())

        self.write_ground_truth_outputs()

        self.history = []
        self.stage_summaries = []
        self.callback_counter = 0
        # The minimizer evaluates objfun at the state it subsequently passes to
        # the callback.  Retain the scalar diagnostics from that evaluation so
        # snapshot() does not run the expensive atomistic model a second time.
        # snapshot() verifies the state exactly and falls back to a fresh model
        # evaluation whenever the callback state is not the cached state.
        self._objective_snapshot_cache = None
        self.objective_snapshot_cache_hits = 0
        self.objective_snapshot_cache_misses = 0
        self.last_heartbeat_time = 0.0
        self.stage_wall_start = None
        self.stage_last_callback_time = None

    def build_hydride_supercell(self):
        symbols = self.ground_truth_atoms.get_chemical_symbols()
        positions = np.asarray(self.ground_truth_atoms.get_positions(), dtype=float)
        full_symbols = list(symbols)
        full_positions = positions.copy()
        full_indices = np.zeros((len(symbols), 3), dtype=int)
        fixed_mask = np.asarray([s not in self.variable_species for s in symbols], dtype=bool)
        fixed_symbols = [s for s, keep in zip(symbols, fixed_mask) if keep]
        fixed_positions = positions[fixed_mask]
        missing_symbols = [s for s, keep in zip(symbols, fixed_mask) if not keep]
        missing_positions = positions[~fixed_mask]
        return (
            full_symbols,
            full_positions,
            full_indices,
            fixed_symbols,
            np.asarray(fixed_positions, dtype=float),
            np.zeros((len(fixed_symbols), 3), dtype=int),
            missing_symbols,
            np.asarray(missing_positions, dtype=float),
            np.zeros((len(missing_symbols), 3), dtype=int),
        )

    def build_feasible_base_grid(self):
        grid = self.feasible_grid.astype(int)
        points = []
        for flat_index in range(int(np.prod(grid))):
            ijk = np.asarray(np.unravel_index(flat_index, tuple(grid.tolist())), dtype=float)
            frac = (ijk + 0.5) / grid.astype(float)
            pos = frac @ self.base_cell
            if len(self.base_fixed_positions):
                fixed_frac = self.base_fixed_positions @ np.linalg.inv(self.base_cell)
                dfrac = fixed_frac - frac[None, :]
                dfrac -= np.round(dfrac)
                distances = np.linalg.norm(dfrac @ self.base_cell, axis=1)
                if float(np.min(distances)) < self.feasible_min_distance_angstrom:
                    continue
            points.append(frac)
        return np.asarray(points, dtype=float)

    def tile_feasible_grid_to_supercell(self, base_frac):
        tiled = []
        for ix in range(int(self.supercell[0])):
            for iy in range(int(self.supercell[1])):
                for iz in range(int(self.supercell[2])):
                    shift = np.asarray([ix, iy, iz], dtype=float)
                    tiled.append((base_frac + shift[None, :]) / self.supercell.astype(float))
        return np.concatenate(tiled, axis=0) if tiled else np.empty((0, 3), dtype=float)

    def full_one_hot_weights(self, symbols):
        weights = np.zeros((len(symbols), len(self.all_species)), dtype=float)
        for i, symbol in enumerate(symbols):
            if str(symbol) not in self.all_species_index:
                raise ValueError(f"Unexpected scaffold species {symbol!r}.")
            weights[i, self.all_species_index[str(symbol)]] = 1.0
        return weights

    def full_counts_from_variable(self, occ_np):
        counts = np.asarray(self.fixed_counts, dtype=float).copy()
        variable_counts = np.asarray(occ_np, dtype=float).sum(axis=0)
        for j, symbol in enumerate(self.variable_species):
            counts[self.all_species_index[symbol]] += variable_counts[j]
        return counts

    def set_site_count(self, n_sites):
        n_sites = int(n_sites)
        if n_sites <= 0:
            raise ValueError("Candidate must contain at least one null site.")
        self.n_sites = n_sites

    def pack_y(self, occ, positions):
        return torch.cat([occ.reshape(-1), positions.reshape(-1)])

    def unpack_y(self, y):
        n_occ = self.n_sites * self.n_species
        occ = y[:n_occ].reshape(self.n_sites, self.n_species)
        positions = y[n_occ:].reshape(self.n_sites, 3)
        return occ, positions

    def project(self, y):
        occ, positions = self.unpack_y(y)
        return self.pack_y(project_capped_simplex(occ), positions)

    def positions_to_frac(self, positions):
        return np.asarray(positions, dtype=float) @ self.inv_cell

    def wrap_positions(self, positions):
        frac = self.positions_to_frac(positions)
        frac[:, self.pbc] = frac[:, self.pbc] - np.floor(frac[:, self.pbc])
        return frac @ self.cell

    def build_atoms(self, occ_np, positions_np):
        positions_np = np.asarray(positions_np, dtype=float)
        all_positions = np.vstack([self.fixed_positions, positions_np])
        weights = np.zeros((len(all_positions), len(self.all_species)), dtype=float)
        weights[: len(self.fixed_symbols)] = self.fixed_weights
        for j, symbol in enumerate(self.variable_species):
            weights[len(self.fixed_symbols):, self.all_species_index[symbol]] = np.asarray(occ_np, dtype=float)[:, j]
        atoms = Atoms(symbols=[SYM_DUMMY] * len(all_positions), positions=all_positions, cell=self.cell, pbc=self.pbc)
        set_multi_species_eat_weights(atoms, weights, self.all_atomic_numbers)
        return atoms

    def composition_penalty_term(self, occ_np):
        counts = self.full_counts_from_variable(occ_np)
        total = float(counts.sum())
        if total <= 1e-12:
            fractions = np.zeros(len(self.all_species), dtype=float)
            grad_counts = np.zeros(len(self.all_species), dtype=float)
        else:
            fractions = counts / total
            diff = fractions - self.target_composition
            objective = float(self.composition_penalty * np.dot(diff, diff))
            grad_counts = np.zeros(len(self.all_species), dtype=float)
            for j in range(len(self.all_species)):
                basis = np.zeros(len(self.all_species), dtype=float)
                basis[j] = 1.0
                grad_counts[j] = 2.0 * self.composition_penalty / total * float(np.dot(diff, basis - fractions))
            grad_occ = np.zeros((self.n_sites, self.n_species), dtype=float)
            for j, symbol in enumerate(self.variable_species):
                grad_occ[:, j] = grad_counts[self.all_species_index[symbol]]
            return {
                "objective_term": objective,
                "grad_occ": grad_occ,
                "fractions": fractions,
                "counts": counts,
            }
        return {
            "objective_term": 0.0,
            "grad_occ": np.zeros((self.n_sites, self.n_species), dtype=float),
            "fractions": fractions,
            "counts": counts,
        }

    def hole_boundary_penalty_term(self, occ_np, positions_np):
        occ_sum = np.asarray(occ_np, dtype=float).sum(axis=1)
        frac = self.positions_to_frac(positions_np)
        below = np.minimum(frac - self.hole_min_frac[None, :], 0.0)
        above = np.maximum(frac - self.hole_max_frac[None, :], 0.0)
        violation = below + above
        per_site = np.sum(violation * violation, axis=1)
        objective = float(self.hole_boundary_penalty * np.dot(occ_sum, per_site))
        grad_occ = np.repeat((self.hole_boundary_penalty * per_site)[:, None], self.n_species, axis=1)
        grad_frac = 2.0 * self.hole_boundary_penalty * occ_sum[:, None] * violation
        grad_pos = grad_frac @ self.inv_cell.T
        inside = np.all((frac >= self.hole_min_frac[None, :]) & (frac <= self.hole_max_frac[None, :]), axis=1)
        return {
            "objective_term": objective,
            "grad_occ": grad_occ,
            "grad_pos": grad_pos,
            "inside_count": int(np.count_nonzero(inside)),
            "max_violation_frac": float(np.max(np.linalg.norm(violation, axis=1))) if len(violation) else 0.0,
        }

    def occupancy_lower_bound_penalty_term_np(self, occ_np):
        sumx = float(np.asarray(occ_np, dtype=float).sum())
        deficit = float(self.occupancy_lower_bound - sumx)
        if (not self.occupancy_lower_bound_enable) or self.occupancy_lower_bound_penalty <= 0.0:
            return {
                "objective_term": 0.0,
                "grad_occ": np.zeros((self.n_sites, self.n_species), dtype=float),
                "estimated_atoms": float(self.estimated_search_region_atoms),
                "lower_bound": float(self.occupancy_lower_bound),
                "deficit": deficit,
            }
        sharpness = float(self.occupancy_lower_bound_softplus_sharpness)
        x = sharpness * deficit
        softplus = np.logaddexp(0.0, x) / sharpness
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
        objective = float(self.occupancy_lower_bound_penalty * softplus)
        grad_value = float(-self.occupancy_lower_bound_penalty * sigmoid)
        return {
            "objective_term": objective,
            "grad_occ": np.full((self.n_sites, self.n_species), grad_value, dtype=float),
            "estimated_atoms": float(self.estimated_search_region_atoms),
            "lower_bound": float(self.occupancy_lower_bound),
            "deficit": deficit,
        }

    def objfun(self, y, beta):
        self.objective_evaluation_count += 1
        objective_input_y = y.detach().clone()
        y = self.project(y)
        occ, positions = self.unpack_y(y)
        occ_np = occ.detach().cpu().numpy()
        positions_np = positions.detach().cpu().numpy()
        atoms = self.build_atoms(occ_np, positions_np)
        energy, eat_grad_all, forces_all = get_energy_gradients_and_forces(atoms, self.calc, self.atomic_numbers, want_grad=True)
        eat_grad = eat_grad_all[len(self.fixed_symbols):]
        forces = forces_all[len(self.fixed_symbols):]
        counts = self.full_counts_from_variable(occ_np)
        n_total = max(float(counts.sum()), 1e-12)
        numerator = energy - float(np.dot(counts, self.element_ref_energies))
        eform = numerator / n_total
        grad_occ_energy = (eat_grad - self.variable_ref_energies[None, :] - eform) / n_total
        grad_pos_energy = -forces / n_total

        comp_info = self.composition_penalty_term(occ_np)
        boundary_info = self.hole_boundary_penalty_term(occ_np, positions_np)
        occupancy_info = self.occupancy_lower_bound_penalty_term_np(occ_np)
        entropy = tsallis_entropy_term(occ, self.tsallis_q)

        f = (
            torch.tensor(eform, device=y.device, dtype=y.dtype)
            + beta * entropy
            + torch.tensor(comp_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(boundary_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(occupancy_info["objective_term"], device=y.device, dtype=y.dtype)
        )
        grad_occ = (
            torch.tensor(grad_occ_energy, device=y.device, dtype=y.dtype)
            + beta * tsallis_entropy_grad(occ, self.tsallis_q)
            + torch.tensor(comp_info["grad_occ"], device=y.device, dtype=y.dtype)
            + torch.tensor(boundary_info["grad_occ"], device=y.device, dtype=y.dtype)
            + torch.tensor(occupancy_info["grad_occ"], device=y.device, dtype=y.dtype)
        )
        grad_pos = torch.tensor(grad_pos_energy + boundary_info["grad_pos"], device=y.device, dtype=y.dtype)
        grad = self.pack_y(grad_occ, grad_pos)

        # Cache only the data needed by snapshot(), not the large gradient
        # arrays.  Keeping the projected state makes reuse correctness-safe even
        # if the optimizer changes its callback/line-search behavior later.
        self._objective_snapshot_cache = {
            "input_y": objective_input_y,
            "y": y.detach().clone(),
            "beta": float(beta),
            "n_sites": int(self.n_sites),
            "eform": float(eform),
            "composition": {
                "objective_term": float(comp_info["objective_term"]),
                "fractions": np.asarray(comp_info["fractions"], dtype=float).copy(),
                "counts": np.asarray(comp_info["counts"], dtype=float).copy(),
            },
            "boundary": {
                "objective_term": float(boundary_info["objective_term"]),
                "inside_count": int(boundary_info["inside_count"]),
                "max_violation_frac": float(boundary_info["max_violation_frac"]),
            },
            "occupancy_lower_bound": {
                "objective_term": float(occupancy_info["objective_term"]),
                "estimated_atoms": float(occupancy_info["estimated_atoms"]),
                "lower_bound": float(occupancy_info["lower_bound"]),
                "deficit": float(occupancy_info["deficit"]),
            },
        }
        return f, grad

    def projected_gradient_error(self, y, g):
        projected = self.project(y - g)
        return float(torch.norm(projected - y).item())

    def snapshot(self, y, stage, iteration, beta, f_objective, err, run_kind, precomputed_energy=None, energy_source=None):
        cache = self._objective_snapshot_cache
        cache_hit = bool(
            precomputed_energy is None
            and cache is not None
            and cache["n_sites"] == int(self.n_sites)
            and cache["y"].shape == y.shape
            and cache["y"].device == y.device
            and cache["y"].dtype == y.dtype
            and (torch.equal(cache["y"], y) or torch.equal(cache["input_y"], y))
        )
        # On a hit, use the exact projected tensor evaluated by objfun.  The
        # input-state match covers the minimizer's reset step, where objfun does
        # the projection internally; the evaluated-state match covers accepted
        # line-search steps.
        y = cache["y"] if cache_hit else self.project(y)
        occ, positions = self.unpack_y(y)
        occ_np = occ.detach().cpu().numpy()
        positions_np = positions.detach().cpu().numpy()
        if precomputed_energy is not None:
            counts = self.full_counts_from_variable(occ_np)
            n_total = max(float(counts.sum()), 1e-12)
            eform = (float(precomputed_energy) - float(np.dot(counts, self.element_ref_energies))) / n_total
            comp_info = self.composition_penalty_term(occ_np)
            boundary_info = self.hole_boundary_penalty_term(occ_np, positions_np)
            occupancy_info = self.occupancy_lower_bound_penalty_term_np(occ_np)
            resolved_energy_source = str(energy_source or "precomputed")
        elif cache_hit:
            eform = cache["eform"]
            comp_info = cache["composition"]
            boundary_info = cache["boundary"]
            occupancy_info = cache["occupancy_lower_bound"]
            self.objective_snapshot_cache_hits += 1
            resolved_energy_source = "objective_cache"
        else:
            # This path preserves the original behavior for any callback state
            # that was not the most recent objective evaluation.
            atoms = self.build_atoms(occ_np, positions_np)
            energy, _, _ = get_energy_gradients_and_forces(atoms, self.calc, self.atomic_numbers, want_grad=False)
            counts = self.full_counts_from_variable(occ_np)
            n_total = max(float(counts.sum()), 1e-12)
            eform = (energy - float(np.dot(counts, self.element_ref_energies))) / n_total
            comp_info = self.composition_penalty_term(occ_np)
            boundary_info = self.hole_boundary_penalty_term(occ_np, positions_np)
            occupancy_info = self.occupancy_lower_bound_penalty_term_np(occ_np)
            self.objective_snapshot_cache_misses += 1
            resolved_energy_source = "fresh_model_evaluation"
        entropy_total = float(tsallis_entropy_term(occ, self.tsallis_q).item())
        ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
        f_guided_no_entropy = eform + comp_info["objective_term"] + boundary_info["objective_term"] + occupancy_info["objective_term"]
        rounded = self.rounded_state(y)
        rounded_occ, rounded_pos = self.unpack_y(rounded)
        return {
            "stage": int(stage + 1),
            "iteration": int(iteration),
            "run_kind": str(run_kind),
            "objective_snapshot_cache_hit": bool(cache_hit),
            "energy_evaluation_source": resolved_energy_source,
            "beta": float(beta),
            "f_objective": float(f_objective.detach().cpu().item() if torch.is_tensor(f_objective) else f_objective),
            "f_guided_no_entropy": float(f_guided_no_entropy),
            "formation_energy_per_atom": float(eform),
            "formation_energy_difference_from_reference": float(eform - self.full_reference_formation_energy_per_atom),
            "composition_term": float(comp_info["objective_term"]),
            "hole_boundary_term": float(boundary_info["objective_term"]),
            "occupancy_lower_bound_term": float(occupancy_info["objective_term"]),
            "occupancy_lower_bound": float(occupancy_info["lower_bound"]),
            "estimated_search_region_atoms": float(occupancy_info["estimated_atoms"]),
            "occupancy_lower_bound_deficit": float(occupancy_info["deficit"]),
            "sumx": float(occ_np.sum()),
            "composition_fractions": comp_info["fractions"].tolist(),
            "composition_counts": comp_info["counts"].tolist(),
            "entropy_total": entropy_total,
            "entropy_avg": float(ent_avg),
            "max_min": float(max_min),
            "err": float(err),
            "n_sites": int(self.n_sites),
            "hole_inside_count": int(boundary_info["inside_count"]),
            "hole_max_violation_frac": float(boundary_info["max_violation_frac"]),
            "species_occupancies": occ_np.tolist(),
            "cart_positions_angstrom": positions_np.tolist(),
            "frac_coords": self.positions_to_frac(positions_np).tolist(),
            "rounded_species_occupancies": rounded_occ.detach().cpu().numpy().tolist(),
            "rounded_cart_positions_angstrom": rounded_pos.detach().cpu().numpy().tolist(),
        }

    def maybe_write_heartbeat(self, message):
        interval = float(getattr(self.args, "heartbeat_interval_seconds", 0.0) or 0.0)
        if interval <= 0.0:
            return
        now = time.time()
        if now - float(self.last_heartbeat_time) < interval:
            return
        append_running_log(self.args, message)
        self.last_heartbeat_time = now

    def run_stage(self, y0, beta, stage, run_kind="main", record_history=True):
        f_history = []
        local_history = []
        capture_trajectory = not bool(self.args.disable_trajectory_history)
        self.stage_wall_start = time.time()
        self.stage_last_callback_time = None
        self._objective_snapshot_cache = None
        cache_hits_before = int(self.objective_snapshot_cache_hits)
        cache_misses_before = int(self.objective_snapshot_cache_misses)

        def callback(y, f, err, k, converged, beta_local):
            now = time.time()
            elapsed_s = now - float(self.stage_wall_start or now)
            dt_s = 0.0 if self.stage_last_callback_time is None else now - float(self.stage_last_callback_time)
            self.stage_last_callback_time = now
            snap = self.snapshot(y, stage, k, beta_local, f, err, run_kind)
            snap["wall_elapsed_seconds"] = float(elapsed_s)
            snap["wall_dt_seconds"] = float(dt_s)
            f_history.append(snap["f_guided_no_entropy"])
            if capture_trajectory:
                local_history.append(snap)
            if record_history and capture_trajectory:
                if self.callback_counter % max(1, int(self.args.history_every)) == 0:
                    self.history.append(snap)
                self.callback_counter += 1
            self.maybe_write_heartbeat(
                f"relaxing run_kind={run_kind} stage={stage+1} iter={k} "
                f"Eform={snap['formation_energy_per_atom']:+.6f} score={snap['f_guided_no_entropy']:+.6f} err={err:.3e}"
            )
            terminal_print(
                self.args,
                f"[stage={stage+1:02d}] iter={k:4d} beta={beta_local:.3e} "
                f"elapsed_s={elapsed_s:.2f} dt_s={dt_s:.2f} "
                f"sumx={snap['sumx']:.3f} "
                f"Eform={snap['formation_energy_per_atom']:+.6f} comp={snap['composition_term']:+.6f} "
                f"boundary={snap['hole_boundary_term']:+.6f} occ_lb={snap['occupancy_lower_bound_term']:+.6f} "
                f"score={snap['f_guided_no_entropy']:+.6f} entropy/site={snap['entropy_avg']:.6e} "
                f"err={err:.3e} conv={converged}",
            )

        opt = minimizer(
            fun_jac=self.objfun,
            x0=self.project(y0),
            args=(beta,),
            project=self.project,
            proj_args=(),
            callback=callback,
            BFGS_hist=self.args.bfgs_hist,
            maxit=self.args.maxit,
            tol=self.args.tol,
            max_line_search=self.args.max_line_search,
        )
        f, y = opt.PBFGS()
        occ, _ = self.unpack_y(y)
        ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
        if record_history:
            self.stage_summaries.append({
                "stage": int(stage + 1),
                "beta": float(beta),
                "run_kind": str(run_kind),
                "sumx": float(occ.sum().item()),
                "entropy_avg_final": float(ent_avg),
                "max_min_final": float(max_min),
                "f_guided_final": float(f_history[-1]) if f_history else float(f),
                "accepted_iterations": int(len(f_history)),
                "objective_snapshot_cache_hits": int(self.objective_snapshot_cache_hits - cache_hits_before),
                "objective_snapshot_cache_misses": int(self.objective_snapshot_cache_misses - cache_misses_before),
            })
        return f, y, {"trajectory": local_history, "f_guided_history": f_history}

    def entropy_stop_reached(self, ent_avg, max_min):
        if bool(self.args.also_require_binary_stop):
            return ent_avg <= float(self.args.entropy_stop_avg) and max_min <= float(self.args.binary_stop_maxmin)
        return ent_avg <= float(self.args.entropy_stop_avg)

    def run(self, y0, run_kind="main"):
        self.history = []
        self.stage_summaries = []
        self.callback_counter = 0
        y = self.project(y0)
        beta = float(self.args.beta0)
        n_stages = max(1, int(self.args.beta_max_stages) + 1)
        for stage in range(n_stages):
            _, y, _ = self.run_stage(y, beta, stage, run_kind=run_kind)
            occ, _ = self.unpack_y(y)
            ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
            if self.entropy_stop_reached(ent_avg, max_min):
                break
            beta *= float(self.args.beta_mult)
        return self.project(y)

    def rounded_state(self, y):
        occ, positions = self.unpack_y(self.project(y))
        null = torch.clamp(1.0 - occ.sum(dim=1, keepdim=True), min=0.0)
        probs = torch.cat([occ, null], dim=1)
        winners = torch.argmax(probs, dim=1)
        rounded = torch.zeros_like(occ)
        real = winners < self.n_species
        if bool(real.any().item()):
            rounded[torch.arange(self.n_sites, device=occ.device)[real], winners[real]] = 1.0
        return self.pack_y(rounded, positions)

    def realized_atoms(self, y, rounded=True):
        if rounded:
            y = self.rounded_state(y)
        occ, positions = self.unpack_y(self.project(y))
        occ_np = occ.detach().cpu().numpy()
        pos_np = positions.detach().cpu().numpy()
        keep = np.where(occ_np.sum(axis=1) > float(self.args.keep_thresh))[0]
        symbols = list(self.fixed_symbols)
        coords = [p for p in self.fixed_positions]
        for idx in keep:
            species_idx = int(np.argmax(occ_np[idx]))
            symbols.append(self.species[species_idx])
            coords.append(pos_np[idx])
        atoms = Atoms(symbols=symbols, positions=np.asarray(coords, dtype=float), cell=self.cell, pbc=self.pbc)
        atoms.wrap()
        return atoms

    def write_structure(self, name, y, rounded=True):
        atoms = self.realized_atoms(y, rounded=rounded)
        write(self.output_dir / f"{name}.xyz", atoms)
        write(self.output_dir / f"{name}.xsf", atoms)

    def write_ground_truth_outputs(self):
        """Write the relaxed reference before any EAT/GA optimization begins."""
        write(self.output_dir / "ground_truth_relaxed_supercell.xsf", self.ground_truth_atoms)
        write(self.output_dir / "ground_truth_relaxed_supercell.xyz", self.ground_truth_atoms)
        scaffold = Atoms(
            symbols=self.fixed_symbols,
            positions=self.fixed_positions,
            cell=self.cell,
            pbc=self.pbc,
        )
        generic_scaffold_xsf = self.output_dir / "fixed_heavy_atom_scaffold.xsf"
        generic_scaffold_xyz = self.output_dir / "fixed_heavy_atom_scaffold.xyz"
        write(generic_scaffold_xsf, scaffold)
        write(generic_scaffold_xyz, scaffold)
        legacy_scaffold_xsf = self.output_dir / "fixed_Mg_Fe_scaffold.xsf"
        legacy_scaffold_xyz = self.output_dir / "fixed_Mg_Fe_scaffold.xyz"
        # Keep these legacy names for the original MP/Mg2FeH6 workflow.  For
        # an experimental file source, the element-neutral scaffold names
        # above avoid implying that the heavy atoms are Mg/Fe.
        if self.ground_truth_source_type == "materials_project":
            write(legacy_scaffold_xsf, scaffold)
            write(legacy_scaffold_xyz, scaffold)
        formula = self.mp_metadata.get("formula_pretty", "") or str(self.args.material_formula)
        ground_truth_reference = {
            "description": f"Relaxed {formula} reference before variable-sublattice removal and EAT.",
            "ground_truth_source": self.ground_truth_source_type,
            "ground_truth_source_file": self.ground_truth_source_file,
            "material_id": self.mp_metadata.get("material_id", ""),
            "formula_pretty": formula,
            "mp_energy_above_hull_eV_per_atom": self.mp_metadata.get("energy_above_hull"),
            "supercell": self.supercell.astype(int).tolist(),
            "n_atoms": int(len(self.ground_truth_atoms)),
            "composition_counts": {
                symbol: float(count)
                for symbol, count in zip(self.all_species, self.full_counts)
            },
            "energy_total_eV": float(self.full_reference_energy_total),
            "formation_energy_total_eV": float(self.full_reference_formation_energy_total),
            "formation_energy_per_atom_eV": float(self.full_reference_formation_energy_per_atom),
            "element_reference_energies_per_atom_eV": {
                symbol: float(energy)
                for symbol, energy in zip(self.all_species, self.element_ref_energies)
            },
            "unit_cell_relaxation": self.unit_relaxation_info,
            "supercell_relaxation": self.supercell_relaxation_info,
            "xsf_file": str(self.output_dir / "ground_truth_relaxed_supercell.xsf"),
            "xyz_file": str(self.output_dir / "ground_truth_relaxed_supercell.xyz"),
            "scaffold_xsf_file": str(generic_scaffold_xsf),
            "scaffold_xyz_file": str(generic_scaffold_xyz),
            "legacy_scaffold_xsf_file": (
                str(legacy_scaffold_xsf)
                if self.ground_truth_source_type == "materials_project"
                else None
            ),
        }
        self.ground_truth_reference = ground_truth_reference
        with (self.output_dir / "ground_truth_reference_energy.json").open("w", encoding="utf-8") as handle:
            json.dump(ground_truth_reference, handle, indent=2)

    def final_snapshot(self, y, run_kind, precomputed_energy=None, energy_source=None):
        return self.snapshot(
            y,
            stage=-1,
            iteration=-1,
            beta=0.0,
            f_objective=0.0,
            err=0.0,
            run_kind=run_kind,
            precomputed_energy=precomputed_energy,
            energy_source=energy_source,
        )

    def realized_structure_payload(self, y, rounded=True):
        if rounded:
            y = self.rounded_state(y)
        occ, positions = self.unpack_y(self.project(y))
        occ_np = occ.detach().cpu().numpy()
        pos_np = positions.detach().cpu().numpy()
        keep = np.where(occ_np.sum(axis=1) > float(self.args.keep_thresh))[0]
        null_symbols = [self.species[int(np.argmax(occ_np[idx]))] for idx in keep]
        return {
            "fixed_symbols": list(self.fixed_symbols),
            "fixed_positions_angstrom": self.fixed_positions.tolist(),
            "null_symbols": null_symbols,
            "null_positions_angstrom": pos_np[keep].tolist(),
            "null_species_occupancies": occ_np[keep].tolist(),
            "all_symbols": list(self.realized_atoms(y, rounded=rounded).get_chemical_symbols()),
            "cell_angstrom": self.cell.tolist(),
        }

    def write_outputs(self, y_cont, y_round, seed):
        self.write_structure("continuous", y_cont, rounded=False)
        self.write_structure("rounded", y_round, rounded=True)
        self.write_ground_truth_outputs()
        payload = {
            "generator": str(Path(__file__).resolve()),
            "seed": int(seed),
            "species": list(self.all_species),
            "variable_species": list(self.variable_species),
            "mp_metadata": self.mp_metadata,
            "ground_truth_source": self.ground_truth_source_type,
            "ground_truth_source_file": self.ground_truth_source_file,
            "geometry_mode": str(self.geometry_mode),
            "supercell": self.supercell.astype(int).tolist(),
            "feasible_grid": self.feasible_grid.astype(int).tolist(),
            "feasible_min_distance_angstrom": float(self.feasible_min_distance_angstrom),
            "n_feasible_grid_points": int(len(self.feasible_grid_positions)),
            "n_fixed_atoms": int(len(self.fixed_symbols)),
            "n_missing_atoms": int(len(self.missing_symbols)),
            "n_null_sites": int(self.n_sites),
            "material_atom_density_per_angstrom3": float(self.material_atom_density_per_angstrom3),
            "estimated_search_region_atoms": float(self.estimated_search_region_atoms),
            "occupancy_lower_bound_enable": bool(self.occupancy_lower_bound_enable),
            "occupancy_lower_bound_gamma": float(self.occupancy_lower_bound_gamma),
            "occupancy_lower_bound": float(self.occupancy_lower_bound),
            "occupancy_lower_bound_penalty": float(self.occupancy_lower_bound_penalty),
            "occupancy_lower_bound_softplus_sharpness": float(self.occupancy_lower_bound_softplus_sharpness),
            "ground_truth_reference": self.ground_truth_reference,
            "full_reference_energy_total_eV": float(self.full_reference_energy_total),
            "full_reference_formation_energy_total_eV": float(self.full_reference_formation_energy_total),
            "full_reference_formation_energy_per_atom": float(self.full_reference_formation_energy_per_atom),
            "element_references": self.element_reference_metadata,
            "final_continuous_state": self.final_snapshot(y_cont, "final_continuous"),
            "final_rounded_state": self.final_snapshot(y_round, "final_rounded"),
            "trajectory": [] if bool(self.args.disable_trajectory_history) else self.history,
            "stage_summaries": self.stage_summaries,
        }
        with (self.output_dir / "inpainting_optimization_history.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def sample_initial_positions(self, n_sites):
        mode = str(self.args.init_position_mode).strip().lower()
        n_sites = int(n_sites)
        if mode in ("ground_truth_jittered", "truth_jittered", "missing_jittered"):
            reps = int(np.ceil(n_sites / max(1, len(self.missing_positions))))
            base = np.tile(self.missing_positions, (reps, 1))[:n_sites]
            sigma = float(self.args.init_position_jitter_angstrom)
            return self.wrap_positions(base + sigma * np.random.normal(size=base.shape))
        if mode in ("feasible_grid", "feasible", "subcell_grid", "subcell", "grid"):
            pool = np.asarray(self.feasible_grid_positions, dtype=float).copy()
            np.random.shuffle(pool)
            selected = []
            cutoff2 = float(self.initial_h_h_min_distance_angstrom) ** 2
            for candidate in pool:
                if not selected:
                    selected.append(candidate)
                else:
                    # Compare this candidate against the already-selected
                    # points. The previous implementation computed the
                    # selected-selected matrix and took its last row, which
                    # always contains the zero self-distance and therefore
                    # rejected every candidate after the first one.
                    selected_array = np.asarray(selected, dtype=float)
                    candidate_distances2 = periodic_pair_distance2(
                        np.vstack([selected_array, candidate[None, :]]),
                        self.cell,
                        self.pbc,
                    )[-1, :-1]
                    if float(np.min(candidate_distances2)) + 1e-12 >= cutoff2:
                        selected.append(candidate)
                if len(selected) >= n_sites:
                    break
            if len(selected) < n_sites:
                raise ValueError(
                    f"Could only place {len(selected)} of {n_sites} initial H sites on the feasible grid. "
                    "Increase crystal.feasible_grid or lower crystal.initial_h_h_min_distance_angstrom."
                )
            return self.wrap_positions(np.asarray(selected[:n_sites], dtype=float))
        if mode in ("legacy_subcell_grid",):
            grid = np.asarray(self.subcell_grid, dtype=int)
            n_subcells = int(np.prod(grid))
            counts = np.full(n_subcells, n_sites // n_subcells, dtype=int)
            remainder = int(n_sites % n_subcells)
            if remainder > 0:
                counts[np.random.choice(n_subcells, size=remainder, replace=False)] += 1
            lo = self.hole_min_frac
            span = self.hole_max_frac - self.hole_min_frac
            points = []
            for flat_index, count in enumerate(counts):
                if int(count) <= 0:
                    continue
                ijk = np.asarray(np.unravel_index(int(flat_index), tuple(grid.tolist())), dtype=float)
                local = (ijk[None, :] + np.random.random((int(count), 3))) / grid[None, :]
                points.append(lo[None, :] + local * span[None, :])
            frac = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=float)
            if frac.shape[0] > 1:
                frac = frac[np.random.permutation(frac.shape[0])]
            return frac @ self.cell
        if mode not in ("random_uniform", "uniform", "random"):
            raise ValueError("initialization.init_position_mode must be subcell_grid, random_uniform, or ground_truth_jittered.")
        lo = self.hole_min_frac
        hi = self.hole_max_frac
        frac = lo[None, :] + np.random.random((n_sites, 3)) * (hi - lo)[None, :]
        return frac @ self.cell


def initial_species_probability_matrix(problem, n_sites, args):
    alpha = float(args.init_species_dirichlet_alpha)
    if alpha <= 0.0:
        raise ValueError("initialization.init_species_dirichlet_alpha must be positive.")
    return np.random.dirichlet(np.full(problem.n_species, alpha, dtype=float), size=int(n_sites))


def make_initial_state(problem, args):
    """Legacy diffuse/fractional EAT initialization."""
    n_sites = int(problem.baseline_n_sites)
    problem.set_site_count(n_sites)
    occupancy = max(0.0, min(1.0, float(args.init_occ_total)))
    occupancy_tensor = torch.full((n_sites, 1), occupancy, device=problem.device, dtype=problem.dtype)
    if float(args.init_occ_noise) > 0.0:
        occupancy_tensor = occupancy_tensor + float(args.init_occ_noise) * torch.randn_like(occupancy_tensor)
    species_probs = torch.tensor(initial_species_probability_matrix(problem, n_sites, args), device=problem.device, dtype=problem.dtype)
    occ = combine_occupancy_and_species_probs(occupancy_tensor, species_probs)
    positions = torch.tensor(problem.sample_initial_positions(n_sites), device=problem.device, dtype=problem.dtype)
    return problem.project(problem.pack_y(occ, positions))


def minimum_distance_to_references(problem, point, references):
    references = np.asarray(references, dtype=float).reshape(-1, 3)
    if not len(references):
        return float("inf")
    combined = np.vstack([np.asarray(point, dtype=float).reshape(1, 3), references])
    distances2 = periodic_pair_distance2(combined, problem.cell, problem.pbc)[0, 1:]
    return float(np.sqrt(np.min(distances2)))


def sample_clear_discrete_positions(problem, count, existing, cutoff, attempts):
    existing = np.asarray(existing, dtype=float).reshape(-1, 3)
    accepted = []
    trials = 0
    while len(accepted) < int(count) and trials < int(attempts):
        batch_size = max(8, 2 * (int(count) - len(accepted)))
        sampled = np.asarray(problem.sample_initial_positions(batch_size), dtype=float)
        for point in sampled:
            trials += 1
            references = [problem.fixed_positions, existing]
            if accepted:
                references.append(np.asarray(accepted, dtype=float))
            populated = [item for item in references if len(item)]
            combined = np.vstack(populated) if populated else np.empty((0, 3))
            if cutoff <= 0.0 or minimum_distance_to_references(problem, point, combined) >= cutoff:
                accepted.append(point)
                if len(accepted) >= int(count):
                    break
            if trials >= int(attempts):
                break
    if len(accepted) != int(count):
        raise RuntimeError(
            f"Could place only {len(accepted)}/{int(count)} discrete atoms after "
            f"{int(attempts)} attempts"
        )
    return problem.wrap_positions(np.asarray(accepted, dtype=float))


def prune_discrete_positions(problem, positions, cutoff):
    positions = np.asarray(problem.wrap_positions(positions), dtype=float).reshape(-1, 3)
    if len(positions) < 2 or cutoff <= 0.0:
        return positions
    kept = []
    for index in np.random.permutation(len(positions)):
        point = positions[int(index)]
        references = [problem.fixed_positions]
        if kept:
            references.append(np.asarray(kept, dtype=float))
        populated = [item for item in references if len(item)]
        combined = np.vstack(populated) if populated else np.empty((0, 3))
        if minimum_distance_to_references(problem, point, combined) >= cutoff:
            kept.append(point)
    return np.asarray(kept, dtype=float).reshape(-1, 3)


def repair_discrete_positions(problem, positions, target_count, args):
    cutoff = float(args.ga_superposition_prune_distance_angstrom)
    repaired = prune_discrete_positions(problem, positions, cutoff)
    if len(repaired) > int(target_count):
        repaired = repaired[
            np.random.choice(len(repaired), size=int(target_count), replace=False)
        ]
    required = int(target_count) - len(repaired)
    if required > 0:
        additions = sample_clear_discrete_positions(
            problem,
            required,
            repaired,
            cutoff,
            int(args.discrete_repair_attempts),
        )
        repaired = np.vstack([repaired, additions])
    return np.asarray(problem.wrap_positions(repaired), dtype=float)


def make_discrete_initial_state(problem, args):
    """Ordinary-style hard initialization: fixed stoichiometric H count."""
    count = int(problem.target_missing_count)
    positions = np.asarray(problem.sample_initial_positions(count), dtype=float)
    positions = repair_discrete_positions(problem, positions, count, args)
    problem.set_site_count(len(positions))
    species_probabilities = initial_species_probability_matrix(problem, len(positions), args)
    winners = np.argmax(species_probabilities, axis=1)
    occupancies = np.zeros((len(positions), problem.n_species), dtype=float)
    occupancies[np.arange(len(positions)), winners] = 1.0
    return problem.pack_y(
        torch.tensor(occupancies, device=problem.device, dtype=problem.dtype),
        torch.tensor(positions, device=problem.device, dtype=problem.dtype),
    )


def make_random_member_initial_state(problem, args):
    mode = str(args.random_member_initialization).strip().lower()
    if mode in ("eat", "legacy_eat", "fractional"):
        return make_initial_state(problem, args)
    if mode in ("discrete", "ordinary_discrete", "poisson_discrete"):
        return make_discrete_initial_state(problem, args)
    raise ValueError(
        "initialization.random_member_initialization must be 'eat' or 'discrete'"
    )


def add_inactive_sites_to_baseline(problem, y):
    occ, positions = problem.unpack_y(y)
    old_n = int(problem.n_sites)
    add_count = max(0, int(problem.baseline_n_sites) - old_n)
    if add_count <= 0:
        return y, 0
    add_occ = torch.zeros((add_count, problem.n_species), device=occ.device, dtype=occ.dtype)
    add_pos = torch.tensor(problem.sample_initial_positions(add_count), device=positions.device, dtype=positions.dtype)
    problem.set_site_count(old_n + add_count)
    return problem.project(problem.pack_y(torch.cat([occ, add_occ], dim=0), torch.cat([positions, add_pos], dim=0))), add_count


def prune_inactive_to_baseline(problem, y, args):
    occ, positions = problem.unpack_y(y)
    old_n = int(problem.n_sites)
    occ_sum = occ.sum(dim=1).detach().cpu().numpy()
    active = np.where(occ_sum > float(args.ga_superposition_prune_occ_thresh))[0]
    inactive = np.where(occ_sum <= float(args.ga_superposition_prune_occ_thresh))[0]
    inactive = inactive[np.argsort(occ_sum[inactive])[::-1]]
    need_inactive = max(0, int(problem.baseline_n_sites) - len(active))
    keep = np.sort(np.concatenate([active, inactive[:need_inactive]])).astype(int)
    keep_t = torch.as_tensor(keep, device=occ.device, dtype=torch.long)
    problem.set_site_count(len(keep))
    y_new = problem.project(problem.pack_y(occ[keep_t], positions[keep_t]))
    y_new, added = add_inactive_sites_to_baseline(problem, y_new)
    return y_new, max(0, old_n - len(keep)), added


def prune_close_pairs_state(problem, y, args):
    cutoff = float(args.ga_superposition_prune_distance_angstrom)
    if cutoff <= 0.0 or problem.n_sites < 2:
        return y, 0
    occ, positions = problem.unpack_y(y)
    occ_np = occ.detach().cpu().numpy()
    occ_sum = occ_np.sum(axis=1)
    active_indices = np.where(occ_sum > float(args.ga_superposition_prune_occ_thresh))[0]
    if active_indices.size < 2:
        return y, 0
    pos_np = positions.detach().cpu().numpy()
    dist2 = periodic_pair_distance2(pos_np[active_indices], problem.cell, problem.pbc)
    cutoff2 = cutoff * cutoff
    removed = set()
    for i in range(len(active_indices)):
        site_i = int(active_indices[i])
        if site_i in removed:
            continue
        for j in range(i + 1, len(active_indices)):
            site_j = int(active_indices[j])
            if site_j in removed or dist2[i, j] >= cutoff2:
                continue
            occ_i = float(occ_sum[site_i])
            occ_j = float(occ_sum[site_j])
            if occ_i < occ_j:
                removed.add(site_i)
                break
            if occ_j < occ_i:
                removed.add(site_j)
                continue
            removed.add(site_j if np.random.random() < 0.5 else site_i)
            if site_i in removed:
                break
    if not removed:
        return y, 0
    old_n = int(problem.n_sites)
    keep = np.asarray([idx for idx in range(old_n) if idx not in removed], dtype=int)
    keep_t = torch.as_tensor(keep, device=occ.device, dtype=torch.long)
    problem.set_site_count(len(keep))
    return problem.project(problem.pack_y(occ[keep_t], positions[keep_t])), len(removed)


def shock_state(problem, y, args):
    occ, positions = problem.unpack_y(y)
    fallback = torch.full(
        (problem.n_sites, problem.n_species),
        1.0 / max(1, problem.n_species),
        device=occ.device,
        dtype=occ.dtype,
    )
    occupancy, species_probs = split_occupancy_and_species_probs(occ, fallback)
    if float(args.ga_crossover_occ_sigma) > 0.0:
        occupancy = occupancy + float(args.ga_crossover_occ_sigma) * torch.randn_like(occupancy)
    if float(args.ga_crossover_species_sigma) > 0.0:
        species_probs = project_simplex_rows(species_probs + float(args.ga_crossover_species_sigma) * torch.randn_like(species_probs))
    occ_new = combine_occupancy_and_species_probs(occupancy, species_probs)
    if float(args.ga_crossover_pos_sigma) > 0.0:
        positions = positions + float(args.ga_crossover_pos_sigma) * torch.randn_like(positions)
    return problem.project(problem.pack_y(occ_new, positions))


def crossover_state(problem, parent_a_y, parent_b_y, args):
    scale = max(0.0, min(1.0, float(args.ga_superposition_occ_scale)))
    problem.set_site_count(parent_a_y["n_sites"])
    occ_a, pos_a = problem.unpack_y(parent_a_y["y"])
    problem.set_site_count(parent_b_y["n_sites"])
    occ_b, pos_b = problem.unpack_y(parent_b_y["y"])
    problem.set_site_count(occ_a.shape[0] + occ_b.shape[0])
    child = problem.project(problem.pack_y(torch.cat([scale * occ_a, scale * occ_b], dim=0), torch.cat([pos_a, pos_b], dim=0)))
    child, _ = prune_close_pairs_state(problem, child, args)
    child, _ = add_inactive_sites_to_baseline(problem, child)
    child = shock_state(problem, child, args)
    child, _ = prune_close_pairs_state(problem, child, args)
    child, _ = add_inactive_sites_to_baseline(problem, child)
    return child


def ga_log_path(problem, args):
    path = str(getattr(args, "ga_log_file", "") or "").strip()
    return Path(path).expanduser() if path else problem.output_dir / "genetic_optimization_log.txt"


def append_ga_log(path, record):
    pieces = [str(record.get("event", "event"))]
    for key, value in record.items():
        if key == "event":
            continue
        if isinstance(value, float):
            pieces.append(f"{key}={value:+.6f}")
        else:
            pieces.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(" | ".join(pieces) + "\n")


def structures_db_path(problem):
    return problem.output_dir / "structures.json"


def read_structures_database(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": 1, "structures": []}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"schema_version": 1, "structures": data}
    data.setdefault("schema_version", 1)
    data.setdefault("structures", [])
    return data


def write_structures_database(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp.replace(path)


def append_structure_record(path, record):
    """Append one record without reading or rewriting the existing database.

    The file remains a normal JSON object with a ``structures`` array.  Only
    the small closing ``]}`` tail is replaced, making each checkpoint O(size of
    the new record) rather than O(size of the full run history).
    """
    path = Path(path)
    if not path.exists():
        write_structures_database(path, {"schema_version": 1, "structures": []})
    rendered = json.dumps(record, indent=2)
    rendered = "\n".join("    " + line for line in rendered.splitlines()).encode("utf-8")
    with path.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        tail_size = min(file_size, 65536)
        handle.seek(file_size - tail_size)
        tail = handle.read(tail_size)
        array_close = tail.rfind(b"]")
        if array_close < 0 or tail[array_close + 1:].strip() != b"}":
            raise ValueError(f"Cannot append to malformed structures database: {path}")
        content_end = array_close
        while content_end > 0 and tail[content_end - 1] in b" \t\r\n":
            content_end -= 1
        if content_end <= 0:
            raise ValueError(f"Cannot locate structures array contents in: {path}")
        is_empty = tail[content_end - 1:content_end] == b"["
        absolute_content_end = file_size - tail_size + content_end
        handle.seek(absolute_content_end)
        handle.truncate()
        handle.write((b"" if is_empty else b",") + b"\n" + rendered + b"\n  ]\n}\n")
        handle.flush()


def y_to_list(y):
    return y.detach().cpu().numpy().tolist()


def serialize_ga_result_for_worker(result):
    packed = dict(result)
    packed["y"] = y_to_list(result["y"])
    packed["y_rounded"] = y_to_list(result["y_rounded"])
    return packed


def restore_ga_result_tensors(problem, result):
    restored = dict(result)
    restored["y"] = torch.tensor(restored["y"], device=problem.device, dtype=problem.dtype)
    restored["y_rounded"] = torch.tensor(restored["y_rounded"], device=problem.device, dtype=problem.dtype)
    return restored


def args_to_worker_dict(args):
    return dict(vars(args))


def structure_database_record(problem, result, args, seed):
    problem.set_site_count(result["n_sites"])
    rounded_snapshot = result["rounded_snapshot"]
    rounded_payload = problem.realized_structure_payload(result["y_rounded"], rounded=True)
    return {
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "kind": str(result["kind"]),
        "parent_refs": list(result.get("parent_refs", [])),
        "score": float(result["score"]),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "composition_penalty": float(result["composition_term"]),
        "hole_boundary_penalty": float(result["hole_boundary_term"]),
        "occupancy_lower_bound_penalty": float(result.get("occupancy_lower_bound_term", 0.0)),
        "native_objective_score": float(result.get("native_objective_score", result["score"])),
        "random_member_initialization": str(result.get("random_member_initialization", "")),
        "realized_atom_count": int(result.get("realized_atom_count", rounded_snapshot.get("sumx", 0))),
        "final_relaxation": dict(result.get("final_relaxation", {})),
        "compute_metrics": dict(result.get("compute_metrics", {})),
        "seed": int(seed),
        "n_sites": int(result["n_sites"]),
        "cell_angstrom": problem.cell.tolist(),
        "fixed_symbols": list(problem.fixed_symbols),
        "fixed_positions_angstrom": problem.fixed_positions.tolist(),
        "null_symbols": rounded_payload["null_symbols"],
        "rounded_structure": rounded_payload,
        "continuous_state": {
            "species_occupancies": result["continuous_snapshot"]["species_occupancies"],
            "cart_positions_angstrom": result["continuous_snapshot"]["cart_positions_angstrom"],
        },
    }


def load_imported_parent_results(problem, args):
    path_text = str(getattr(args, "ga_import_structures_json", "") or "").strip()
    if not path_text:
        return []
    payload = read_structures_database(Path(path_text).expanduser())
    records = sorted(payload.get("structures", []), key=lambda item: float(item.get("score", np.inf)))
    limit = int(getattr(args, "ga_import_parent_count", 0) or 0)
    if limit > 0:
        records = records[:limit]
    else:
        records = records[: int(args.ga_population_size)]
    imported = []
    for rank, record in enumerate(records):
        state = record.get("continuous_state", {})
        occ = np.asarray(state.get("species_occupancies", []), dtype=float)
        pos = np.asarray(state.get("cart_positions_angstrom", []), dtype=float)
        if occ.ndim != 2 or pos.ndim != 2 or occ.shape[0] != pos.shape[0] or occ.shape[1] != problem.n_species:
            continue
        problem.set_site_count(occ.shape[0])
        y = problem.project(problem.pack_y(
            torch.tensor(occ, device=problem.device, dtype=problem.dtype),
            torch.tensor(pos, device=problem.device, dtype=problem.dtype),
        ))
        y, _, _ = prune_inactive_to_baseline(problem, y, args)
        y_round = problem.rounded_state(y)
        y_relaxed, relaxation = hard_fire_relax_state(problem, y_round, args)
        snap = problem.final_snapshot(
            y_relaxed,
            f"imported_rank_{rank}_hard_fire_relaxed",
            precomputed_energy=relaxation.get("final_energy_eV"),
            energy_source=relaxation.get("final_energy_source"),
        )
        score = float(snap["f_guided_no_entropy"])
        if relaxation["converged"] is False:
            score += float(args.final_relax_unconverged_penalty)
        imported.append({
            "generation": -1,
            "member_index": rank,
            "kind": "imported_parent",
            "parent_refs": [],
            "y": y_relaxed.detach().clone(),
            "y_rounded": y_relaxed.detach().clone(),
            "n_sites": int(problem.n_sites),
            "score": score,
            "native_objective_score": float(snap["f_guided_no_entropy"]),
            "formation_energy_per_atom": float(snap["formation_energy_per_atom"]),
            "composition_term": float(snap["composition_term"]),
            "hole_boundary_term": float(snap["hole_boundary_term"]),
            "occupancy_lower_bound_term": float(snap["occupancy_lower_bound_term"]),
            "sumx": float(snap["sumx"]),
            "continuous_snapshot": snap,
            "rounded_snapshot": snap,
            "final_relaxation": relaxation,
            "realized_atom_count": int(relaxation["realized_atom_count"]),
            "random_member_initialization": str(args.random_member_initialization),
        })
    return imported


def hard_fire_relax_state(problem, y_rounded, args):
    """Relax the realized hard atoms while keeping scaffold atoms and cell fixed."""
    y_hard = problem.rounded_state(y_rounded).detach().clone()
    occ, positions = problem.unpack_y(y_hard)
    active_indices = np.where(
        occ.detach().cpu().numpy().sum(axis=1) > float(args.keep_thresh)
    )[0]
    metadata = {
        "performed": bool(args.final_relax_enable),
        "converged": None,
        "steps": 0,
        "max_force_eV_per_angstrom": 0.0,
        "realized_atom_count": int(len(active_indices)),
    }
    if not bool(args.final_relax_enable) or not len(active_indices):
        if not len(active_indices):
            metadata["converged"] = True
        return y_hard, metadata

    atoms = problem.realized_atoms(y_hard, rounded=True)
    fixed_count = len(problem.fixed_symbols)
    atoms.set_constraint(FixAtoms(indices=np.arange(fixed_count, dtype=int)))
    atoms.calc = problem.calc
    optimizer = FIRE(
        atoms,
        logfile=None,
        trajectory=None,
        maxstep=float(args.final_relax_maxstep_angstrom),
    )
    converged = bool(
        optimizer.run(
            fmax=float(args.final_relax_fmax),
            steps=int(args.final_relax_steps),
        )
    )
    atoms.wrap()
    relaxed_positions = np.asarray(atoms.get_positions()[fixed_count:], dtype=float)
    if len(relaxed_positions) != len(active_indices):
        raise RuntimeError(
            "Hard FIRE relaxation changed the number of realized candidate atoms"
        )
    updated_positions = positions.detach().clone()
    updated_positions[torch.as_tensor(active_indices, device=positions.device)] = torch.tensor(
        relaxed_positions,
        device=positions.device,
        dtype=positions.dtype,
    )
    raw_forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)[fixed_count:]
    # FIRE's final force request has already populated the calculator results,
    # so this energy read is an ASE cache hit rather than another model call.
    final_energy = float(atoms.get_potential_energy())
    max_force = (
        float(np.max(np.linalg.norm(raw_forces, axis=1)))
        if len(raw_forces)
        else 0.0
    )
    metadata.update({
        "converged": converged,
        "steps": int(getattr(optimizer, "nsteps", 0)),
        "max_force_eV_per_angstrom": max_force,
        "final_energy_eV": final_energy,
        "final_energy_source": "hard_atom_FIRE_calculator_cache",
    })
    return problem.pack_y(occ.detach().clone(), updated_positions), metadata


def evaluate_ga_member(problem, y0, args, seed, generation, member_index, parent_refs=None, kind="random"):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    n_sites = int(len(y0) // (problem.n_species + 3))
    problem.set_site_count(n_sites)
    tracker = problem.model_evaluation_tracker
    tracker_before = tracker.snapshot()
    objective_evaluations_before = int(problem.objective_evaluation_count)
    member_wall_start = time.perf_counter()
    previous_phase = tracker.phase
    phase_wall_seconds = {}
    try:
        tracker.phase = "syntropization"
        phase_start = time.perf_counter()
        y_final = problem.run(y0, run_kind=f"ga_generation_{generation}_member_{member_index}_{kind}")
        phase_wall_seconds["syntropization"] = time.perf_counter() - phase_start

        preparation_start = time.perf_counter()
        y_final, pruned, added = prune_inactive_to_baseline(problem, y_final, args)
        y_rounded = problem.rounded_state(y_final)
        phase_wall_seconds["state_preparation"] = time.perf_counter() - preparation_start

        # The former continuous and pre-FIRE snapshots were not used for
        # ranking or persisted output, but each caused a full model evaluation.
        tracker.phase = "final_FIRE"
        phase_start = time.perf_counter()
        y_relaxed, relaxation = hard_fire_relax_state(problem, y_rounded, args)
        phase_wall_seconds["final_FIRE"] = time.perf_counter() - phase_start

        tracker.phase = "final_scoring"
        phase_start = time.perf_counter()
        hard_snapshot = problem.final_snapshot(
            y_relaxed,
            f"ga_generation_{generation}_member_{member_index}_hard_fire_relaxed",
            precomputed_energy=relaxation.get("final_energy_eV"),
            energy_source=relaxation.get("final_energy_source"),
        )
        phase_wall_seconds["final_scoring"] = time.perf_counter() - phase_start
    finally:
        tracker.phase = previous_phase

    compute_metrics = ModelEvaluationTracker.difference(tracker.snapshot(), tracker_before)
    compute_metrics.update({
        "objective_evaluations": int(problem.objective_evaluation_count - objective_evaluations_before),
        "accepted_optimizer_iterations": int(sum(item.get("accepted_iterations", 0) for item in problem.stage_summaries)),
        "member_wall_seconds": float(time.perf_counter() - member_wall_start),
        "phase_wall_seconds": {key: float(value) for key, value in phase_wall_seconds.items()},
    })
    native_score = float(hard_snapshot["f_guided_no_entropy"])
    score = native_score
    if relaxation["converged"] is False:
        score += float(args.final_relax_unconverged_penalty)
    return {
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "parent_refs": list(parent_refs or []),
        "seed": int(seed),
        "y": y_relaxed.detach().clone(),
        "y_rounded": y_relaxed.detach().clone(),
        "n_sites": int(problem.n_sites),
        "score": float(score),
        "native_objective_score": native_score,
        "formation_energy_per_atom": float(hard_snapshot["formation_energy_per_atom"]),
        "composition_term": float(hard_snapshot["composition_term"]),
        "hole_boundary_term": float(hard_snapshot["hole_boundary_term"]),
        "occupancy_lower_bound_term": float(hard_snapshot["occupancy_lower_bound_term"]),
        "sumx": float(hard_snapshot["sumx"]),
        "post_relax_pruned_inactive_sites": int(pruned),
        "post_relax_added_inactive_sites": int(added),
        "continuous_snapshot": hard_snapshot,
        "rounded_snapshot": hard_snapshot,
        "final_relaxation": relaxation,
        "compute_metrics": compute_metrics,
        "realized_atom_count": int(relaxation["realized_atom_count"]),
        "random_member_initialization": str(args.random_member_initialization),
        "history": [] if bool(args.disable_trajectory_history) else list(problem.history),
        "stage_summaries": list(problem.stage_summaries),
    }


def parse_parallel_gpu_ids(text):
    ids = [part.strip() for part in str(text).split(",") if part.strip()]
    return ids or ["0"]


def ga_worker_loop(worker_index, gpu_id, args_dict, task_queue, result_queue):
    try:
        configured_device = str(args_dict.get("device", "auto") or "auto").strip().lower()
        if configured_device.startswith("cuda") and str(gpu_id).lower() != "cpu":
            if torch.cuda.is_available():
                torch.cuda.set_device(int(gpu_id))
            else:
                terminal_print(SimpleNamespace(**args_dict), f"[ga worker {worker_index}] CUDA not available; falling back to CPU")
        args = SimpleNamespace(**args_dict)
        worker_seed = int(args.seed or 0) + 7919 * (int(worker_index) + 1)
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed % 2**32)
        problem = InpaintingProblem(args)
        terminal_print(args, f"[ga worker {worker_index}] ready gpu={gpu_id} device={problem.device}")
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = str(task["task_id"])
            try:
                member_seed = int(task["seed"])
                torch.manual_seed(member_seed)
                np.random.seed(member_seed % 2**32)
                y0 = torch.tensor(task["y0"], device=problem.device, dtype=problem.dtype)
                result = evaluate_ga_member(
                    problem,
                    y0,
                    args,
                    member_seed,
                    int(task["generation"]),
                    int(task["member_index"]),
                    parent_refs=task.get("parent_refs", []),
                    kind=str(task.get("kind", "random")),
                )
                for key, value in dict(task.get("result_metadata", {})).items():
                    result[key] = value
                result_queue.put({
                    "task_id": task_id,
                    "ok": True,
                    "worker_index": int(worker_index),
                    "gpu_id": str(gpu_id),
                    "result": serialize_ga_result_for_worker(result),
                })
            except Exception:
                result_queue.put({
                    "task_id": task_id,
                    "ok": False,
                    "worker_index": int(worker_index),
                    "gpu_id": str(gpu_id),
                    "traceback": traceback.format_exc(),
                })
    except Exception:
        result_queue.put({
            "task_id": "__worker_init__",
            "ok": False,
            "worker_index": int(worker_index),
            "gpu_id": str(gpu_id),
            "traceback": traceback.format_exc(),
        })


def start_ga_workers(args, worker_count):
    gpu_ids = parse_parallel_gpu_ids(args.ga_parallel_gpu_ids)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = []
    args_dict = args_to_worker_dict(args)
    for worker_index in range(int(worker_count)):
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        process = ctx.Process(
            target=ga_worker_loop,
            args=(worker_index, gpu_id, args_dict, task_queue, result_queue),
            daemon=True,
        )
        process.start()
        workers.append(process)
        terminal_print(args, f"[ga parallel] started worker={worker_index} gpu={gpu_id} pid={process.pid}")
    return task_queue, result_queue, workers


def stop_ga_workers(args, task_queue, workers):
    if task_queue is not None:
        for _ in workers:
            task_queue.put(None)
    for process in workers:
        process.join()
        terminal_print(args, f"[ga parallel] stopped pid={process.pid} exitcode={process.exitcode}")


def make_evaluation_task(task_id, y0, member_seed, generation, member_index, kind, parent_refs=None, result_metadata=None):
    return {
        "task_id": str(task_id),
        "y0": y_to_list(y0),
        "seed": int(member_seed),
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "parent_refs": [] if parent_refs is None else list(parent_refs),
        "result_metadata": {} if result_metadata is None else dict(result_metadata),
    }


def collect_parallel_ga_tasks(problem, args, seed, db_path, log_path, all_generations, generation_results, result_queue, expected_count):
    completed = 0
    last_main_heartbeat = 0.0
    interval = float(getattr(args, "heartbeat_interval_seconds", 0.0) or 0.0)
    while completed < int(expected_count):
        try:
            timeout = max(1.0, min(interval, 60.0)) if interval > 0.0 else None
            message = result_queue.get(timeout=timeout)
        except queue_module.Empty:
            now = time.time()
            if interval > 0.0 and now - last_main_heartbeat >= interval:
                append_running_log(
                    args,
                    f"main waiting for generation members completed={completed}/{int(expected_count)}; workers still running",
                )
                last_main_heartbeat = now
            continue
        if not message.get("ok", False):
            raise RuntimeError(
                f"GA worker failed on task {message.get('task_id')} "
                f"(worker={message.get('worker_index')} gpu={message.get('gpu_id')}):\n"
                f"{message.get('traceback', '')}"
            )
        result = restore_ga_result_tensors(problem, message["result"])
        terminal_print(
            args,
            f"[ga parallel] completed generation={result['generation']} member={result['member_index']} "
            f"kind={result['kind']} worker={message.get('worker_index')} gpu={message.get('gpu_id')} "
            f"score={result['score']:+.6f}"
        )
        record_completed_ga_result(
            problem,
            args,
            seed,
            db_path,
            log_path,
            all_generations,
            generation_results,
            result,
            event="member_done",
        )
        completed += 1


def parent_ref(parent):
    return f"g{int(parent.get('generation', -1))}_m{int(parent.get('member_index', -1))}_{parent.get('kind', 'parent')}"


def allocate_generation_counts(population_size, args):
    fractions = np.asarray([float(args.ga_elite_fraction), float(args.ga_random_fraction), float(args.ga_crossover_fraction)], dtype=float)
    if np.any(fractions < 0.0):
        raise ValueError("GA fractions must be non-negative.")
    if float(fractions.sum()) <= 0.0:
        raise ValueError("At least one GA fraction must be positive.")
    raw = fractions / fractions.sum() * int(population_size)
    counts = np.floor(raw).astype(int)
    remainder = int(population_size - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {"elite": int(counts[0]), "random": int(counts[1]), "crossover": int(counts[2])}


def member_spatial_species_vector(problem, member):
    problem.set_site_count(int(member["n_sites"]))
    y = member.get("y_rounded", member["y"])
    occ, positions = problem.unpack_y(y)
    occ_np = occ.detach().cpu().numpy()
    pos_np = positions.detach().cpu().numpy()
    grid = np.asarray(problem.subcell_grid, dtype=int)
    vector = np.zeros((int(np.prod(grid)), problem.n_species), dtype=float)
    frac = problem.positions_to_frac(pos_np)
    rel = (frac - problem.hole_min_frac[None, :]) / np.maximum(problem.hole_max_frac - problem.hole_min_frac, 1e-12)[None, :]
    bins = np.floor(rel * grid[None, :]).astype(int)
    valid = np.all((bins >= 0) & (bins < grid[None, :]), axis=1)
    if np.any(valid):
        flat = np.ravel_multi_index(bins[valid].T, tuple(grid.tolist()))
        np.add.at(vector, flat, occ_np[valid])
    return vector.reshape(-1)


def member_diversity_distance(problem, member_a, member_b):
    return float(np.sum(np.abs(member_spatial_species_vector(problem, member_a) - member_spatial_species_vector(problem, member_b))))


def tournament_parent(parent_pool, tournament_size, exclude=None):
    candidates = [parent for parent in parent_pool if parent is not exclude]
    if not candidates:
        raise ValueError("No eligible tournament parents remain.")
    size = min(max(1, int(tournament_size)), len(candidates))
    sampled = list(np.random.choice(candidates, size=size, replace=False))
    return min(sampled, key=lambda item: float(item["score"]))


def choose_tournament_pair(parent_pool, tournament_size):
    if len(parent_pool) < 2:
        raise ValueError("Need at least two parents for crossover.")
    parent_a = tournament_parent(parent_pool, tournament_size)
    parent_b = tournament_parent(parent_pool, tournament_size, exclude=parent_a)
    return parent_a, parent_b


def live_plot_path(problem, args):
    path = str(getattr(args, "ga_live_plot_file", "") or "").strip()
    return Path(path).expanduser() if path else problem.output_dir / "genetic_optimization_live_scores_by_generation.png"


def stable_generation_jitter(generation, member_index, jitter):
    phase = ((int(generation) + 1) * 73856093 + (int(member_index) + 1) * 19349663) % 10000
    return (phase / 9999.0 - 0.5) * 2.0 * float(jitter)


def update_live_ga_plot(problem, args, generations, current_generation=None):
    if not bool(args.ga_live_plot_enable):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    metric = str(args.ga_live_plot_metric or "score")
    groups = {
        "random": {"x": [], "y": [], "color": "red"},
        "crossover": {"x": [], "y": [], "color": "green"},
        "elite/imported": {"x": [], "y": [], "color": "gray"},
        "other": {"x": [], "y": [], "color": "black"},
    }
    all_generation_lists = list(generations)
    if current_generation is not None:
        all_generation_lists.append(current_generation)
    best_by_generation = []
    for gen_idx, members in enumerate(all_generation_lists):
        if not members:
            continue
        best_by_generation.append((gen_idx, min(float(m["score"]) for m in members)))
        for member in members:
            kind = str(member.get("kind", "other"))
            if "random" in kind:
                label = "random"
            elif "crossover" in kind:
                label = "crossover"
            elif "elite" in kind or "imported" in kind:
                label = "elite/imported"
            else:
                label = "other"
            groups[label]["x"].append(gen_idx + stable_generation_jitter(gen_idx, member.get("member_index", 0), args.ga_live_plot_jitter))
            groups[label]["y"].append(float(member.get(metric, member.get("score", np.nan))))
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in groups.items():
        if group["x"]:
            ax.scatter(group["x"], group["y"], s=46, c=group["color"], alpha=float(args.ga_live_plot_alpha), edgecolors="none", label=label)
    if best_by_generation:
        x_best, y_best = zip(*best_by_generation)
        ax.plot(x_best, y_best, color="black", linewidth=1.6, alpha=0.75, label="best per generation")
    ax.set_xlabel("Generation", fontsize=18)
    ax.set_ylabel(metric.replace("_", " "), fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=13)
    ax.set_title("GA Population Scores by Generation", fontsize=18, pad=14)
    formula = problem.mp_metadata.get("formula_pretty", "") or str(problem.args.material_formula)
    completed_members = [member for members in all_generation_lists for member in members]
    live_energy_evaluations = sum(int(member.get("compute_metrics", {}).get("model_energy_evaluations", 0)) for member in completed_members)
    live_model_seconds = sum(float(member.get("compute_metrics", {}).get("model_evaluation_seconds", 0.0)) for member in completed_members)
    ax.text(
        0.0,
        -0.14,
        f"{formula} H-sublattice   ranking=objective-no-entropy   "
        f"energy evaluations={live_energy_evaluations:,}   model time={live_model_seconds / 3600.0:.2f} h",
        transform=ax.transAxes,
        fontsize=10,
        color="0.35",
    )
    path = live_plot_path(problem, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate_compute_metrics(members):
    """Sum candidate-compute metrics without double-counting copied elites."""
    total = {
        "model_energy_evaluations": 0,
        "model_evaluation_seconds": 0.0,
        "objective_evaluations": 0,
        "accepted_optimizer_iterations": 0,
        "member_wall_seconds_sum": 0.0,
        "by_phase": {},
    }
    for member in members:
        metrics = member.get("compute_metrics", {})
        total["model_energy_evaluations"] += int(metrics.get("model_energy_evaluations", 0))
        total["model_evaluation_seconds"] += float(metrics.get("model_evaluation_seconds", 0.0))
        total["objective_evaluations"] += int(metrics.get("objective_evaluations", 0))
        total["accepted_optimizer_iterations"] += int(metrics.get("accepted_optimizer_iterations", 0))
        total["member_wall_seconds_sum"] += float(metrics.get("member_wall_seconds", 0.0))
        for phase, values in metrics.get("by_phase", {}).items():
            entry = total["by_phase"].setdefault(phase, {"count": 0, "seconds": 0.0})
            entry["count"] += int(values.get("count", 0))
            entry["seconds"] += float(values.get("seconds", 0.0))
    return total


def empty_compute_metrics():
    return {
        "model_energy_evaluations": 0,
        "model_evaluation_seconds": 0.0,
        "objective_evaluations": 0,
        "accepted_optimizer_iterations": 0,
        "member_wall_seconds": 0.0,
        "phase_wall_seconds": {},
        "by_phase": {},
    }


def ga_member_log_record(result, event):
    metrics = result.get("compute_metrics", {})
    phases = metrics.get("by_phase", {})
    phase_wall = metrics.get("phase_wall_seconds", {})
    return {
        "event": event,
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "kind": str(result["kind"]),
        "score": float(result["score"]),
        "native_objective_score": float(result.get("native_objective_score", result["score"])),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "composition_term": float(result["composition_term"]),
        "hole_boundary_term": float(result["hole_boundary_term"]),
        "occupancy_lower_bound_term": float(result.get("occupancy_lower_bound_term", 0.0)),
        "n_sites": int(result["n_sites"]),
        "realized_atom_count": int(result.get("realized_atom_count", 0)),
        "random_member_initialization": str(result.get("random_member_initialization", "")),
        "final_relaxation_converged": result.get("final_relaxation", {}).get("converged"),
        "final_relaxation_steps": int(result.get("final_relaxation", {}).get("steps", 0)),
        "final_max_force_eV_per_angstrom": float(result.get("final_relaxation", {}).get("max_force_eV_per_angstrom", 0.0)),
        "energy_evaluations": int(metrics.get("model_energy_evaluations", 0)),
        "energy_evaluation_seconds": float(metrics.get("model_evaluation_seconds", 0.0)),
        "member_wall_seconds": float(metrics.get("member_wall_seconds", 0.0)),
        "objective_evaluations": int(metrics.get("objective_evaluations", 0)),
        "accepted_optimizer_iterations": int(metrics.get("accepted_optimizer_iterations", 0)),
        "syntropization_energy_evaluations": int(phases.get("syntropization", {}).get("count", 0)),
        "syntropization_energy_evaluation_seconds": float(phases.get("syntropization", {}).get("seconds", 0.0)),
        "syntropization_wall_seconds": float(phase_wall.get("syntropization", 0.0)),
        "final_FIRE_energy_evaluations": int(phases.get("final_FIRE", {}).get("count", 0)),
        "final_FIRE_energy_evaluation_seconds": float(phases.get("final_FIRE", {}).get("seconds", 0.0)),
        "final_FIRE_wall_seconds": float(phase_wall.get("final_FIRE", 0.0)),
        "final_scoring_energy_evaluations": int(phases.get("final_scoring", {}).get("count", 0)),
        "final_scoring_energy_evaluation_seconds": float(phases.get("final_scoring", {}).get("seconds", 0.0)),
        "final_scoring_wall_seconds": float(phase_wall.get("final_scoring", 0.0)),
    }


def write_ga_summary(problem, args, log_path, generations, best, ga_wall_seconds):
    total_compute = aggregate_compute_metrics([member for generation in generations for member in generation])
    payload = {
        "generator": str(Path(__file__).resolve()),
        "config_path": str(args.config_path),
        "ga_enabled": True,
        "ga_population_size": int(args.ga_population_size),
        "ga_generations": int(args.ga_generations),
        "ga_elite_fraction": float(args.ga_elite_fraction),
        "ga_random_fraction": float(args.ga_random_fraction),
        "ga_crossover_fraction": float(args.ga_crossover_fraction),
        "ga_parent_pool_size": int(args.ga_parent_pool_size),
        "ga_parent_selection": "two_independent_best_of_tournament",
        "ga_tournament_size": int(args.ga_parent_a_tournament_size),
        "ga_elite_selection": "best_score",
        "ga_superposition_occ_scale": float(args.ga_superposition_occ_scale),
        "ga_superposition_prune_distance_angstrom": float(args.ga_superposition_prune_distance_angstrom),
        "ga_score": "hard_FIRE_relaxed_formation_energy_plus_composition_plus_boundary_no_entropy",
        "ga_score_energy_source": "final hard-atom FIRE calculator result reused without an additional model evaluation",
        "ga_random_member_initialization": str(args.random_member_initialization),
        "ga_discrete_count_min": int(args.discrete_count_min),
        "ga_discrete_count_max_factor": float(args.discrete_count_max_factor),
        "ga_discrete_repair_attempts": int(args.discrete_repair_attempts),
        "ga_candidate_relaxation": {
            "enable": bool(args.final_relax_enable),
            "fmax": float(args.final_relax_fmax),
            "steps": int(args.final_relax_steps),
            "maxstep_angstrom": float(args.final_relax_maxstep_angstrom),
            "unconverged_penalty": float(args.final_relax_unconverged_penalty),
            "fixed_cell": True,
            "fixed_scaffold": True,
        },
        "ga_execution_backend": str(args.ga_execution_backend),
        "ga_parallel_workers": int(args.ga_parallel_workers),
        "ga_parallel_gpu_ids": str(args.ga_parallel_gpu_ids),
        "compute_metrics_documentation": {
            "energy_evaluation_definition": "One actual FAIR-Chem calculator.calculate execution. A joint energy/forces/EAT-gradient calculation counts once; ASE cache hits count zero.",
            "energy_evaluation_seconds_definition": "Summed wall time spent inside FAIR-Chem calculator.calculate across all workers; this is aggregate device work time, not elapsed GA wall time.",
            "member_wall_seconds_sum_definition": "Sum of individual evaluated-member elapsed times. With parallel workers it intentionally exceeds elapsed GA wall time.",
            "objective_evaluation_definition": "One PBFGS objective/gradient call, including line-search trials.",
            "accepted_optimizer_iteration_definition": "One optimizer callback after an accepted PBFGS step.",
            "copied_elites": "Elite copies have zero compute metrics because their original evaluation is not repeated.",
            "scope": "Candidate metrics exclude one-time model loading, reference construction/relaxation, and imported-parent preprocessing. GA wall time includes orchestration, plotting, logging, and database appends.",
            "member_wall_seconds_definition": "Elapsed worker time from the start of candidate syntropization through final scoring; it excludes main-process live plotting and database output.",
        },
        "total_candidate_compute_metrics": total_compute,
        "ga_wall_seconds": float(ga_wall_seconds),
        "generation_compute_metrics": [aggregate_compute_metrics(generation) for generation in generations],
        "geometry_mode": str(problem.geometry_mode),
        "ground_truth_source": problem.ground_truth_source_type,
        "ground_truth_source_file": problem.ground_truth_source_file,
        "mp_metadata": problem.mp_metadata,
        "slab_size": None if problem.slab_size is None else problem.slab_size.astype(int).tolist(),
        "vacuum_size_z": int(problem.vacuum_size_z),
        "supercell": problem.supercell.astype(int).tolist(),
        "search_region_start": problem.hole_start.astype(int).tolist(),
        "search_region_size": problem.hole_size.astype(int).tolist(),
        "n_fixed_atoms": int(len(problem.fixed_symbols)),
        "n_missing_atoms": int(len(problem.missing_symbols)),
        "material_atom_density_per_angstrom3": float(problem.material_atom_density_per_angstrom3),
        "estimated_search_region_atoms": float(problem.estimated_search_region_atoms),
        "occupancy_lower_bound_enable": bool(problem.occupancy_lower_bound_enable),
        "occupancy_lower_bound_gamma": float(problem.occupancy_lower_bound_gamma),
        "occupancy_lower_bound": float(problem.occupancy_lower_bound),
        "occupancy_lower_bound_penalty": float(problem.occupancy_lower_bound_penalty),
        "occupancy_lower_bound_softplus_sharpness": float(problem.occupancy_lower_bound_softplus_sharpness),
        "ground_truth_filled_energy_total_eV": float(problem.full_reference_energy_total),
        "ground_truth_filled_formation_energy_total_eV": float(problem.full_reference_formation_energy_total),
        "full_reference_formation_energy_per_atom": float(problem.full_reference_formation_energy_per_atom),
        "best": ga_member_log_record(best, "best"),
        "generations": [
            [ga_member_log_record(member, "member") for member in members]
            for members in generations
        ],
        "ga_log_file": str(log_path),
        "ga_live_plot_file": str(live_plot_path(problem, args)),
    }
    with (problem.output_dir / "genetic_optimization_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def copy_parent_to_generation(parent, generation, member_index, kind="elite_copy"):
    copied = dict(parent)
    copied["generation"] = int(generation)
    copied["member_index"] = int(member_index)
    copied["kind"] = str(kind)
    copied["parent_refs"] = [parent_ref(parent)]
    copied["y"] = parent["y"].detach().clone()
    copied["y_rounded"] = parent["y_rounded"].detach().clone()
    copied["compute_metrics"] = empty_compute_metrics()
    return copied


def record_completed_ga_result(problem, args, seed, db_path, log_path, generations, generation_results, result, event="member_done"):
    append_structure_record(db_path, structure_database_record(problem, result, args, seed))
    append_ga_log(log_path, ga_member_log_record(result, event))
    generation_results.append(result)
    update_live_ga_plot(problem, args, generations, generation_results)
    metrics = result.get("compute_metrics", {})
    append_running_log(
        args,
        f"{event} generation={result['generation']} member={result['member_index']} kind={result['kind']} "
        f"score={result['score']:+.6f} energy_evaluations={int(metrics.get('model_energy_evaluations', 0))} "
        f"energy_eval_s={float(metrics.get('model_evaluation_seconds', 0.0)):.2f} "
        f"member_wall_s={float(metrics.get('member_wall_seconds', 0.0)):.2f}",
    )


def run_genetic_algorithm(problem, args, seed):
    population_size = int(args.ga_population_size)
    generations_count = int(args.ga_generations)
    if population_size <= 0 or generations_count <= 0:
        raise ValueError("GA population_size and generations must be positive.")
    execution_backend = str(getattr(args, "ga_execution_backend", "local")).strip().lower()
    if execution_backend != "local":
        raise ValueError("This inpainting script currently supports genetic.execution_backend: local only.")
    parallel_workers = max(1, int(getattr(args, "ga_parallel_workers", 1)))
    parallel_workers = min(parallel_workers, population_size)
    counts = allocate_generation_counts(population_size, args)
    log_path = ga_log_path(problem, args)
    log_path.write_text("", encoding="utf-8")
    db_path = structures_db_path(problem)
    write_structures_database(db_path, {"schema_version": 1, "structures": []})
    running_path = running_log_path_from_args(args)
    running_path.parent.mkdir(parents=True, exist_ok=True)
    running_path.write_text(f"{timestamp_now()} | ga_start seed={int(seed)}\n", encoding="utf-8")
    append_ga_log(log_path, {
        "event": "ga_start",
        "seed": int(seed),
        "population_size": population_size,
        "generations": generations_count,
        "elite_fraction": float(args.ga_elite_fraction),
        "random_fraction": float(args.ga_random_fraction),
        "crossover_fraction": float(args.ga_crossover_fraction),
        "parent_selection": "two_independent_best_of_tournament",
        "tournament_size": int(args.ga_parent_a_tournament_size),
        "random_member_initialization": str(args.random_member_initialization),
        "final_relax_enable": bool(args.final_relax_enable),
        "final_relax_fmax": float(args.final_relax_fmax),
        "final_relax_steps": int(args.final_relax_steps),
        "final_relax_maxstep_angstrom": float(args.final_relax_maxstep_angstrom),
        "crossover_prune_distance": float(args.ga_superposition_prune_distance_angstrom),
        "execution_backend": execution_backend,
        "parallel_workers": int(parallel_workers),
        "parallel_gpu_ids": str(args.ga_parallel_gpu_ids),
        "energy_evaluation_definition": "one_actual_FAIRChem_calculator.calculate_call",
    })
    ga_wall_start = time.perf_counter()
    all_generations = []
    previous_population = []
    imported_parents = load_imported_parent_results(problem, args)
    seed_generation_from_imports = len(imported_parents) > 0
    best = None
    member_seed_base = int(seed)
    task_queue = None
    result_queue = None
    workers = []

    def submit_or_run(task):
        if parallel_workers > 1:
            task_queue.put(task)
            return 1
        y0_local = torch.tensor(task["y0"], device=problem.device, dtype=problem.dtype)
        result_local = evaluate_ga_member(
            problem,
            y0_local,
            args,
            int(task["seed"]),
            int(task["generation"]),
            int(task["member_index"]),
            parent_refs=task.get("parent_refs", []),
            kind=str(task.get("kind", "random")),
        )
        for key, value in dict(task.get("result_metadata", {})).items():
            result_local[key] = value
        record_completed_ga_result(problem, args, seed, db_path, log_path, all_generations, current_generation_results, result_local)
        return 0

    try:
        if parallel_workers > 1:
            task_queue, result_queue, workers = start_ga_workers(args, parallel_workers)
            terminal_print(args, f"[ga parallel] enabled workers={parallel_workers} gpu_ids={parse_parallel_gpu_ids(args.ga_parallel_gpu_ids)}")

        for generation in range(generations_count):
            generation_wall_start = time.perf_counter()
            append_running_log(args, f"generation_start generation={generation}")
            append_ga_log(log_path, {"event": "generation_start", "generation": generation})
            current_generation_results = []
            pending_count = 0

            if generation == 0:
                member_index = 0
                if seed_generation_from_imports:
                    for imported in imported_parents[:population_size]:
                        copied = copy_parent_to_generation(imported, generation, member_index, kind="imported_seed")
                        record_completed_ga_result(problem, args, seed, db_path, log_path, all_generations, current_generation_results, copied, event="imported_seed_used")
                        member_index += 1
                while member_index < population_size:
                    member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                    torch.manual_seed(member_seed)
                    np.random.seed(member_seed)
                    problem.set_site_count(problem.baseline_n_sites)
                    y0 = make_random_member_initial_state(problem, args)
                    append_ga_log(log_path, {"event": "member_start", "generation": generation, "member_index": member_index, "kind": "random", "seed": int(member_seed)})
                    pending_count += submit_or_run(make_evaluation_task(f"g{generation}_m{member_index}", y0, member_seed, generation, member_index, "random"))
                    member_index += 1
            else:
                parents = sorted(previous_population, key=lambda item: item["score"])
                parent_pool = parents[: min(int(args.ga_parent_pool_size), len(parents))]
                member_index = 0
                for elite in parents[: counts["elite"]]:
                    copied = copy_parent_to_generation(elite, generation, member_index, kind="elite_copy")
                    record_completed_ga_result(problem, args, seed, db_path, log_path, all_generations, current_generation_results, copied, event="elite_copied")
                    member_index += 1
                random_stop = min(population_size, member_index + counts["random"])
                while member_index < random_stop:
                    member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                    torch.manual_seed(member_seed)
                    np.random.seed(member_seed)
                    problem.set_site_count(problem.baseline_n_sites)
                    y0 = make_random_member_initial_state(problem, args)
                    append_ga_log(log_path, {"event": "member_start", "generation": generation, "member_index": member_index, "kind": "random_immigrant", "seed": int(member_seed)})
                    pending_count += submit_or_run(make_evaluation_task(f"g{generation}_m{member_index}", y0, member_seed, generation, member_index, "random_immigrant"))
                    member_index += 1
                while member_index < population_size:
                    member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                    torch.manual_seed(member_seed)
                    np.random.seed(member_seed)
                    parent_a, parent_b = choose_tournament_pair(parent_pool, args.ga_parent_a_tournament_size)
                    diversity = member_diversity_distance(problem, parent_a, parent_b)
                    refs = [parent_ref(parent_a), parent_ref(parent_b)]
                    append_ga_log(log_path, {
                        "event": "member_start",
                        "generation": generation,
                        "member_index": member_index,
                        "kind": "crossover_child",
                        "parent_a": refs[0],
                        "parent_b": refs[1],
                        "parent_diversity": diversity,
                        "seed": int(member_seed),
                    })
                    y0 = crossover_state(problem, parent_a, parent_b, args)
                    pending_count += submit_or_run(make_evaluation_task(
                        f"g{generation}_m{member_index}",
                        y0,
                        member_seed,
                        generation,
                        member_index,
                        "crossover_child",
                        parent_refs=refs,
                        result_metadata={"parent_diversity": float(diversity)},
                    ))
                    member_index += 1

            if parallel_workers > 1 and pending_count > 0:
                collect_parallel_ga_tasks(
                    problem,
                    args,
                    seed,
                    db_path,
                    log_path,
                    all_generations,
                    current_generation_results,
                    result_queue,
                    pending_count,
                )

            current_generation_results.sort(key=lambda item: item["score"])
            previous_population = current_generation_results
            all_generations.append(current_generation_results)
            generation_best = current_generation_results[0]
            if best is None or generation_best["score"] < best["score"]:
                best = generation_best
            generation_compute = aggregate_compute_metrics(current_generation_results)
            generation_wall_seconds = time.perf_counter() - generation_wall_start
            append_ga_log(log_path, {
                "event": "generation_done",
                "generation": generation,
                "best_score": float(generation_best["score"]),
                "global_best_score": float(best["score"]),
                "global_best_generation": int(best["generation"]),
                "global_best_member_index": int(best["member_index"]),
                "energy_evaluations": int(generation_compute["model_energy_evaluations"]),
                "energy_evaluation_seconds": float(generation_compute["model_evaluation_seconds"]),
                "objective_evaluations": int(generation_compute["objective_evaluations"]),
                "accepted_optimizer_iterations": int(generation_compute["accepted_optimizer_iterations"]),
                "evaluated_member_wall_seconds_sum": float(generation_compute["member_wall_seconds_sum"]),
                "generation_wall_seconds": float(generation_wall_seconds),
            })
            append_running_log(
                args,
                f"generation_done generation={generation} best={generation_best['score']:+.6f} "
                f"global_best={best['score']:+.6f} energy_evaluations={generation_compute['model_energy_evaluations']} "
                f"energy_eval_s={generation_compute['model_evaluation_seconds']:.2f} generation_wall_s={generation_wall_seconds:.2f}",
            )
    finally:
        if workers:
            stop_ga_workers(args, task_queue, workers)

    ga_wall_seconds = time.perf_counter() - ga_wall_start
    total_compute = aggregate_compute_metrics([member for generation in all_generations for member in generation])
    append_ga_log(log_path, ga_member_log_record(best, "ga_done_best"))
    append_ga_log(log_path, {
        "event": "ga_done_compute",
        "energy_evaluations": int(total_compute["model_energy_evaluations"]),
        "energy_evaluation_seconds": float(total_compute["model_evaluation_seconds"]),
        "objective_evaluations": int(total_compute["objective_evaluations"]),
        "accepted_optimizer_iterations": int(total_compute["accepted_optimizer_iterations"]),
        "evaluated_member_wall_seconds_sum": float(total_compute["member_wall_seconds_sum"]),
        "ga_wall_seconds": float(ga_wall_seconds),
    })
    write_ga_summary(problem, args, log_path, all_generations, best, ga_wall_seconds)
    problem.set_site_count(best["n_sites"])
    return best["y"]


def main():
    cli_args = parse_cli_args()
    args = get_settings(cli_args.config)
    validate_new_ga_controls(args)
    create_new_output_directory_or_fail(args)
    params_path = write_run_parameters(args)
    seed = int(time.time_ns() % 2**32) if args.seed is None else int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    problem = InpaintingProblem(args)

    print(f"[config] {args.config_path}")
    print(f"[run parameters] {params_path}")
    print(f"[seed] {seed}")
    print(f"[device] {problem.device}")
    source_label = problem.ground_truth_source_file if problem.ground_truth_source_file else "Materials Project"
    print(
        f"[system] {problem.mp_metadata.get('formula_pretty', args.material_formula)} "
        f"material_id={problem.mp_metadata.get('material_id', '')} source={source_label} geometry={problem.geometry_mode} "
        f"supercell={problem.supercell.tolist()}"
    )
    print(
        f"[atoms] full={len(problem.full_symbols)} fixed={len(problem.fixed_symbols)} missing={len(problem.missing_symbols)} "
        f"search_volume={problem.hole_volume_angstrom3:.3f} A^3 null_density={problem.null_density_per_angstrom3:.6f} "
        f"baseline_candidate_floor={problem.baseline_n_sites} subcell_grid={problem.subcell_grid.tolist()}"
    )
    print(f"[species] scaffold+variable={problem.all_species} variable={problem.variable_species} atomic_numbers={problem.all_atomic_numbers}")
    print(f"[objective] score=formation + composition + search-boundary, entropy excluded from GA ranking")
    print("[objective] H-sublattice search is periodic; search-boundary penalty disabled")
    print(
        f"[grid] base_grid={problem.feasible_grid.tolist()} feasible_base_points={len(problem.feasible_grid_base_frac)} "
        f"tiled_points={len(problem.feasible_grid_positions)} metal_clearance={problem.feasible_min_distance_angstrom:.3f} A "
        f"initial_H-H_clearance={problem.initial_h_h_min_distance_angstrom:.3f} A"
    )
    print(
        f"[occupancy lower bound] enabled={problem.occupancy_lower_bound_enable} "
        f"density={problem.material_atom_density_per_angstrom3:.6f} atoms/A^3 "
        f"estimated_atoms={problem.estimated_search_region_atoms:.3f} gamma={problem.occupancy_lower_bound_gamma:.3f} "
        f"lower_bound={problem.occupancy_lower_bound:.3f} penalty={problem.occupancy_lower_bound_penalty:.6f} "
        f"sharpness={problem.occupancy_lower_bound_softplus_sharpness:.3f}"
    )
    print("[ground truth filled reference]")
    print(f"  atoms={len(problem.full_symbols)}")
    print(f"  total_energy={problem.full_reference_energy_total:+.6f} eV")
    print(f"  formation_energy_total={problem.full_reference_formation_energy_total:+.6f} eV")
    print(f"  formation_energy_per_atom={problem.full_reference_formation_energy_per_atom:+.6f} eV/atom")
    print(
        "  files=ground_truth_relaxed_supercell.xsf, fixed_heavy_atom_scaffold.xsf, "
        "ground_truth_reference_energy.json"
    )
    for ref in problem.element_reference_metadata:
        print(f"[element reference] {ref['symbol']} {ref['material_id']} E/atom={ref['computed_energy_per_atom_eV']:+.6f} eV")
    print(
        f"[ga] enabled={args.ga_enable} population={args.ga_population_size} generations={args.ga_generations} "
        f"fractions=(elite={args.ga_elite_fraction}, random={args.ga_random_fraction}, crossover={args.ga_crossover_fraction}) "
        f"tournament={args.ga_parent_a_tournament_size}"
    )
    print(f"[ga execution] backend={args.ga_execution_backend}")
    print(f"[ga parallel] workers={args.ga_parallel_workers} gpu_ids={args.ga_parallel_gpu_ids}")
    print(
        f"[random initialization] mode={args.random_member_initialization} "
        f"applies_to=generation_0_and_all_random_immigrants"
    )
    print(
        f"[final hard FIRE] enabled={args.final_relax_enable} fmax={args.final_relax_fmax:.4f} eV/A "
        f"steps={args.final_relax_steps} maxstep={args.final_relax_maxstep_angstrom:.3f} A "
        f"fixed_scaffold=True fixed_cell=True"
    )
    print(
        f"[crossover] occ_scale={args.ga_superposition_occ_scale:.3f} "
        f"prune_distance={args.ga_superposition_prune_distance_angstrom:.3f} A "
        f"pos_sigma={args.ga_crossover_pos_sigma:.3f} occ_sigma={args.ga_crossover_occ_sigma:.3f} species_sigma={args.ga_crossover_species_sigma:.3f} "
        f"site_policy=keep_all_active_pad_only_below_{problem.baseline_n_sites}"
    )

    if args.ga_enable:
        y_cont = run_genetic_algorithm(problem, args, seed)
    else:
        problem.set_site_count(problem.baseline_n_sites)
        y0 = make_random_member_initial_state(problem, args)
        y_cont = problem.run(y0)
    y_round = problem.rounded_state(y_cont)
    problem.write_outputs(y_cont, y_round, seed)


if __name__ == "__main__":
    main()
