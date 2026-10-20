#!/usr/bin/env python3
"""W/Se/null EAT with a Gaussian depth response and no envelope (09/30/26).

FairChem retains xyz periodicity. Imaging is periodic only in xy, with all
native TIFF slices and configurable vacuum padding. Atomic confinement uses explicit
experimental z bounds independently of the measured image ROI. The guided
objective combines formation energy, voxel Huber loss, soft 1:2 composition,
and occupancy-weighted confinement. The depth-sum ellipse penalty is removed.
"""

from __future__ import annotations

import json
import csv
import os
import argparse
import copy
import hashlib
import importlib.util
import math
import sys
import time
import multiprocessing as mp
import traceback
import re
import queue as queue_module
import signal
import shlex
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import write
from fairchem.core import FAIRChemCalculator, pretrained_mlip

from minimizer import minimizer as _PBFGSMinimizer
from gaussian_image_model import atomic_box_bounds, initialize_gaussian_renderer


class minimizer(_PBFGSMinimizer):
    """PBFGS with a defensive repair for a cleared quasi-Newton history.

    Some versions of the external minimizer use ``None`` histories to signal
    that an ill-conditioned BFGS update should be reset, but do not replace
    those sentinels with empty lists before the next update. Normalizing both
    histories here performs the intended reset and prevents ``len(None)``.
    """

    def _construct_B(self, s, y, S_list, Y_list):
        if S_list is None or Y_list is None:
            S_list = []
            Y_list = []
        return super()._construct_B(s, y, S_list, Y_list)


MAX_NUM_ELEMENTS = 100
SYM_DUMMY = "H"
SCRIPT_DIR = Path(__file__).resolve().parent
MULLER_DIR = SCRIPT_DIR.parent
# Defaults used when the YAML file next to this script is absent.
# Prefer editing the YAML config or passing --config path/to/config.yaml.
OUTPUT_DIR = str(SCRIPT_DIR.parent / "09.30/runs_12_slice_gaussian_320")
VACUUM_PADDING_ANGSTROM = "8.0, 8.0, 8.0"
PBC = "true, true, true"
SPECIES = "W, Se"
MP_API_KEY = os.environ.get("MP_API_KEY", "")
MODEL_NAME = "uma-s-1p1"
TASK_NAME = "omat"

COMPOSITION_TARGET = "1, 2"
COMPOSITION_PENALTY = 20.0
EXACT_COMPOSITION_ENFORCEMENT_ENABLE = False
SE_VACANCY_BIAS_ENABLE = False
SE_VACANCY_BIAS_ALPHA_EV = 1.8353
SE_VACANCY_REFERENCE_ATOM_COUNT = 350.0
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
MAX_LINE_SEARCH = 10
STAGE_STAGNATION_PATIENCE = 5
STAGE_STAGNATION_MIN_IMPROVEMENT = 1e-3
FINAL_PBFGS_ENABLE = True
FINAL_PBFGS_MAXIT = 500
FINAL_PBFGS_TOL = 1e-3
FINAL_PBFGS_COST_PLATEAU_TOL = 1e-4
FINAL_PBFGS_COST_PLATEAU_PATIENCE = 30
FINAL_PBFGS_COST_PLATEAU_MIN_ITERATIONS = 50
INIT_OCC_TOTAL = 0.4
INIT_OCC_NOISE = 0.0
INIT_SPECIES_MODE = "dirichlet"  # "blob-symbol", "uniform", "dirichlet", or "random".
INIT_SPECIES_DIRICHLET_ALPHA = 5.0
KEEP_THRESH = 1e-3
SEED = None

DISABLE_TRAJECTORY_HISTORY = True
HISTORY_EVERY = 1

GA_ENABLE = True
GA_POPULATION_SIZE = 20
GA_GENERATIONS = 20
GA_PARENT_POOL_SIZE = 19
GA_LOG_FILE = ""
GA_SCORE = "guided"  # "guided" or "formation".
GA_ELITE_FRACTION = 0.10
GA_CROSSOVER_FRACTION = 0.90
GA_RANDOM_FRACTION = 0.0
GA_MUTATION_FRACTION = 0.0  # Opt-in for older configurations; enabled in the adjacent YAML.
GA_CROSSOVER_POS_SIGMA = 0.12
GA_CROSSOVER_OCC_SIGMA = 0.12
GA_CROSSOVER_SPECIES_SIGMA = 0.12
GA_PARENT_A_TOURNAMENT_SIZE = 4
GA_PARENT_B_TOURNAMENT_SIZE = 2
GA_DIVERSITY_SUBCELLS = "4, 4, 4"
GA_SUPERPOSITION_OCC_SCALE = 0.3
GA_SUPERPOSITION_PRUNE_DISTANCE_ANGSTROM = 0.5
GA_SUPERPOSITION_PRUNE_OCC_THRESH = 1e-3
GA_IMPORT_STRUCTURES_JSON = ""
GA_IMPORT_PARENT_COUNT = 0
GA_LIVE_PLOT_ENABLE = True
GA_LIVE_PLOT_FILE = ""
GA_LIVE_PLOT_METRIC = "score"
GA_LIVE_PLOT_ALPHA = 0.45
GA_LIVE_PLOT_JITTER = 0.16
GA_PARALLEL_WORKERS = 1
GA_PARALLEL_GPU_IDS = "0,1"
GA_EXECUTION_BACKEND = "local"  # "local" or "slurm".

SLURM_PARTITION = "hi_mem2"
SLURM_ACCOUNT = ""
SLURM_CONSTRAINT = ""
SLURM_TIME = "999:00:00"
SLURM_NODES = 1
SLURM_CPUS_PER_TASK = 40
SLURM_MEM = ""
SLURM_VENV = os.environ.get("HEAT_VENV_ACTIVATE", "")
SLURM_PYTHON = "python"
SLURM_JOB_NAME = "blobGA"
SLURM_POLL_SECONDS = 30.0
SLURM_RESOURCE_LOG_INTERVAL_SECONDS = 60.0
SLURM_STREAM_WORKER_STDOUT = True
SLURM_USE_ARRAY = False

TERMINAL_VERBOSE = False
HEARTBEAT_INTERVAL_SECONDS = 600.0
RUNNING_LOG_FILE = ""
LOCAL_RESOURCE_LOG_INTERVAL_SECONDS = 60.0
WORKER_HEALTH_CHECK_INTERVAL_SECONDS = 10.0
MEMBER_PROGRESS_CHECKPOINT_ENABLE = True
MEMBER_PROGRESS_CHECKPOINT_INTERVAL_ITERATIONS = 10
MEMBER_PROGRESS_CHECKPOINT_INTERVAL_SECONDS = 60.0
RETAIN_COMPLETED_MEMBER_CHECKPOINTS = False

EXPERIMENTAL_TIFF = str(SCRIPT_DIR.parents[2] / "08.2026/blob_EAT/08.24/objp_zstack_iter0200.tif")
CROP_XY_SIZE_PX = 320
TARGET_XY_PIXELS = 320
SOURCE_VOXEL_SIZE_A = "1.0, 0.152564, 0.152564"
TARGET_Z_SLICES = 12
HEAVY_SITE_DENSITY_PER_ANGSTROM3 = 0.20
INITIAL_CYLINDER_LATERAL_RADIUS_A = 0.75
INITIAL_CYLINDER_Z_RADIUS_A = 2.0
INITIAL_JITTER_A = 0.10

TARGET_BACKGROUND_PERCENTILE = 1.0
TARGET_NORMALIZATION_PERCENTILE = 99.5
MASK_SMOOTHING_SIGMA_PX = "0.5, 1.0, 1.0"
MASK_THRESHOLD_MODE = "percentile"
MASK_THRESHOLD = 50.0
LEVEL_SET_MARGIN_A = 0.0
LEVEL_SET_BOUNDARY_PENALTY = 0.0
LEVEL_SET_BOUNDARY_SHARPNESS = 8.0
SAMPLE_BOX_CONFINEMENT_PENALTY = 50.0
SAMPLE_BOX_CONFINEMENT_SHARPNESS = 8.0
SAMPLE_BOX_CONFINEMENT_MARGIN_A = 0.0

IMAGE_PENALTY = 100.0
IMAGE_HUBER_DELTA = 0.20
ELLIPSE_COMPOSITION_PENALTY = 0.0
ELLIPSE_DETECTION_HALO_SOURCE_PIXELS = 16
LOG_MIN_SIGMA_PIXELS = 0.6
LOG_MAX_SIGMA_PIXELS = 2.5
LOG_NUM_SIGMA = 16
LOG_THRESHOLD_REL = 0.05
LOG_MINIMUM_SEPARATION_A = 0.45
ELLIPSE_KERNEL_SUPPORT_SIGMA = 4.5
ELLIPSE_MINIMUM_RETAINED_AMPLITUDE_SE_UNITS = 0.45
ELLIPSE_PRUNING_ROUNDS = 3
ELLIPSE_POSITION_REFINEMENT_ROUNDS = 2
ELLIPSE_POSITION_STEP_PIXELS = 0.50
ELLIPSE_MINIMUM_FITTED_SEPARATION_A = 0.28
ELLIPSE_MERGE_DISTANCE_A = 0.55
ELLIPSE_RESIDUAL_ADDITION_ROUNDS = 1
ELLIPSE_RESIDUAL_THRESHOLD_REL = 0.05
ELLIPSE_RESIDUAL_MINIMUM_SEPARATION_A = 0.35
ELLIPSE_MAXIMUM_RESIDUAL_PEAKS = 20
ELLIPSE_MAXIMUM_SE_COUNT = 6
ELLIPSE_MAXIMUM_W_COUNT = 6
ELLIPSE_MAXIMUM_TOTAL_ATOMS = 8
ELLIPSE_INITIAL_I_SE = 0.12117379665687308
ELLIPSE_SCALE_FIT_MAXIMUM_ITERATIONS = 100
ELLIPSE_LOCAL_SEARCH_RADIUS_A = 1.40
ELLIPSE_FOREGROUND_RELATIVE_THRESHOLD = 0.12
ELLIPSE_MINIMUM_COVARIANCE_SIGMA_A = 0.12
ELLIPSE_AXIS_SCALE_MINIMUM = 0.65
ELLIPSE_AXIS_SCALE_MAXIMUM = 1.80
ELLIPSE_AXIS_SCALE_STEPS = 18
ELLIPSE_ANGLE_SEARCH_HALF_RANGE_DEGREES = 20.0
ELLIPSE_ANGLE_SEARCH_STEPS = 9
ELLIPSE_INTENSITY_COVERAGE_WEIGHT = 0.35
ELLIPSE_MEMBERSHIP_BOUNDARY_SOFTNESS = 0.10
ELLIPSE_MEMBERSHIP_BACKGROUND_SCORE = 0.02
ELLIPSE_LOW_CONFIDENCE_VISIBLE_FRACTION = 0.70
IMAGE_TARGET_TRANSFORM = "linear"
GAUSSIAN_EPSILON = 1e-12
GAUSSIAN_TRANSFORM_EPSILON = 1e-6
GAUSSIAN_ADDITIONAL_LATERAL_BLUR_A = 0.08
GAUSSIAN_ELEMENT_FIT_CSV = str(SCRIPT_DIR / "element_double_gaussian_fits.csv")
GAUSSIAN_RELATIVE_INTENSITY_REFERENCE_SPECIES = "W"
GAUSSIAN_CLIP_EXPERIMENTAL_TO_UNIT_INTERVAL = False
GAUSSIAN_NORMALIZE_RELATIVE_REFERENCE_PEAK_TO_ONE = True
# Retain the Gaussian image-domain signal scale while replacing all relative
# intensities and lateral shapes with isolated-atom phase-table fits.
GAUSSIAN_REFERENCE_INTEGRATED_AMPLITUDE = 15.283720016479492
GAUSSIAN_SHARED_SIGMA_Z_A = 4.0
RANDOM_IMMIGRANT_IMAGE_PREFIT_ENABLE = True
RANDOM_IMMIGRANT_IMAGE_PREFIT_INITIAL_OCCUPANCY_MODE = "uniform"
RANDOM_IMMIGRANT_IMAGE_PREFIT_INITIAL_LOGIT_NOISE = 0.20
RANDOM_IMMIGRANT_IMAGE_PREFIT_SOFT_STEPS = 500
RANDOM_IMMIGRANT_IMAGE_PREFIT_FINAL_POSITION_STEPS = 70
RANDOM_IMMIGRANT_IMAGE_PREFIT_POSITION_LEARNING_RATE = 0.025
RANDOM_IMMIGRANT_IMAGE_PREFIT_OCCUPANCY_LEARNING_RATE = 0.07
RANDOM_IMMIGRANT_IMAGE_PREFIT_ENTROPY_FINAL_WEIGHT = 1.0
RANDOM_IMMIGRANT_IMAGE_PREFIT_REPULSION_WEIGHT = 2.0
RANDOM_IMMIGRANT_IMAGE_PREFIT_MINIMUM_DISTANCE_A = 1.0
RANDOM_IMMIGRANT_IMAGE_PREFIT_OXYGEN_SITE_DENSITY_PER_ANGSTROM3 = 0.30
RANDOM_IMMIGRANT_IMAGE_PREFIT_COMPOSITION_PENALTY_ENABLE = True
RANDOM_IMMIGRANT_IMAGE_PREFIT_MAXIMUM_FORMULA_UNITS = 12
RANDOM_IMMIGRANT_IMAGE_PREFIT_VOLUME_PRIOR_ENABLE = True
RANDOM_IMMIGRANT_IMAGE_PREFIT_MAGPIE_VOLUME_MODULE = str(
    SCRIPT_DIR / "magpie_atomic_volumes.py"
)
RANDOM_IMMIGRANT_IMAGE_PREFIT_VOLUME_PRIOR_RELATIVE_UNCERTAINTY = 0.20
RANDOM_IMMIGRANT_IMAGE_PREFIT_VOLUME_PRIOR_PENALTY_WEIGHT = 0.02
RANDOM_IMMIGRANT_IMAGE_PREFIT_COMPACT_NULL_SITES_BEFORE_FAIRCHEM = True
INITIAL_DIAGNOSTICS_ENABLE = True
INITIAL_DIAGNOSTICS_EXIT_AFTER = False
INITIAL_DIAGNOSTICS_MAX_ISOSURFACE_POINTS = 60000
MEMBER_SLICE_COMPARISONS_ENABLE = True

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
    },
    "system": {
        "vacuum_padding_angstrom": VACUUM_PADDING_ANGSTROM,
        "pbc": PBC,
        "species": SPECIES,
    },
    "experimental": {
        "tiff_file": EXPERIMENTAL_TIFF,
        "crop_xy_size_source_pixels": CROP_XY_SIZE_PX,
        "target_xy_pixels": TARGET_XY_PIXELS,
        "source_voxel_size_angstrom": SOURCE_VOXEL_SIZE_A,
        "target_z_slices": TARGET_Z_SLICES,
        "target_background_percentile": TARGET_BACKGROUND_PERCENTILE,
        "target_normalization_percentile": TARGET_NORMALIZATION_PERCENTILE,
    },
    "feasible_region": {
        "mask_smoothing_sigma_pixels": MASK_SMOOTHING_SIGMA_PX,
        "threshold_mode": MASK_THRESHOLD_MODE,
        "threshold": MASK_THRESHOLD,
        "level_set_margin_angstrom": LEVEL_SET_MARGIN_A,
        "boundary_penalty": LEVEL_SET_BOUNDARY_PENALTY,
        "boundary_sharpness": LEVEL_SET_BOUNDARY_SHARPNESS,
    },
    "sample_box": {
        "confinement_penalty": SAMPLE_BOX_CONFINEMENT_PENALTY,
        "confinement_sharpness": SAMPLE_BOX_CONFINEMENT_SHARPNESS,
        "confinement_margin_angstrom": SAMPLE_BOX_CONFINEMENT_MARGIN_A,
        "z_min_angstrom": 0.0,
        "z_max_angstrom": 12.0,
    },
    "imaging": {
        "shared_sigma_z_angstrom": 4.0,
        "penalty": IMAGE_PENALTY,
        "huber_delta": IMAGE_HUBER_DELTA,
        "ellipse_composition_penalty": ELLIPSE_COMPOSITION_PENALTY,
        "target_transform": IMAGE_TARGET_TRANSFORM,
        "epsilon": GAUSSIAN_EPSILON,
        "transform_epsilon": GAUSSIAN_TRANSFORM_EPSILON,
        "additional_lateral_blur_angstrom": GAUSSIAN_ADDITIONAL_LATERAL_BLUR_A,
        "element_fit_csv": GAUSSIAN_ELEMENT_FIT_CSV,
        "relative_intensity_reference_species": GAUSSIAN_RELATIVE_INTENSITY_REFERENCE_SPECIES,
        "reference_integrated_amplitude": GAUSSIAN_REFERENCE_INTEGRATED_AMPLITUDE,
        "clip_experimental_to_unit_interval": GAUSSIAN_CLIP_EXPERIMENTAL_TO_UNIT_INTERVAL,
        "normalize_relative_intensity_reference_peak_to_one": GAUSSIAN_NORMALIZE_RELATIVE_REFERENCE_PEAK_TO_ONE,
    },
    "experimental_peak_ellipses": {
        "detection_halo_source_pixels": ELLIPSE_DETECTION_HALO_SOURCE_PIXELS,
        "min_sigma_pixels": LOG_MIN_SIGMA_PIXELS,
        "max_sigma_pixels": LOG_MAX_SIGMA_PIXELS,
        "num_sigma": LOG_NUM_SIGMA,
        "threshold_rel": LOG_THRESHOLD_REL,
        "minimum_separation_angstrom": LOG_MINIMUM_SEPARATION_A,
        "kernel_support_sigma": ELLIPSE_KERNEL_SUPPORT_SIGMA,
        "minimum_retained_amplitude_Se_units": ELLIPSE_MINIMUM_RETAINED_AMPLITUDE_SE_UNITS,
        "pruning_rounds": ELLIPSE_PRUNING_ROUNDS,
        "position_refinement_rounds": ELLIPSE_POSITION_REFINEMENT_ROUNDS,
        "position_step_pixels": ELLIPSE_POSITION_STEP_PIXELS,
        "minimum_fitted_separation_angstrom": ELLIPSE_MINIMUM_FITTED_SEPARATION_A,
        "merge_distance_angstrom": ELLIPSE_MERGE_DISTANCE_A,
        "residual_addition_rounds": ELLIPSE_RESIDUAL_ADDITION_ROUNDS,
        "residual_peak_threshold_rel": ELLIPSE_RESIDUAL_THRESHOLD_REL,
        "residual_minimum_separation_angstrom": ELLIPSE_RESIDUAL_MINIMUM_SEPARATION_A,
        "maximum_residual_peaks_per_round": ELLIPSE_MAXIMUM_RESIDUAL_PEAKS,
        "maximum_Se_count": ELLIPSE_MAXIMUM_SE_COUNT,
        "maximum_W_count": ELLIPSE_MAXIMUM_W_COUNT,
        "maximum_total_atoms": ELLIPSE_MAXIMUM_TOTAL_ATOMS,
        "initial_I_Se_for_scale_fit": ELLIPSE_INITIAL_I_SE,
        "scale_fit_maximum_iterations": ELLIPSE_SCALE_FIT_MAXIMUM_ITERATIONS,
        "local_search_radius_angstrom": ELLIPSE_LOCAL_SEARCH_RADIUS_A,
        "foreground_relative_threshold": ELLIPSE_FOREGROUND_RELATIVE_THRESHOLD,
        "minimum_covariance_sigma_angstrom": ELLIPSE_MINIMUM_COVARIANCE_SIGMA_A,
        "axis_scale_minimum": ELLIPSE_AXIS_SCALE_MINIMUM,
        "axis_scale_maximum": ELLIPSE_AXIS_SCALE_MAXIMUM,
        "axis_scale_steps": ELLIPSE_AXIS_SCALE_STEPS,
        "angle_search_half_range_degrees": ELLIPSE_ANGLE_SEARCH_HALF_RANGE_DEGREES,
        "angle_search_steps": ELLIPSE_ANGLE_SEARCH_STEPS,
        "intensity_coverage_weight": ELLIPSE_INTENSITY_COVERAGE_WEIGHT,
        "membership_boundary_softness": ELLIPSE_MEMBERSHIP_BOUNDARY_SOFTNESS,
        "membership_background_score": ELLIPSE_MEMBERSHIP_BACKGROUND_SCORE,
        "low_confidence_visible_fraction": ELLIPSE_LOW_CONFIDENCE_VISIBLE_FRACTION,
    },
    "random_immigrant_image_prefit": {
        "enable": RANDOM_IMMIGRANT_IMAGE_PREFIT_ENABLE,
        "initial_occupancy_mode": RANDOM_IMMIGRANT_IMAGE_PREFIT_INITIAL_OCCUPANCY_MODE,
        "initial_logit_noise": RANDOM_IMMIGRANT_IMAGE_PREFIT_INITIAL_LOGIT_NOISE,
        "soft_steps": RANDOM_IMMIGRANT_IMAGE_PREFIT_SOFT_STEPS,
        "final_position_steps": RANDOM_IMMIGRANT_IMAGE_PREFIT_FINAL_POSITION_STEPS,
        "position_learning_rate": RANDOM_IMMIGRANT_IMAGE_PREFIT_POSITION_LEARNING_RATE,
        "occupancy_learning_rate": RANDOM_IMMIGRANT_IMAGE_PREFIT_OCCUPANCY_LEARNING_RATE,
        "entropy_final_weight": RANDOM_IMMIGRANT_IMAGE_PREFIT_ENTROPY_FINAL_WEIGHT,
        "repulsion_weight": RANDOM_IMMIGRANT_IMAGE_PREFIT_REPULSION_WEIGHT,
        "minimum_distance_angstrom": RANDOM_IMMIGRANT_IMAGE_PREFIT_MINIMUM_DISTANCE_A,
        "composition_penalty_enable": RANDOM_IMMIGRANT_IMAGE_PREFIT_COMPOSITION_PENALTY_ENABLE,
        "compact_null_sites_before_fairchem": RANDOM_IMMIGRANT_IMAGE_PREFIT_COMPACT_NULL_SITES_BEFORE_FAIRCHEM,
    },
    "diagnostics": {
        "initial_diagnostics_enable": INITIAL_DIAGNOSTICS_ENABLE,
        "exit_after_initial_diagnostics": INITIAL_DIAGNOSTICS_EXIT_AFTER,
        "max_isosurface_points": INITIAL_DIAGNOSTICS_MAX_ISOSURFACE_POINTS,
        "member_slice_comparisons_enable": MEMBER_SLICE_COMPARISONS_ENABLE,
    },
    "effective_atoms": {
        "candidate_site_density_per_angstrom3": HEAVY_SITE_DENSITY_PER_ANGSTROM3,
        "initial_cylinder_lateral_radius_angstrom": INITIAL_CYLINDER_LATERAL_RADIUS_A,
        "initial_cylinder_z_radius_angstrom": INITIAL_CYLINDER_Z_RADIUS_A,
        "initial_jitter_angstrom": INITIAL_JITTER_A,
    },
    "composition": {
        "composition_target": COMPOSITION_TARGET,
        "composition_penalty": COMPOSITION_PENALTY,
        "exact_composition_enforcement_enable": EXACT_COMPOSITION_ENFORCEMENT_ENABLE,
    },
    "se_vacancy_bias": {
        "enable": SE_VACANCY_BIAS_ENABLE,
        "alpha_eV_per_vacancy": SE_VACANCY_BIAS_ALPHA_EV,
        "reference_atom_count": SE_VACANCY_REFERENCE_ATOM_COUNT,
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
        "stage_stagnation_patience": STAGE_STAGNATION_PATIENCE,
        "stage_stagnation_min_improvement": STAGE_STAGNATION_MIN_IMPROVEMENT,
        "final_pbfgs_enable": FINAL_PBFGS_ENABLE,
        "final_pbfgs_maxit": FINAL_PBFGS_MAXIT,
        "final_pbfgs_tol": FINAL_PBFGS_TOL,
        "final_pbfgs_cost_plateau_tol": FINAL_PBFGS_COST_PLATEAU_TOL,
        "final_pbfgs_cost_plateau_patience": FINAL_PBFGS_COST_PLATEAU_PATIENCE,
        "final_pbfgs_cost_plateau_min_iterations": FINAL_PBFGS_COST_PLATEAU_MIN_ITERATIONS,
    },
    "initialization": {
        "init_occ_total": INIT_OCC_TOTAL,
        "init_occ_noise": INIT_OCC_NOISE,
        "init_species_mode": INIT_SPECIES_MODE,
        "init_species_dirichlet_alpha": INIT_SPECIES_DIRICHLET_ALPHA,
        "keep_thresh": KEEP_THRESH,
        "seed": SEED,
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
        "mutation_fraction": GA_MUTATION_FRACTION,
        "parent_pool_size": GA_PARENT_POOL_SIZE,
        "import_parent_count": GA_IMPORT_PARENT_COUNT,
        "crossover_pos_sigma": GA_CROSSOVER_POS_SIGMA,
        "crossover_occ_sigma": GA_CROSSOVER_OCC_SIGMA,
        "crossover_species_sigma": GA_CROSSOVER_SPECIES_SIGMA,
        "parent_a_tournament_size": GA_PARENT_A_TOURNAMENT_SIZE,
        "parent_b_tournament_size": GA_PARENT_B_TOURNAMENT_SIZE,
        "diversity_subcells": GA_DIVERSITY_SUBCELLS,
        "superposition_occ_scale": GA_SUPERPOSITION_OCC_SCALE,
        "superposition_prune_distance_angstrom": GA_SUPERPOSITION_PRUNE_DISTANCE_ANGSTROM,
        "superposition_prune_occ_thresh": GA_SUPERPOSITION_PRUNE_OCC_THRESH,
        "live_plot_enable": GA_LIVE_PLOT_ENABLE,
        "live_plot_file": GA_LIVE_PLOT_FILE,
        "live_plot_metric": GA_LIVE_PLOT_METRIC,
        "live_plot_alpha": GA_LIVE_PLOT_ALPHA,
        "live_plot_jitter": GA_LIVE_PLOT_JITTER,
        "parallel_workers": GA_PARALLEL_WORKERS,
        "parallel_gpu_ids": GA_PARALLEL_GPU_IDS,
        "execution_backend": GA_EXECUTION_BACKEND,
        "log_file": GA_LOG_FILE,
        "score": GA_SCORE,
    },
    "slurm": {
        "partition": SLURM_PARTITION,
        "account": SLURM_ACCOUNT,
        "constraint": SLURM_CONSTRAINT,
        "time": SLURM_TIME,
        "nodes": SLURM_NODES,
        "cpus_per_task": SLURM_CPUS_PER_TASK,
        "mem": SLURM_MEM,
        "venv": SLURM_VENV,
        "python": SLURM_PYTHON,
        "job_name": SLURM_JOB_NAME,
        "poll_seconds": SLURM_POLL_SECONDS,
        "resource_log_interval_seconds": SLURM_RESOURCE_LOG_INTERVAL_SECONDS,
        "stream_worker_stdout": SLURM_STREAM_WORKER_STDOUT,
        "use_array": SLURM_USE_ARRAY,
    },
    "runtime": {
        "terminal_verbose": TERMINAL_VERBOSE,
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "running_log_file": RUNNING_LOG_FILE,
        "local_resource_log_interval_seconds": LOCAL_RESOURCE_LOG_INTERVAL_SECONDS,
        "worker_health_check_interval_seconds": WORKER_HEALTH_CHECK_INTERVAL_SECONDS,
    },
    "checkpointing": {
        "member_progress_enable": MEMBER_PROGRESS_CHECKPOINT_ENABLE,
        "member_progress_interval_iterations": MEMBER_PROGRESS_CHECKPOINT_INTERVAL_ITERATIONS,
        "member_progress_interval_seconds": MEMBER_PROGRESS_CHECKPOINT_INTERVAL_SECONDS,
        "retain_completed_member_checkpoints": RETAIN_COMPLETED_MEMBER_CHECKPOINTS,
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
    },
    "system": {
        "vacuum_padding_angstrom": "vacuum_padding_angstrom",
        "pbc": "pbc",
        "species": "species",
    },
    "experimental": {
        "tiff_file": "experimental_tiff",
        "crop_xy_size_source_pixels": "crop_xy_size_source_pixels",
        "target_xy_pixels": "target_xy_pixels",
        "source_voxel_size_angstrom": "source_voxel_size_angstrom",
        "target_z_slices": "target_z_slices",
        "target_background_percentile": "target_background_percentile",
        "target_normalization_percentile": "target_normalization_percentile",
    },
    "feasible_region": {
        "mask_smoothing_sigma_pixels": "mask_smoothing_sigma_pixels",
        "threshold_mode": "mask_threshold_mode",
        "threshold": "mask_threshold",
        "level_set_margin_angstrom": "level_set_margin_angstrom",
        "boundary_penalty": "level_set_boundary_penalty",
        "boundary_sharpness": "level_set_boundary_sharpness",
    },
    "sample_box": {
        "confinement_penalty": "sample_box_confinement_penalty",
        "confinement_sharpness": "sample_box_confinement_sharpness",
        "confinement_margin_angstrom": "sample_box_confinement_margin_angstrom",
        "z_min_angstrom": "sample_box_z_min_angstrom",
        "z_max_angstrom": "sample_box_z_max_angstrom",
    },
    "imaging": {
        "shared_sigma_z_angstrom": "gaussian_shared_sigma_z_angstrom",
        "penalty": "image_penalty",
        "huber_delta": "image_huber_delta",
        "ellipse_composition_penalty": "ellipse_composition_penalty",
        "target_transform": "image_target_transform",
        "epsilon": "gaussian_epsilon",
        "transform_epsilon": "gaussian_transform_epsilon",
        "additional_lateral_blur_angstrom": "gaussian_additional_lateral_blur_angstrom",
        "element_fit_csv": "gaussian_element_fit_csv",
        "relative_intensity_reference_species": "gaussian_relative_intensity_reference_species",
        "reference_integrated_amplitude": "gaussian_reference_integrated_amplitude",
        "clip_experimental_to_unit_interval": "gaussian_clip_experimental_to_unit_interval",
        "normalize_relative_intensity_reference_peak_to_one": "gaussian_normalize_relative_intensity_reference_peak_to_one",
    },
    "experimental_peak_ellipses": {
        "detection_halo_source_pixels": "ellipse_detection_halo_source_pixels",
        "min_sigma_pixels": "log_min_sigma_pixels",
        "max_sigma_pixels": "log_max_sigma_pixels",
        "num_sigma": "log_num_sigma",
        "threshold_rel": "log_threshold_rel",
        "minimum_separation_angstrom": "log_minimum_separation_angstrom",
        "kernel_support_sigma": "ellipse_kernel_support_sigma",
        "minimum_retained_amplitude_Se_units": "ellipse_minimum_retained_amplitude_Se_units",
        "pruning_rounds": "ellipse_pruning_rounds",
        "position_refinement_rounds": "ellipse_position_refinement_rounds",
        "position_step_pixels": "ellipse_position_step_pixels",
        "minimum_fitted_separation_angstrom": "ellipse_minimum_fitted_separation_angstrom",
        "merge_distance_angstrom": "ellipse_merge_distance_angstrom",
        "residual_addition_rounds": "ellipse_residual_addition_rounds",
        "residual_peak_threshold_rel": "ellipse_residual_peak_threshold_rel",
        "residual_minimum_separation_angstrom": "ellipse_residual_minimum_separation_angstrom",
        "maximum_residual_peaks_per_round": "ellipse_maximum_residual_peaks_per_round",
        "maximum_Se_count": "ellipse_maximum_Se_count",
        "maximum_W_count": "ellipse_maximum_W_count",
        "maximum_total_atoms": "ellipse_maximum_total_atoms",
        "initial_I_Se_for_scale_fit": "ellipse_initial_I_Se_for_scale_fit",
        "scale_fit_maximum_iterations": "ellipse_scale_fit_maximum_iterations",
        "local_search_radius_angstrom": "ellipse_local_search_radius_angstrom",
        "foreground_relative_threshold": "ellipse_foreground_relative_threshold",
        "minimum_covariance_sigma_angstrom": "ellipse_minimum_covariance_sigma_angstrom",
        "axis_scale_minimum": "ellipse_axis_scale_minimum",
        "axis_scale_maximum": "ellipse_axis_scale_maximum",
        "axis_scale_steps": "ellipse_axis_scale_steps",
        "angle_search_half_range_degrees": "ellipse_angle_search_half_range_degrees",
        "angle_search_steps": "ellipse_angle_search_steps",
        "intensity_coverage_weight": "ellipse_intensity_coverage_weight",
        "membership_boundary_softness": "ellipse_membership_boundary_softness",
        "membership_background_score": "ellipse_membership_background_score",
        "low_confidence_visible_fraction": "ellipse_low_confidence_visible_fraction",
    },
    "random_immigrant_image_prefit": {
        "enable": "random_immigrant_image_prefit_enable",
        "initial_occupancy_mode": "random_immigrant_image_prefit_initial_occupancy_mode",
        "initial_logit_noise": "random_immigrant_image_prefit_initial_logit_noise",
        "soft_steps": "random_immigrant_image_prefit_soft_steps",
        "final_position_steps": "random_immigrant_image_prefit_final_position_steps",
        "position_learning_rate": "random_immigrant_image_prefit_position_learning_rate",
        "occupancy_learning_rate": "random_immigrant_image_prefit_occupancy_learning_rate",
        "entropy_final_weight": "random_immigrant_image_prefit_entropy_final_weight",
        "repulsion_weight": "random_immigrant_image_prefit_repulsion_weight",
        "minimum_distance_angstrom": "random_immigrant_image_prefit_minimum_distance_angstrom",
        "composition_penalty_enable": "random_immigrant_image_prefit_composition_penalty_enable",
        "compact_null_sites_before_fairchem": "random_immigrant_image_prefit_compact_null_sites_before_fairchem",
    },
    "diagnostics": {
        "initial_diagnostics_enable": "initial_diagnostics_enable",
        "exit_after_initial_diagnostics": "exit_after_initial_diagnostics",
        "max_isosurface_points": "initial_diagnostics_max_isosurface_points",
        "member_slice_comparisons_enable": "member_slice_comparisons_enable",
    },
    "effective_atoms": {
        "candidate_site_density_per_angstrom3": "heavy_site_density_per_angstrom3",
        "initial_cylinder_lateral_radius_angstrom": "initial_cylinder_lateral_radius_angstrom",
        "initial_cylinder_z_radius_angstrom": "initial_cylinder_z_radius_angstrom",
        "initial_jitter_angstrom": "initial_jitter_angstrom",
    },
    "composition": {
        "composition_target": "composition_target",
        "composition_penalty": "composition_penalty",
        "exact_composition_enforcement_enable": "exact_composition_enforcement_enable",
    },
    "se_vacancy_bias": {
        "enable": "se_vacancy_bias_enable",
        "alpha_eV_per_vacancy": "se_vacancy_bias_alpha_eV",
        "reference_atom_count": "se_vacancy_reference_atom_count",
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
        "stage_stagnation_patience": "stage_stagnation_patience",
        "stage_stagnation_min_improvement": "stage_stagnation_min_improvement",
        "final_pbfgs_enable": "final_pbfgs_enable",
        "final_pbfgs_maxit": "final_pbfgs_maxit",
        "final_pbfgs_tol": "final_pbfgs_tol",
        "final_pbfgs_cost_plateau_tol": "final_pbfgs_cost_plateau_tol",
        "final_pbfgs_cost_plateau_patience": "final_pbfgs_cost_plateau_patience",
        "final_pbfgs_cost_plateau_min_iterations": "final_pbfgs_cost_plateau_min_iterations",
    },
    "initialization": {
        "init_occ_total": "init_occ_total",
        "init_occ_noise": "init_occ_noise",
        "init_species_mode": "init_species_mode",
        "init_species_dirichlet_alpha": "init_species_dirichlet_alpha",
        "keep_thresh": "keep_thresh",
        "seed": "seed",
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
        "mutation_fraction": "ga_mutation_fraction",
        "random_fraction": "ga_random_fraction",
        "parent_pool_size": "ga_parent_pool_size",
        "import_parent_count": "ga_import_parent_count",
        "crossover_pos_sigma": "ga_crossover_pos_sigma",
        "crossover_occ_sigma": "ga_crossover_occ_sigma",
        "crossover_species_sigma": "ga_crossover_species_sigma",
        "parent_a_tournament_size": "ga_parent_a_tournament_size",
        "parent_b_tournament_size": "ga_parent_b_tournament_size",
        "diversity_subcells": "ga_diversity_subcells",
        "superposition_occ_scale": "ga_superposition_occ_scale",
        "superposition_prune_distance_angstrom": "ga_superposition_prune_distance_angstrom",
        "superposition_prune_occ_thresh": "ga_superposition_prune_occ_thresh",
        "live_plot_enable": "ga_live_plot_enable",
        "live_plot_file": "ga_live_plot_file",
        "live_plot_metric": "ga_live_plot_metric",
        "live_plot_alpha": "ga_live_plot_alpha",
        "live_plot_jitter": "ga_live_plot_jitter",
        "parallel_workers": "ga_parallel_workers",
        "parallel_gpu_ids": "ga_parallel_gpu_ids",
        "execution_backend": "ga_execution_backend",
        "log_file": "ga_log_file",
        "score": "ga_score",
    },
    "slurm": {
        "partition": "slurm_partition",
        "account": "slurm_account",
        "constraint": "slurm_constraint",
        "time": "slurm_time",
        "nodes": "slurm_nodes",
        "cpus_per_task": "slurm_cpus_per_task",
        "mem": "slurm_mem",
        "venv": "slurm_venv",
        "python": "slurm_python",
        "job_name": "slurm_job_name",
        "poll_seconds": "slurm_poll_seconds",
        "resource_log_interval_seconds": "slurm_resource_log_interval_seconds",
        "stream_worker_stdout": "slurm_stream_worker_stdout",
        "use_array": "slurm_use_array",
    },
    "runtime": {
        "terminal_verbose": "terminal_verbose",
        "heartbeat_interval_seconds": "heartbeat_interval_seconds",
        "running_log_file": "running_log_file",
        "local_resource_log_interval_seconds": "local_resource_log_interval_seconds",
        "worker_health_check_interval_seconds": "worker_health_check_interval_seconds",
    },
    "checkpointing": {
        "member_progress_enable": "member_progress_checkpoint_enable",
        "member_progress_interval_iterations": "member_progress_checkpoint_interval_iterations",
        "member_progress_interval_seconds": "member_progress_checkpoint_interval_seconds",
        "retain_completed_member_checkpoints": "retain_completed_member_checkpoints",
    },
}


def strip_yaml_comment(line):
    in_quote = None
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in ("'", '"'):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
            continue
        if ch == "#" and in_quote is None:
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


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text)).strip("_")


def timestamp_now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def running_log_path_from_args(args):
    path_text = str(getattr(args, "running_log_file", "") or "").strip()
    if path_text:
        return Path(path_text).expanduser().resolve()
    return Path(args.output_dir).expanduser().resolve() / "running_log.txt"


def append_running_log(args, message):
    path = running_log_path_from_args(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp_now()} | {message}\n")


def terminal_print(args, *values, **kwargs):
    if bool(getattr(args, "terminal_verbose", False)):
        print(*values, **kwargs)


def read_process_rss_gb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return None
    return None


def read_node_memory_gb():
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                pieces = raw_value.strip().split()
                if pieces:
                    values[key] = float(pieces[0]) / 1024.0 / 1024.0
    except OSError:
        return None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    return {
        "total_gb": float(total),
        "available_gb": float(available),
        "used_gb": float(total - available),
    }


def read_cuda_memory_gb():
    """Return this process's PyTorch CUDA allocator state when available."""
    try:
        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        scale = float(1024**3)
        return {
            "allocated_gb": float(torch.cuda.memory_allocated(device)) / scale,
            "reserved_gb": float(torch.cuda.memory_reserved(device)) / scale,
            "max_allocated_gb": float(torch.cuda.max_memory_allocated(device)) / scale,
            "max_reserved_gb": float(torch.cuda.max_memory_reserved(device)) / scale,
        }
    except Exception:
        return None


def start_resource_monitor(path, interval_seconds, label):
    interval = float(interval_seconds or 0.0)
    if not path or interval <= 0.0:
        return None, None
    stop_event = threading.Event()
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    def monitor():
        def fmt(value):
            return "" if value is None else f"{float(value):.6f}"

        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8") as handle:
            if needs_header:
                handle.write(
                    "timestamp\tlabel\tprocess_rss_gb\tnode_used_gb\t"
                    "node_available_gb\tnode_total_gb\tcuda_allocated_gb\t"
                    "cuda_reserved_gb\tcuda_max_allocated_gb\t"
                    "cuda_max_reserved_gb\n"
                )
            while not stop_event.is_set():
                rss = read_process_rss_gb()
                node = read_node_memory_gb() or {}
                cuda = read_cuda_memory_gb() or {}
                handle.write(
                    f"{timestamp_now()}\t{label}\t"
                    f"{fmt(rss)}\t"
                    f"{fmt(node.get('used_gb'))}\t"
                    f"{fmt(node.get('available_gb'))}\t"
                    f"{fmt(node.get('total_gb'))}\t"
                    f"{fmt(cuda.get('allocated_gb'))}\t"
                    f"{fmt(cuda.get('reserved_gb'))}\t"
                    f"{fmt(cuda.get('max_allocated_gb'))}\t"
                    f"{fmt(cuda.get('max_reserved_gb'))}\n"
                )
                handle.flush()
                stop_event.wait(interval)

    thread = threading.Thread(target=monitor, name="resource-monitor", daemon=True)
    thread.start()
    return stop_event, thread


def stop_resource_monitor(stop_event, thread):
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=5.0)


def configure_cpu_threads(args):
    threads = int(getattr(args, "slurm_cpus_per_task", 0) or 0)
    env_threads = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_threads:
        try:
            threads = int(env_threads)
        except ValueError:
            pass
    if threads <= 0:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def disable_optional_torch_native_bmm_override():
    """Use standard ATen bmm when the optional Triton override cannot build."""
    global _BMM_OVERRIDE_ATTEMPTED
    if _BMM_OVERRIDE_ATTEMPTED:
        return
    _BMM_OVERRIDE_ATTEMPTED = True
    try:
        from torch._native.registry import deregister_op_overrides

        deregister_op_overrides(disable_op_symbols="bmm")
        print(
            "[fairchem] disabled optional torch._native Triton bmm override; "
            "using standard ATen bmm",
            flush=True,
        )
    except (ImportError, AttributeError):
        pass


def read_simple_yaml(path):
    data = {}
    current_section = None
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = strip_yaml_comment(raw_line).rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            if "\t" in raw_line[:indent]:
                raise ValueError(f"{path}:{line_number}: tabs are not supported in this config YAML.")
            text = line.strip()
            if ":" not in text:
                raise ValueError(f"{path}:{line_number}: expected 'key: value'.")
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if indent == 0:
                if value:
                    data[key] = parse_yaml_scalar(value)
                    current_section = None
                else:
                    data[key] = {}
                    current_section = key
            else:
                if current_section is None or not isinstance(data.get(current_section), dict):
                    raise ValueError(f"{path}:{line_number}: nested keys must belong to a section.")
                data[current_section][key] = parse_yaml_scalar(value)
    return data


def deep_update_config(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update_config(base[key], value)
        else:
            base[key] = value
    return base


def flatten_config(config):
    flat = {}
    for section, keys in CONFIG_KEY_MAP.items():
        values = config.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Config section {section!r} must be a mapping.")
        for source_key, target_key in keys.items():
            if source_key in values:
                flat[target_key] = values[source_key]
    if not flat.get("mp_api_key"):
        flat["mp_api_key"] = os.environ.get("MP_API_KEY", MP_API_KEY)
    return flat


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run multi-species blob EAT genetic optimization.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config path. Defaults to a .yaml file next to this script.",
    )
    parser.add_argument("--slurm-worker-task", default="", help=argparse.SUPPRESS)
    parser.add_argument("--slurm-worker-result", default="", help=argparse.SUPPRESS)
    parser.add_argument("--slurm-worker-resource-log", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def get_settings(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser()
    if path.exists():
        override = read_simple_yaml(path)
        deep_update_config(config, override)
    elif path.resolve() != DEFAULT_CONFIG_PATH.expanduser().resolve():
        raise FileNotFoundError(f"Config file not found: {path}")
    if "depth_calibration_file" in config.get("imaging", {}):
        raise ValueError("imaging.depth_calibration_file is no longer used; configure shared_sigma_z_angstrom instead.")
    if "raw_gain" in config.get("imaging", {}):
        raise ValueError("imaging.raw_gain is no longer used; the isolated W kernel peak is normalized to one.")
    flat = flatten_config(config)
    flat["mp_api_key"] = flat.get("mp_api_key") or os.environ.get("MP_API_KEY", "")
    output_path = Path(str(flat["output_dir"])).expanduser()
    if not output_path.is_absolute():
        # Relative outputs belong to the directory from which the user launches
        # the driver. Slurm subdirectories are derived from this output path.
        output_path = Path.cwd() / output_path
    flat["output_dir"] = str(output_path.resolve())
    import_path = str(flat.get("ga_import_structures_json", "") or "").strip()
    if import_path:
        import_path_obj = Path(import_path).expanduser()
        if not import_path_obj.is_absolute():
            import_path_obj = path.resolve().parent / import_path_obj
        flat["ga_import_structures_json"] = str(import_path_obj.resolve())
    # Scientific calibration assets are part of the script bundle, not run
    # inputs. Always load them from the resolved Python-script directory even
    # when the script is launched elsewhere or a YAML is copied elsewhere.
    fit_csv_path = (SCRIPT_DIR / "element_double_gaussian_fits.csv").resolve()
    magpie_volume_path = (SCRIPT_DIR / "magpie_atomic_volumes.py").resolve()
    if not fit_csv_path.is_file():
        raise FileNotFoundError(
            f"Bundled double-Gaussian fit CSV not found: {fit_csv_path}"
        )
    if not magpie_volume_path.is_file():
        raise FileNotFoundError(
            f"Bundled Magpie atomic-volume module not found: {magpie_volume_path}"
        )
    flat["gaussian_element_fit_csv"] = str(fit_csv_path)
    for key in ("gaussian_shared_sigma_z_angstrom",):
        if not np.isfinite(float(flat[key])) or float(flat[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if float(flat["ellipse_composition_penalty"]) != 0.0:
        raise ValueError("The depth-sum ellipse penalty has been removed; set it to zero or omit it.")
    if not flat["gaussian_normalize_relative_intensity_reference_peak_to_one"]:
        raise ValueError("Set imaging.normalize_relative_intensity_reference_peak_to_one to true.")
    if flat["gaussian_relative_intensity_reference_species"] != "W":
        raise ValueError("The image model uses W as intensity reference.")
    if flat["gaussian_clip_experimental_to_unit_interval"] or flat["image_target_transform"] != "linear":
        raise ValueError("The Gaussian image model requires unclipped linear target intensities.")
    if int(flat["target_z_slices"]) <= 1 or int(flat["target_xy_pixels"]) != int(flat["crop_xy_size_source_pixels"]):
        raise ValueError("Use at least two slices and native XY resolution with this workflow.")
    flat["random_immigrant_image_prefit_magpie_volume_module"] = str(
        magpie_volume_path
    )
    return SimpleNamespace(config_path=str(path.resolve()), **flat)


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


def split_species_list_value(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [piece.strip() for piece in re.split(r"[|,/;]+", value) if piece.strip()]
    if isinstance(value, (list, tuple)):
        return [str(piece).strip() for piece in value if str(piece).strip()]
    return [str(value).strip()] if str(value).strip() else []


def allowed_elements_from_blob(blob):
    for key in ("source_circle_allowed_elements", "allowed_elements", "allowed_species"):
        if key in blob:
            allowed = split_species_list_value(blob.get(key))
            return allowed if allowed else None
    return None


def allowed_elements_from_blobs(blobs):
    return [allowed_elements_from_blob(blob) for blob in blobs]


def unique_species_from_allowed_elements(blob_allowed_elements):
    species = []
    for allowed in blob_allowed_elements:
        if not allowed:
            continue
        for symbol in allowed:
            if symbol not in species:
                species.append(symbol)
    return species


def parse_species_arg(text, blob_symbols, blob_allowed_elements=None):
    if text.strip():
        species = [s.strip() for s in text.split(",") if s.strip()]
    elif blob_allowed_elements is not None:
        species = unique_species_from_allowed_elements(blob_allowed_elements)
        if not species:
            species = list(dict.fromkeys(blob_symbols))
    else:
        species = list(dict.fromkeys(blob_symbols))
    if len(species) < 1:
        raise ValueError(
            "This optimizer expects at least one real species. Provide per-blob "
            "source_circle_allowed_elements in the blob JSON or set system.species."
        )
    if len(set(species)) != len(species):
        raise ValueError(f"Species list contains duplicates: {species}")
    for symbol in species:
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element symbol {symbol!r}")
    return species


def parse_composition_target(text, species, blob_symbols):
    if text.strip():
        values = np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)
        if values.shape != (len(species),):
            raise ValueError(f"--composition-target must have {len(species)} entries.")
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("--composition-target must have positive sum.")
        return values / total
    counts = np.asarray([sum(1 for symbol in blob_symbols if symbol == species_name) for species_name in species], dtype=float)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("Could not infer composition target from blob symbols.")
    return counts / total


def composition_symbols_from_blob_data(blob_data, blobs):
    original_atoms = blob_data.get("original_atoms", [])
    if original_atoms:
        return [str(atom["symbol"]) for atom in original_atoms]
    symbols = []
    for blob in blobs:
        member_symbols = blob.get("member_symbols")
        if member_symbols:
            symbols.extend(str(symbol) for symbol in member_symbols)
        else:
            symbol = str(blob.get("symbol", "X"))
            if symbol != "X":
                symbols.append(symbol)
    return symbols


def reference_atoms_from_blob_data(blob_data, blobs):
    original_atoms = blob_data.get("original_atoms", [])
    if original_atoms:
        return (
            [str(atom["symbol"]) for atom in original_atoms],
            np.asarray([atom["position_angstrom"] for atom in original_atoms], dtype=float),
        )
    symbols = [str(blob.get("symbol", "X")) for blob in blobs]
    if not symbols or all(symbol == "X" for symbol in symbols):
        return [], np.zeros((0, 3), dtype=float)
    return symbols, np.asarray([blob["center_angstrom"] for blob in blobs], dtype=float)


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


def lattice_vectors_from_blob_data(blob_data):
    for key in ("lattice_vectors_angstrom", "cell_vectors_angstrom", "cell_angstrom"):
        if key in blob_data:
            return np.asarray(blob_data[key], dtype=float)
    raise KeyError("Blob JSON must contain lattice_vectors_angstrom or cell_vectors_angstrom.")


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
        raise RuntimeError("Set --mp-api-key or MP_API_KEY to fetch elemental references from Materials Project.")
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


def blob_species_fraction(blob, species):
    counts = {symbol: 0.0 for symbol in species}
    if "composition_counts" in blob:
        for symbol, count in blob["composition_counts"].items():
            if symbol in counts:
                counts[symbol] += float(count)
    elif blob.get("member_symbols"):
        for symbol in blob["member_symbols"]:
            if symbol in counts:
                counts[symbol] += 1.0
    else:
        symbol = str(blob.get("symbol", "X"))
        if symbol in counts:
            counts[symbol] += 1.0
    values = np.asarray([counts[symbol] for symbol in species], dtype=float)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(len(species), 1.0 / len(species), dtype=float)
    return values / total


def split_species_list(text):
    return split_species_list_value(text)


def parse_species_by_source_type(text, species):
    mapping = {}
    raw = str(text or "").strip()
    if not raw:
        return mapping
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "blobs.species_by_source_type entries must look like "
                "'source_type:Species|Species; other_source:Species'."
            )
        source_type, allowed_text = entry.split(":", 1)
        source_type = source_type.strip()
        if not source_type:
            raise ValueError(f"Empty source type in blobs.species_by_source_type entry {entry!r}.")
        allowed = split_species_list(allowed_text)
        if not allowed:
            raise ValueError(f"No allowed species listed for source type {source_type!r}.")
        unknown = [symbol for symbol in allowed if symbol not in species]
        if unknown:
            raise ValueError(
                f"Unknown species in blobs.species_by_source_type for {source_type!r}: {unknown}. "
                f"Known species are {species}."
            )
        mapping[source_type] = allowed
    return mapping


def blob_source_type(blob):
    return str(blob.get("source_circle_type") or blob.get("source_type") or blob.get("type") or "default")


def allowed_species_mask_from_blobs(blobs, species, text, blob_allowed_elements=None):
    mapping = parse_species_by_source_type(text, species)
    default_allowed = mapping.get("default", list(species))
    if blob_allowed_elements is None:
        blob_allowed_elements = allowed_elements_from_blobs(blobs)
    mask = np.zeros((len(blobs), len(species)), dtype=bool)
    labels = []
    for blob_index, blob in enumerate(blobs):
        source_type = blob_source_type(blob)
        labels.append(source_type)
        allowed = blob_allowed_elements[blob_index] or mapping.get(source_type, default_allowed)
        unknown = [symbol for symbol in allowed if symbol not in species]
        if unknown:
            raise ValueError(
                f"Unknown allowed element(s) for blob {blob_index}: {unknown}. "
                f"Known species are {species}."
            )
        for symbol in allowed:
            mask[blob_index, species.index(symbol)] = True
        if not np.any(mask[blob_index]):
            raise ValueError(
                f"Blob {blob_index} source_type={source_type!r} has no allowed real species. "
                "Use at least one real species; vacuum remains implicit."
            )
    return mask, labels, mapping


def allowed_species_mask_from_blob_source_types(blobs, species, text):
    return allowed_species_mask_from_blobs(blobs, species, text)


def restrict_blob_species_fractions(fractions, allowed_mask):
    fractions = np.asarray(fractions, dtype=float)
    allowed_mask = np.asarray(allowed_mask, dtype=bool)
    restricted = np.where(allowed_mask, np.maximum(fractions, 0.0), 0.0)
    totals = restricted.sum(axis=1)
    for row_index in np.where(totals <= 1e-12)[0]:
        allowed = allowed_mask[row_index]
        restricted[row_index, allowed] = 1.0 / float(np.count_nonzero(allowed))
        totals[row_index] = 1.0
    return restricted / totals[:, None]


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
    ent_avg = float(tsallis_entropy_term(occ, q).item()) / occ.shape[0]
    null = torch.clamp(1.0 - occ.sum(dim=1), min=0.0)
    probs = torch.cat([occ, null[:, None]], dim=1)
    max_nonbinary = float((1.0 - probs.max(dim=1).values).max().item())
    return ent_avg, max_nonbinary


def set_enable_eat_grad(atoms, flag):
    if flag:
        atoms.info["enable_eat_grad"] = True
    else:
        atoms.info.pop("enable_eat_grad", None)


def set_multi_species_eat_weights(atoms, occ_np, atomic_numbers_for_species):
    occ_np = np.asarray(occ_np, dtype=np.float32).reshape(len(atoms), len(atomic_numbers_for_species))
    ew = np.zeros((len(atoms), MAX_NUM_ELEMENTS), dtype=np.float32)
    for species_idx, atomic_number in enumerate(atomic_numbers_for_species):
        ew[:, int(atomic_number)] = occ_np[:, species_idx]
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


class ModelEvaluationTracker:
    """Count and time actual calculator executions by optimization phase."""

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
                    "count": int(
                        after["by_phase"].get(phase, {}).get("count", 0)
                        - before["by_phase"].get(phase, {}).get("count", 0)
                    ),
                    "seconds": float(
                        after["by_phase"].get(phase, {}).get("seconds", 0.0)
                        - before["by_phase"].get(phase, {}).get("seconds", 0.0)
                    ),
                }
                for phase in sorted(phases)
            },
        }


def instrument_calculator(calc, tracker):
    """Instrument ASE ``calculate`` calls without altering ASE caching."""
    original_calculate = calc.calculate

    def counted_calculate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_calculate(*args, **kwargs)
        finally:
            tracker.record(tracker.phase, time.perf_counter() - started)

    calc.calculate = counted_calculate
    return calc


def sample_uniform_unit_ball(n):
    directions = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):
        zero = norms == 0.0
        directions[zero] = np.random.normal(size=(int(zero.sum()), 3))
        norms = np.linalg.norm(directions, axis=1)
    directions = directions / norms[:, None]
    radii = np.random.random(size=n) ** (1.0 / 3.0)
    return directions * radii[:, None]


def sample_unit_ball_axis_slab(radius, axis, lo, hi):
    radius = float(radius)
    if radius <= 0.0:
        return np.zeros(3, dtype=float)
    axis = int(axis)
    lo = max(-radius, float(lo))
    hi = min(radius, float(hi))
    if hi < lo:
        lo, hi = hi, lo
    for _ in range(2000):
        candidate = sample_uniform_unit_ball(1)[0] * radius
        if lo <= candidate[axis] <= hi:
            return candidate

    # Extremely thin end slabs are rare for the current ghosts-per-blob values,
    # but this direct fallback avoids pathological rejection loops.
    coord = np.random.uniform(lo, hi)
    remaining_radius = max(radius * radius - coord * coord, 0.0) ** 0.5
    disk_r = remaining_radius * np.sqrt(np.random.random())
    theta = 2.0 * np.pi * np.random.random()
    other_axes = [idx for idx in range(3) if idx != axis]
    candidate = np.zeros(3, dtype=float)
    candidate[axis] = coord
    candidate[other_axes[0]] = disk_r * np.cos(theta)
    candidate[other_axes[1]] = disk_r * np.sin(theta)
    return candidate


def softplus_stable(x):
    return np.logaddexp(0.0, x)


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


def parse_positive_int_triplet(value, name):
    if isinstance(value, (list, tuple, np.ndarray)):
        pieces = list(value)
    else:
        pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated integers.")
    numeric = np.asarray([float(piece) for piece in pieces], dtype=float)
    result = numeric.astype(int)
    if np.any(numeric != result) or np.any(result <= 0):
        raise ValueError(f"{name} values must be positive integers.")
    return result


def resolve_script_relative_path(value):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def periodic_gaussian_fourier_kernel(shape_zyx, sigma_zyx_pixels, device, dtype):
    """Fourier multiplier for a normalized Gaussian on a periodic ZYX grid."""
    nz, ny, nx = (int(value) for value in shape_zyx)
    sigma_z, sigma_y, sigma_x = (
        float(value) for value in sigma_zyx_pixels
    )
    fz = torch.fft.fftfreq(nz, device=device, dtype=dtype)[:, None, None]
    fy = torch.fft.fftfreq(ny, device=device, dtype=dtype)[None, :, None]
    fx = torch.fft.fftfreq(nx, device=device, dtype=dtype)[None, None, :]
    exponent = -2.0 * math.pi * math.pi * (
        (sigma_z * fz) ** 2
        + (sigma_y * fy) ** 2
        + (sigma_x * fx) ** 2
    )
    return torch.exp(exponent)


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


def display_transform_slice(image, background, normalization):
    """Apply the sqrt display convention used for mask/source diagnostics."""
    shifted = np.maximum(
        np.asarray(image, dtype=np.float32) - float(background),
        0.0,
    )
    return np.sqrt(shifted / max(float(normalization), 1e-12)).astype(np.float32)


def robust_limits(volume, low=0.2, high=99.8):
    values = np.asarray(volume, dtype=np.float32)
    vmin, vmax = np.percentile(values, [float(low), float(high)])
    if vmax <= vmin:
        vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def cell_edge_traces(cell_lengths_A):
    lx, ly, lz = [float(value) for value in cell_lengths_A]
    corners = np.asarray(
        [
            [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0], [0, 0, 0],
            [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz], [0, 0, lz],
        ],
        dtype=float,
    )
    verticals = [
        ([0, 0], [0, 0], [0, lz]),
        ([lx, lx], [0, 0], [0, lz]),
        ([lx, lx], [ly, ly], [0, lz]),
        ([0, 0], [ly, ly], [0, lz]),
    ]
    return corners, verticals


def write_plotly_heavy_boundary_html(path, points_xyz_A, cell_lengths_A):
    """Write a lightweight interactive view without requiring Plotly in Python."""
    points = np.asarray(points_xyz_A, dtype=float)
    corners, verticals = cell_edge_traces(cell_lengths_A)
    traces = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "heavy-atom allowed boundary",
            "x": np.round(points[:, 0], 4).tolist(),
            "y": np.round(points[:, 1], 4).tolist(),
            "z": np.round(points[:, 2], 4).tolist(),
            "marker": {
                "size": 2.3,
                "color": np.round(points[:, 2], 4).tolist(),
                "colorscale": "Viridis",
                "opacity": 0.58,
            },
        },
        {
            "type": "scatter3d",
            "mode": "lines",
            "name": "periodic optimization cell",
            "x": corners[:, 0].tolist(),
            "y": corners[:, 1].tolist(),
            "z": corners[:, 2].tolist(),
            "line": {"color": "black", "width": 5},
        },
    ]
    for x_values, y_values, z_values in verticals:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "showlegend": False,
                "x": x_values,
                "y": y_values,
                "z": z_values,
                "line": {"color": "black", "width": 5},
            }
        )
    layout = {
        "title": "Garnet heavy-atom allowed region from the microscopy threshold",
        "scene": {
            "xaxis": {"title": "x (Angstrom)"},
            "yaxis": {"title": "y (Angstrom)"},
            "zaxis": {"title": "z (Angstrom)"},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "t": 45, "b": 0},
    }
    html = (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
        "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>\n"
        "</head><body><div id=\"plot\" style=\"width:100%;height:95vh;\"></div>\n"
        "<script>\n"
        f"const traces = {json.dumps(traces)};\n"
        f"const layout = {json.dumps(layout)};\n"
        "Plotly.newPlot('plot', traces, layout, {responsive: true});\n"
        "</script></body></html>\n"
    )
    path.write_text(html, encoding="utf-8")


def write_initial_diagnostics(source, payload, args, output_dir):
    """Visualize the exact TIFF crop, z interpolation, threshold, and cell."""
    diagnostics_dir = output_dir / "initial_diagnostics"
    source_slices_dir = diagnostics_dir / "source_slices_with_crop"
    target_slices_dir = diagnostics_dir / "target_slices"
    source_slices_dir.mkdir(parents=True, exist_ok=True)
    target_slices_dir.mkdir(parents=True, exist_ok=True)

    matplotlib_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "eat_matplotlib_cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(payload["target_image"], dtype=np.float32)
    smoothed = np.asarray(payload["smoothed_target_sqrt"], dtype=np.float32)
    image_transform = str(payload["image_transform"])
    heavy_mask = np.asarray(payload["heavy_allowed_mask"], dtype=bool)
    source_voxel_A = np.asarray(payload["source_voxel_size_A"], dtype=float)
    target_voxel_A = np.asarray(payload["target_voxel_size_A"], dtype=float)
    z_coordinates = np.asarray(payload["z_source_coordinates"], dtype=float)
    z_lower = np.floor(z_coordinates).astype(int)
    z_upper = np.minimum(z_lower + 1, source.shape[0] - 1)
    z_fraction = z_coordinates - z_lower
    z0, y0_float, x0_float = np.asarray(payload["crop_origin"], dtype=float)
    x0, y0 = int(x0_float), int(y0_float)
    nz, ny, nx = [int(value) for value in payload["target_shape_zyx"]]
    dz_A, dy_A, dx_A = target_voxel_A
    cell_length_A = float(payload["cell_length_A"])
    threshold = float(payload["threshold_value"])

    # Full-stack projection makes the chosen x/y subregion unmistakable.
    source_background = float(
        np.percentile(source, float(args.target_background_percentile))
    )
    source_shifted = np.maximum(source - source_background, 0.0)
    source_normalization = max(
        float(np.percentile(source_shifted, float(args.target_normalization_percentile))),
        1e-12,
    )
    source_display = np.sqrt(source_shifted / source_normalization).astype(np.float32)
    source_vmin, source_vmax = robust_limits(source_display)
    fig, ax = plt.subplots(figsize=(7.5, 7.2), constrained_layout=True)
    ax.imshow(
        np.max(source_display, axis=0),
        cmap="inferno",
        origin="upper",
        vmin=source_vmin,
        vmax=source_vmax,
    )
    ax.add_patch(
        Rectangle((x0, y0), nx, ny, fill=False, edgecolor="cyan", linewidth=2.2)
    )
    ax.set_title(
        f"Full TIFF max-z projection | selected x={x0}:{x0 + nx}, "
        f"y={y0}:{y0 + ny}"
    )
    ax.set_xlabel("source x pixel")
    ax.set_ylabel("source y pixel")
    fig.savefig(diagnostics_dir / "source_xy_projection_crop_overlay.png", dpi=190)
    plt.close(fig)

    # Record the exact target interpolation weights contributed by source slices.
    interpolation_weights = np.zeros((nz, source.shape[0]), dtype=float)
    for target_index in range(nz):
        interpolation_weights[target_index, z_lower[target_index]] += 1.0 - z_fraction[target_index]
        interpolation_weights[target_index, z_upper[target_index]] += z_fraction[target_index]

    for source_index in range(source.shape[0]):
        contributors = np.where(interpolation_weights[:, source_index] > 1e-10)[0]
        contribution_text = ", ".join(
            f"target {index} (w={interpolation_weights[index, source_index]:.3f})"
            for index in contributors
        )
        fig, ax = plt.subplots(figsize=(7.5, 7.2), constrained_layout=True)
        ax.imshow(
            source_display[source_index],
            cmap="inferno",
            origin="upper",
            vmin=source_vmin,
            vmax=source_vmax,
        )
        ax.add_patch(
            Rectangle(
                (x0, y0),
                nx,
                ny,
                fill=False,
                edgecolor="#ff2d20" if len(contributors) else "#808080",
                linewidth=2.0,
            )
        )
        ax.set_title(
            f"source z={source_index:03d} ({source_index * source_voxel_A[0]:.3f} A) | "
            + (contribution_text if contribution_text else "not used by target planes")
        )
        ax.set_xlabel("source x pixel")
        ax.set_ylabel("source y pixel")
        fig.savefig(
            source_slices_dir / f"source_z{source_index:03d}_crop_overlay.png",
            dpi=170,
        )
        plt.close(fig)

    used_source_indices = np.where(interpolation_weights.sum(axis=0) > 0.0)[0]
    fig_width = max(7.0, 0.48 * max(1, len(used_source_indices)))
    fig, ax = plt.subplots(figsize=(fig_width, 1.1 * nz + 2.2), constrained_layout=True)
    shown_weights = interpolation_weights[:, used_source_indices]
    image_handle = ax.imshow(shown_weights, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for target_index in range(nz):
        for column_index, source_index in enumerate(used_source_indices):
            weight = shown_weights[target_index, column_index]
            if weight > 1e-10:
                ax.text(column_index, target_index, f"{weight:.3f}", ha="center", va="center")
    ax.set_xticks(np.arange(len(used_source_indices)))
    ax.set_xticklabels([str(index) for index in used_source_indices])
    ax.set_yticks(np.arange(nz))
    ax.set_yticklabels(
        [f"target {index} | z={index * dz_A:.3f} A" for index in range(nz)]
    )
    ax.set_xlabel("source TIFF z-slice index")
    ax.set_ylabel("optimization target plane")
    ax.set_title("Exact linear interpolation weights used to build target z planes")
    fig.colorbar(image_handle, ax=ax, label="source-slice weight")
    fig.savefig(diagnostics_dir / "target_to_source_z_interpolation.png", dpi=190)
    plt.close(fig)

    target_vmin, target_vmax = robust_limits(target)
    mask_target_vmin, mask_target_vmax = robust_limits(smoothed)
    extent_xy = (0.0, cell_length_A, 0.0, cell_length_A)
    for z_index in range(nz):
        fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.2), constrained_layout=True)
        axes[0].imshow(
            target[z_index], cmap="inferno", origin="lower", extent=extent_xy,
            vmin=target_vmin, vmax=target_vmax,
        )
        axes[0].set_title(f"{image_transform} target used by image loss")
        axes[1].imshow(
            smoothed[z_index], cmap="inferno", origin="lower", extent=extent_xy,
            vmin=mask_target_vmin, vmax=mask_target_vmax,
        )
        axes[1].set_title("smoothed sqrt target for threshold")
        axes[2].imshow(
            heavy_mask[z_index], cmap="gray", origin="lower", extent=extent_xy,
            vmin=0, vmax=1,
        )
        axes[2].set_title(f"heavy mask: smoothed >= {threshold:.5g}")
        axes[3].imshow(
            target[z_index], cmap="inferno", origin="lower", extent=extent_xy,
            vmin=target_vmin, vmax=target_vmax,
        )
        if np.any(heavy_mask[z_index]) and not np.all(heavy_mask[z_index]):
            axes[3].contour(
                heavy_mask[z_index].astype(float),
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.2,
                extent=extent_xy,
            )
        axes[3].set_title("target + heavy-region contour")
        for axis in axes:
            axis.set_xlabel("x (A)")
            axis.set_ylabel("y (A)")
        fig.suptitle(
            f"target z={z_index:03d} ({z_index * dz_A:.4f} A) | "
            f"source coordinate={z_coordinates[z_index]:.5f}: "
            f"slice {z_lower[z_index]} weight {1.0 - z_fraction[z_index]:.5f}, "
            f"slice {z_upper[z_index]} weight {z_fraction[z_index]:.5f}"
        )
        fig.savefig(target_slices_dir / f"target_z{z_index:03d}_diagnostic.png", dpi=190)
        plt.close(fig)

    cols = min(4, nz)
    rows = int(math.ceil(nz / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, z_index in zip(axes, range(nz)):
        axis.imshow(
            target[z_index], cmap="inferno", origin="lower", extent=extent_xy,
            vmin=target_vmin, vmax=target_vmax,
        )
        if np.any(heavy_mask[z_index]) and not np.all(heavy_mask[z_index]):
            axis.contour(
                heavy_mask[z_index].astype(float), levels=[0.5], colors=["cyan"],
                linewidths=1.0, extent=extent_xy,
            )
        axis.set_title(
            f"target {z_index}: z={z_index * dz_A:.2f} A\n"
            f"source z={z_coordinates[z_index]:.3f}"
        )
        axis.set_xlabel("x (A)")
        axis.set_ylabel("y (A)")
    for axis in axes[nz:]:
        axis.axis("off")
    fig.suptitle("Every optimization target slice; cyan = heavy-atom allowed contour")
    fig.savefig(diagnostics_dir / "target_slices_with_heavy_contours.png", dpi=190)
    plt.close(fig)

    # Orthogonal projections expose accidental truncation or unexpected z geometry.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    projections = [
        (np.max(smoothed, axis=0), np.max(heavy_mask, axis=0), (0, cell_length_A, 0, cell_length_A), "x (A)", "y (A)", "top: max over z"),
        (np.max(smoothed, axis=1), np.max(heavy_mask, axis=1), (0, cell_length_A, 0, cell_length_A), "x (A)", "z (A)", "side: max over y"),
        (np.max(smoothed, axis=2), np.max(heavy_mask, axis=2), (0, cell_length_A, 0, cell_length_A), "y (A)", "z (A)", "side: max over x"),
    ]
    for axis, (image, mask_projection, extent, xlabel, ylabel, title) in zip(axes, projections):
        axis.imshow(image, cmap="inferno", origin="lower", extent=extent, aspect="auto")
        if np.any(mask_projection) and not np.all(mask_projection):
            axis.contour(mask_projection.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.1, extent=extent)
        axis.add_patch(Rectangle((0, 0), cell_length_A, cell_length_A, fill=False, edgecolor="white", linewidth=1.5))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    fig.suptitle("Periodic cubic optimization cell; cyan = projected heavy region")
    fig.savefig(diagnostics_dir / "cell_top_and_side_views_heavy_region.png", dpi=190)
    plt.close(fig)

    # Periodic boundary extraction: a mask voxel is interior only when all six
    # periodically wrapped axial neighbors are also inside.
    interior = heavy_mask.copy()
    for axis_index in range(3):
        interior &= np.roll(heavy_mask, 1, axis=axis_index)
        interior &= np.roll(heavy_mask, -1, axis=axis_index)
    boundary_zyx = np.argwhere(heavy_mask & ~interior)
    max_points = int(
        getattr(
            args,
            "initial_diagnostics_max_isosurface_points",
            INITIAL_DIAGNOSTICS_MAX_ISOSURFACE_POINTS,
        )
    )
    if max_points > 0 and len(boundary_zyx) > max_points:
        rng = np.random.default_rng(12345)
        boundary_zyx = boundary_zyx[
            rng.choice(len(boundary_zyx), size=max_points, replace=False)
        ]
    points_xyz_A = np.column_stack(
        (
            (boundary_zyx[:, 2] + 0.5) * dx_A,
            (boundary_zyx[:, 1] + 0.5) * dy_A,
            (boundary_zyx[:, 0] + 0.5) * dz_A,
        )
    ) if len(boundary_zyx) else np.zeros((0, 3), dtype=float)
    isosurface_path = diagnostics_dir / "heavy_region_isosurface_boundary.html"
    if len(points_xyz_A):
        write_plotly_heavy_boundary_html(
            isosurface_path,
            points_xyz_A,
            np.asarray([cell_length_A] * 3, dtype=float),
        )

    voxel_volume_A3 = float(np.prod(target_voxel_A))
    slice_report = []
    for index in range(nz):
        slice_report.append(
            {
                "target_z_index": int(index),
                "target_z_A": float(index * dz_A),
                "source_z_coordinate": float(z_coordinates[index]),
                "source_lower_index": int(z_lower[index]),
                "source_lower_weight": float(1.0 - z_fraction[index]),
                "source_upper_index": int(z_upper[index]),
                "source_upper_weight": float(z_fraction[index]),
                "heavy_allowed_pixel_fraction": float(heavy_mask[index].mean()),
            }
        )
    report = {
        "source_tiff": str(payload["tiff_path"]),
        "source_shape_zyx": [int(value) for value in payload["source_shape_zyx"]],
        "source_voxel_size_A_zyx": source_voxel_A.tolist(),
        "crop_source_pixels": {
            "x_half_open": [x0, x0 + nx],
            "y_half_open": [y0, y0 + ny],
            "z_start_coordinate": float(z0),
        },
        "target_shape_zyx": [nz, ny, nx],
        "target_voxel_size_A_zyx": target_voxel_A.tolist(),
        "periodic_cell_lengths_A_xyz": [cell_length_A] * 3,
        "target_slices": slice_report,
        "threshold_mode": str(payload["threshold_mode"]),
        "threshold_config": float(payload["threshold_config"]),
        "threshold_value_after_smoothing": threshold,
        "mask_smoothing_sigma_pixels_zyx": parse_float_triplet(
            args.mask_smoothing_sigma_pixels,
            "heavy_region.mask_smoothing_sigma_pixels",
        ).tolist(),
        "heavy_allowed_fraction": float(heavy_mask.mean()),
        "heavy_allowed_volume_A3": float(np.count_nonzero(heavy_mask) * voxel_volume_A3),
        "heavy_boundary_points_written": int(len(points_xyz_A)),
        "outputs": {
            "full_tiff_crop_overlay": str(diagnostics_dir / "source_xy_projection_crop_overlay.png"),
            "source_slice_overlays": str(source_slices_dir),
            "z_interpolation_map": str(diagnostics_dir / "target_to_source_z_interpolation.png"),
            "per_target_slice_panels": str(target_slices_dir),
            "all_target_slices_montage": str(diagnostics_dir / "target_slices_with_heavy_contours.png"),
            "orthogonal_cell_views": str(diagnostics_dir / "cell_top_and_side_views_heavy_region.png"),
            "interactive_heavy_boundary": str(isosurface_path),
        },
    }
    (diagnostics_dir / "initial_diagnostics_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def prepare_garnet_experimental_volume_legacy(args, output_dir, write_diagnostics=True):
    """Crop/resample the TIFF and construct a periodic intensity level set."""
    import tifffile
    from scipy.ndimage import gaussian_filter

    tiff_path = resolve_script_relative_path(args.experimental_tiff)
    if not tiff_path.is_file():
        raise FileNotFoundError(f"Experimental TIFF not found: {tiff_path}")
    source = np.asarray(tifffile.imread(tiff_path))
    if source.ndim != 3:
        raise ValueError(
            f"Experimental TIFF must be a ZYX stack; got shape {source.shape}."
        )
    source = source.astype(np.float32, copy=False)
    source_dz_A, source_dy_A, source_dx_A = parse_float_triplet(
        args.source_voxel_size_angstrom,
        "experimental.source_voxel_size_angstrom",
    )
    if min(source_dz_A, source_dy_A, source_dx_A) <= 0.0:
        raise ValueError("Experimental voxel sizes must be positive.")

    x0 = int(args.crop_x_start_px)
    y0 = int(args.crop_y_start_px)
    nxy = int(args.crop_xy_size_px)
    z0 = float(args.crop_z_start_slice)
    nz = int(args.target_z_slices)
    if nxy <= 1 or nz <= 1:
        raise ValueError("The target crop must have at least two samples per axis.")
    if x0 < 0 or y0 < 0 or x0 + nxy > source.shape[2] or y0 + nxy > source.shape[1]:
        raise ValueError(
            f"Requested xy crop x={x0}:{x0+nxy}, y={y0}:{y0+nxy} "
            f"is outside TIFF shape {source.shape}."
        )

    if args.cell_length_angstrom is None:
        cell_length_A = float(nxy * source_dx_A)
    else:
        cell_length_A = float(args.cell_length_angstrom)
    if cell_length_A <= 0.0:
        raise ValueError("experimental.cell_length_angstrom must be positive.")
    if not np.isclose(source_dx_A, source_dy_A, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"The requested axis-aligned cubic crop requires dx=dy; got "
            f"{source_dx_A} and {source_dy_A} Å."
        )

    target_voxel_size_A = np.asarray(
        [cell_length_A / nz, cell_length_A / nxy, cell_length_A / nxy],
        dtype=float,
    )
    z_source_coordinates = (
        z0
        + np.arange(nz, dtype=float)
        * target_voxel_size_A[0]
        / source_dz_A
    )
    if z_source_coordinates.min() < 0.0 or z_source_coordinates.max() > source.shape[0] - 1:
        raise ValueError(
            "The requested cubic z interval requires source slice coordinates "
            f"{z_source_coordinates.min():.3f}..{z_source_coordinates.max():.3f}, "
            f"outside available slices 0..{source.shape[0]-1}."
        )
    z_lower = np.floor(z_source_coordinates).astype(int)
    z_upper = np.minimum(z_lower + 1, source.shape[0] - 1)
    z_fraction = (z_source_coordinates - z_lower).astype(np.float32)
    xy_crop = source[:, y0 : y0 + nxy, x0 : x0 + nxy]
    target_linear = (
        (1.0 - z_fraction)[:, None, None] * xy_crop[z_lower]
        + z_fraction[:, None, None] * xy_crop[z_upper]
    ).astype(np.float32)

    background = float(
        np.percentile(target_linear, float(args.target_background_percentile))
    )
    target_linear = np.maximum(target_linear - background, 0.0)
    normalization = float(
        np.percentile(
            target_linear,
            float(args.target_normalization_percentile),
        )
    )
    if normalization <= 0.0:
        raise ValueError("Experimental target has zero robust intensity scale.")
    target_linear_normalized = target_linear / normalization
    if bool(args.gaussian_clip_experimental_to_unit_interval):
        target_linear_normalized = np.clip(
            target_linear_normalized, 0.0, 1.0
        ).astype(np.float32)
    target_sqrt = np.sqrt(target_linear_normalized).astype(np.float32)
    image_transform = normalize_image_transform_name(args.image_target_transform)
    target_image = numpy_image_transform(
        target_linear_normalized,
        image_transform,
        float(args.gaussian_transform_epsilon),
    ).astype(np.float32)

    mask_sigma = parse_float_triplet(
        args.mask_smoothing_sigma_pixels,
        "heavy_region.mask_smoothing_sigma_pixels",
    )
    smoothed_target = gaussian_filter(
        target_sqrt,
        sigma=mask_sigma,
        mode="wrap",
    ).astype(np.float32)
    threshold_mode = str(args.mask_threshold_mode).strip().lower()
    if threshold_mode == "percentile":
        threshold_value = float(
            np.percentile(smoothed_target, float(args.mask_threshold))
        )
    elif threshold_mode in ("absolute", "value"):
        threshold_value = float(args.mask_threshold)
    elif threshold_mode in ("fraction_of_max", "max_fraction"):
        threshold_value = float(args.mask_threshold) * float(smoothed_target.max())
    else:
        raise ValueError(
            "heavy_region.threshold_mode must be percentile, absolute, "
            "or fraction_of_max."
        )
    feasible_mask = smoothed_target >= threshold_value
    feasible_count = int(np.count_nonzero(feasible_mask))
    if feasible_count == 0:
        raise ValueError("The selected threshold produced an empty heavy-atom mask.")

    level_set_A, level_set_metadata = distance_scaled_level_set(
        smoothed_target,
        threshold_value,
        target_voxel_size_A,
    )

    payload = {
        "tiff_path": tiff_path,
        "source_shape_zyx": tuple(int(v) for v in source.shape),
        "source_voxel_size_A": np.asarray(
            [source_dz_A, source_dy_A, source_dx_A],
            dtype=float,
        ),
        "crop_origin": np.asarray([z0, y0, x0], dtype=float),
        "cell_length_A": cell_length_A,
        "cell": np.diag([cell_length_A] * 3).astype(float),
        "target_shape_zyx": (nz, nxy, nxy),
        "target_voxel_size_A": target_voxel_size_A,
        "z_source_coordinates": z_source_coordinates,
        "target_background": background,
        "target_normalization": normalization,
        "target_linear_normalized": target_linear_normalized.astype(np.float32),
        "target_sqrt": target_sqrt,
        "target_image": target_image,
        "image_transform": image_transform,
        "image_transform_epsilon": float(args.gaussian_transform_epsilon),
        "smoothed_target_sqrt": smoothed_target,
        "threshold_mode": threshold_mode,
        "threshold_config": float(args.mask_threshold),
        "threshold_value": threshold_value,
        "heavy_allowed_mask": feasible_mask,
        "heavy_allowed_fraction": float(feasible_mask.mean()),
        "heavy_level_set_A": level_set_A,
        "feasible_mask": feasible_mask,
        "feasible_fraction": float(feasible_mask.mean()),
        "level_set_A": level_set_A,
        "level_set_metadata": level_set_metadata,
    }
    if write_diagnostics:
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir = output_dir / "initial_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        image_display_vmax = max(
            float(np.percentile(target_image, 99.5)),
            1e-12,
        )
        image_display_u16 = np.round(
            np.clip(target_image / image_display_vmax, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
        tifffile.imwrite(
            output_dir / f"experimental_target_{image_transform}_ZYX.tif",
            image_display_u16,
            imagej=True,
            metadata={
                "axes": "ZYX",
                "unit": "angstrom",
                "spacing": float(target_voxel_size_A[0]),
                "display_vmin_raw": 0.0,
                "display_vmax_raw": image_display_vmax,
                "image_transform": image_transform,
            },
        )
        target_display_vmax = max(
            float(np.percentile(target_sqrt, 99.5)),
            1e-12,
        )
        target_display_u16 = np.round(
            np.clip(target_sqrt / target_display_vmax, 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
        tifffile.imwrite(
            output_dir / "experimental_target_sqrt_for_heavy_mask_ZYX.tif",
            target_display_u16,
            imagej=True,
            metadata={
                "axes": "ZYX",
                "unit": "angstrom",
                "spacing": float(target_voxel_size_A[0]),
                "display_vmin_raw": 0.0,
                "display_vmax_raw": target_display_vmax,
            },
        )
        tifffile.imwrite(
            output_dir / "heavy_allowed_mask_ZYX.tif",
            (feasible_mask.astype(np.uint8) * 255),
            imagej=True,
            metadata={
                "axes": "ZYX",
                "unit": "angstrom",
                "spacing": float(target_voxel_size_A[0]),
                "threshold": threshold_value,
            },
        )
        tifffile.imwrite(
            output_dir / "heavy_level_set_A_ZYX.tif",
            level_set_A,
            imagej=True,
            metadata={
                "axes": "ZYX",
                "unit": "angstrom",
                "spacing": float(target_voxel_size_A[0]),
            },
        )
        np.savez_compressed(
            output_dir / "experimental_target_and_heavy_region.npz",
            target_linear_normalized=target_linear_normalized.astype(np.float32),
            target_sqrt=target_sqrt,
            target_image=target_image,
            image_transform=np.asarray(image_transform),
            image_transform_epsilon=np.asarray(float(args.gaussian_transform_epsilon)),
            smoothed_target_sqrt=smoothed_target,
            heavy_allowed_mask=feasible_mask,
            heavy_level_set_A=level_set_A,
            target_voxel_size_A=target_voxel_size_A,
            cell=np.diag([cell_length_A] * 3),
            crop_origin_zyx=np.asarray([z0, y0, x0], dtype=float),
            z_source_coordinates=z_source_coordinates,
            threshold_value=np.asarray(threshold_value),
            threshold_mode=np.asarray(threshold_mode),
            source_tiff=np.asarray(str(tiff_path)),
        )
        if bool(getattr(args, "initial_diagnostics_enable", True)):
            write_initial_diagnostics(source, payload, args, output_dir)
    return payload


def _ellipse_full_normalized_template(shape, center_yx, pixel_size_A, kernel, support_sigma):
    """Visible crop of a double-Gaussian normalized over its complete support."""
    ny, nx = map(int, shape)
    y, x = map(float, center_yx)
    radius = int(
        math.ceil(
            float(support_sigma)
            * float(kernel["tail_sigma_A"])
            / float(pixel_size_A)
        )
    )
    full_y0 = int(math.floor(y)) - radius
    full_y1 = int(math.floor(y)) + radius + 2
    full_x0 = int(math.floor(x)) - radius
    full_x1 = int(math.floor(x)) + radius + 2
    yy, xx = np.indices((full_y1 - full_y0, full_x1 - full_x0), dtype=float)
    dy_A = (yy + full_y0 - y) * pixel_size_A
    dx_A = (xx + full_x0 - x) * pixel_size_A
    squared_A = dx_A**2 + dy_A**2
    patch = np.zeros_like(squared_A)
    for prefix in ("core", "tail"):
        sigma_A = float(kernel[f"{prefix}_sigma_A"])
        fraction = float(kernel[f"{prefix}_fraction"])
        patch += fraction * np.exp(-0.5 * squared_A / sigma_A**2) / (
            2.0 * math.pi * sigma_A**2
        )
    normalizer = float(np.sum(patch) * pixel_size_A**2)
    if normalizer <= 0.0:
        raise RuntimeError("Ellipse-fit double-Gaussian template has zero integral.")
    patch /= normalizer
    template = np.zeros((ny, nx), dtype=np.float64)
    y0, y1 = max(0, full_y0), min(ny, full_y1)
    x0, x1 = max(0, full_x0), min(nx, full_x1)
    if y0 < y1 and x0 < x1:
        template[y0:y1, x0:x1] = patch[
            y0 - full_y0 : y1 - full_y0,
            x0 - full_x0 : x1 - full_x0,
        ]
    return template


def _ellipse_fit_amplitudes(image, centers, pixel_size_A, kernel, support_sigma):
    from scipy.optimize import nnls

    templates = [
        _ellipse_full_normalized_template(
            image.shape, center, pixel_size_A, kernel, support_sigma
        )
        for center in centers
    ]
    if not templates:
        raise RuntimeError("No LoG candidates remain for ellipse fitting.")
    design = np.column_stack([template.ravel() for template in templates])
    amplitudes, residual_norm = nnls(
        design, np.asarray(image, dtype=float).ravel(), maxiter=20 * design.shape[1]
    )
    reconstruction = (design @ amplitudes).reshape(image.shape)
    return amplitudes, reconstruction, templates, float(residual_norm)


def _ellipse_prune_and_refit(
    image,
    centers,
    minimum_amplitude,
    rounds,
    pixel_size_A,
    kernel,
    support_sigma,
):
    centers = np.asarray(centers, dtype=float)
    history = []
    for round_index in range(max(1, int(rounds))):
        amplitudes, reconstruction, templates, residual_norm = _ellipse_fit_amplitudes(
            image, centers, pixel_size_A, kernel, support_sigma
        )
        keep = amplitudes >= float(minimum_amplitude)
        history.append(
            {
                "round": int(round_index),
                "candidate_count": int(len(centers)),
                "retained_count": int(np.count_nonzero(keep)),
                "residual_l2": residual_norm,
            }
        )
        if np.all(keep):
            break
        if not np.any(keep):
            raise RuntimeError("Joint ellipse fit pruned every LoG candidate.")
        centers = centers[keep]
    amplitudes, reconstruction, templates, residual_norm = _ellipse_fit_amplitudes(
        image, centers, pixel_size_A, kernel, support_sigma
    )
    return centers, amplitudes, reconstruction, templates, residual_norm, history


def _ellipse_refine_positions(
    image,
    centers,
    amplitudes,
    reconstruction,
    templates,
    pixel_size_A,
    kernel,
    support_sigma,
    step_pixels,
    minimum_separation_pixels,
):
    centers = np.asarray(centers, dtype=float).copy()
    reconstruction = np.asarray(reconstruction, dtype=float).copy()
    templates = list(templates)
    offsets = (-float(step_pixels), 0.0, float(step_pixels))
    ny, nx = image.shape
    accepted = 0
    for index in np.argsort(amplitudes)[::-1]:
        amplitude = float(amplitudes[index])
        old_template = templates[index]
        image_without_others = image - reconstruction + amplitude * old_template
        best_error = float(
            np.sum((image_without_others - amplitude * old_template) ** 2)
        )
        best_center = centers[index].copy()
        best_template = old_template
        for dy in offsets:
            for dx in offsets:
                if dx == 0.0 and dy == 0.0:
                    continue
                candidate = centers[index] + np.asarray([dy, dx])
                if not (0.0 <= candidate[0] < ny and 0.0 <= candidate[1] < nx):
                    continue
                if len(centers) > 1:
                    squared = np.sum((centers - candidate[None, :]) ** 2, axis=1)
                    squared[index] = np.inf
                    if float(np.min(squared)) < float(minimum_separation_pixels) ** 2:
                        continue
                candidate_template = _ellipse_full_normalized_template(
                    image.shape, candidate, pixel_size_A, kernel, support_sigma
                )
                error = float(
                    np.sum((image_without_others - amplitude * candidate_template) ** 2)
                )
                if error < best_error:
                    best_error = error
                    best_center = candidate
                    best_template = candidate_template
        if not np.array_equal(best_center, centers[index]):
            reconstruction += amplitude * (best_template - old_template)
            centers[index] = best_center
            templates[index] = best_template
            accepted += 1
    return centers, reconstruction, templates, int(accepted)


def _ellipse_residual_candidates(
    residual,
    existing_centers,
    pixel_size_A,
    threshold_relative,
    minimum_separation_A,
    maximum_count,
):
    from scipy.ndimage import gaussian_filter, maximum_filter

    positive = np.maximum(np.asarray(residual, dtype=float), 0.0)
    smoothed = gaussian_filter(positive, sigma=0.65, mode="nearest")
    local_maximum = maximum_filter(smoothed, size=3, mode="nearest")
    cutoff = float(threshold_relative) * float(np.max(positive))
    candidates = np.argwhere((smoothed == local_maximum) & (smoothed >= cutoff))
    candidates = sorted(candidates, key=lambda yx: smoothed[tuple(yx)], reverse=True)
    minimum_pixels = float(minimum_separation_A) / pixel_size_A
    comparison = [np.asarray(center, dtype=float) for center in existing_centers]
    accepted = []
    for y, x in candidates:
        center = np.asarray([float(y), float(x)])
        if any(np.linalg.norm(center - other) < minimum_pixels for other in comparison):
            continue
        accepted.append(center)
        comparison.append(center)
        if len(accepted) >= int(maximum_count):
            break
    return accepted, float(cutoff)


def _ellipse_group_close_centers(centers, amplitudes, merge_distance_pixels):
    centers = np.asarray(centers, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    parents = np.arange(len(centers), dtype=int)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return int(index)

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[second] = first

    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            if np.linalg.norm(centers[first] - centers[second]) < merge_distance_pixels:
                union(first, second)
    groups = {}
    for index in range(len(centers)):
        groups.setdefault(find(index), []).append(index)
    memberships = list(groups.values())
    group_amplitudes = np.asarray(
        [float(np.sum(amplitudes[members])) for members in memberships], dtype=float
    )
    group_centers = np.asarray(
        [
            np.average(centers[members], axis=0, weights=amplitudes[members])
            for members in memberships
        ],
        dtype=float,
    )
    return group_centers, group_amplitudes, memberships


def _ellipse_boolean_mask(shape, center_yx, major_A, minor_A, angle_degrees, pixel_size_A):
    yy, xx = np.indices(shape, dtype=float)
    dy_A = (yy - float(center_yx[0])) * pixel_size_A
    dx_A = (xx - float(center_yx[1])) * pixel_size_A
    angle = math.radians(float(angle_degrees))
    major_coordinate = math.cos(angle) * dx_A + math.sin(angle) * dy_A
    minor_coordinate = -math.sin(angle) * dx_A + math.cos(angle) * dy_A
    return (
        (major_coordinate / max(float(major_A), 1.0e-12)) ** 2
        + (minor_coordinate / max(float(minor_A), 1.0e-12)) ** 2
        <= 1.0
    )


def _fit_adaptive_ellipses(
    image,
    reconstruction,
    group_components,
    fallback_centers,
    pixel_size_A,
    args,
):
    scales = np.linspace(
        float(args.ellipse_axis_scale_minimum),
        float(args.ellipse_axis_scale_maximum),
        int(args.ellipse_axis_scale_steps),
    )
    angle_offsets = np.linspace(
        -float(args.ellipse_angle_search_half_range_degrees),
        float(args.ellipse_angle_search_half_range_degrees),
        int(args.ellipse_angle_search_steps),
    )
    local_radius_A = float(args.ellipse_local_search_radius_angstrom)
    threshold = float(args.ellipse_foreground_relative_threshold)
    coverage_weight = float(args.ellipse_intensity_coverage_weight)
    ellipses = []
    for group_index, (component, fallback_center) in enumerate(
        zip(group_components, fallback_centers)
    ):
        radius_pixels = int(math.ceil(local_radius_A / pixel_size_A))
        fallback_y, fallback_x = map(float, fallback_center)
        y0 = max(0, int(math.floor(fallback_y)) - radius_pixels)
        y1 = min(image.shape[0], int(math.floor(fallback_y)) + radius_pixels + 2)
        x0 = max(0, int(math.floor(fallback_x)) - radius_pixels)
        x1 = min(image.shape[1], int(math.floor(fallback_x)) + radius_pixels + 2)
        yy_local, xx_local = np.indices((y1 - y0, x1 - x0), dtype=float)
        yy, xx = yy_local + y0, xx_local + x0
        component_local = component[y0:y1, x0:x1]
        responsibility = component_local / np.maximum(
            reconstruction[y0:y1, x0:x1], 1.0e-12
        )
        attributed = image[y0:y1, x0:x1] * responsibility
        radial_A = np.sqrt(
            ((yy - fallback_y) * pixel_size_A) ** 2
            + ((xx - fallback_x) * pixel_size_A) ** 2
        )
        local = radial_A <= local_radius_A
        local_peak = float(np.max(attributed[local])) if np.any(local) else 0.0
        foreground = local & (attributed >= threshold * local_peak) & (attributed > 0.0)
        if np.count_nonzero(foreground) < 3:
            foreground = local & (component_local > 0.05 * float(np.max(component_local)))
        weights = np.where(foreground, attributed, 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            weights = np.where(local, component_local, 0.0)
            weight_sum = float(np.sum(weights))
        center_y = float(np.sum(weights * yy) / weight_sum)
        center_x = float(np.sum(weights * xx) / weight_sum)
        dy_A = (yy - center_y) * pixel_size_A
        dx_A = (xx - center_x) * pixel_size_A
        covariance = np.asarray(
            [
                [np.sum(weights * dx_A * dx_A), np.sum(weights * dx_A * dy_A)],
                [np.sum(weights * dx_A * dy_A), np.sum(weights * dy_A * dy_A)],
            ]
        ) / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(
            eigenvalues, float(args.ellipse_minimum_covariance_sigma_angstrom) ** 2
        )
        major_index = int(np.argmax(eigenvalues))
        minor_index = 1 - major_index
        base_major_A = 2.0 * math.sqrt(float(eigenvalues[major_index]))
        base_minor_A = 2.0 * math.sqrt(float(eigenvalues[minor_index]))
        major_vector = eigenvectors[:, major_index]
        base_angle = math.degrees(math.atan2(major_vector[1], major_vector[0]))
        target_intensity = float(np.sum(attributed[foreground]))
        best = None
        for major_scale in scales:
            for minor_scale in scales:
                major_A = max(base_major_A * major_scale, base_minor_A * minor_scale)
                minor_A = min(base_major_A * major_scale, base_minor_A * minor_scale)
                for angle_offset in angle_offsets:
                    angle = base_angle + angle_offset
                    mask = _ellipse_boolean_mask(
                        attributed.shape,
                        (center_y - y0, center_x - x0),
                        major_A,
                        minor_A,
                        angle,
                        pixel_size_A,
                    )
                    intersection = int(np.count_nonzero(mask & foreground))
                    union = int(np.count_nonzero(mask | foreground))
                    iou = float(intersection / max(union, 1))
                    coverage = float(np.sum(attributed[mask & foreground])) / max(
                        target_intensity, 1.0e-12
                    )
                    score = (1.0 - coverage_weight) * iou + coverage_weight * coverage
                    candidate = (
                        score,
                        iou,
                        coverage,
                        -int(np.count_nonzero(mask)),
                        major_A,
                        minor_A,
                        angle,
                    )
                    if best is None or candidate[:4] > best[:4]:
                        best = candidate
        _, iou, coverage, _, major_A, minor_A, angle = best
        ellipses.append(
            {
                "region_id": int(group_index),
                "center_y_pixel": center_y,
                "center_x_pixel": center_x,
                "semi_major_angstrom": float(major_A),
                "semi_minor_angstrom": float(minor_A),
                "angle_degrees_from_x": float(angle),
                "foreground_iou": float(iou),
                "foreground_intensity_coverage": float(coverage),
            }
        )
    return ellipses


def _calibrate_ellipse_database_scale(amplitudes, database_ratio, args):
    pairs = []
    for n_Se in range(int(args.ellipse_maximum_Se_count) + 1):
        for n_W in range(int(args.ellipse_maximum_W_count) + 1):
            if n_Se + n_W == 0 or n_Se + n_W > int(args.ellipse_maximum_total_atoms):
                continue
            pairs.append((n_Se, n_W))
    pairs = np.asarray(pairs, dtype=int)
    coefficients = pairs[:, 0] + float(database_ratio) * pairs[:, 1]
    I_Se = float(args.ellipse_initial_I_Se_for_scale_fit)
    previous = None
    history = []
    measured = np.asarray(amplitudes, dtype=float)
    for iteration in range(int(args.ellipse_scale_fit_maximum_iterations)):
        assignments = np.argmin(
            np.abs(measured[:, None] - I_Se * coefficients[None, :]), axis=1
        )
        selected = coefficients[assignments]
        I_Se_new = float(np.dot(selected, measured) / np.dot(selected, selected))
        residual = measured - I_Se_new * selected
        history.append(
            {
                "iteration": int(iteration),
                "I_Se": I_Se_new,
                "I_W": I_Se_new * float(database_ratio),
                "rmse": float(np.sqrt(np.mean(residual**2))),
            }
        )
        if previous is not None and np.array_equal(assignments, previous) and abs(I_Se_new - I_Se) < 1.0e-12:
            I_Se = I_Se_new
            break
        previous = assignments.copy()
        I_Se = I_Se_new
    return I_Se, I_Se * float(database_ratio), pairs, history


def build_experimental_log_ellipse_regions(
    source_volume,
    crop_origin_yx,
    crop_size_source_pixels,
    target_linear_volume,
    voxel_size_zyx_A,
    args,
):
    """Build fixed experimental regions with halo LoG and joint unmixing."""
    from scipy.ndimage import gaussian_laplace, maximum_filter, zoom

    source_volume = np.asarray(source_volume, dtype=float)
    target_linear_volume = np.asarray(target_linear_volume, dtype=float)
    target_ny, target_nx = target_linear_volume.shape[1:]
    if target_nx != target_ny:
        raise ValueError("Ellipse composition preprocessing requires a square XY target.")
    pixel_size_A = math.sqrt(float(voxel_size_zyx_A[1] * voxel_size_zyx_A[2]))
    crop_y0, crop_x0 = map(int, crop_origin_yx)
    crop_size = int(crop_size_source_pixels)
    # Match the validated standalone analysis exactly: the composition field is
    # fitted to the background-subtracted *raw depth sum*, independently of the
    # 3D voxel-loss normalization/transform. This preserves integrated units.
    source_crop_sum = np.sum(
        source_volume[
            :,
            crop_y0 : crop_y0 + crop_size,
            crop_x0 : crop_x0 + crop_size,
        ],
        axis=0,
    )
    if (target_ny, target_nx) != (crop_size, crop_size):
        source_crop_sum = zoom(
            source_crop_sum,
            (target_ny / crop_size, target_nx / crop_size),
            order=1,
            mode="nearest",
            prefilter=False,
        )
    depth_sum_background = float(
        np.percentile(source_crop_sum, args.target_background_percentile)
    )
    depth_sum = np.maximum(source_crop_sum - depth_sum_background, 0.0)
    halo = int(args.ellipse_detection_halo_source_pixels)
    if halo < 0:
        raise ValueError("experimental_peak_ellipses detection halo must be nonnegative.")
    halo_y0 = max(0, crop_y0 - halo)
    halo_x0 = max(0, crop_x0 - halo)
    halo_y1 = min(source_volume.shape[1], crop_y0 + crop_size + halo)
    halo_x1 = min(source_volume.shape[2], crop_x0 + crop_size + halo)
    halo_sum = np.sum(source_volume[:, halo_y0:halo_y1, halo_x0:halo_x1], axis=0)
    halo_background = float(np.percentile(halo_sum, args.target_background_percentile))
    halo_corrected = np.maximum(halo_sum - halo_background, 0.0)
    halo_scale = float(np.percentile(halo_corrected, args.target_normalization_percentile))
    if halo_scale <= 0.0:
        raise RuntimeError("Halo LoG image has no positive robust intensity scale.")
    halo_normalized = np.clip(halo_corrected / halo_scale, 0.0, 1.0)
    sigmas = np.linspace(
        float(args.log_min_sigma_pixels),
        float(args.log_max_sigma_pixels),
        int(args.log_num_sigma),
    )
    responses = np.stack(
        [-sigma**2 * gaussian_laplace(halo_normalized, sigma, mode="nearest") for sigma in sigmas]
    )
    maxima = maximum_filter(responses, size=(3, 3, 3), mode="nearest")
    cutoff = float(args.log_threshold_rel) * float(np.max(responses))
    candidates = np.argwhere((responses == maxima) & (responses >= cutoff))
    candidates = sorted(candidates, key=lambda item: responses[tuple(item)], reverse=True)
    source_voxel_A = parse_float_triplet(
        args.source_voxel_size_angstrom,
        "experimental.source_voxel_size_angstrom",
    )
    source_lateral_A = math.sqrt(float(source_voxel_A[1] * source_voxel_A[2]))
    minimum_source_pixels = float(args.log_minimum_separation_angstrom) / source_lateral_A
    halo_peaks = []
    for scale_index, y_halo, x_halo in candidates:
        if any(
            (float(y_halo) - peak[0]) ** 2 + (float(x_halo) - peak[1]) ** 2
            < minimum_source_pixels**2
            for peak in halo_peaks
        ):
            continue
        halo_peaks.append((float(y_halo), float(x_halo), int(scale_index)))
    source_to_target = (
        (target_nx - 1) / float(crop_size - 1) if crop_size > 1 else 1.0
    )
    initial_centers = []
    for y_halo, x_halo, _ in halo_peaks:
        y_source = y_halo + halo_y0
        x_source = x_halo + halo_x0
        y_target = (y_source - crop_y0) * source_to_target
        x_target = (x_source - crop_x0) * source_to_target
        if 0.0 <= y_target < target_ny and 0.0 <= x_target < target_nx:
            initial_centers.append((y_target, x_target))
    initial_centers = np.asarray(initial_centers, dtype=float)
    if not len(initial_centers):
        raise RuntimeError("Halo LoG found no candidates inside the selected crop.")

    _, element_fits = load_double_gaussian_element_fits(args.gaussian_element_fit_csv)
    per_species_kernel = []
    for symbol in ("Se", "W"):
        fit = element_fits[symbol]
        total = float(fit["total_integrated_weight"])
        per_species_kernel.append(
            {
                "core_fraction": float(fit["core_integrated_weight"]) / total,
                "tail_fraction": float(fit["tail_integrated_weight"]) / total,
                "core_sigma_A": math.sqrt(
                    float(fit["core_sigma_xy_A"]) ** 2
                    + float(args.gaussian_additional_lateral_blur_angstrom) ** 2
                ),
                "tail_sigma_A": math.sqrt(
                    float(fit["tail_sigma_xy_A"]) ** 2
                    + float(args.gaussian_additional_lateral_blur_angstrom) ** 2
                ),
            }
        )
    kernel = {
        key: float(np.mean([entry[key] for entry in per_species_kernel]))
        for key in ("core_fraction", "tail_fraction", "core_sigma_A", "tail_sigma_A")
    }
    support_sigma = float(args.ellipse_kernel_support_sigma)
    minimum_amplitude = float(args.ellipse_initial_I_Se_for_scale_fit) * float(
        args.ellipse_minimum_retained_amplitude_Se_units
    )
    centers, amplitudes, reconstruction, templates, residual_norm, pruning_history = _ellipse_prune_and_refit(
        depth_sum,
        initial_centers,
        minimum_amplitude,
        int(args.ellipse_pruning_rounds),
        pixel_size_A,
        kernel,
        support_sigma,
    )
    position_history = []
    for round_index in range(int(args.ellipse_position_refinement_rounds)):
        centers, reconstruction, templates, accepted = _ellipse_refine_positions(
            depth_sum,
            centers,
            amplitudes,
            reconstruction,
            templates,
            pixel_size_A,
            kernel,
            support_sigma,
            float(args.ellipse_position_step_pixels),
            float(args.ellipse_minimum_fitted_separation_angstrom) / pixel_size_A,
        )
        amplitudes, reconstruction, templates, residual_norm = _ellipse_fit_amplitudes(
            depth_sum, centers, pixel_size_A, kernel, support_sigma
        )
        position_history.append({"round": round_index, "accepted_moves": accepted})
    residual_history = []
    for round_index in range(int(args.ellipse_residual_addition_rounds)):
        additions, residual_cutoff = _ellipse_residual_candidates(
            depth_sum - reconstruction,
            centers,
            pixel_size_A,
            float(args.ellipse_residual_peak_threshold_rel),
            float(args.ellipse_residual_minimum_separation_angstrom),
            int(args.ellipse_maximum_residual_peaks_per_round),
        )
        residual_history.append(
            {"round": round_index, "addition_count": len(additions), "cutoff": residual_cutoff}
        )
        if not additions:
            break
        centers = np.vstack([centers, np.asarray(additions)])
        centers, amplitudes, reconstruction, templates, residual_norm, history = _ellipse_prune_and_refit(
            depth_sum,
            centers,
            minimum_amplitude,
            int(args.ellipse_pruning_rounds),
            pixel_size_A,
            kernel,
            support_sigma,
        )
        residual_history[-1]["pruning"] = history
    if residual_history:
        centers, reconstruction, templates, accepted = _ellipse_refine_positions(
            depth_sum,
            centers,
            amplitudes,
            reconstruction,
            templates,
            pixel_size_A,
            kernel,
            support_sigma,
            float(args.ellipse_position_step_pixels),
            float(args.ellipse_minimum_fitted_separation_angstrom) / pixel_size_A,
        )
        amplitudes, reconstruction, templates, residual_norm = _ellipse_fit_amplitudes(
            depth_sum, centers, pixel_size_A, kernel, support_sigma
        )
        position_history.append(
            {"round": "post_residual", "accepted_moves": accepted}
        )
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    centers = centers[order]
    amplitudes, reconstruction, templates, residual_norm = _ellipse_fit_amplitudes(
        depth_sum, centers, pixel_size_A, kernel, support_sigma
    )
    visible_fractions = np.asarray(
        [np.sum(template) * pixel_size_A**2 for template in templates], dtype=float
    )
    group_centers, group_amplitudes, memberships = _ellipse_group_close_centers(
        centers,
        amplitudes,
        float(args.ellipse_merge_distance_angstrom) / pixel_size_A,
    )
    group_order = np.lexsort((group_centers[:, 0], group_centers[:, 1]))
    group_centers = group_centers[group_order]
    group_amplitudes = group_amplitudes[group_order]
    memberships = [memberships[index] for index in group_order]
    kernel_group_index = np.empty(len(centers), dtype=int)
    for group_index, members in enumerate(memberships):
        kernel_group_index[members] = group_index
    group_components = np.zeros((len(group_centers), *depth_sum.shape), dtype=float)
    for kernel_index, (template, amplitude) in enumerate(zip(templates, amplitudes)):
        group_components[kernel_group_index[kernel_index]] += amplitude * template
    group_visible_fractions = np.asarray(
        [np.average(visible_fractions[m], weights=amplitudes[m]) for m in memberships],
        dtype=float,
    )
    ellipses = _fit_adaptive_ellipses(
        depth_sum,
        reconstruction,
        group_components,
        group_centers,
        pixel_size_A,
        args,
    )
    database_Se = float(element_fits["Se"]["total_integrated_weight"])
    database_W = float(element_fits["W"]["total_integrated_weight"])
    I_Se, I_W, pairs, scale_history = _calibrate_ellipse_database_scale(
        group_amplitudes, database_W / database_Se, args
    )
    centers_xy_A = np.asarray(
        [[ellipse["center_x_pixel"] * voxel_size_zyx_A[2], ellipse["center_y_pixel"] * voxel_size_zyx_A[1]] for ellipse in ellipses],
        dtype=float,
    )
    for index, ellipse in enumerate(ellipses):
        ellipse["target_full_fitted_integrated_intensity"] = float(group_amplitudes[index])
        ellipse["target_intensity_Se_units"] = float(group_amplitudes[index] / I_Se)
        ellipse["full_kernel_visible_fraction"] = float(group_visible_fractions[index])
        ellipse["edge_low_confidence"] = bool(
            group_visible_fractions[index]
            < float(args.ellipse_low_confidence_visible_fraction)
        )
    residual = depth_sum - reconstruction
    image_r2 = float(
        1.0 - np.sum(residual**2) / np.sum((depth_sum - depth_sum.mean()) ** 2)
    )
    return {
        "depth_sum": depth_sum.astype(np.float32),
        "reconstruction": reconstruction.astype(np.float32),
        "residual": residual.astype(np.float32),
        "initial_centers_yx_pixels": initial_centers,
        "fitted_centers_yx_pixels": centers,
        "group_centers_yx_pixels": group_centers,
        "centers_xy_A": centers_xy_A,
        "semi_major_A": np.asarray([item["semi_major_angstrom"] for item in ellipses]),
        "semi_minor_A": np.asarray([item["semi_minor_angstrom"] for item in ellipses]),
        "angles_degrees": np.asarray([item["angle_degrees_from_x"] for item in ellipses]),
        "target_integrals": group_amplitudes.astype(np.float32),
        "I_Se": float(I_Se),
        "I_W": float(I_W),
        "database_Se_weight": database_Se,
        "database_W_weight": database_W,
        "database_W_to_Se_ratio": database_W / database_Se,
        "membership_boundary_softness": float(
            args.ellipse_membership_boundary_softness
        ),
        "membership_background_score": float(
            args.ellipse_membership_background_score
        ),
        "candidate_pairs_Se_W": pairs,
        "ellipses": ellipses,
        "group_memberships": memberships,
        "group_visible_fractions": group_visible_fractions,
        "kernel": kernel,
        "halo_normalized_depth_sum": halo_normalized.astype(np.float32),
        "halo_candidate_centers_yx_pixels": np.asarray(
            [[peak[0], peak[1]] for peak in halo_peaks], dtype=float
        ),
        "halo": {
            "requested_source_pixels_each_side": halo,
            "shape_yx": list(halo_normalized.shape),
            "origin_yx_source_pixels": [halo_y0, halo_x0],
            "crop_bounds_yx_in_halo_pixels": [
                crop_y0 - halo_y0,
                crop_x0 - halo_x0,
                crop_y0 - halo_y0 + crop_size,
                crop_x0 - halo_x0 + crop_size,
            ],
            "candidate_count_before_crop_back": int(len(halo_peaks)),
            "candidate_count_after_crop_back": int(len(initial_centers)),
        },
        "fit": {
            "subkernel_count": int(len(centers)),
            "group_count": int(len(group_centers)),
            "image_r_squared": image_r2,
            "residual_l2": residual_norm,
            "pruning_history": pruning_history,
            "position_history": position_history,
            "residual_addition_history": residual_history,
            "scale_fit_history": scale_history,
        },
    }


def write_log_ellipse_diagnostics(regions, voxel_size_zyx_A, output_dir):
    """Write global diagnostics for halo-LoG, unmixing, and adaptive ellipses."""
    matplotlib_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "eat_matplotlib_cache"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    diagnostics_dir = Path(output_dir) / "initial_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(regions["depth_sum"], dtype=float)
    reconstruction = np.asarray(regions["reconstruction"], dtype=float)
    residual = image - reconstruction
    vmax = max(float(np.percentile(image, 99.8)), 1.0e-12)
    residual_limit = max(float(np.percentile(np.abs(residual), 99.5)), 1.0e-12)
    dy_A, dx_A = map(float, voxel_size_zyx_A[1:])
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), constrained_layout=True)
    axes[0, 0].imshow(image, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
    axes[0, 0].scatter(
        regions["initial_centers_yx_pixels"][:, 1],
        regions["initial_centers_yx_pixels"][:, 0],
        s=10,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.45,
    )
    axes[0, 0].set_title(
        f"Halo-LoG candidates mapped into crop ({len(regions['initial_centers_yx_pixels'])})"
    )
    axes[0, 1].imshow(reconstruction, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
    axes[0, 1].set_title(
        f"Joint double-Gaussian reconstruction, R²={regions['fit']['image_r_squared']:.4f}"
    )
    shown = axes[1, 0].imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    fig.colorbar(shown, ax=axes[1, 0], fraction=0.045)
    axes[1, 0].set_title("Experimental - joint reconstruction")
    axes[1, 1].imshow(image, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
    records = []
    for index, ellipse in enumerate(regions["ellipses"]):
        color = "magenta" if ellipse["edge_low_confidence"] else "lime"
        axes[1, 1].add_patch(
            Ellipse(
                (ellipse["center_x_pixel"], ellipse["center_y_pixel"]),
                width=2.0 * ellipse["semi_major_angstrom"] / dx_A,
                height=2.0 * ellipse["semi_minor_angstrom"] / dy_A,
                angle=ellipse["angle_degrees_from_x"],
                fill=False,
                edgecolor=color,
                linestyle="--" if ellipse["edge_low_confidence"] else "-",
                linewidth=0.8,
            )
        )
        label = f"{index}\n{ellipse['target_intensity_Se_units']:.2f} Se-u"
        axes[1, 1].text(
            ellipse["center_x_pixel"],
            ellipse["center_y_pixel"],
            label,
            color="white",
            fontsize=4,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.2, "edgecolor": "none"},
        )
        records.append(dict(ellipse))
    axes[1, 1].set_title(
        f"Adaptive occupancy regions ({len(records)}); dashed magenta=edge-low-confidence"
    )
    for axis in axes.flat:
        axis.set_xlabel("x pixel in experimental crop")
        axis.set_ylabel("y pixel in experimental crop")
    fig.suptitle(
        "Experimental halo-LoG / joint-unmixing ellipse composition field\n"
        f"I_Se={regions['I_Se']:.8g}, I_W={regions['I_W']:.8g}, "
        f"W/Se={regions['I_W']/regions['I_Se']:.8g} (database fixed)"
    )
    fig.savefig(
        diagnostics_dir / "experimental_log_joint_fit_adaptive_ellipses.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    halo_image = np.asarray(regions["halo_normalized_depth_sum"], dtype=float)
    halo_centers = np.asarray(regions["halo_candidate_centers_yx_pixels"], dtype=float)
    crop_y0, crop_x0, crop_y1, crop_x1 = regions["halo"][
        "crop_bounds_yx_in_halo_pixels"
    ]
    fig, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.imshow(halo_image, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    axis.scatter(
        halo_centers[:, 1], halo_centers[:, 0], s=9, facecolors="none",
        edgecolors="cyan", linewidths=0.4,
    )
    axis.add_patch(
        Rectangle(
            (crop_x0, crop_y0), crop_x1 - crop_x0, crop_y1 - crop_y0,
            fill=False, edgecolor="yellow", linewidth=1.2,
        )
    )
    axis.set_title(
        f"LoG detection on {halo_image.shape[1]}x{halo_image.shape[0]} source halo\n"
        "yellow box is the retained analysis crop"
    )
    axis.set_xlabel("x pixel in halo")
    axis.set_ylabel("y pixel in halo")
    fig.savefig(
        diagnostics_dir / "experimental_log_detection_halo.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    yy, xx = np.indices(image.shape, dtype=float)
    centers_xy_A = np.asarray(regions["centers_xy_A"], dtype=float)
    dx_grid = xx[:, :, None] * dx_A - centers_xy_A[None, None, :, 0]
    dy_grid = yy[:, :, None] * dy_A - centers_xy_A[None, None, :, 1]
    angle = np.deg2rad(np.asarray(regions["angles_degrees"], dtype=float))
    major_coordinate = np.cos(angle)[None, None, :] * dx_grid + np.sin(angle)[None, None, :] * dy_grid
    minor_coordinate = -np.sin(angle)[None, None, :] * dx_grid + np.cos(angle)[None, None, :] * dy_grid
    distance_squared = (
        major_coordinate / np.asarray(regions["semi_major_A"])[None, None, :]
    ) ** 2 + (
        minor_coordinate / np.asarray(regions["semi_minor_A"])[None, None, :]
    ) ** 2
    raw = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (distance_squared - 1.0)
                / float(regions["membership_boundary_softness"]),
                -80.0,
                80.0,
            )
        )
    )
    denominator = raw.sum(axis=2) + float(regions["membership_background_score"])
    membership_sum = raw.sum(axis=2) / denominator
    background_membership = float(regions["membership_background_score"]) / denominator
    owner = np.argmax(raw, axis=2).astype(float)
    owner[np.max(raw / denominator[:, :, None], axis=2) <= background_membership] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(image, origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
    axes[0].set_title("Experimental depth sum")
    axes[1].imshow(owner, origin="lower", cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title("Dominant smooth ellipse owner (background is blank)")
    shown = axes[2].imshow(membership_sum, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(shown, ax=axes[2], fraction=0.045)
    axes[2].set_title("Sum of regional q values (always <= 1)")
    for axis in axes:
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
    fig.savefig(
        diagnostics_dir / "experimental_ellipse_membership_field.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    with (diagnostics_dir / "experimental_log_joint_fit_adaptive_ellipses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        diagnostics_dir / "experimental_log_joint_fit_adaptive_ellipses.npz",
        experimental_depth_sum=image.astype(np.float32),
        reconstruction=reconstruction.astype(np.float32),
        residual=residual.astype(np.float32),
        initial_centers_yx_pixels=regions["initial_centers_yx_pixels"],
        fitted_centers_yx_pixels=regions["fitted_centers_yx_pixels"],
        group_centers_yx_pixels=regions["group_centers_yx_pixels"],
        centers_xy_A=regions["centers_xy_A"],
        semi_major_A=regions["semi_major_A"],
        semi_minor_A=regions["semi_minor_A"],
        angles_degrees=regions["angles_degrees"],
        target_integrals=regions["target_integrals"],
        I_Se=np.asarray(regions["I_Se"]),
        I_W=np.asarray(regions["I_W"]),
        group_visible_fractions=regions["group_visible_fractions"],
        halo_normalized_depth_sum=halo_image.astype(np.float32),
        halo_candidate_centers_yx_pixels=halo_centers,
    )
    summary = {
        key: value
        for key, value in regions.items()
        if key
        not in {
            "depth_sum",
            "reconstruction",
            "residual",
            "initial_centers_yx_pixels",
            "fitted_centers_yx_pixels",
            "group_centers_yx_pixels",
            "halo_normalized_depth_sum",
            "halo_candidate_centers_yx_pixels",
        }
    }
    (diagnostics_dir / "experimental_log_joint_fit_adaptive_ellipses.json").write_text(
        json.dumps(json_safe_value(summary), indent=2) + "\n", encoding="utf-8"
    )


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
    # Empty legacy metadata keeps old serializers readable; no detection or loss.
    log_ellipse_regions = {
        "ellipses": [], "centers_xy_A": np.empty((0, 2)),
        "semi_major_A": np.empty(0), "semi_minor_A": np.empty(0),
        "angles_degrees": np.empty(0), "target_integrals": np.empty(0),
        "halo": {}, "I_Se": 0.5824533131834269, "I_W": 1.0,
        "database_W_to_Se_ratio": 1.0 / 0.5824533131834269,
    }

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
    # Include vacuum in the image loss, but exclude it from candidate seeding.
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
        "log_ellipse_regions": log_ellipse_regions,
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
            "ellipse_composition_loss": {"removed": True, "region_count": 0},
            "candidate_volume_excludes_vacuum": True,
            "exact_composition_enforcement": False,
        }
        (diagnostics_dir / "geometry_and_mask_summary.json").write_text(
            json.dumps(geometry_summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


class WSe2VolumeProblem:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lightweight_slurm_driver = (
            str(getattr(args, "ga_execution_backend", "local")).strip().lower() == "slurm"
            and not bool(getattr(args, "slurm_worker_mode", False))
        )
        self.device = torch.device("cpu" if self.lightweight_slurm_driver else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float32

        write_diagnostics = (
            mp.current_process().name == "MainProcess"
            and not bool(getattr(args, "slurm_worker_mode", False))
            and not bool(getattr(args, "ga_resuming_from_checkpoint", False))
        )
        self.experimental = prepare_experimental_volume(
            args,
            self.output_dir,
            write_diagnostics=write_diagnostics,
        )
        self.blob_path = self.experimental["tiff_path"]
        self.blob_data = {}
        self.original_cell = np.asarray(self.experimental["sample_cell"], dtype=float)
        self.cell = np.asarray(self.experimental["cell"], dtype=float)
        self.requested_vacuum_padding = np.asarray(
            self.experimental["requested_vacuum_padding_A"], dtype=float
        )
        self.vacuum_padding = np.asarray(
            self.experimental["vacuum_padding_A"], dtype=float
        )
        self.vacuum_position_shift = np.asarray(
            self.experimental["vacuum_position_shift_A"], dtype=float
        )
        self.pbc = np.asarray(self.experimental["pbc"], dtype=bool)
        if not np.all(self.pbc):
            raise ValueError("WSe2 FairChem requires pbc=true,true,true.")
        self.diversity_subcells_xyz = parse_positive_int_triplet(
            args.ga_diversity_subcells,
            "genetic.diversity_subcells",
        )
        # Score-tournament parents are crossed in their stored common cell;
        # subcell-diversity translation alignment is intentionally disabled.
        self.periodic_z_parent_alignment = False

        cell_lengths_A = np.diag(self.cell).astype(float)
        self.blobs = [
            {
                "symbol": "WSe2Site",
                "center_angstrom": (0.5 * cell_lengths_A).tolist(),
                "sigma_angstrom": cell_lengths_A.tolist(),
                "allowed_elements": ["W", "Se"],
                "source_type": "wse2_feasible_site_class",
            },
        ]
        self.blob_symbols = ["WSe2Site"]
        self.blob_allowed_elements_from_json = [["W", "Se"]]
        self.composition_source_symbols = ["W", "Se", "Se"]
        species_text = str(args.species or "W, Se")
        self.species = split_species_list(species_text)
        if set(self.species) != {"W", "Se"} or len(self.species) != 2:
            raise ValueError(
                "The WSe2 imaging workflow requires system.species to contain "
                "exactly W and Se; null is represented implicitly."
            )
        self.atomic_numbers = [atomic_numbers[symbol] for symbol in self.species]
        self.n_species = len(self.species)
        # Keep the former heavy aliases internally where they represent all
        # real occupancy; there is only one W/Se/null site class.
        self.heavy_species_indices = np.arange(self.n_species, dtype=int)
        self.effective_species_indices = self.heavy_species_indices.copy()
        self.heavy_site_class_index = 0
        self.effective_site_class_index = 0
        self.blob_allowed_species_mask = np.ones((1, self.n_species), dtype=bool)
        self.blob_source_types = ["wse2_feasible_site_class"]
        self.species_by_source_type = {}
        self.tsallis_q = float(args.tsallis_q)
        if self.tsallis_q <= 1.0:
            raise ValueError("--tsallis-q must be > 1.")

        self.heavy_site_density_per_angstrom3 = float(
            args.heavy_site_density_per_angstrom3
        )
        # Zero-valued legacy fields keep historical serializers readable; no
        # oxygen class or oxygen-only sites exist in this workflow.
        self.oxygen_site_density_per_angstrom3 = 0.0
        self.initial_oxygen_cell_sites = 0
        self.initial_oxygen_min_distance_A = 0.0
        self.oxygen_site_class_index = -1
        if self.heavy_site_density_per_angstrom3 <= 0.0:
            raise ValueError(
                "effective_atoms.candidate_site_density_per_angstrom3 must be positive."
            )
        feasible_voxel_volume_A3 = float(
            np.prod(
                np.asarray(
                    self.experimental["target_voxel_size_A"],
                    dtype=float,
                )
            )
        )
        self.feasible_volume_angstrom3 = float(
            np.count_nonzero(self.experimental["feasible_mask"])
            * feasible_voxel_volume_A3
        )
        if self.feasible_volume_angstrom3 <= 0.0:
            raise ValueError("The thresholded feasible-region volume is empty.")
        self.cell_volume_angstrom3 = float(abs(np.linalg.det(self.cell)))
        self.sample_volume_angstrom3 = float(abs(np.linalg.det(self.original_cell)))
        self.initial_heavy_region_sites = max(
            1,
            int(
                round(
                    self.heavy_site_density_per_angstrom3
                    * self.feasible_volume_angstrom3
                )
            ),
        )
        self.initial_n_sites = int(self.initial_heavy_region_sites)
        self.initial_cylinder_lateral_radius_A = float(
            args.initial_cylinder_lateral_radius_angstrom
        )
        self.initial_cylinder_z_radius_A = float(
            args.initial_cylinder_z_radius_angstrom
        )
        self.initial_jitter_A = float(args.initial_jitter_angstrom)
        if self.initial_cylinder_lateral_radius_A < 0.0:
            raise ValueError(
                "effective_atoms.initial_cylinder_lateral_radius_angstrom "
                "must be nonnegative."
            )
        if self.initial_cylinder_z_radius_A < 0.0:
            raise ValueError(
                "effective_atoms.initial_cylinder_z_radius_angstrom must be nonnegative."
            )
        if self.initial_jitter_A < 0.0:
            raise ValueError(
                "effective_atoms.initial_jitter_angstrom must be nonnegative."
            )
        self.blob_species_fractions = np.full(
            (1, self.n_species), 1.0 / self.n_species, dtype=float
        )
        self.ref_positions_blobs = np.asarray([0.5 * cell_lengths_A], dtype=float)
        self.ref_blob_sigmas = np.asarray([cell_lengths_A], dtype=float)
        self.initial_site_region_indices = np.full(
            self.initial_heavy_region_sites,
            self.effective_site_class_index,
            dtype=int,
        )
        self.set_site_layout(self.initial_site_region_indices)

        self.boundary_penalty = float(args.level_set_boundary_penalty)
        self.boundary_sharpness = float(args.level_set_boundary_sharpness)
        if self.boundary_penalty < 0.0:
            raise ValueError("heavy_region.boundary_penalty must be nonnegative.")
        if self.boundary_sharpness <= 0.0:
            raise ValueError("heavy_region.boundary_sharpness must be positive.")
        self.level_set_margin_A = float(args.level_set_margin_angstrom)

        self.sample_box_confinement_penalty = float(
            args.sample_box_confinement_penalty
        )
        self.sample_box_confinement_sharpness = float(
            args.sample_box_confinement_sharpness
        )
        self.sample_box_confinement_margin_A = float(
            args.sample_box_confinement_margin_angstrom
        )
        self.sample_box_lower_A, self.sample_box_upper_A = atomic_box_bounds(
            args, np.diag(self.original_cell), self.vacuum_position_shift
        )
        if self.sample_box_confinement_penalty < 0.0:
            raise ValueError("sample_box.confinement_penalty must be nonnegative.")
        if self.sample_box_confinement_sharpness <= 0.0:
            raise ValueError("sample_box.confinement_sharpness must be positive.")
        if self.sample_box_confinement_margin_A < 0.0:
            raise ValueError(
                "sample_box.confinement_margin_angstrom must be nonnegative."
            )
        if np.any(
            2.0 * self.sample_box_confinement_margin_A
            >= np.diag(self.original_cell)
        ):
            raise ValueError(
                "sample_box.confinement_margin_angstrom leaves an empty sample box."
            )

        self.reference_symbols = []
        self.reference_positions = np.zeros((0, 3), dtype=float)
        self.composition_target = parse_composition_target(args.composition_target, self.species, self.composition_source_symbols)
        self.composition_penalty = float(args.composition_penalty)
        if self.composition_penalty < 0.0:
            raise ValueError("--composition-penalty must be non-negative.")
        if bool(args.exact_composition_enforcement_enable):
            raise ValueError(
                "WSe2 requires composition.exact_composition_enforcement_enable=false; "
                "the 1:2 ratio is a soft guide only."
            )
        self.se_vacancy_bias_enable = bool(args.se_vacancy_bias_enable)
        self.se_vacancy_bias_alpha_eV = float(args.se_vacancy_bias_alpha_eV)
        self.se_vacancy_reference_atom_count = float(
            args.se_vacancy_reference_atom_count
        )
        if self.se_vacancy_bias_alpha_eV < 0.0:
            raise ValueError(
                "se_vacancy_bias.alpha_eV_per_vacancy must be nonnegative."
            )
        if self.se_vacancy_reference_atom_count <= 0.0:
            raise ValueError(
                "se_vacancy_bias.reference_atom_count must be positive."
            )
        if "Se" not in self.species:
            raise ValueError("The Se-vacancy bias requires Se in system.species.")
        self.se_species_index = int(self.species.index("Se"))
        self.se_vacancy_reference_se_count = float(
            self.se_vacancy_reference_atom_count
            * self.composition_target[self.se_species_index]
        )
        self.has_compound_reference_atoms = len(self.reference_symbols) > 0
        ref_counts = np.asarray([sum(1 for symbol in self.reference_symbols if symbol == species_name) for species_name in self.species], dtype=float)
        if float(ref_counts.sum()) > 0.0:
            self.compound_reference_composition = ref_counts / ref_counts.sum()
        else:
            self.compound_reference_composition = np.asarray(self.composition_target, dtype=float)
        self.element_reference_metadata = []
        self.model_evaluation_tracker = ModelEvaluationTracker()
        self.objective_evaluation_count = 0
        if self.lightweight_slurm_driver:
            self.calc = None
            self.E_ref_total = float("nan")
            self.E_ref_per_atom = float("nan")
            self.element_ref_energies = np.full(len(self.species), np.nan, dtype=float)
            self.compound_reference_formation_energy_per_atom = float("nan")
        else:
            disable_optional_torch_native_bmm_override()
            predictor = pretrained_mlip.get_predict_unit(args.model_name)
            self.calc = instrument_calculator(
                FAIRChemCalculator(predictor, task_name=args.task_name),
                self.model_evaluation_tracker,
            )
            element_ref_energies = []
            for symbol in self.species:
                ref = fetch_elemental_reference_atoms(symbol, args.mp_api_key)
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
                    "cell_angstrom": ref["atoms"].cell.array.tolist(),
                })
            self.element_ref_energies = np.asarray(element_ref_energies, dtype=float)
            if self.has_compound_reference_atoms:
                ref_atoms = Atoms(symbols=self.reference_symbols, positions=self.reference_positions, cell=self.cell, pbc=self.pbc)
                ref_atoms.wrap()
                ref_atoms.calc = self.calc
                self.E_ref_total = float(ref_atoms.get_potential_energy())
                self.E_ref_per_atom = self.E_ref_total / len(ref_atoms)
                self.compound_reference_formation_energy_per_atom = (
                    self.E_ref_total - float(np.dot(ref_counts, self.element_ref_energies))
                ) / float(ref_counts.sum())
            else:
                self.E_ref_total = float("nan")
                self.E_ref_per_atom = float("nan")
                self.compound_reference_formation_energy_per_atom = float("nan")

        self.target_shape_zyx = tuple(self.experimental["target_shape_zyx"])
        self.target_voxel_size_A = np.asarray(
            self.experimental["target_voxel_size_A"],
            dtype=float,
        )
        self.target_image_t = torch.tensor(
            self.experimental["target_image"],
            device=self.device,
            dtype=self.dtype,
        )
        self.image_loss_mask_t = torch.tensor(
            self.experimental["image_loss_mask"],
            device=self.device,
            dtype=torch.bool,
        )
        self.image_loss_voxel_count = int(self.image_loss_mask_t.sum().item())
        expected_loss_voxels = int(np.prod(self.experimental["sample_shape_zyx"]))
        if self.image_loss_voxel_count != expected_loss_voxels:
            raise RuntimeError(
                "The padded image-loss mask does not contain exactly the measured "
                f"sample voxels: {self.image_loss_voxel_count} != {expected_loss_voxels}."
            )
        ellipse_regions = self.experimental["log_ellipse_regions"]
        ellipse_centers_xy_A = np.asarray(
            ellipse_regions["centers_xy_A"], dtype=float
        ) + self.sample_box_lower_A[None, :2]
        self.ellipse_centers_xy_A_t = torch.tensor(
            ellipse_centers_xy_A,
            device=self.device,
            dtype=self.dtype,
        )
        self.ellipse_semi_major_A_t = torch.tensor(
            ellipse_regions["semi_major_A"], device=self.device, dtype=self.dtype
        )
        self.ellipse_semi_minor_A_t = torch.tensor(
            ellipse_regions["semi_minor_A"], device=self.device, dtype=self.dtype
        )
        self.ellipse_angles_radians_t = torch.tensor(
            np.deg2rad(ellipse_regions["angles_degrees"]),
            device=self.device,
            dtype=self.dtype,
        )
        self.ellipse_target_integrals_t = torch.tensor(
            ellipse_regions["target_integrals"], device=self.device, dtype=self.dtype
        )
        self.ellipse_region_count = int(self.ellipse_target_integrals_t.shape[0])
        self.ellipse_I_Se = float(ellipse_regions["I_Se"])
        self.ellipse_I_W = float(ellipse_regions["I_W"])
        self.ellipse_species_intensities_t = torch.tensor(
            [
                self.ellipse_I_W if symbol == "W" else self.ellipse_I_Se
                for symbol in self.species
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.ellipse_composition_penalty = 0.0
        self.ellipse_membership_boundary_softness = float(
            args.ellipse_membership_boundary_softness
        )
        self.ellipse_membership_background_score = float(
            args.ellipse_membership_background_score
        )
        self.sample_roi_start_zyx = tuple(
            int(value) for value in self.experimental["sample_roi_start_zyx"]
        )
        self.sample_roi_stop_zyx = tuple(
            int(value) for value in self.experimental["sample_roi_stop_zyx"]
        )
        self.level_set_coefficients_t = torch.tensor(
            periodic_cubic_bspline_coefficients(self.experimental["level_set_A"]),
            device=self.device,
            dtype=self.dtype,
        )
        self.image_penalty = float(args.image_penalty)
        self.image_huber_delta = float(args.image_huber_delta)
        self.image_target_transform = normalize_image_transform_name(
            args.image_target_transform
        )
        self.gaussian_epsilon = float(args.gaussian_epsilon)
        self.gaussian_transform_epsilon = float(args.gaussian_transform_epsilon)
        self.gaussian_additional_lateral_blur_A = float(
            args.gaussian_additional_lateral_blur_angstrom
        )
        self.gaussian_element_fit_csv, element_fits = load_double_gaussian_element_fits(
            args.gaussian_element_fit_csv
        )
        self.gaussian_element_fits = element_fits
        missing_species = [symbol for symbol in self.species if symbol not in element_fits]
        if missing_species:
            raise ValueError(
                "Double-Gaussian element fit CSV does not contain configured species: "
                f"{missing_species}"
            )
        self.gaussian_relative_intensity_reference_species = str(
            args.gaussian_relative_intensity_reference_species
        ).strip()
        if self.gaussian_relative_intensity_reference_species not in element_fits:
            raise ValueError(
                "imaging.relative_intensity_reference_species is absent from the "
                f"element fit CSV: {self.gaussian_relative_intensity_reference_species!r}"
            )
        self.gaussian_normalize_relative_reference_peak_to_one = bool(
            args.gaussian_normalize_relative_intensity_reference_peak_to_one
        )
        raw_component_amplitudes = np.asarray(
            [
                [
                    element_fits[symbol]["core_integrated_weight"],
                    element_fits[symbol]["tail_integrated_weight"],
                ]
                for symbol in self.species
            ],
            dtype=float,
        )
        self.gaussian_fitted_sigma_xy_A = np.asarray(
            [
                [
                    element_fits[symbol]["core_sigma_xy_A"],
                    element_fits[symbol]["tail_sigma_xy_A"],
                ]
                for symbol in self.species
            ],
            dtype=float,
        )
        if (
            not np.isfinite(self.gaussian_additional_lateral_blur_A)
            or self.gaussian_additional_lateral_blur_A < 0.0
        ):
            raise ValueError(
                "imaging.additional_lateral_blur_angstrom must be finite and nonnegative."
            )
        self.gaussian_sigma_xy_A = np.sqrt(
            self.gaussian_fitted_sigma_xy_A**2
            + self.gaussian_additional_lateral_blur_A**2
        )
        if self.image_penalty < 0.0 or self.image_huber_delta <= 0.0:
            raise ValueError("Imaging penalty must be nonnegative and Huber delta positive.")
        if self.ellipse_composition_penalty < 0.0:
            raise ValueError(
                "The ellipse-composition penalty must be nonnegative."
            )
        if self.ellipse_membership_boundary_softness <= 0.0:
            raise ValueError("Ellipse membership boundary softness must be positive.")
        if self.ellipse_membership_background_score <= 0.0:
            raise ValueError("Ellipse membership background score must be positive.")
        if self.gaussian_epsilon <= 0.0:
            raise ValueError("imaging.epsilon must be positive.")
        if (
            not np.isfinite(self.gaussian_transform_epsilon)
            or self.gaussian_transform_epsilon <= 0.0
        ):
            raise ValueError("imaging.transform_epsilon must be finite and positive.")
        if np.any(~np.isfinite(self.gaussian_sigma_xy_A)) or np.any(
            self.gaussian_sigma_xy_A <= 0.0
        ):
            raise ValueError("All Gaussian lateral sigma values must be finite and positive.")
        initialize_gaussian_renderer(self)
        if str(args.ga_score).strip().lower() != "guided":
            raise ValueError(
                "This workflow requires genetic.score=guided so Gaussian image, level-set, "
                "and composition terms all rank GA members."
            )
        self._last_aux_cache = None
        self._objective_result_cache = None

        self.history = []
        self.stage_summaries = []
        self.callback_counter = 0
        self.last_heartbeat_time = 0.0
        self.stage_wall_start = None
        self.stage_last_callback_time = None
        self.active_member_checkpoint_context = None
        self._member_checkpoint_last_phase = None
        self._member_checkpoint_last_iteration = None
        self._member_checkpoint_last_time = 0.0
        self.exact_composition_projection_summary = None

    def ensure_calculator(self):
        if self.calc is None:
            disable_optional_torch_native_bmm_override()
            predictor = pretrained_mlip.get_predict_unit(self.args.model_name)
            self.calc = instrument_calculator(
                FAIRChemCalculator(predictor, task_name=self.args.task_name),
                self.model_evaluation_tracker,
            )

    def release_calculator(self):
        self.calc = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def site_region_count(self):
        return int(len(self.blobs))

    def set_site_layout(self, assigned_site_region_indices):
        regions = np.asarray(assigned_site_region_indices, dtype=int).reshape(-1)
        if regions.size <= 0:
            raise ValueError("Candidate site layout must contain at least one ghost site.")
        if np.any(regions < 0) or np.any(regions >= self.site_region_count()):
            raise ValueError("Candidate site layout contains invalid site-class indices.")
        self.assigned_site_region_indices = regions.copy()
        self.assigned_blob_indices = regions.copy()
        self.assigned_blob_symbols = [self.blob_symbols[i] for i in self.assigned_blob_indices]
        self.assigned_blob_species_fractions = self.blob_species_fractions[self.assigned_blob_indices]
        self.assigned_allowed_species_mask = self.blob_allowed_species_mask[self.assigned_blob_indices]
        self.n_sites = int(len(self.assigned_blob_indices))
        self.ref_positions = self.ref_positions_blobs[self.assigned_blob_indices]
        self.blob_sigma = self.ref_blob_sigmas[self.assigned_blob_indices]

    def initial_site_layout(self):
        return self.initial_site_region_indices.copy()

    def reset_run_records(self):
        self.history = []
        self.stage_summaries = []
        self.callback_counter = 0
        self.stage_wall_start = None
        self.stage_last_callback_time = None
        self._member_checkpoint_last_phase = None
        self._member_checkpoint_last_iteration = None
        self._member_checkpoint_last_time = 0.0
        self.exact_composition_projection_summary = None
        self._last_aux_cache = None
        self._objective_result_cache = None

    def pack_y(self, occ, z):
        return torch.cat([occ.reshape(-1), z.reshape(-1)])

    def unpack_y(self, y):
        n_occ = self.n_sites * self.n_species
        occ = y[:n_occ].reshape(self.n_sites, self.n_species)
        z = y[n_occ:].reshape(self.n_sites, 3)
        return occ, z

    def project(self, y):
        occ, z = self.unpack_y(y)
        # Keep the PBFGS position state continuous and unwrapped.  Periodicity
        # belongs in physical evaluations (energy, imaging, level set, and output),
        # not in this optimizer projection: a modulo discontinuity here makes
        # crossing a cell boundary look like an O(cell length) optimizer step.
        return self.pack_y(self.project_allowed_occupancies(occ), z)

    def project_allowed_occupancies(self, occ):
        allowed = torch.as_tensor(self.assigned_allowed_species_mask, device=occ.device, dtype=occ.dtype)
        return project_capped_simplex(occ * allowed)

    def zero_forbidden_grad(self, grad_occ):
        allowed = torch.as_tensor(self.assigned_allowed_species_mask, device=grad_occ.device, dtype=grad_occ.dtype)
        return grad_occ * allowed

    def z_to_positions(self, z_np):
        lengths = np.diag(self.cell)
        return np.mod(np.asarray(z_np, dtype=float), lengths[None, :])

    def positions_to_frac(self, positions):
        return np.linalg.solve(self.cell.T, np.asarray(positions, dtype=float).T).T

    def assigned_blob_penalty_term(self, occ_np, z_np):
        positions_t = torch.tensor(
            self.z_to_positions(z_np),
            device=self.device,
            dtype=self.dtype,
        )
        with torch.no_grad():
            level_set = periodic_cubic_bspline_sample_zyx(
                self.level_set_coefficients_t,
                positions_t,
                self.target_voxel_size_A,
            )
            violation = torch.nn.functional.softplus(
                self.boundary_sharpness * (self.level_set_margin_A - level_set)
            ) / self.boundary_sharpness
        level_set_np = level_set.cpu().numpy()
        violation_np = violation.cpu().numpy()
        heavy_occ = np.asarray(occ_np, dtype=float)[
            :, self.heavy_species_indices
        ].sum(axis=1)
        per_site = violation_np * violation_np
        heavy_occupancy_sum = max(float(np.sum(heavy_occ)), 1e-8)
        objective = float(
            self.boundary_penalty
            * np.dot(heavy_occ, per_site)
            / heavy_occupancy_sum
        )
        inside = level_set_np >= self.level_set_margin_A
        return {
            "objective_term": objective,
            "grad_occ": np.zeros_like(occ_np, dtype=float),
            "grad_z": np.zeros_like(z_np, dtype=float),
            "level_set_A": level_set_np,
            "inside": inside,
            "inside_count": int(np.count_nonzero(inside)),
        }

    def composition_penalty_term(self, occ_np):
        occ_np = np.asarray(occ_np, dtype=float)
        species_totals = occ_np.sum(axis=0)
        total_real = float(species_totals.sum())
        target = np.asarray(self.composition_target, dtype=float)
        if total_real <= 1e-12:
            fraction = np.zeros_like(target)
            diff = fraction - target
            return {
                "objective_term": float(self.composition_penalty * np.sum(diff * diff)),
                "grad_occ": np.zeros_like(occ_np),
                "species_totals": species_totals,
                "species_fractions": fraction,
                "diff": diff,
            }
        fraction = species_totals / total_real
        diff = fraction - target
        objective_term = float(self.composition_penalty * np.sum(diff * diff))
        dot_diff_totals = float(np.dot(diff, species_totals))
        grad_by_species = 2.0 * self.composition_penalty * (
            diff * total_real - dot_diff_totals
        ) / (total_real**2)
        grad_occ = np.repeat(grad_by_species[None, :], occ_np.shape[0], axis=0)
        return {
            "objective_term": objective_term,
            "grad_occ": grad_occ,
            "species_totals": species_totals,
            "species_fractions": fraction,
            "diff": diff,
        }

    def se_vacancy_bias_terms(self, occ_np):
        """Linear Se-deficiency bias with a fixed, member-independent scale.

        The conventional formation energy per current atom is deliberately
        untouched.  This separate score contribution is

            -alpha * (N_Se,reference - N_Se) / N_reference.

        For soft EAT states, N_Se is the summed Se occupancy.  For rounded
        states, the same expression automatically becomes an exact atom count.
        """
        occ_np = np.asarray(occ_np, dtype=float)
        se_count = float(occ_np[:, self.se_species_index].sum())
        se_deficiency = float(self.se_vacancy_reference_se_count - se_count)
        alpha = self.se_vacancy_bias_alpha_eV if self.se_vacancy_bias_enable else 0.0
        objective_term = float(
            -alpha * se_deficiency / self.se_vacancy_reference_atom_count
        )
        grad_occ = np.zeros_like(occ_np, dtype=float)
        if alpha != 0.0:
            grad_occ[:, self.se_species_index] = (
                alpha / self.se_vacancy_reference_atom_count
            )
        return {
            "objective_term": objective_term,
            "grad_occ": grad_occ,
            "enabled": bool(self.se_vacancy_bias_enable),
            "alpha_eV_per_vacancy": float(self.se_vacancy_bias_alpha_eV),
            "reference_atom_count": float(self.se_vacancy_reference_atom_count),
            "reference_se_count": float(self.se_vacancy_reference_se_count),
            "se_count": se_count,
            "se_deficiency": se_deficiency,
        }

    def build_atoms(self, occ_np, z_np):
        atoms = Atoms(
            symbols=[SYM_DUMMY] * self.n_sites,
            positions=self.z_to_positions(z_np),
            cell=self.cell,
            pbc=self.pbc,
        )
        atoms.wrap()
        set_multi_species_eat_weights(atoms, occ_np, self.atomic_numbers)
        return atoms

    def dominant_species(self, occ_np):
        occ_np = np.asarray(occ_np, dtype=float)
        total = occ_np.sum(axis=1)
        winners = np.argmax(occ_np, axis=1)
        return [self.species[int(w)] if total[i] > 1e-8 else "X" for i, w in enumerate(winners)]

    def render_differentiable_gaussian_image(self, occ, positions):
        """Compatibility entry point; all GA phases use the Gaussian renderer."""
        predicted_linear, species_components = self.gaussian_renderer.render(occ, positions)
        return predicted_linear, predicted_linear, species_components

    def ellipse_composition_terms(self, occ, positions):
        """Retired term: exact zero for legacy metrics, never part of an objective."""
        zero = occ.sum() * 0.0
        empty = occ.new_empty(0)
        return {"objective_term": zero, "loss_raw": zero,
                "predicted_integrals": empty, "target_integrals": empty,
                "residuals_Se_units": empty,
                "membership": occ.new_zeros((len(occ), 0)),
                "background_membership": occ.new_ones(len(occ)),
                "region_species_occupancies": occ.new_zeros((0, self.n_species))}

    def sample_box_confinement_terms(self, occ, positions):
        """Return an occupancy-normalized smooth penalty outside the sample box."""
        lower = torch.as_tensor(
            self.sample_box_lower_A + self.sample_box_confinement_margin_A,
            device=positions.device,
            dtype=positions.dtype,
        )
        upper = torch.as_tensor(
            self.sample_box_upper_A - self.sample_box_confinement_margin_A,
            device=positions.device,
            dtype=positions.dtype,
        )
        sharpness = self.sample_box_confinement_sharpness
        lower_violation = torch.nn.functional.softplus(
            sharpness * (lower[None, :] - positions)
        ) / sharpness
        upper_violation = torch.nn.functional.softplus(
            sharpness * (positions - upper[None, :])
        ) / sharpness
        per_site_squared_violation_A2 = torch.sum(
            lower_violation**2 + upper_violation**2,
            dim=1,
        )
        real_occupancy = occ.sum(dim=1)
        total_occupancy = real_occupancy.sum().clamp_min(1.0e-8)
        loss_raw = torch.sum(
            real_occupancy * per_site_squared_violation_A2
        ) / total_occupancy
        objective = self.sample_box_confinement_penalty * loss_raw
        outside = torch.any(
            (positions < lower[None, :]) | (positions > upper[None, :]),
            dim=1,
        )
        return objective, loss_raw, outside, per_site_squared_violation_A2

    def auxiliary_imaging_and_boundary_terms(self, occ, z, want_grad):
        occ_work = occ.detach().clone().requires_grad_(bool(want_grad))
        z_work = z.detach().clone().requires_grad_(bool(want_grad))
        cell_lengths = torch.tensor(
            np.diag(self.cell),
            device=z_work.device,
            dtype=z_work.dtype,
        )
        positions = torch.remainder(z_work, cell_lengths)
        predicted_image, predicted_linear, gaussian_components = (
            self.render_differentiable_gaussian_image(occ_work, positions)
        )
        per_voxel_image_loss = torch.nn.functional.huber_loss(
            predicted_image,
            self.target_image_t,
            delta=self.image_huber_delta,
            reduction="none",
        )
        image_loss = per_voxel_image_loss[self.image_loss_mask_t].mean()
        image_objective = self.image_penalty * image_loss
        ellipse_composition_info = self.ellipse_composition_terms(
            occ_work, positions
        )
        ellipse_composition_objective = ellipse_composition_info["objective_term"]

        level_set = periodic_cubic_bspline_sample_zyx(
            self.level_set_coefficients_t,
            positions,
            self.target_voxel_size_A,
        )
        violation = torch.nn.functional.softplus(
            self.boundary_sharpness
            * (self.level_set_margin_A - level_set)
        ) / self.boundary_sharpness
        heavy_indices = torch.as_tensor(
            self.heavy_species_indices,
            device=occ_work.device,
            dtype=torch.long,
        )
        heavy_occupancy = occ_work.index_select(1, heavy_indices).sum(dim=1)
        heavy_occupancy_sum = torch.clamp(torch.sum(heavy_occupancy), min=1e-8)
        boundary_objective = self.boundary_penalty * (
            torch.sum(heavy_occupancy * violation * violation)
            / heavy_occupancy_sum
        )
        (
            box_objective,
            box_loss_raw,
            outside_sample_box,
            box_per_site_squared_violation_A2,
        ) = self.sample_box_confinement_terms(occ_work, positions)
        auxiliary_objective = (
            image_objective
            + boundary_objective
            + box_objective
        )

        grad_occ = grad_z = None
        if want_grad:
            grad_occ, grad_z = torch.autograd.grad(
                auxiliary_objective,
                (occ_work, z_work),
                create_graph=False,
                retain_graph=False,
            )

        with torch.no_grad():
            prediction_roi = predicted_image[self.image_loss_mask_t]
            target_roi = self.target_image_t[self.image_loss_mask_t]
            pred_centered = prediction_roi - prediction_roi.mean()
            target_centered = target_roi - target_roi.mean()
            correlation = torch.sum(pred_centered * target_centered) / torch.sqrt(
                torch.sum(pred_centered * pred_centered)
                * torch.sum(target_centered * target_centered)
                + 1e-20
            )
            level_set_np = level_set.detach().cpu().numpy()
            inside = level_set_np >= self.level_set_margin_A
            heavy_occupancy_np = heavy_occupancy.detach().cpu().numpy()
            active_heavy = heavy_occupancy_np > max(
                float(self.args.keep_thresh),
                1e-3,
            )
            outside_sample_box_np = outside_sample_box.detach().cpu().numpy()
            box_per_site_np = (
                box_per_site_squared_violation_A2.detach().cpu().numpy()
            )
            box_info = {
                "objective_term": float(box_objective.detach().cpu()),
                "loss_raw_A2": float(box_loss_raw.detach().cpu()),
                "normalization": "total_W_plus_Se_occupancy",
                "total_real_occupancy": float(heavy_occupancy_sum.detach().cpu()),
                "lower_bound_xyz_A": self.sample_box_lower_A.tolist(),
                "upper_bound_xyz_A": self.sample_box_upper_A.tolist(),
                "margin_A": float(self.sample_box_confinement_margin_A),
                "outside": outside_sample_box_np,
                "active_outside_count": int(
                    np.count_nonzero(active_heavy & outside_sample_box_np)
                ),
                "outside_occupancy_mass": float(
                    np.sum(heavy_occupancy_np[outside_sample_box_np])
                ),
                "maximum_squared_violation_A2": float(
                    np.max(box_per_site_np) if len(box_per_site_np) else 0.0
                ),
            }
            boundary_info = {
                "objective_term": float(boundary_objective.detach().cpu()),
                "grad_occ": np.zeros((self.n_sites, self.n_species), dtype=float),
                "grad_z": np.zeros((self.n_sites, 3), dtype=float),
                "level_set_A": level_set_np,
                "inside": inside,
                "inside_count": int(np.count_nonzero(active_heavy & inside)),
                "active_heavy_count": int(np.count_nonzero(active_heavy)),
                "active_heavy_outside_count": int(
                    np.count_nonzero(active_heavy & ~inside)
                ),
                "outside_heavy_occupancy_mass": float(
                    np.sum(heavy_occupancy_np[~inside])
                ),
            }
            image_info = {
                "objective_term": float(
                    image_objective.detach().cpu()
                ),
                "voxel_objective_term": float(image_objective.detach().cpu()),
                "loss_raw": float(image_loss.detach().cpu()),
                "ellipse_composition_objective_term": float(
                    ellipse_composition_objective.detach().cpu()
                ),
                "ellipse_composition_loss_raw": float(
                    ellipse_composition_info["loss_raw"].detach().cpu()
                ),
                "ellipse_mean_absolute_residual_Se_units": 0.0,
                "ellipse_maximum_absolute_residual_Se_units": 0.0,
                "ellipse_mean_background_membership": 1.0,
                "correlation": float(correlation.detach().cpu()),
                "predicted_image_min": float(predicted_image.min().detach().cpu()),
                "predicted_image_max": float(predicted_image.max().detach().cpu()),
            }
        return {
            "objective_term": float(auxiliary_objective.detach().cpu()),
            "boundary": boundary_info,
            "sample_box": box_info,
            "image": image_info,
            "grad_occ": None if grad_occ is None else grad_occ.detach(),
            "grad_z": None if grad_z is None else grad_z.detach(),
            "predicted_image": predicted_image.detach(),
            "predicted_linear": predicted_linear.detach(),
            "gaussian_components": gaussian_components.detach(),
        }

    def objfun(self, y, beta):
        self.ensure_calculator()
        input_y = y.detach().clone()
        occ, z = self.unpack_y(y)
        occ = self.project_allowed_occupancies(occ)
        occ_np = occ.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        atoms = self.build_atoms(occ_np, z_np)
        self.objective_evaluation_count += 1
        E, dE_dw, forces = get_energy_gradients_and_forces(atoms, self.calc, self.atomic_numbers, want_grad=True)
        auxiliary = self.auxiliary_imaging_and_boundary_terms(occ, z, want_grad=True)
        blob_info = auxiliary["boundary"]
        box_info = auxiliary["sample_box"]
        image_info = auxiliary["image"]
        comp_info = self.composition_penalty_term(occ_np)
        se_vacancy_info = self.se_vacancy_bias_terms(occ_np)

        xsum = occ.sum()
        xsum_value = max(float(xsum.item()), 1e-8)
        E_t = torch.tensor(E, device=y.device, dtype=y.dtype)
        species_totals = occ_np.sum(axis=0)
        ref_sum = float(np.dot(species_totals, self.element_ref_energies))
        formation_numerator = float(E) - ref_sum
        f = torch.tensor(formation_numerator / xsum_value, device=y.device, dtype=y.dtype)
        g_occ_np = (np.asarray(dE_dw, dtype=float) - self.element_ref_energies[None, :]) / xsum_value
        g_occ_np = g_occ_np - formation_numerator / (xsum_value**2)

        ent = tsallis_entropy_term(occ, self.tsallis_q)
        f = (
            f
            + beta * ent
            + torch.tensor(blob_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(box_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(image_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(comp_info["objective_term"], device=y.device, dtype=y.dtype)
            + torch.tensor(se_vacancy_info["objective_term"], device=y.device, dtype=y.dtype)
        )
        g_occ = (
            torch.tensor(g_occ_np, device=y.device, dtype=y.dtype)
            + beta * tsallis_entropy_grad(occ, self.tsallis_q)
            + torch.tensor(blob_info["grad_occ"], device=y.device, dtype=y.dtype)
            + auxiliary["grad_occ"].to(device=y.device, dtype=y.dtype)
            + torch.tensor(comp_info["grad_occ"], device=y.device, dtype=y.dtype)
            + torch.tensor(se_vacancy_info["grad_occ"], device=y.device, dtype=y.dtype)
        )
        g_occ = self.zero_forbidden_grad(g_occ)

        grad_cart = -np.asarray(forces, dtype=float) / xsum_value
        grad_z = grad_cart + auxiliary["grad_z"].detach().cpu().numpy()
        g_z = torch.tensor(grad_z, device=y.device, dtype=y.dtype)
        self._last_aux_cache = {
            "occ": occ_np.copy(),
            "z": z_np.copy(),
            "boundary": blob_info,
            "sample_box": box_info,
            "image": image_info,
            "se_vacancy_bias": se_vacancy_info,
            "predicted_image": auxiliary["predicted_image"].cpu().numpy(),
        }
        gradient = self.pack_y(g_occ, g_z)
        self._objective_result_cache = {
            "input_y": input_y,
            "evaluated_y": self.pack_y(occ, z).detach().clone(),
            "f": float(f.detach().cpu()),
            "gradient": gradient.detach().clone(),
        }
        return f, gradient

    def snapshot(self, y, stage, iteration, beta, f_objective, err, run_kind):
        occ, z = self.unpack_y(y)
        occ = self.project_allowed_occupancies(occ)
        occ_np = occ.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        positions = self.z_to_positions(z_np)
        cache = self._last_aux_cache
        if (
            cache is not None
            and cache["occ"].shape == occ_np.shape
            and cache["z"].shape == z_np.shape
            and np.allclose(cache["occ"], occ_np, atol=1e-7, rtol=0.0)
            and np.allclose(cache["z"], z_np, atol=1e-7, rtol=0.0)
        ):
            blob_info = cache["boundary"]
            box_info = cache["sample_box"]
            image_info = cache["image"]
        else:
            auxiliary = self.auxiliary_imaging_and_boundary_terms(
                occ,
                z,
                want_grad=False,
            )
            blob_info = auxiliary["boundary"]
            box_info = auxiliary["sample_box"]
            image_info = auxiliary["image"]
        comp_info = self.composition_penalty_term(occ_np)
        se_vacancy_info = self.se_vacancy_bias_terms(occ_np)
        occ_total = occ_np.sum(axis=1)
        null_occ = np.maximum(1.0 - occ_total, 0.0)
        species_totals = occ_np.sum(axis=0)
        xsum = max(float(species_totals.sum()), 1e-12)
        reference_mix_per_atom = float(np.dot(species_totals / xsum, self.element_ref_energies))
        entropy_penalty = float(beta) * float(tsallis_entropy_term(occ, self.tsallis_q).item())
        f_guided = float(f_objective) - entropy_penalty
        f_energy = (
            f_guided
            - float(blob_info["objective_term"])
            - float(box_info["objective_term"])
            - float(image_info["objective_term"])
            - float(comp_info["objective_term"])
            - float(se_vacancy_info["objective_term"])
        )
        formation_diff = f_energy - self.compound_reference_formation_energy_per_atom
        ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
        return {
            "stage": int(stage + 1),
            "iter": int(iteration),
            "run_kind": str(run_kind),
            "beta": float(beta),
            "sumx": float(occ_np.sum()),
            "entropy_avg": ent_avg,
            "max_min": max_min,
            "f_objective": float(f_objective),
            "formation_energy_per_atom": f_energy,
            "compound_reference_formation_energy_per_atom": float(self.compound_reference_formation_energy_per_atom),
            "formation_energy_difference_from_reference": float(formation_diff),
            "f_energy_no_entropy": f_energy,
            "f_guided_no_entropy": f_guided,
            "element_reference_energies_per_atom": self.element_ref_energies.tolist(),
            "reference_mix_energy_per_atom": reference_mix_per_atom,
            "assigned_blob_term": float(blob_info["objective_term"]),
            "heavy_region_term": float(blob_info["objective_term"]),
            "sample_box_term": float(box_info["objective_term"]),
            "sample_box_loss_raw_A2": float(box_info["loss_raw_A2"]),
            "sample_box_active_outside_count": int(
                box_info["active_outside_count"]
            ),
            "sample_box_outside_occupancy_mass": float(
                box_info["outside_occupancy_mass"]
            ),
            "image_term": float(image_info["objective_term"]),
            "voxel_image_term": float(image_info["voxel_objective_term"]),
            "image_loss_raw": float(image_info["loss_raw"]),
            "ellipse_composition_term": float(
                image_info["ellipse_composition_objective_term"]
            ),
            "ellipse_composition_loss_raw": float(
                image_info["ellipse_composition_loss_raw"]
            ),
            "ellipse_mean_absolute_residual_Se_units": float(
                image_info["ellipse_mean_absolute_residual_Se_units"]
            ),
            "image_correlation": float(image_info["correlation"]),
            "image_predicted_min": float(image_info["predicted_image_min"]),
            "image_predicted_max": float(image_info["predicted_image_max"]),
            "composition_term": float(comp_info["objective_term"]),
            "composition_target": self.composition_target.tolist(),
            "composition_fractions": comp_info["species_fractions"].tolist(),
            "composition_species_totals": comp_info["species_totals"].tolist(),
            "composition_diff": comp_info["diff"].tolist(),
            "se_vacancy_bias_term": float(se_vacancy_info["objective_term"]),
            "se_vacancy_bias_enabled": bool(se_vacancy_info["enabled"]),
            "se_vacancy_bias_alpha_eV_per_vacancy": float(
                se_vacancy_info["alpha_eV_per_vacancy"]
            ),
            "se_vacancy_reference_atom_count": float(
                se_vacancy_info["reference_atom_count"]
            ),
            "se_vacancy_reference_se_count": float(
                se_vacancy_info["reference_se_count"]
            ),
            "se_occupancy_count": float(se_vacancy_info["se_count"]),
            "se_vacancy_count_relative_to_reference": float(
                se_vacancy_info["se_deficiency"]
            ),
            "assigned_blob_inside_count": int(blob_info["inside_count"]),
            "active_heavy_count": int(blob_info.get("active_heavy_count", 0)),
            "active_heavy_outside_count": int(
                blob_info.get("active_heavy_outside_count", 0)
            ),
            "outside_heavy_occupancy_mass": float(
                blob_info.get("outside_heavy_occupancy_mass", 0.0)
            ),
            "error": float(err),
            "lattice_vectors_angstrom": self.cell.tolist(),
            "cart_positions_angstrom": positions.tolist(),
            "frac_coords": self.positions_to_frac(positions).tolist(),
            "occupancies": occ_total.tolist(),
            "species_occupancies": occ_np.tolist(),
            "null_occupancies": null_occ.tolist(),
            "species": list(self.species),
            "dominant_species": self.dominant_species(occ_np),
            "assigned_blob_indices": self.assigned_blob_indices.astype(int).tolist(),
            "assigned_site_region_indices": self.assigned_site_region_indices.astype(int).tolist(),
            "position_state_angstrom": z_np.tolist(),
            "inside_heavy_level_set": blob_info["inside"].astype(bool).tolist(),
            "level_set_angstrom": blob_info["level_set_A"].tolist(),
            "reference_blob_centers_angstrom": self.ref_positions_blobs.tolist(),
            "reference_blob_sigmas_angstrom": self.ref_blob_sigmas.tolist(),
            "reference_blob_symbols": self.blob_symbols,
            "reference_blob_source_types": self.blob_source_types,
            "reference_blob_allowed_elements": self.blob_allowed_elements_from_json,
            "reference_blob_allowed_species_mask": self.blob_allowed_species_mask.astype(bool).tolist(),
            "species_by_source_type": {key: list(value) for key, value in self.species_by_source_type.items()},
        }

    def sample_uniform_cell_z(self, count, avoid_positions=None):
        """Sample continuous Cartesian positions throughout the periodic cell."""
        count = int(count)
        if count <= 0:
            return np.zeros((0, 3), dtype=float)
        cell_lengths = np.diag(self.cell).astype(float)
        avoid = np.asarray(
            np.zeros((0, 3), dtype=float)
            if avoid_positions is None
            else avoid_positions,
            dtype=float,
        ).reshape(-1, 3)

        def periodic_distance_ok(candidate, points, cutoff2):
            if cutoff2 <= 0.0 or len(points) == 0:
                return True
            delta = np.asarray(points, dtype=float) - candidate[None, :]
            delta -= np.round(delta / cell_lengths[None, :]) * cell_lengths[None, :]
            return bool(np.all(np.sum(delta * delta, axis=1) >= cutoff2))

        preferred_cutoff = float(self.initial_oxygen_min_distance_A)
        selected = []
        effective_cutoff = preferred_cutoff
        for cutoff_fraction in (1.0, 0.90, 0.75, 0.50, 0.25, 0.0):
            effective_cutoff = preferred_cutoff * cutoff_fraction
            cutoff2 = effective_cutoff * effective_cutoff
            attempts = 0
            max_attempts = max(5000, 250 * count)
            while len(selected) < count and attempts < max_attempts:
                candidate = np.random.random(3) * cell_lengths
                attempts += 1
                if not periodic_distance_ok(candidate, avoid, cutoff2):
                    continue
                if not periodic_distance_ok(candidate, selected, cutoff2):
                    continue
                selected.append(candidate)
            if len(selected) >= count:
                break

        selected = np.asarray(selected[:count], dtype=float)
        if effective_cutoff + 1e-12 < preferred_cutoff:
            message = (
                f"uniform-cell initialization requested {count} sites; "
                f"placement relaxed the periodic separation from "
                f"{preferred_cutoff:.3f} A to {effective_cutoff:.3f} A"
            )
            append_running_log(self.args, message)
            terminal_print(self.args, f"[initialization] {message}")
        return selected

    def sample_feasible_region_z(self, count):
        """Initialize W/Se/null sites with the deterministic cylinder scheme.

        The mask voxels are pruned in experimental-intensity order with the
        configured cylindrical exclusion geometry.  A uniform Cartesian jitter
        in Angstrom is then proposed once per site.  Proposals outside the exact
        periodic cubic-spline level set are discarded, leaving their original
        feasible voxel centers in place.
        """
        count = int(count)
        if count <= 0:
            return np.zeros((0, 3), dtype=float)
        selected_zyx, summary = cylinder_pruned_feasible_sites_zyx(
            self.experimental["feasible_mask"],
            self.target_voxel_size_A,
            self.initial_cylinder_lateral_radius_A,
            self.initial_cylinder_z_radius_A,
            intensity=self.experimental["smoothed_target_sqrt"],
        )
        if len(selected_zyx) < count:
            raise ValueError(
                "The configured cylindrical initializer produced only "
                f"{len(selected_zyx)} candidate sites but {count} were requested. "
                "Reduce candidate_site_density_per_angstrom3 or one of the "
                "initial cylinder radii."
            )
        selected_zyx = selected_zyx[:count]
        dz_A, dy_A, dx_A = self.target_voxel_size_A
        selected = np.column_stack(
            (
                selected_zyx[:, 2] * dx_A,
                selected_zyx[:, 1] * dy_A,
                selected_zyx[:, 0] * dz_A,
            )
        ).astype(float)

        accepted_jitter_count = 0
        rejected_jitter_count = 0
        if self.initial_jitter_A > 0.0:
            cell_lengths = np.diag(self.cell).astype(float)
            proposals = np.mod(
                selected
                + np.random.uniform(
                    -self.initial_jitter_A,
                    self.initial_jitter_A,
                    size=selected.shape,
                ),
                cell_lengths[None, :],
            )
            proposals_t = torch.tensor(
                proposals,
                device=self.device,
                dtype=self.dtype,
            )
            with torch.no_grad():
                proposal_level_set = periodic_cubic_bspline_sample_zyx(
                    self.level_set_coefficients_t,
                    proposals_t,
                    self.target_voxel_size_A,
                ).cpu().numpy()
            accept = proposal_level_set >= self.level_set_margin_A
            selected[accept] = proposals[accept]
            accepted_jitter_count = int(np.count_nonzero(accept))
            rejected_jitter_count = int(count - accepted_jitter_count)

        summary.update(
            {
                "requested_count": int(count),
                "returned_count": int(len(selected)),
                "jitter_A": float(self.initial_jitter_A),
                "accepted_jitter_count": accepted_jitter_count,
                "rejected_outside_jitter_count": rejected_jitter_count,
            }
        )
        self._last_feasible_initialization_summary = summary
        message = (
            "intensity-prioritized cylinder initialization "
            f"requested={count} available={summary['selected_before_count_limit']} "
            f"lateral_radius={self.initial_cylinder_lateral_radius_A:.3f} A "
            f"z_radius={self.initial_cylinder_z_radius_A:.3f} A "
            f"jitter=+/-{self.initial_jitter_A:.3f} A "
            f"jitter_accepted={accepted_jitter_count} "
            f"jitter_rejected_outside={rejected_jitter_count}"
        )
        append_running_log(self.args, message)
        terminal_print(self.args, f"[initialization] {message}")
        return selected

    def projected_gradient_error(self, y, g):
        y_projected = self.project(y)
        return torch.norm(self.project(y_projected - g) - y_projected).item()

    def maybe_write_heartbeat(self, message):
        interval = float(getattr(self.args, "heartbeat_interval_seconds", 0.0) or 0.0)
        if interval <= 0.0:
            return
        now = time.time()
        if now - float(self.last_heartbeat_time) < interval:
            return
        append_running_log(self.args, message)
        self.last_heartbeat_time = now

    def run_stage(
        self,
        y0,
        beta,
        stage,
        run_kind="main",
        attempt_index=0,
        maxit_override=None,
        tol_override=None,
        stagnation_patience_override=None,
        fixed_occupancies=None,
        cost_plateau_tol_override=0.0,
        cost_plateau_patience_override=0,
        cost_plateau_min_iterations_override=0,
        record_history=True,
    ):
        f_history = []
        objective_history = []
        local_history = []
        capture_trajectory = not bool(self.args.disable_trajectory_history)
        self.stage_wall_start = time.time()
        self.stage_last_callback_time = None
        maxit_value = (
            int(self.args.maxit)
            if maxit_override is None
            else int(maxit_override)
        )
        tol_value = (
            float(self.args.tol)
            if tol_override is None
            else float(tol_override)
        )
        stagnation_patience = (
            int(self.args.stage_stagnation_patience)
            if stagnation_patience_override is None
            else int(stagnation_patience_override)
        )
        stagnation_min_improvement = float(
            self.args.stage_stagnation_min_improvement
        )
        cost_plateau_tol = float(cost_plateau_tol_override)
        cost_plateau_patience = int(cost_plateau_patience_override)
        cost_plateau_min_iterations = int(
            cost_plateau_min_iterations_override
        )
        if stagnation_patience < 0:
            raise ValueError("minimizer.stage_stagnation_patience must be nonnegative.")
        if stagnation_min_improvement < 0.0:
            raise ValueError(
                "minimizer.stage_stagnation_min_improvement must be nonnegative."
            )
        if maxit_value <= 0:
            raise ValueError("PBFGS maxit must be positive.")
        if tol_value <= 0.0:
            raise ValueError("PBFGS tolerance must be positive.")
        if cost_plateau_tol < 0.0:
            raise ValueError("Final PBFGS cost plateau tolerance must be nonnegative.")
        if cost_plateau_patience < 0 or cost_plateau_min_iterations < 0:
            raise ValueError("Final PBFGS plateau iteration counts must be nonnegative.")
        if cost_plateau_tol > 0.0 and cost_plateau_patience <= 0:
            raise ValueError(
                "A positive final PBFGS cost plateau tolerance requires positive patience."
            )

        fixed_occ = None
        if fixed_occupancies is not None:
            fixed_occ = self.project_allowed_occupancies(
                fixed_occupancies.detach().clone().to(
                    device=y0.device,
                    dtype=y0.dtype,
                )
            )
            if tuple(fixed_occ.shape) != (self.n_sites, self.n_species):
                raise ValueError("Fixed final occupancies have the wrong shape.")

        def stage_project(y):
            if fixed_occ is None:
                return self.project(y)
            _, z = self.unpack_y(y)
            return self.pack_y(fixed_occ, z)

        def stage_objfun(y, beta_local):
            y_eval = stage_project(y)
            if fixed_occ is None:
                return self.objfun(y_eval, beta_local)
            f_value, gradient = self.objfun(y_eval, 0.0)
            _, grad_z = self.unpack_y(gradient)
            return f_value, self.pack_y(torch.zeros_like(fixed_occ), grad_z)

        optimizer_beta = 0.0 if fixed_occ is not None else float(beta)
        stagnation_state = {}
        plateau_state = {}
        best_state = {}
        best_objective_history = []
        last_projected_gradient_error = {"value": None}

        class StageStagnationStop(RuntimeError):
            pass

        class StageCostPlateauStop(RuntimeError):
            pass

        def callback(y, f, err, k, converged, beta_local):
            y = stage_project(y)
            last_projected_gradient_error["value"] = float(err)
            now = time.time()
            elapsed_s = now - float(self.stage_wall_start or now)
            dt_s = 0.0 if self.stage_last_callback_time is None else now - float(self.stage_last_callback_time)
            self.stage_last_callback_time = now
            objective_value = (
                float(f.detach().cpu())
                if isinstance(f, torch.Tensor)
                else float(f)
            )
            objective_history.append(objective_value)
            if (
                np.isfinite(objective_value)
                and (
                    not best_state
                    or objective_value < float(best_state["f"])
                )
            ):
                best_state.update(
                    {
                        "y": y.detach().clone(),
                        "f": objective_value,
                        "error": float(err),
                        "iteration": int(k),
                    }
                )
            best_objective_history.append(
                float(best_state.get("f", objective_value))
            )
            snap = self.snapshot(y, stage, k, beta_local, f, err, run_kind)
            snap["attempt_index"] = int(attempt_index)
            snap["wall_elapsed_seconds"] = float(elapsed_s)
            snap["wall_dt_seconds"] = float(dt_s)
            f_history.append(snap["f_guided_no_entropy"])
            if capture_trajectory:
                local_history.append(snap)
            if record_history and capture_trajectory:
                if self.callback_counter % max(1, int(self.args.history_every)) == 0:
                    self.history.append(snap)
                self.callback_counter += 1
            write_active_member_progress(
                self,
                y,
                stage,
                beta_local,
                k,
                run_kind,
            )
            self.maybe_write_heartbeat(
                f"relaxing run_kind={run_kind} stage={stage+1} iter={k} "
                f"Eform={snap['formation_energy_per_atom']:+.6f} Fguided={snap['f_guided_no_entropy']:+.6f} "
                f"err={err:.3e}"
            )
            terminal_print(
                self.args,
                f"[stage={stage+1:02d}] iter={k:4d} beta={beta_local:.3e} "
                f"elapsed_s={elapsed_s:.2f} dt_s={dt_s:.2f} "
                f"sumx={snap['sumx']:.3f} Eform={snap['formation_energy_per_atom']:+.6f} "
                f"dEform={snap['formation_energy_difference_from_reference']:+.6f} "
                f"heavy_level_set={snap['assigned_blob_term']:+.6f} "
                f"box={snap['sample_box_term']:+.6f} "
                f"box_out={snap['sample_box_active_outside_count']} "
                f"comp={snap['composition_term']:+.6f} "
                f"Se_vac_bias={snap['se_vacancy_bias_term']:+.6f} "
                f"Se_def={snap['se_vacancy_count_relative_to_reference']:+.3f} "
                f"image={snap['image_term']:+.6f} corr={snap['image_correlation']:+.4f} "
                f"xfrac={snap['composition_fractions']} "
                f"heavy_inside={snap['assigned_blob_inside_count']}/{snap['active_heavy_count']} "
                f"Fguided={snap['f_guided_no_entropy']:+.6f} "
                f"entropy/site={snap['entropy_avg']:.6e} max_nonbinary={snap['max_min']:.3e} "
                f"err={err:.3e} conv={converged}"
            )

            if (
                cost_plateau_tol > 0.0
                and int(k) >= cost_plateau_min_iterations
                and len(best_objective_history) >= cost_plateau_patience + 1
            ):
                improvement = float(
                    best_objective_history[-cost_plateau_patience - 1]
                    - best_objective_history[-1]
                )
                if improvement <= cost_plateau_tol:
                    plateau_state.update(
                        {
                            "y": best_state["y"].detach().clone(),
                            "f": float(best_state["f"]),
                            "error": float(best_state["error"]),
                            "iteration": int(k),
                            "best_iteration": int(best_state["iteration"]),
                            "improvement": improvement,
                        }
                    )
                    message = (
                        f"stage_cost_plateau_stop run_kind={run_kind} "
                        f"stage={stage + 1} iter={k} best_objective="
                        f"{float(best_state['f']):+.9e} best_improvement_over_"
                        f"{cost_plateau_patience}={improvement:.3e} "
                        f"threshold={cost_plateau_tol:.3e}"
                    )
                    append_running_log(self.args, message)
                    terminal_print(self.args, f"[cost plateau] {message}")
                    raise StageCostPlateauStop(message)

            # The projected-gradient norm can have a nonzero noise floor for
            # the coupled FairChem/level-set/image objective. At fixed beta, stop a
            # stage when even its best objective value over the last N steps
            # fails to improve sufficiently relative to N steps ago.
            if (
                stagnation_patience > 0
                and len(objective_history) >= stagnation_patience + 1
                and np.all(
                    np.isfinite(objective_history[-stagnation_patience - 1 :])
                )
            ):
                window_start = objective_history[-stagnation_patience - 1]
                best_recent = min(objective_history[-stagnation_patience:])
                improvement = float(window_start - best_recent)
                if improvement < stagnation_min_improvement:
                    stagnation_state.update(
                        {
                            "y": y.detach().clone(),
                            "f": objective_value,
                            "iteration": int(k),
                            "improvement": improvement,
                        }
                    )
                    message = (
                        f"stage_stagnation_stop run_kind={run_kind} "
                        f"stage={stage + 1} iter={k} objective={objective_value:+.9e} "
                        f"best_improvement_over_{stagnation_patience}="
                        f"{improvement:.3e} threshold="
                        f"{stagnation_min_improvement:.3e}"
                    )
                    append_running_log(self.args, message)
                    terminal_print(self.args, f"[stagnation] {message}")
                    raise StageStagnationStop(message)

        opt = minimizer(
            fun_jac=stage_objfun,
            x0=stage_project(y0),
            args=(optimizer_beta,),
            project=stage_project,
            proj_args=(),
            callback=callback,
            BFGS_hist=self.args.bfgs_hist,
            maxit=maxit_value,
            tol=tol_value,
            max_line_search=self.args.max_line_search,
        )
        stage_stagnated = False
        cost_plateau_reached = False
        try:
            f, y = opt.PBFGS()
        except StageStagnationStop:
            stage_stagnated = True
            f = float(stagnation_state["f"])
            y = stagnation_state["y"]
        except StageCostPlateauStop:
            cost_plateau_reached = True
            f = float(plateau_state["f"])
            y = plateau_state["y"]
            last_projected_gradient_error["value"] = float(
                plateau_state["error"]
            )
        y = stage_project(y)
        returned_f = (
            float(f.detach().cpu()) if isinstance(f, torch.Tensor) else float(f)
        )
        if (
            fixed_occ is not None
            and best_state
            and float(best_state["f"]) < returned_f
        ):
            f = float(best_state["f"])
            y = best_state["y"]
            last_projected_gradient_error["value"] = float(best_state["error"])
        occ, _ = self.unpack_y(y)
        ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
        tolerance_reached = bool(
            last_projected_gradient_error["value"] is not None
            and float(last_projected_gradient_error["value"]) < tol_value
            and not stage_stagnated
            and not cost_plateau_reached
        )
        termination_reason = (
            "cost_plateau"
            if cost_plateau_reached
            else "stage_stagnation"
            if stage_stagnated
            else "gradient_tolerance"
            if tolerance_reached
            else "max_iterations"
        )
        run_info = {
            "run_kind": str(run_kind),
            "attempt_index": int(attempt_index),
            "maxit": int(maxit_value),
            "tol": float(tol_value),
            "stagnation_patience": int(stagnation_patience),
            "positions_only": bool(fixed_occ is not None),
            "cost_plateau_tol": float(cost_plateau_tol),
            "cost_plateau_patience": int(cost_plateau_patience),
            "cost_plateau_min_iterations": int(cost_plateau_min_iterations),
            "cost_plateau_reached": bool(cost_plateau_reached),
            "cost_plateau_iteration": int(plateau_state.get("iteration", -1)),
            "cost_plateau_best_iteration": int(
                plateau_state.get("best_iteration", -1)
            ),
            "cost_plateau_improvement": plateau_state.get("improvement"),
            "termination_reason": termination_reason,
            "final_projected_gradient_error": last_projected_gradient_error["value"],
            "tolerance_reached": tolerance_reached,
            "f_guided_history": list(f_history),
            "trajectory": list(local_history),
            "f_guided_final": (
                float(f)
                if fixed_occ is not None
                else float(f_history[-1])
                if f_history
                else float(f)
            ),
            "entropy_avg_final": float(ent_avg),
            "max_min_final": float(max_min),
            "objective_history": list(objective_history),
            "stagnated": bool(stage_stagnated),
            "stagnation_iteration": int(
                stagnation_state.get("iteration", -1)
            ),
            "stagnation_improvement": stagnation_state.get("improvement"),
            "accepted_iterations": int(len(objective_history)),
            "stage_wall_seconds": float(
                time.time() - float(self.stage_wall_start or time.time())
            ),
        }
        if record_history:
            self.stage_summaries.append({
                "stage": int(stage + 1),
                "beta": float(beta),
                "sumx": float(occ.sum().item()),
                "entropy_avg": ent_avg,
                "max_min": max_min,
                "run_kind": str(run_kind),
                "attempt_index": int(attempt_index),
                "maxit": int(maxit_value),
                "tol": float(tol_value),
                "stagnation_patience": int(stagnation_patience),
                "positions_only": bool(fixed_occ is not None),
                "cost_plateau_tol": float(cost_plateau_tol),
                "cost_plateau_patience": int(cost_plateau_patience),
                "cost_plateau_min_iterations": int(cost_plateau_min_iterations),
                "cost_plateau_reached": bool(cost_plateau_reached),
                "cost_plateau_iteration": int(
                    plateau_state.get("iteration", -1)
                ),
                "cost_plateau_best_iteration": int(
                    plateau_state.get("best_iteration", -1)
                ),
                "cost_plateau_improvement": plateau_state.get("improvement"),
                "termination_reason": termination_reason,
                "final_projected_gradient_error": run_info[
                    "final_projected_gradient_error"
                ],
                "tolerance_reached": bool(run_info["tolerance_reached"]),
                "stagnated": bool(stage_stagnated),
                "stagnation_iteration": int(
                    stagnation_state.get("iteration", -1)
                ),
                "stagnation_improvement": stagnation_state.get("improvement"),
                "accepted_iterations": int(run_info["accepted_iterations"]),
                "stage_wall_seconds": float(run_info["stage_wall_seconds"]),
                "f_guided_final": run_info["f_guided_final"],
                "entropy_avg_final": run_info["entropy_avg_final"],
                "max_min_final": run_info["max_min_final"],
            })
        return f, y, run_info

    def entropy_stop_reached(self, ent_avg, max_min):
        entropy_ok = float(ent_avg) < float(self.args.entropy_stop_avg)
        if not bool(self.args.also_require_binary_stop):
            return entropy_ok
        return entropy_ok and float(max_min) < float(self.args.binary_stop_maxmin)

    def run_fixed_occupancy_positions(
        self,
        y0,
        run_kind="random_immigrant_prefit",
        resume_state=None,
    ):
        """FairChem-relax an image-reconstructed immigrant without changing atoms.

        The image prefit has independently rounded hard W/Se/null identities
        without enforcing exact WSe2 counts. This path deletes exact-null
        proposals, freezes the selected count/species, and minimizes the same
        beta=0 guided objective over positions only.
        """
        y = self.project(y0)
        y = self.remove_rounded_null_sites(
            y,
            context="random_immigrant_before_fixed_occupancy_fairchem",
        )
        fixed_occ, _ = self.unpack_y(y)
        counts = np.rint(
            fixed_occ.detach().sum(dim=0).cpu().numpy()
        ).astype(int)
        self.exact_composition_projection_summary = {
            "enabled": False,
            "source": "wse2_random_immigrant_independent_argmax",
            "reason": "exact_composition_is_intentionally_disabled",
            "actual_edit_count": 0,
            "counts": {
                symbol: int(counts[index])
                for index, symbol in enumerate(self.species)
            },
        }
        stage = int(resume_state.get("stage", 0)) if resume_state else 0
        effective_run_kind = (
            str(resume_state.get("run_kind", ""))
            if resume_state
            else ""
        ) or f"{run_kind}_final_positions"
        terminal_print(
            self.args,
            "[random-immigrant FairChem] freezing reconstructed atom count, "
            "species, and occupancies; optimizing positions only with the full "
            f"guided image objective (sites={self.n_sites}, no exact composition)",
        )
        self.model_evaluation_tracker.phase = "final_PBFGS"
        _, y, run_info = self.run_stage(
            y,
            0.0,
            stage,
            run_kind=effective_run_kind,
            attempt_index=1,
            maxit_override=int(self.args.final_pbfgs_maxit),
            tol_override=float(self.args.final_pbfgs_tol),
            stagnation_patience_override=0,
            fixed_occupancies=fixed_occ,
            cost_plateau_tol_override=float(
                self.args.final_pbfgs_cost_plateau_tol
            ),
            cost_plateau_patience_override=int(
                self.args.final_pbfgs_cost_plateau_patience
            ),
            cost_plateau_min_iterations_override=int(
                self.args.final_pbfgs_cost_plateau_min_iterations
            ),
            record_history=True,
        )
        terminal_print(
            self.args,
            "[random-immigrant FairChem done] "
            f"termination={run_info['termination_reason']} "
            f"projected_gradient_error={run_info['final_projected_gradient_error']}",
        )
        return y

    def run(self, y0, run_kind="main", resume_state=None):
        y = y0
        self.model_evaluation_tracker.phase = "syntropization"
        resume_run_kind = (
            str(resume_state.get("run_kind", ""))
            if resume_state is not None
            else ""
        )
        beta = (
            float(resume_state["beta"])
            if resume_state is not None
            else float(self.args.beta0)
        )
        stage = (
            int(resume_state["stage"])
            if resume_state is not None
            else 0
        )
        max_stages = int(self.args.beta_max_stages)

        if resume_run_kind.endswith("_final_positions"):
            self.model_evaluation_tracker.phase = "final_PBFGS"
            y = self.remove_rounded_null_sites(
                y,
                context="resumed_final_positions",
            )
            fixed_occ, _ = self.unpack_y(y)
            _, y, final_pbfgs_info = self.run_stage(
                y,
                0.0,
                stage,
                run_kind=resume_run_kind,
                attempt_index=1,
                maxit_override=int(self.args.final_pbfgs_maxit),
                tol_override=float(self.args.final_pbfgs_tol),
                stagnation_patience_override=0,
                fixed_occupancies=fixed_occ,
                cost_plateau_tol_override=float(
                    self.args.final_pbfgs_cost_plateau_tol
                ),
                cost_plateau_patience_override=int(
                    self.args.final_pbfgs_cost_plateau_patience
                ),
                cost_plateau_min_iterations_override=int(
                    self.args.final_pbfgs_cost_plateau_min_iterations
                ),
                record_history=True,
            )
            terminal_print(
                self.args,
                "[final PBFGS resumed done] "
                f"termination={final_pbfgs_info['termination_reason']} "
                f"projected_gradient_error="
                f"{final_pbfgs_info['final_projected_gradient_error']}",
            )
            return y

        while True:
            if max_stages > 0 and stage >= max_stages:
                occ, _ = self.unpack_y(y)
                ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
                terminal_print(
                    self.args,
                    f"[warning] reached --beta-max-stages={max_stages} before entropy target: "
                    f"entropy/site={ent_avg:.6e}, target={float(self.args.entropy_stop_avg):.6e}, "
                    f"max_nonbinary={max_min:.3e}. Returning current state."
                )
                break

            terminal_print(self.args, "\n" + "=" * 88)
            terminal_print(self.args, f"[stage {stage+1}] beta={beta:.3e}")
            terminal_print(self.args, "=" * 88)

            _, y, _ = self.run_stage(
                y,
                beta,
                stage,
                run_kind=run_kind,
                attempt_index=0,
                record_history=True,
            )

            occ, _ = self.unpack_y(y)
            ent_avg, max_min = binarity_metrics(occ, self.tsallis_q)
            terminal_print(
                self.args,
                f"[stage {stage+1} done] entropy/site={ent_avg:.6e}, "
                f"target={float(self.args.entropy_stop_avg):.6e}, max_nonbinary={max_min:.3e}"
            )
            if self.entropy_stop_reached(ent_avg, max_min):
                break
            beta *= float(self.args.beta_mult)
            stage += 1

        y_soft = y
        y = self.rounded_state(y_soft)
        exact_composition_enforced = bool(
            self.args.exact_composition_enforcement_enable
        )
        if exact_composition_enforced:
            y = self.exact_composition_project_rounded_state(y_soft, y)
        else:
            self.exact_composition_projection_summary = {
                "enabled": False,
                "reason": "disabled_by_configuration",
            }
            message = (
                "exact_composition_projection disabled; retaining ordinary "
                "rounded occupancies"
            )
            append_running_log(self.args, message)
            terminal_print(self.args, f"[composition projection] {message}")
        y = self.remove_rounded_null_sites(
            y,
            context="before_final_positions",
        )
        if bool(self.args.final_pbfgs_enable):
            self.model_evaluation_tracker.phase = "final_PBFGS"
            fixed_occ, _ = self.unpack_y(y)
            final_occupancy_description = (
                "exact composition projected"
                if exact_composition_enforced
                else "ordinary rounded composition retained"
            )
            terminal_print(
                self.args,
                f"[final PBFGS] {final_occupancy_description}; freezing all "
                "occupancies/species and optimizing positions only with beta=0, "
                f"tol={float(self.args.final_pbfgs_tol):.3e}, "
                f"maxit={int(self.args.final_pbfgs_maxit)}, "
                f"cost_plateau_tol="
                f"{float(self.args.final_pbfgs_cost_plateau_tol):.3e}, "
                f"cost_plateau_patience="
                f"{int(self.args.final_pbfgs_cost_plateau_patience)}, "
                f"cost_plateau_min_iterations="
                f"{int(self.args.final_pbfgs_cost_plateau_min_iterations)}",
            )
            _, y, final_pbfgs_info = self.run_stage(
                y,
                0.0,
                stage,
                run_kind=f"{run_kind}_final_positions",
                attempt_index=1,
                maxit_override=int(self.args.final_pbfgs_maxit),
                tol_override=float(self.args.final_pbfgs_tol),
                stagnation_patience_override=0,
                fixed_occupancies=fixed_occ,
                cost_plateau_tol_override=float(
                    self.args.final_pbfgs_cost_plateau_tol
                ),
                cost_plateau_patience_override=int(
                    self.args.final_pbfgs_cost_plateau_patience
                ),
                cost_plateau_min_iterations_override=int(
                    self.args.final_pbfgs_cost_plateau_min_iterations
                ),
                record_history=True,
            )
            terminal_print(
                self.args,
                "[final PBFGS done] "
                f"termination={final_pbfgs_info['termination_reason']} "
                f"tolerance_reached="
                f"{bool(final_pbfgs_info['tolerance_reached'])} "
                f"cost_plateau_reached="
                f"{bool(final_pbfgs_info['cost_plateau_reached'])} "
                f"projected_gradient_error="
                f"{final_pbfgs_info['final_projected_gradient_error']}",
            )
        return y

    def rounded_state(self, y):
        occ, z = self.unpack_y(y)
        occ = self.project_allowed_occupancies(occ)
        null = torch.clamp(1.0 - occ.sum(dim=1), min=0.0)
        probs = torch.cat([occ, null[:, None]], dim=1)
        winners = torch.argmax(probs, dim=1)
        rounded = torch.zeros_like(occ)
        real = winners < self.n_species
        if bool(real.any().item()):
            rounded[real, winners[real]] = 1.0
        return self.project(self.pack_y(rounded, z))

    def remove_rounded_null_sites(self, y, context):
        """Delete exact-null ghosts after occupancies have become immutable.

        A rounded site with zero real occupancy contributes no energy, image,
        composition, or boundary signal. Keeping it in the positions-only
        FairChem relaxation only enlarges the neighbor graph.
        """
        occ, z = self.unpack_y(y)
        keep = occ.sum(dim=1) > 0.5
        kept_count = int(keep.sum().item())
        original_count = int(self.n_sites)
        if kept_count <= 0:
            raise RuntimeError(
                "Rounded structure contains no occupied sites; cannot run the "
                "final positions-only relaxation."
            )
        if kept_count == original_count:
            return y
        keep_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
        keep_numpy = keep_indices.detach().cpu().numpy().astype(int)
        compact_layout = self.assigned_site_region_indices[keep_numpy]
        compact_occ = occ.index_select(0, keep_indices)
        compact_z = z.index_select(0, keep_indices)
        self.set_site_layout(compact_layout)
        compact_y = self.project(self.pack_y(compact_occ, compact_z))
        message = (
            f"removed_exact_null_sites context={context} "
            f"before={original_count} occupied={kept_count} "
            f"removed={original_count - kept_count}"
        )
        append_running_log(self.args, message)
        terminal_print(self.args, f"[null-site compaction] {message}")
        return compact_y

    def exact_composition_project_rounded_state(self, y_soft, y_rounded):
        """Minimally edit a rounded GGG/null state to exact Gd3Ga5O12 counts."""
        from scipy.optimize import linear_sum_assignment

        soft_occ, _ = self.unpack_y(y_soft)
        hard_occ, z = self.unpack_y(y_rounded)
        soft_occ = self.project_allowed_occupancies(soft_occ)
        hard_occ = self.project_allowed_occupancies(hard_occ)
        soft_null = torch.clamp(1.0 - soft_occ.sum(dim=1), min=0.0)
        hard_null = torch.clamp(1.0 - hard_occ.sum(dim=1), min=0.0)
        soft_probs = torch.cat([soft_occ, soft_null[:, None]], dim=1).detach().cpu().numpy()
        hard_probs = torch.cat([hard_occ, hard_null[:, None]], dim=1)
        hard_labels = torch.argmax(hard_probs, dim=1).detach().cpu().numpy().astype(int)
        null_index = int(self.n_species)
        gd_index = int(self.species.index("Gd"))
        ga_index = int(self.species.index("Ga"))
        oxygen_index = int(self.species.index("O"))
        layout = np.asarray(self.assigned_blob_indices, dtype=int)
        heavy_sites = np.where(layout == self.heavy_site_class_index)[0]
        oxygen_sites = np.where(layout == self.oxygen_site_class_index)[0]

        current_counts = np.bincount(
            hard_labels[hard_labels < self.n_species],
            minlength=self.n_species,
        ).astype(int)
        current_heavy = int(current_counts[gd_index] + current_counts[ga_index])
        current_oxygen = int(current_counts[oxygen_index])
        max_formula_units = min(len(heavy_sites) // 8, len(oxygen_sites) // 12)
        if max_formula_units < 1:
            raise ValueError(
                "The current garnet site inventory cannot represent even one "
                "Gd3Ga5O12 formula unit."
            )

        feasible = []
        for formula_units in range(1, max_formula_units + 1):
            target_gd = 3 * formula_units
            target_ga = 5 * formula_units
            target_oxygen = 12 * formula_units
            heavy_edits_twice = (
                abs(int(current_counts[gd_index]) - target_gd)
                + abs(int(current_counts[ga_index]) - target_ga)
                + abs(current_heavy - (target_gd + target_ga))
            )
            if heavy_edits_twice % 2 != 0:
                raise RuntimeError("Internal exact-composition edit count is not integral.")
            edit_count = heavy_edits_twice // 2 + abs(current_oxygen - target_oxygen)
            feasible.append((int(edit_count), int(formula_units)))
        minimum_edits = min(item[0] for item in feasible)
        tied_formula_units = [
            formula_units
            for edit_count, formula_units in feasible
            if edit_count == minimum_edits
        ]

        confidence_cost = -np.log(np.clip(soft_probs, 1.0e-12, 1.0))
        edit_weight = 1.0e6

        def assignment_for_formula_units(formula_units):
            target_gd = 3 * formula_units
            target_ga = 5 * formula_units
            target_oxygen = 12 * formula_units
            heavy_slot_labels = np.asarray(
                [gd_index] * target_gd
                + [ga_index] * target_ga
                + [null_index] * (len(heavy_sites) - target_gd - target_ga),
                dtype=int,
            )
            heavy_before = hard_labels[heavy_sites]
            heavy_cost = (
                edit_weight
                * (heavy_before[:, None] != heavy_slot_labels[None, :])
                + confidence_cost[heavy_sites[:, None], heavy_slot_labels[None, :]]
            )
            row_indices, column_indices = linear_sum_assignment(heavy_cost)
            heavy_after = np.full(len(heavy_sites), null_index, dtype=int)
            heavy_after[row_indices] = heavy_slot_labels[column_indices]

            oxygen_before = hard_labels[oxygen_sites]
            oxygen_cost = (
                edit_weight * (oxygen_before != oxygen_index)
                + confidence_cost[oxygen_sites, oxygen_index]
            )
            null_cost = (
                edit_weight * (oxygen_before != null_index)
                + confidence_cost[oxygen_sites, null_index]
            )
            oxygen_priority = oxygen_cost - null_cost
            selected_oxygen = np.argsort(oxygen_priority, kind="stable")[:target_oxygen]
            oxygen_after = np.full(len(oxygen_sites), null_index, dtype=int)
            oxygen_after[selected_oxygen] = oxygen_index

            labels_after = np.full(self.n_sites, null_index, dtype=int)
            labels_after[heavy_sites] = heavy_after
            labels_after[oxygen_sites] = oxygen_after
            edits = int(np.count_nonzero(labels_after != hard_labels))
            selected_confidence_cost = float(
                np.sum(confidence_cost[np.arange(self.n_sites), labels_after])
            )
            return edits, selected_confidence_cost, labels_after

        candidates = []
        for formula_units in tied_formula_units:
            edits, selected_confidence_cost, labels_after = (
                assignment_for_formula_units(formula_units)
            )
            candidates.append(
                (
                    int(edits),
                    float(selected_confidence_cost),
                    int(formula_units),
                    labels_after,
                )
            )
        edits, selected_confidence_cost, formula_units, labels_after = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if edits != minimum_edits:
            raise RuntimeError(
                "Exact-composition assignment did not achieve the predicted "
                "minimum number of edits."
            )

        exact_occ = torch.zeros_like(hard_occ)
        real_sites = np.where(labels_after < self.n_species)[0]
        if len(real_sites):
            rows = torch.as_tensor(real_sites, device=exact_occ.device, dtype=torch.long)
            columns = torch.as_tensor(
                labels_after[real_sites],
                device=exact_occ.device,
                dtype=torch.long,
            )
            exact_occ[rows, columns] = 1.0
        projected = self.project(self.pack_y(exact_occ, z))
        counts_after = np.bincount(
            labels_after[labels_after < self.n_species],
            minlength=self.n_species,
        ).astype(int)
        target_counts = np.zeros(self.n_species, dtype=int)
        target_counts[gd_index] = 3 * formula_units
        target_counts[ga_index] = 5 * formula_units
        target_counts[oxygen_index] = 12 * formula_units
        if not np.array_equal(counts_after, target_counts):
            raise RuntimeError("Exact garnet composition projection failed its count check.")

        before_real = hard_labels < self.n_species
        after_real = labels_after < self.n_species
        summary = {
            "enabled": True,
            "formula_units": int(formula_units),
            "minimum_edit_count": int(minimum_edits),
            "actual_edit_count": int(edits),
            "additions": int(np.count_nonzero(~before_real & after_real)),
            "removals": int(np.count_nonzero(before_real & ~after_real)),
            "species_changes": int(
                np.count_nonzero(
                    before_real & after_real & (hard_labels != labels_after)
                )
            ),
            "counts_before": {
                symbol: int(current_counts[index])
                for index, symbol in enumerate(self.species)
            },
            "counts_after": {
                symbol: int(counts_after[index])
                for index, symbol in enumerate(self.species)
            },
            "target_counts": {
                symbol: int(target_counts[index])
                for index, symbol in enumerate(self.species)
            },
            "candidate_formula_units": [
                {"formula_units": int(k), "minimum_edits": int(edit_count)}
                for edit_count, k in feasible
            ],
            "pre_round_confidence_cost": float(selected_confidence_cost),
        }
        self.exact_composition_projection_summary = summary
        message = (
            "exact_composition_projection "
            f"formula_units={formula_units} edits={edits} "
            f"additions={summary['additions']} removals={summary['removals']} "
            f"species_changes={summary['species_changes']} "
            f"counts_before={summary['counts_before']} "
            f"counts_after={summary['counts_after']}"
        )
        append_running_log(self.args, message)
        terminal_print(self.args, f"[composition projection] {message}")
        return projected

    def write_structure(self, path_base, y, rounded):
        occ, z = self.unpack_y(y)
        occ = self.project_allowed_occupancies(occ)
        occ_np = occ.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        total = occ_np.sum(axis=1)
        winners = np.argmax(occ_np, axis=1)
        if rounded:
            null = np.maximum(1.0 - total, 0.0)
            keep = np.where(np.max(occ_np, axis=1) > null)[0]
        else:
            keep = np.where(total > float(self.args.keep_thresh))[0]
        symbols = [self.species[int(winners[i])] for i in keep]
        atoms = Atoms(symbols=symbols, positions=self.z_to_positions(z_np)[keep], cell=self.cell, pbc=self.pbc)
        atoms.wrap()
        write(self.output_dir / f"{path_base}_structure.xyz", atoms)
        write(self.output_dir / f"{path_base}_structure.cif", atoms)

    def final_snapshot(self, y, run_kind):
        cache = self._objective_result_cache
        cache_hit = bool(
            cache is not None
            and cache["evaluated_y"].shape == y.shape
            and cache["evaluated_y"].device == y.device
            and cache["evaluated_y"].dtype == y.dtype
            and (
                torch.equal(cache["evaluated_y"], y)
                or torch.equal(cache["input_y"], y)
            )
        )
        if cache_hit:
            f = torch.tensor(cache["f"], device=y.device, dtype=y.dtype)
            g = cache["gradient"].detach().clone()
        else:
            f, g = self.objfun(y, 0.0)
        snap = self.snapshot(
            y,
            stage=-1,
            iteration=-1,
            beta=0.0,
            f_objective=f,
            err=self.projected_gradient_error(y, g),
            run_kind=run_kind,
        )
        snap["is_final_state"] = True
        snap["energy_source"] = "objective_cache" if cache_hit else "fresh_model_evaluation"
        return snap

    def realized_structure_payload(self, y, rounded):
        occ, z = self.unpack_y(y)
        occ = self.project_allowed_occupancies(occ)
        occ_np = occ.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        total = occ_np.sum(axis=1)
        winners = np.argmax(occ_np, axis=1)
        if rounded:
            null = np.maximum(1.0 - total, 0.0)
            keep = np.where(np.max(occ_np, axis=1) > null)[0]
        else:
            keep = np.where(total > float(self.args.keep_thresh))[0]
        positions = self.z_to_positions(z_np)[keep]
        symbols = [self.species[int(winners[i])] for i in keep]
        return {
            "rounded": bool(rounded),
            "symbols": symbols,
            "cart_positions_angstrom": positions.tolist(),
            "frac_coords": self.positions_to_frac(positions).tolist() if len(keep) else [],
            "cell_angstrom": self.cell.tolist(),
            "kept_site_indices": keep.astype(int).tolist(),
            "n_atoms": int(len(keep)),
        }

    def frame_matches_state(self, frame, state, tol=1e-7):
        try:
            a_pos = np.asarray(frame["cart_positions_angstrom"], dtype=float)
            b_pos = np.asarray(state["cart_positions_angstrom"], dtype=float)
            a_occ = np.asarray(frame.get("species_occupancies", frame["occupancies"]), dtype=float)
            b_occ = np.asarray(state.get("species_occupancies", state["occupancies"]), dtype=float)
        except Exception:
            return False
        return a_pos.shape == b_pos.shape and a_occ.shape == b_occ.shape and np.allclose(a_pos, b_pos, atol=tol, rtol=0.0) and np.allclose(a_occ, b_occ, atol=tol, rtol=0.0)

    def write_member_slice_comparisons(self, result):
        """Write final continuous/syntropized image diagnostics for one GA member."""
        if not bool(getattr(self.args, "member_slice_comparisons_enable", True)):
            result["slice_comparison_dir"] = ""
            result["slice_comparison_written"] = False
            result["slice_comparison_disabled"] = True
            return ""
        generation = int(result["generation"])
        member_index = int(result["member_index"])
        kind_raw = str(result.get("kind", "member"))
        kind = safe_name(kind_raw) or "member"
        member_dir = (
            self.output_dir
            / "slice_comparisons"
            / f"generation_{generation:03d}_member_{member_index:03d}_{kind}"
        )
        member_dir.mkdir(parents=True, exist_ok=True)
        completion_marker = member_dir / "slice_comparison_complete.json"
        if completion_marker.exists():
            try:
                marker = read_json_file(completion_marker)
                marker_matches = (
                    int(marker.get("generation", -1)) == generation
                    and int(marker.get("member_index", -1)) == member_index
                    and str(marker.get("kind", "")) == kind_raw
                    and int(marker.get("seed", -1))
                    == int(result.get("seed", -2))
                    and math.isclose(
                        float(marker.get("score", float("nan"))),
                        float(result.get("score", float("nan"))),
                        rel_tol=1e-7,
                        abs_tol=1e-9,
                    )
                )
            except Exception:
                marker_matches = False
            if marker_matches:
                result["slice_comparison_dir"] = str(member_dir)
                result["slice_comparison_written"] = True
                terminal_print(
                    self.args,
                    f"[slice comparisons] generation={generation} "
                    f"member={member_index} already complete; keeping={member_dir}",
                )
                return str(member_dir)

        matplotlib_cache_dir = (
            Path(os.environ.get("TMPDIR", "/tmp"))
            / "eat_matplotlib_cache"
        )
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        layout = np.asarray(
            result.get(
                "assigned_site_region_indices",
                result.get("assigned_blob_indices", self.initial_site_layout()),
            ),
            dtype=int,
        ).reshape(-1)
        self.set_site_layout(layout)
        state = result["y"]
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device, dtype=self.dtype)
        else:
            state = state.to(device=self.device, dtype=self.dtype)
        occ, z = self.unpack_y(state)
        occ = self.project_allowed_occupancies(occ)
        auxiliary = self.auxiliary_imaging_and_boundary_terms(
            occ,
            z,
            want_grad=False,
        )

        target = np.asarray(self.experimental["target_image"], dtype=np.float32)
        predicted = (
            auxiliary["predicted_image"].detach().cpu().numpy().astype(np.float32)
        )
        roi = tuple(slice(int(lo), int(hi)) for lo, hi in zip(
            self.sample_roi_start_zyx, self.sample_roi_stop_zyx))
        target = target[roi]
        predicted = predicted[roi]
        residual = predicted - target
        mask = np.asarray(self.experimental["feasible_mask"], dtype=bool)[roi]
        # Plot and export these overlay coordinates in the TIFF crop frame.
        positions = self.z_to_positions(z.detach().cpu().numpy()) - self.vacuum_position_shift
        occ_np = occ.detach().cpu().numpy()
        site_occupancy = occ_np.sum(axis=1)
        heavy_occupancy = occ_np[:, self.heavy_species_indices].sum(axis=1)
        oxygen_occupancy = np.zeros_like(site_occupancy)
        dominant = np.argmax(occ_np, axis=1)
        level_set_A = np.asarray(
            auxiliary["boundary"]["level_set_A"],
            dtype=float,
        )
        active_threshold = max(float(self.args.keep_thresh), 1e-3)
        active = site_occupancy > active_threshold
        active_heavy = heavy_occupancy > active_threshold
        inside = level_set_A >= self.level_set_margin_A

        nz, ny, nx = target.shape
        dz_A, dy_A, dx_A = self.target_voxel_size_A
        x_A = np.arange(nx, dtype=float) * dx_A
        y_A = np.arange(ny, dtype=float) * dy_A
        slice_assignment = np.rint(positions[:, 2] / dz_A).astype(int)
        species_colors = {
            "W": "#d62728",
            "Se": "#2ca02c",
        }

        display_vmax = max(
            float(np.percentile(target, 99.5)),
            float(np.percentile(predicted, 99.5)),
            1e-12,
        )
        residual_limit = max(
            float(np.percentile(np.abs(residual), 99.5)),
            1e-12,
        )

        snapshot = result.get("final_snapshot", {})
        headline = (
            f"generation {generation:03d}, member {member_index:03d}, {kind} | "
            f"GA score={float(result.get('score', np.nan)):+.6f} | "
            f"image corr={float(auxiliary['image']['correlation']):+.4f} | "
            f"voxel Huber={float(auxiliary['image']['loss_raw']):.6g} | "
            
            f"weighted image={float(auxiliary['image']['objective_term']):.6g}"
        )
        detail_line = (
            f"Eform/atom={float(result.get('formation_energy_per_atom', np.nan)):+.6f} eV | "
            f"guided={float(result.get('f_guided_no_entropy', np.nan)):+.6f} | "
            f"Se-vacancy bias={float(result.get('se_vacancy_bias_term', 0.0)):+.6f} "
            f"(deficiency={float(result.get('se_vacancy_count_relative_to_reference', 0.0)):+.2f}) | "
            f"W/Se level-set term={float(auxiliary['boundary']['objective_term']):.6g} | "
            f"sample-box term={float(auxiliary['sample_box']['objective_term']):.6g} "
            f"(outside={int(auxiliary['sample_box']['active_outside_count'])}) | "
            f"sum occupancy={float(site_occupancy.sum()):.3f} | "
            f"active heavy inside={int(np.count_nonzero(active_heavy & inside))}/"
            f"{int(np.count_nonzero(active_heavy))}"
        )

        # Keep the diagnostic header inside the exported figure width.
        import textwrap
        headline = textwrap.fill(headline, width=130)
        detail_line = textwrap.fill(detail_line, width=130)

        def draw_contour(axis, z_index):
            plane = mask[z_index].astype(float)
            if plane.min() < 0.5 < plane.max():
                axis.contour(
                    x_A,
                    y_A,
                    plane,
                    levels=[0.5],
                    colors=["#00ffff"],
                    linewidths=1.0,
                )

        def draw_atoms(axis, z_index):
            indices = np.where(active & (slice_assignment == z_index))[0]
            for index in indices:
                symbol = self.species[int(dominant[index])]
                violates_heavy_mask = bool(active_heavy[index] and not inside[index])
                axis.scatter(
                    positions[index, 0],
                    positions[index, 1],
                    s=18.0 + 34.0 * float(site_occupancy[index]),
                    facecolor=species_colors.get(symbol, "white"),
                    edgecolor="#ffff00" if violates_heavy_mask else "white",
                    linewidth=1.8 if violates_heavy_mask else 0.6,
                    alpha=max(0.20, min(1.0, float(site_occupancy[index]))),
                    marker="X" if violates_heavy_mask else "o",
                    zorder=5,
                )

        slice_metrics = []
        for z_index in range(nz):
            diff = residual[z_index].astype(np.float64)
            abs_diff = np.abs(diff)
            delta = float(self.image_huber_delta)
            huber = np.where(
                abs_diff <= delta,
                0.5 * diff * diff,
                delta * (abs_diff - 0.5 * delta),
            )
            target_flat = target[z_index].ravel().astype(np.float64)
            predicted_flat = predicted[z_index].ravel().astype(np.float64)
            if target_flat.std() > 0.0 and predicted_flat.std() > 0.0:
                correlation = float(np.corrcoef(target_flat, predicted_flat)[0, 1])
            else:
                correlation = 0.0
            slice_metric = {
                "z_index": int(z_index),
                "z_angstrom": float(z_index * dz_A),
                "huber_loss": float(huber.mean()),
                "mae": float(abs_diff.mean()),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "correlation": correlation,
                "active_sites_assigned_to_slice": int(
                    np.count_nonzero(active & (slice_assignment == z_index))
                ),
                "active_heavy_outside_sites_assigned_to_slice": int(
                    np.count_nonzero(
                        active_heavy & ~inside & (slice_assignment == z_index)
                    )
                ),
            }
            slice_metrics.append(slice_metric)

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(13.2, 4.4),
                constrained_layout=True,
            )
            axes[0].imshow(
                target[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="gray",
                vmin=0.0,
                vmax=display_vmax,
            )
            axes[1].imshow(
                predicted[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="gray",
                vmin=0.0,
                vmax=display_vmax,
            )
            residual_image = axes[2].imshow(
                residual[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
            )
            for axis in axes:
                draw_contour(axis, z_index)
                axis.set_xlabel("x (Å)")
                axis.set_ylabel("y (Å)")
                axis.set_aspect("equal")
            draw_atoms(axes[1], z_index)
            axes[0].set_title(f"Experimental {self.image_target_transform} intensity")
            axes[1].set_title(
                f"Calibrated {self.image_target_transform} model + atoms"
            )
            axes[2].set_title("Simulated − experimental")
            fig.colorbar(
                residual_image,
                ax=axes[2],
                shrink=0.82,
                label=f"{self.image_target_transform}-intensity residual",
            )
            fig.suptitle(
                headline
                + "\n"
                + detail_line
                + "\n"
                + (
                    f"z={z_index} ({z_index * dz_A:.3f} Å) | "
                    f"slice corr={correlation:+.4f}, "
                    f"Huber={slice_metric['huber_loss']:.6g}, "
                    f"MAE={slice_metric['mae']:.6g}"
                ),
                fontsize=9.5,
            )
            fig.savefig(
                member_dir / f"slice_z{z_index:03d}_comparison.png",
                dpi=170,
            )
            plt.close(fig)

        overview, overview_axes = plt.subplots(
            nz,
            3,
            figsize=(12.0, max(3.0 * nz, 4.0)),
            constrained_layout=True,
            squeeze=False,
        )
        for z_index in range(nz):
            overview_axes[z_index, 0].imshow(
                target[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="gray",
                vmin=0.0,
                vmax=display_vmax,
            )
            overview_axes[z_index, 1].imshow(
                predicted[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="gray",
                vmin=0.0,
                vmax=display_vmax,
            )
            overview_axes[z_index, 2].imshow(
                residual[z_index],
                origin="lower",
                extent=(0.0, nx * dx_A, 0.0, ny * dy_A),
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
            )
            for column in range(3):
                axis = overview_axes[z_index, column]
                draw_contour(axis, z_index)
                axis.set_ylabel(f"z{z_index} ({z_index * dz_A:.2f} Å)")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal")
            draw_atoms(overview_axes[z_index, 1], z_index)
        overview_axes[0, 0].set_title(
            f"Experimental {self.image_target_transform}"
        )
        overview_axes[0, 1].set_title(
            f"Calibrated {self.image_target_transform} model"
        )
        overview_axes[0, 2].set_title("Residual")
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                label=symbol,
                markersize=7,
            )
            for symbol, color in species_colors.items()
        ]
        legend_handles.append(
            Line2D([0], [0], color="#00ffff", linewidth=1.5, label="heavy-allowed contour")
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#ffff00",
                label="active W/Se weight outside level set",
                markersize=7,
            )
        )
        overview.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=5,
            frameon=False,
        )
        overview.suptitle(headline + "\n" + detail_line, fontsize=10)
        overview.savefig(
            member_dir / "all_slices_overview.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(overview)

        metrics = {
            "generation": generation,
            "member_index": member_index,
            "kind": str(result.get("kind", "member")),
            "seed": int(result.get("seed", -1)),
            "state": "final_continuous_syntropized",
            "score": float(result.get("score", np.nan)),
            "formation_energy_per_atom": float(
                result.get("formation_energy_per_atom", np.nan)
            ),
            "f_guided_no_entropy": float(
                result.get("f_guided_no_entropy", np.nan)
            ),
            "composition_penalty": float(
                result.get("composition_penalty", np.nan)
            ),
            "se_vacancy_bias_term": float(
                result.get("se_vacancy_bias_term", 0.0)
            ),
            "se_vacancy_count_relative_to_reference": float(
                result.get("se_vacancy_count_relative_to_reference", 0.0)
            ),
            "se_occupancy_count": float(result.get("se_occupancy_count", 0.0)),
            "composition_fractions": list(
                result.get("composition_fractions", [])
            ),
            "sum_occupancy": float(site_occupancy.sum()),
            "entropy_avg": float(result.get("entropy_avg", np.nan)),
            "max_nonbinary": float(result.get("max_min", np.nan)),
            "image_loss_raw_huber": float(auxiliary["image"]["loss_raw"]),
            "voxel_image_term_weighted": float(
                auxiliary["image"]["voxel_objective_term"]
            ),
            "ellipse_composition_loss_raw_mse": float(
                auxiliary["image"]["ellipse_composition_loss_raw"]
            ),
            "ellipse_composition_term_weighted": float(
                auxiliary["image"]["ellipse_composition_objective_term"]
            ),
            "image_term_weighted": float(auxiliary["image"]["objective_term"]),
            "image_correlation": float(auxiliary["image"]["correlation"]),
            "heavy_region_term": float(
                auxiliary["boundary"]["objective_term"]
            ),
            "sample_box_term": float(
                auxiliary["sample_box"]["objective_term"]
            ),
            "sample_box_loss_raw_A2": float(
                auxiliary["sample_box"]["loss_raw_A2"]
            ),
            "sample_box_active_outside_count": int(
                auxiliary["sample_box"]["active_outside_count"]
            ),
            "sample_box_outside_occupancy_mass": float(
                auxiliary["sample_box"]["outside_occupancy_mass"]
            ),
            "level_set_margin_angstrom": float(self.level_set_margin_A),
            "candidate_site_count": int(self.n_sites),
            "heavy_class_site_count": int(
                np.count_nonzero(layout == self.heavy_site_class_index)
            ),
            "oxygen_class_site_count": 0,
            "active_site_threshold": active_threshold,
            "active_site_count": int(np.count_nonzero(active)),
            "active_heavy_site_count": int(np.count_nonzero(active_heavy)),
            "active_heavy_inside_count": int(
                np.count_nonzero(active_heavy & inside)
            ),
            "active_heavy_outside_count": int(
                np.count_nonzero(active_heavy & ~inside)
            ),
            "outside_heavy_occupancy_mass": float(
                np.sum(heavy_occupancy[~inside])
            ),
            "oxygen_occupancy_outside_heavy_region": float(
                np.sum(oxygen_occupancy[~inside])
            ),
            "minimum_active_heavy_level_set_angstrom": (
                float(np.min(level_set_A[active_heavy]))
                if np.any(active_heavy)
                else None
            ),
            "target_shape_zyx": list(target.shape),
            "target_voxel_size_zyx_angstrom": self.target_voxel_size_A.tolist(),
            "shared_intensity_display_vmax": display_vmax,
            "residual_display_limit": residual_limit,
            "slice_metrics": slice_metrics,
            "final_snapshot": {
                key: snapshot[key]
                for key in (
                    "image_term",
                    "image_loss_raw",
                    "image_correlation",
                    "assigned_blob_term",
                    "composition_term",
                    "se_vacancy_bias_term",
                    "se_vacancy_count_relative_to_reference",
                    "se_occupancy_count",
                    "sumx",
                    "entropy_avg",
                    "max_min",
                )
                if key in snapshot
            },
        }
        (member_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            member_dir / "comparison_fields.npz",
            experimental_target_image=target,
            simulated_transformed_gaussian=predicted,
            simulated_linear_gaussian=auxiliary["predicted_linear"].detach().cpu().numpy(),
            image_transform=np.asarray(self.image_target_transform),
            residual_sim_minus_experiment=residual,
            heavy_allowed_mask=mask,
            positions_xyz_angstrom=positions,
            position_coordinate_frame=np.asarray("TIFF_crop_without_added_padding"),
            species_occupancies=occ_np,
            total_occupancy=site_occupancy,
            heavy_occupancy=heavy_occupancy,
            oxygen_occupancy=oxygen_occupancy,
            dominant_species=np.asarray(
                [self.species[int(index)] for index in dominant]
            ),
            level_set_angstrom=level_set_A,
            assigned_z_slice=slice_assignment,
            site_class_index=layout,
            site_class_name=np.asarray(
                [self.blob_symbols[int(index)] for index in layout]
            ),
            target_voxel_size_zyx_angstrom=self.target_voxel_size_A,
        )
        write_json_atomic(
            completion_marker,
            {
                "complete": True,
                "generation": generation,
                "member_index": member_index,
                "kind": kind_raw,
                "seed": int(result.get("seed", -1)),
                "score": float(result.get("score", np.nan)),
                "completed_at": timestamp_now(),
            },
        )
        result["slice_comparison_dir"] = str(member_dir)
        result["slice_comparison_written"] = True
        terminal_print(
            self.args,
            f"[slice comparisons] generation={generation} member={member_index} "
            f"saved={member_dir}",
        )
        return str(member_dir)

    def write_final_image_fields(self, y_cont, y_round):
        import tifffile

        fields = {}
        ellipse_state = {}
        for label, state in (("continuous", y_cont), ("rounded", y_round)):
            occ, z = self.unpack_y(state)
            occ = self.project_allowed_occupancies(occ)
            auxiliary = self.auxiliary_imaging_and_boundary_terms(
                occ,
                z,
                want_grad=False,
            )
            predicted_image = auxiliary["predicted_image"].cpu().numpy().astype(np.float32)
            predicted_linear = auxiliary["predicted_linear"].cpu().numpy().astype(np.float32)
            fields[f"{label}_predicted_image"] = predicted_image
            fields[f"{label}_predicted_linear"] = predicted_linear
            wrapped_positions = torch.remainder(
                z,
                torch.as_tensor(
                    np.diag(self.cell).copy(), device=z.device, dtype=z.dtype
                ),
            )
            ellipse_terms = self.ellipse_composition_terms(
                occ, wrapped_positions
            )
            ellipse_state[label] = {
                key: value.detach().cpu().numpy()
                for key, value in ellipse_terms.items()
                if torch.is_tensor(value)
            }
            fields[f"{label}_ellipse_predicted_integrals"] = ellipse_state[label][
                "predicted_integrals"
            ]
            fields[f"{label}_ellipse_residuals_Se_units"] = ellipse_state[label][
                "residuals_Se_units"
            ]
            fields[f"{label}_ellipse_region_species_occupancies"] = ellipse_state[label][
                "region_species_occupancies"
            ]
            display_vmax = max(
                float(np.percentile(self.experimental["target_image"], 99.5)),
                1e-12,
            )
            display_u16 = np.round(
                np.clip(predicted_image / display_vmax, 0.0, 1.0) * 65535.0
            ).astype(np.uint16)
            tifffile.imwrite(
                self.output_dir
                / f"final_{label}_predicted_{self.image_target_transform}_ZYX.tif",
                display_u16,
                imagej=True,
                metadata={
                    "axes": "ZYX",
                    "unit": "angstrom",
                    "spacing": float(self.target_voxel_size_A[0]),
                    "image_correlation": float(auxiliary["image"]["correlation"]),
                    "display_vmin_raw": 0.0,
                    "display_vmax_raw": display_vmax,
                    "image_transform": self.image_target_transform,
                },
            )
        np.savez_compressed(
            self.output_dir / "final_gaussian_image_comparison.npz",
            experimental_target_image=self.experimental["target_image"],
            experimental_image_loss_mask=self.experimental["image_loss_mask"],
            sample_roi_start_zyx=self.experimental["sample_roi_start_zyx"],
            sample_roi_stop_zyx=self.experimental["sample_roi_stop_zyx"],
            sample_roi_z_slices=np.asarray(
                [
                    self.experimental["sample_roi_z_start"],
                    self.experimental["sample_roi_z_stop"],
                ],
                dtype=int,
            ),
            experimental_target_linear_normalized=self.experimental[
                "target_linear_normalized"
            ],
            experimental_target_sqrt_for_heavy_mask=self.experimental["target_sqrt"],
            image_transform=np.asarray(self.image_target_transform),
            image_transform_epsilon=np.asarray(self.gaussian_transform_epsilon),
            heavy_allowed_mask=self.experimental["heavy_allowed_mask"],
            heavy_level_set_A=self.experimental["heavy_level_set_A"],
            target_voxel_size_A=self.target_voxel_size_A,
            cell=self.cell,
            gaussian_species=np.asarray(self.species),
            gaussian_total_integrated_amplitudes=self.gaussian_amplitudes,
            gaussian_component_integrated_amplitudes=self.gaussian_component_amplitudes,
            gaussian_fitted_component_sigma_xy_angstrom=self.gaussian_fitted_sigma_xy_A,
            gaussian_component_sigma_xy_angstrom=self.gaussian_sigma_xy_A,
            gaussian_additional_lateral_blur_angstrom=np.asarray(
                self.gaussian_additional_lateral_blur_A
            ),
            depth_response_rms_angstrom=np.asarray(self.gaussian_shared_sigma_z_A),
            gaussian_shared_sigma_z_angstrom=np.asarray(self.gaussian_shared_sigma_z_A),
            depth_response_model=np.asarray("analytic_gaussian_no_envelope"),
            gaussian_kernel_peak_scale=np.asarray(self.gaussian_renderer.gain),
            gaussian_intensity_normalization=np.asarray("unit_isolated_grid_aligned_W_peak"),
            target_normalization=np.asarray(self.experimental["target_normalization"]),
            gaussian_element_fit_csv=np.asarray(str(self.gaussian_element_fit_csv)),
            gaussian_relative_intensity_reference_species=np.asarray(
                self.gaussian_relative_intensity_reference_species
            ),
            gaussian_reference_integrated_amplitude=np.asarray(
                self.gaussian_reference_integrated_amplitude
            ),
            gaussian_phase_table_to_image_amplitude_scale=np.asarray(
                self.gaussian_phase_table_to_image_amplitude_scale
            ),
            ellipse_centers_xy_angstrom=self.experimental["log_ellipse_regions"][
                "centers_xy_A"
            ],
            ellipse_semi_major_angstrom=self.experimental["log_ellipse_regions"][
                "semi_major_A"
            ],
            ellipse_semi_minor_angstrom=self.experimental["log_ellipse_regions"][
                "semi_minor_A"
            ],
            ellipse_angles_degrees=self.experimental["log_ellipse_regions"][
                "angles_degrees"
            ],
            ellipse_target_integrals=self.experimental["log_ellipse_regions"][
                "target_integrals"
            ],
            ellipse_I_Se=np.asarray(self.ellipse_I_Se),
            ellipse_I_W=np.asarray(self.ellipse_I_W),
            **fields,
        )
        if self.ellipse_region_count == 0:
            return
        W_index = self.species.index("W")
        Se_index = self.species.index("Se")
        region_source = self.experimental["log_ellipse_regions"]
        rows = []
        for region_index, ellipse in enumerate(region_source["ellipses"]):
            row = {
                "region_id": int(region_index),
                "center_x_pixel": float(ellipse["center_x_pixel"]),
                "center_y_pixel": float(ellipse["center_y_pixel"]),
                "semi_major_angstrom": float(ellipse["semi_major_angstrom"]),
                "semi_minor_angstrom": float(ellipse["semi_minor_angstrom"]),
                "angle_degrees_from_x": float(ellipse["angle_degrees_from_x"]),
                "target_integrated_intensity": float(
                    region_source["target_integrals"][region_index]
                ),
                "target_intensity_Se_units": float(
                    region_source["target_integrals"][region_index] / self.ellipse_I_Se
                ),
                "edge_low_confidence": bool(ellipse["edge_low_confidence"]),
            }
            for label in ("continuous", "rounded"):
                counts = ellipse_state[label]["region_species_occupancies"][region_index]
                row[f"{label}_W_occupancy"] = float(counts[W_index])
                row[f"{label}_Se_occupancy"] = float(counts[Se_index])
                row[f"{label}_predicted_integrated_intensity"] = float(
                    ellipse_state[label]["predicted_integrals"][region_index]
                )
                row[f"{label}_residual_Se_units"] = float(
                    ellipse_state[label]["residuals_Se_units"][region_index]
                )
            rows.append(row)
        with (self.output_dir / "final_ellipse_composition_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_lightweight_final_ellipse_comparison(self, y_cont, y_round):
        """Legacy compatibility; there are no ellipse regions in the new model."""
        if self.ellipse_region_count == 0:
            return
        states = {}
        for label, state in (("continuous", y_cont), ("rounded", y_round)):
            occ, z = self.unpack_y(state)
            occ = self.project_allowed_occupancies(occ)
            positions = torch.remainder(
                z,
                torch.as_tensor(np.diag(self.cell).copy(), device=z.device, dtype=z.dtype),
            )
            terms = self.ellipse_composition_terms(occ, positions)
            states[label] = {
                key: value.detach().cpu().numpy()
                for key, value in terms.items()
                if torch.is_tensor(value)
            }
        source = self.experimental["log_ellipse_regions"]
        W_index, Se_index = self.species.index("W"), self.species.index("Se")
        rows = []
        for region_index, ellipse in enumerate(source["ellipses"]):
            row = {
                "region_id": int(region_index),
                "center_x_pixel": float(ellipse["center_x_pixel"]),
                "center_y_pixel": float(ellipse["center_y_pixel"]),
                "semi_major_angstrom": float(ellipse["semi_major_angstrom"]),
                "semi_minor_angstrom": float(ellipse["semi_minor_angstrom"]),
                "angle_degrees_from_x": float(ellipse["angle_degrees_from_x"]),
                "target_integrated_intensity": float(source["target_integrals"][region_index]),
                "target_intensity_Se_units": float(source["target_integrals"][region_index] / self.ellipse_I_Se),
                "edge_low_confidence": bool(ellipse["edge_low_confidence"]),
            }
            for label in ("continuous", "rounded"):
                counts = states[label]["region_species_occupancies"][region_index]
                row[f"{label}_W_occupancy"] = float(counts[W_index])
                row[f"{label}_Se_occupancy"] = float(counts[Se_index])
                row[f"{label}_predicted_integrated_intensity"] = float(
                    states[label]["predicted_integrals"][region_index]
                )
                row[f"{label}_residual_Se_units"] = float(
                    states[label]["residuals_Se_units"][region_index]
                )
            rows.append(row)
        with (self.output_dir / "final_ellipse_composition_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            self.output_dir / "final_ellipse_composition_comparison.npz",
            target_integrals=source["target_integrals"],
            centers_xy_A=source["centers_xy_A"],
            semi_major_A=source["semi_major_A"],
            semi_minor_A=source["semi_minor_A"],
            angles_degrees=source["angles_degrees"],
            I_Se=np.asarray(self.ellipse_I_Se),
            I_W=np.asarray(self.ellipse_I_W),
            continuous_region_species_occupancies=states["continuous"]["region_species_occupancies"],
            continuous_predicted_integrals=states["continuous"]["predicted_integrals"],
            continuous_residuals_Se_units=states["continuous"]["residuals_Se_units"],
            rounded_region_species_occupancies=states["rounded"]["region_species_occupancies"],
            rounded_predicted_integrals=states["rounded"]["predicted_integrals"],
            rounded_residuals_Se_units=states["rounded"]["residuals_Se_units"],
        )

    def write_outputs(self, y_cont, y_round, seed):
        self.write_structure("continuous", y_cont, rounded=False)
        self.write_structure("rounded", y_round, rounded=True)
        if bool(getattr(self, "lightweight_slurm_driver", False)):
            self.write_lightweight_final_ellipse_comparison(y_cont, y_round)
            payload = {
                "generator": str(Path(__file__).resolve()),
                "seed": int(seed),
                "blob_json": str(self.blob_path),
                "species": list(self.species),
                "atomic_numbers": [int(z) for z in self.atomic_numbers],
                "target_blobs": int(len(self.blobs)),
                "candidate_site_density_per_angstrom3": float(
                    self.heavy_site_density_per_angstrom3
                ),
                "initialization_scheme": "intensity_prioritized_cylinder_with_angstrom_jitter",
                "initial_cylinder_lateral_radius_angstrom": float(
                    self.initial_cylinder_lateral_radius_A
                ),
                "initial_cylinder_z_radius_angstrom": float(
                    self.initial_cylinder_z_radius_A
                ),
                "initial_jitter_angstrom": float(self.initial_jitter_A),
                "initial_candidate_sites": int(self.initial_heavy_region_sites),
                "feasible_sample_volume_angstrom3": float(
                    self.feasible_volume_angstrom3
                ),
                "sample_volume_angstrom3": float(self.sample_volume_angstrom3),
                "padded_cell_volume_angstrom3": float(self.cell_volume_angstrom3),
                "initial_n_sites": int(self.initial_n_sites),
                "n_sites": int(self.n_sites),
                "requested_vacuum_padding_angstrom": (
                    self.requested_vacuum_padding.tolist()
                ),
                "vacuum_padding_angstrom": self.vacuum_padding.tolist(),
                "pbc": self.pbc.astype(bool).tolist(),
                "species_by_source_type": {key: list(value) for key, value in self.species_by_source_type.items()},
                "se_vacancy_bias_enable": bool(self.se_vacancy_bias_enable),
                "se_vacancy_bias_alpha_eV_per_vacancy": float(
                    self.se_vacancy_bias_alpha_eV
                ),
                "se_vacancy_reference_atom_count": float(
                    self.se_vacancy_reference_atom_count
                ),
                "se_vacancy_reference_se_count": float(
                    self.se_vacancy_reference_se_count
                ),
                "reference_blob_source_types": list(self.blob_source_types),
                "reference_blob_allowed_elements": self.blob_allowed_elements_from_json,
                "reference_blob_allowed_species_mask": self.blob_allowed_species_mask.astype(bool).tolist(),
                "ga_enabled": bool(self.args.ga_enable),
                "ga_execution_backend": str(self.args.ga_execution_backend),
                "note": "Lightweight Slurm driver output: final objective snapshots are not recomputed on the submit machine. Use structures.json and genetic_optimization_summary.json for worker-computed scores.",
                "final_continuous_structure": self.realized_structure_payload(y_cont, rounded=False),
                "final_rounded_structure": self.realized_structure_payload(y_round, rounded=True),
                "trajectory_history_disabled": bool(self.args.disable_trajectory_history),
            }
            with (self.output_dir / "blob_optimization_history.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote lightweight Slurm driver history: {self.output_dir / 'blob_optimization_history.json'}")
            return
        final_continuous = self.final_snapshot(y_cont, "final_continuous")
        final_rounded = self.final_snapshot(y_round, "final_rounded")
        self.write_final_image_fields(y_cont, y_round)
        if (
            not bool(self.args.disable_trajectory_history)
            and not bool(self.args.ga_enable)
            and (not self.history or not self.frame_matches_state(self.history[-1], final_continuous))
        ):
            self.history.append(final_continuous)
        trajectory = [] if (bool(self.args.disable_trajectory_history) or bool(self.args.ga_enable)) else self.history
        payload = {
            "generator": str(Path(__file__).resolve()),
            "seed": int(seed),
            "experimental_tiff": str(self.blob_path),
            "experimental_crop_origin_zyx": self.experimental["crop_origin"].tolist(),
            "experimental_source_voxel_size_A": self.experimental["source_voxel_size_A"].tolist(),
            "target_shape_zyx": list(self.target_shape_zyx),
            "sample_shape_zyx": list(self.experimental["sample_shape_zyx"]),
            "target_voxel_size_A": self.target_voxel_size_A.tolist(),
            "sample_cell_lengths_A": self.experimental["sample_cell_lengths_A"].tolist(),
            "padded_cell_lengths_A": np.diag(self.cell).tolist(),
            "sample_roi_start_zyx": self.experimental[
                "sample_roi_start_zyx"
            ].tolist(),
            "sample_roi_stop_zyx": self.experimental[
                "sample_roi_stop_zyx"
            ].tolist(),
            "sample_roi_z_slices": [
                int(self.experimental["sample_roi_z_start"]),
                int(self.experimental["sample_roi_z_stop"]),
            ],
            "image_loss_voxel_count": int(self.image_loss_voxel_count),
            "feasible_threshold_mode": str(self.experimental["threshold_mode"]),
            "feasible_threshold_value": float(self.experimental["threshold_value"]),
            "feasible_sample_fraction": float(self.experimental["heavy_allowed_fraction"]),
            "image_penalty": float(self.image_penalty),
            "image_huber_delta": float(self.image_huber_delta),
            "image_target_transform": self.image_target_transform,
            "image_target_domain": (
                f"{self.image_target_transform}_of_background_subtracted_"
                "normalized_linear_TIFF"
            ),
            "image_transform_epsilon": float(self.gaussian_transform_epsilon),
            "image_forward_model": "species_resolved_double_xy_gaussian_gaussian_depth_no_envelope",
            "gaussian_total_integrated_amplitudes": {
                symbol: float(self.gaussian_amplitudes[index])
                for index, symbol in enumerate(self.species)
            },
            "gaussian_component_integrated_amplitudes": {
                symbol: {
                    "core": float(self.gaussian_component_amplitudes[index, 0]),
                    "tail": float(self.gaussian_component_amplitudes[index, 1]),
                }
                for index, symbol in enumerate(self.species)
            },
            "gaussian_component_sigma_xy_angstrom": {
                symbol: {
                    "core": float(self.gaussian_sigma_xy_A[index, 0]),
                    "tail": float(self.gaussian_sigma_xy_A[index, 1]),
                }
                for index, symbol in enumerate(self.species)
            },
            "gaussian_fitted_component_sigma_xy_angstrom": {
                symbol: {
                    "core": float(self.gaussian_fitted_sigma_xy_A[index, 0]),
                    "tail": float(self.gaussian_fitted_sigma_xy_A[index, 1]),
                }
                for index, symbol in enumerate(self.species)
            },
            "gaussian_additional_lateral_blur_angstrom": float(
                self.gaussian_additional_lateral_blur_A
            ),
            "gaussian_image_model": self.image_model_metadata,
            "gaussian_element_fit_csv": str(self.gaussian_element_fit_csv),
            "gaussian_relative_intensity_reference_species": self.gaussian_relative_intensity_reference_species,
            "gaussian_intensity_scale_source": self.gaussian_intensity_scale_source,
            "gaussian_rendered_single_atom_peak_amplitudes": {
                symbol: float(
                    self.gaussian_rendered_single_atom_peak_amplitudes[index]
                )
                for index, symbol in enumerate(self.species)
            },
            "gaussian_reference_integrated_amplitude": float(
                self.gaussian_reference_integrated_amplitude
            ),
            "gaussian_phase_table_to_image_amplitude_scale": float(
                self.gaussian_phase_table_to_image_amplitude_scale
            ),
            "depth_response_rms_angstrom": float(
                self.gaussian_shared_sigma_z_A
            ),
            "species": list(self.species),
            "atomic_numbers": [int(z) for z in self.atomic_numbers],
            "target_blobs": int(len(self.blobs)),
            "candidate_site_density_per_angstrom3": float(
                self.heavy_site_density_per_angstrom3
            ),
            "initialization_scheme": "intensity_prioritized_cylinder_with_angstrom_jitter",
            "initial_cylinder_lateral_radius_angstrom": float(
                self.initial_cylinder_lateral_radius_A
            ),
            "initial_cylinder_z_radius_angstrom": float(
                self.initial_cylinder_z_radius_A
            ),
            "initial_jitter_angstrom": float(self.initial_jitter_A),
            "initial_candidate_sites": int(self.initial_heavy_region_sites),
            "feasible_sample_volume_angstrom3": float(
                self.feasible_volume_angstrom3
            ),
            "sample_volume_angstrom3": float(self.sample_volume_angstrom3),
            "padded_cell_volume_angstrom3": float(self.cell_volume_angstrom3),
            "initial_n_sites": int(self.initial_n_sites),
            "n_sites": int(self.n_sites),
            "level_set_boundary_penalty": float(self.boundary_penalty),
            "level_set_boundary_sharpness": float(self.boundary_sharpness),
            "level_set_margin_angstrom": float(self.level_set_margin_A),
            "level_set_boundary_normalization": "total_W_plus_Se_occupancy",
            "level_set_interpolation": "periodic_cubic_bspline_C2",
            "sample_box_confinement_penalty": float(
                self.sample_box_confinement_penalty
            ),
            "sample_box_confinement_sharpness": float(
                self.sample_box_confinement_sharpness
            ),
            "sample_box_confinement_margin_angstrom": float(
                self.sample_box_confinement_margin_A
            ),
            "sample_box_lower_xyz_angstrom": self.sample_box_lower_A.tolist(),
            "sample_box_upper_xyz_angstrom": self.sample_box_upper_A.tolist(),
            "sample_box_confinement_normalization": "total_W_plus_Se_occupancy",
            "diversity_subcells_xyz": self.diversity_subcells_xyz.tolist(),
            "periodic_z_parent_alignment": bool(
                self.periodic_z_parent_alignment
            ),
            "composition_target": self.composition_target.tolist(),
            "composition_penalty": float(self.composition_penalty),
            "se_vacancy_bias_enable": bool(self.se_vacancy_bias_enable),
            "se_vacancy_bias_alpha_eV_per_vacancy": float(
                self.se_vacancy_bias_alpha_eV
            ),
            "se_vacancy_reference_atom_count": float(
                self.se_vacancy_reference_atom_count
            ),
            "se_vacancy_reference_se_count": float(
                self.se_vacancy_reference_se_count
            ),
            "se_vacancy_bias_definition": (
                "-alpha*(reference_Se_count-summed_Se_occupancy)/reference_atom_count"
            ),
            "species_by_source_type": {key: list(value) for key, value in self.species_by_source_type.items()},
            "tsallis_q": float(self.tsallis_q),
            "relaxation_shocks_enabled": False,
            "ga_enabled": bool(self.args.ga_enable),
            "ga_population_size": int(self.args.ga_population_size),
            "ga_generations": int(self.args.ga_generations),
            "ga_elite_fraction": float(self.args.ga_elite_fraction),
            "ga_random_fraction": float(self.args.ga_random_fraction),
            "ga_crossover_fraction": float(self.args.ga_crossover_fraction),
            "ga_parent_pool_size": int(self.args.ga_parent_pool_size),
            "ga_crossover_pos_sigma": float(self.args.ga_crossover_pos_sigma),
            "ga_crossover_occ_sigma": float(self.args.ga_crossover_occ_sigma),
            "ga_crossover_species_sigma": float(self.args.ga_crossover_species_sigma),
            "ga_parent_a_tournament_size": int(self.args.ga_parent_a_tournament_size),
            "ga_parent_b_tournament_size": int(self.args.ga_parent_b_tournament_size),
            "ga_superposition_occ_scale": float(self.args.ga_superposition_occ_scale),
            "ga_superposition_prune_distance_angstrom": float(self.args.ga_superposition_prune_distance_angstrom),
            "ga_superposition_prune_occ_thresh": float(self.args.ga_superposition_prune_occ_thresh),
            "ga_score": str(self.args.ga_score),
            "ga_parallel_workers": int(self.args.ga_parallel_workers),
            "ga_parallel_gpu_ids": str(self.args.ga_parallel_gpu_ids),
            "terminal_verbose": bool(self.args.terminal_verbose),
            "heartbeat_interval_seconds": float(self.args.heartbeat_interval_seconds),
            "running_log_file": str(running_log_path_from_args(self.args)),
            "local_resource_log_interval_seconds": float(
                self.args.local_resource_log_interval_seconds
            ),
            "worker_health_check_interval_seconds": float(
                self.args.worker_health_check_interval_seconds
            ),
            "member_progress_checkpoint_enable": bool(
                self.args.member_progress_checkpoint_enable
            ),
            "member_progress_checkpoint_interval_iterations": int(
                self.args.member_progress_checkpoint_interval_iterations
            ),
            "member_progress_checkpoint_interval_seconds": float(
                self.args.member_progress_checkpoint_interval_seconds
            ),
            "retain_completed_member_checkpoints": bool(
                self.args.retain_completed_member_checkpoints
            ),
            "compound_reference_energy_total_eV": self.E_ref_total,
            "compound_reference_energy_per_atom_eV": self.E_ref_per_atom,
            "compound_reference_composition": self.compound_reference_composition.tolist(),
            "compound_reference_formation_energy_per_atom_eV": float(self.compound_reference_formation_energy_per_atom),
            "element_reference_energies_per_atom_eV": self.element_ref_energies.tolist(),
            "element_references": self.element_reference_metadata,
            "reference_atom_symbols": list(self.reference_symbols),
            "reference_atom_positions_angstrom": self.reference_positions.tolist(),
            "composition_source_symbols": list(self.composition_source_symbols),
            "requested_vacuum_padding_angstrom": (
                self.requested_vacuum_padding.tolist()
            ),
            "vacuum_padding_angstrom": self.vacuum_padding.tolist(),
            "vacuum_position_shift_angstrom": self.vacuum_position_shift.tolist(),
            "pbc": self.pbc.astype(bool).tolist(),
            "original_lattice_vectors_angstrom": self.original_cell.tolist(),
            "lattice_vectors_angstrom": self.cell.tolist(),
            "reference_blob_centers_angstrom": self.ref_positions_blobs.tolist(),
            "reference_blob_sigmas_angstrom": self.ref_blob_sigmas.tolist(),
            "reference_blob_symbols": self.blob_symbols,
            "reference_blob_source_types": self.blob_source_types,
            "reference_blob_allowed_elements": self.blob_allowed_elements_from_json,
            "reference_blob_allowed_species_mask": self.blob_allowed_species_mask.astype(bool).tolist(),
            "reference_blob_species_fractions": self.blob_species_fractions.tolist(),
            "final_continuous_state": final_continuous,
            "final_rounded_state": final_rounded,
            "trajectory": trajectory,
            "trajectory_history_disabled": bool(self.args.disable_trajectory_history),
            "trajectory_stored_in_member_history_files": bool(self.args.ga_enable) and not bool(self.args.disable_trajectory_history),
            "stage_summaries": self.stage_summaries,
            "exact_composition_projection": copy.deepcopy(
                self.exact_composition_projection_summary
            ),
        }
        with (self.output_dir / "blob_optimization_history.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote history: {self.output_dir / 'blob_optimization_history.json'}")


def initial_species_probability_matrix(problem, args):
    mode = str(args.init_species_mode).strip().lower()
    if mode == "uniform":
        probs = np.full((problem.n_sites, problem.n_species), 1.0 / problem.n_species, dtype=float)
    elif mode in ("random", "dirichlet"):
        alpha = float(getattr(args, "init_species_dirichlet_alpha", INIT_SPECIES_DIRICHLET_ALPHA))
        if alpha <= 0.0:
            raise ValueError("initialization.init_species_dirichlet_alpha must be positive.")
        probs = np.random.dirichlet(np.full(problem.n_species, alpha, dtype=float), size=problem.n_sites)
    elif mode in ("blob-symbol", "blob_symbol", "blob"):
        probs = np.asarray(problem.assigned_blob_species_fractions, dtype=float)
    else:
        raise ValueError("initialization.init_species_mode must be blob-symbol, uniform, dirichlet, or random.")
    return restrict_blob_species_fractions(np.asarray(probs, dtype=float), problem.assigned_allowed_species_mask)


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


def make_initial_state(problem, args):
    occupancy = max(0.0, min(1.0, float(args.init_occ_total)))
    species_probs_np = initial_species_probability_matrix(problem, args)
    z_np = problem.sample_feasible_region_z(problem.n_sites)

    occupancy_tensor = torch.full((problem.n_sites, 1), occupancy, device=problem.device, dtype=problem.dtype)
    if float(args.init_occ_noise) > 0.0:
        occupancy_tensor = occupancy_tensor + float(args.init_occ_noise) * torch.randn_like(occupancy_tensor)
    species_probs = torch.tensor(species_probs_np, device=problem.device, dtype=problem.dtype)
    occ = combine_occupancy_and_species_probs(occupancy_tensor, species_probs)
    z = torch.tensor(z_np, device=problem.device, dtype=problem.dtype)
    return problem.project(problem.pack_y(occ, z))


def reduced_integer_formula_counts(composition_target, n_species):
    values = np.asarray(
        [
            float(piece.strip())
            for piece in str(composition_target).split(",")
            if piece.strip()
        ],
        dtype=float,
    )
    if values.shape != (int(n_species),):
        raise ValueError(
            "composition.composition_target must contain one positive formula "
            "count per species."
        )
    counts = np.rint(values).astype(int)
    if np.any(counts <= 0) or not np.allclose(
        values, counts, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            "Random-immigrant automatic k selection requires positive integer "
            "composition formula counts."
        )
    divisor = int(counts[0])
    for count in counts[1:]:
        divisor = math.gcd(divisor, int(count))
    return counts // max(divisor, 1)


def random_immigrant_image_prefit_site_layout(problem, args):
    """Use one W/Se/null proposal inventory based only on feasible sample volume."""
    return np.full(
        int(problem.initial_heavy_region_sites),
        problem.effective_site_class_index,
        dtype=int,
    )


def random_immigrant_image_prefit_state_legacy(problem, y0, args, seed, member_index):
    """Create one random immigrant using image-only EAT before FairChem.

    This mirrors ``initialization_tests/image_prefit_population_advanced.py``:
    optimize soft Gd/Ga/O/null probabilities and positions against the same
    differentiable image, composition, heavy-region, repulsion, and increasing
    entropy terms; enumerate every feasible integer formula-unit count, briefly
    polish and score each hard projection, select the best count, and then refine
    its positions. The selected hard atom count and identities are then frozen
    for the subsequent positions-only FairChem relaxation, so exact-null proposal
    sites can be deleted immediately instead of enlarging the neighbor graph.
    """
    if not bool(args.random_immigrant_image_prefit_enable):
        return y0, {"enabled": False, "reason": "disabled_by_configuration"}

    soft_steps = int(args.random_immigrant_image_prefit_soft_steps)
    k_search_steps = int(
        args.random_immigrant_image_prefit_k_search_position_steps
    )
    final_steps = int(args.random_immigrant_image_prefit_final_position_steps)
    position_lr = float(args.random_immigrant_image_prefit_position_learning_rate)
    occupancy_lr = float(args.random_immigrant_image_prefit_occupancy_learning_rate)
    entropy_final = float(args.random_immigrant_image_prefit_entropy_final_weight)
    repulsion_weight = float(args.random_immigrant_image_prefit_repulsion_weight)
    minimum_distance = float(
        args.random_immigrant_image_prefit_minimum_distance_angstrom
    )
    composition_penalty_enable = bool(
        args.random_immigrant_image_prefit_composition_penalty_enable
    )
    composition_penalty_weight = (
        float(problem.composition_penalty)
        if composition_penalty_enable
        else 0.0
    )
    maximum_formula_units_requested = int(
        args.random_immigrant_image_prefit_maximum_formula_units_to_test
    )
    volume_prior_enable = bool(
        args.random_immigrant_image_prefit_volume_prior_enable
    )
    volume_prior_relative_uncertainty = float(
        args.random_immigrant_image_prefit_volume_prior_relative_uncertainty
    )
    volume_prior_penalty_weight = float(
        args.random_immigrant_image_prefit_volume_prior_penalty_weight
    )
    compact_null_sites_before_fairchem = bool(
        args.random_immigrant_image_prefit_compact_null_sites_before_fairchem
    )
    logit_noise = float(args.random_immigrant_image_prefit_initial_logit_noise)
    initial_mode = str(
        args.random_immigrant_image_prefit_initial_occupancy_mode
    ).strip().lower()
    if soft_steps < 0 or k_search_steps < 0 or final_steps < 0:
        raise ValueError("Random-immigrant image-prefit step counts must be nonnegative.")
    if position_lr <= 0.0 or occupancy_lr <= 0.0:
        raise ValueError("Random-immigrant image-prefit learning rates must be positive.")
    if entropy_final < 0.0 or repulsion_weight < 0.0:
        raise ValueError(
            "Random-immigrant image-prefit entropy and repulsion weights must be nonnegative."
        )
    if minimum_distance < 0.0 or logit_noise < 0.0:
        raise ValueError(
            "Random-immigrant image-prefit minimum distance and logit noise must be nonnegative."
        )
    if maximum_formula_units_requested < 0:
        raise ValueError(
            "random_immigrant_image_prefit.maximum_formula_units_to_test must be nonnegative."
        )
    if (
        not np.isfinite(volume_prior_relative_uncertainty)
        or volume_prior_relative_uncertainty <= 0.0
    ):
        raise ValueError(
            "random_immigrant_image_prefit.volume_prior_relative_uncertainty must be positive."
        )
    if (
        not np.isfinite(volume_prior_penalty_weight)
        or volume_prior_penalty_weight < 0.0
    ):
        raise ValueError(
            "random_immigrant_image_prefit.volume_prior_penalty_weight must be nonnegative."
        )
    if initial_mode not in ("uniform", "current_state"):
        raise ValueError(
            "random_immigrant_image_prefit.initial_occupancy_mode must be uniform "
            "or current_state."
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    y0 = problem.project(y0.detach().clone())
    initial_occ, initial_z = problem.unpack_y(y0)
    layout = np.asarray(problem.assigned_site_region_indices, dtype=int)
    heavy_rows_np = np.where(layout == problem.heavy_site_class_index)[0]
    oxygen_rows_np = np.where(layout == problem.oxygen_site_class_index)[0]
    if len(heavy_rows_np) == 0 or len(oxygen_rows_np) == 0:
        raise ValueError(
            "Random-immigrant image prefit requires both heavy-class and oxygen-class sites."
        )
    heavy_rows = torch.as_tensor(
        heavy_rows_np, device=problem.device, dtype=torch.long
    )
    oxygen_rows = torch.as_tensor(
        oxygen_rows_np, device=problem.device, dtype=torch.long
    )
    heavy_species = torch.as_tensor(
        problem.heavy_species_indices, device=problem.device, dtype=torch.long
    )

    if initial_mode == "uniform":
        heavy_initial_probabilities = torch.full(
            (len(heavy_rows_np), len(problem.heavy_species_indices) + 1),
            1.0 / float(len(problem.heavy_species_indices) + 1),
            device=problem.device,
            dtype=problem.dtype,
        )
        oxygen_initial_probabilities = torch.full(
            (len(oxygen_rows_np), 2),
            0.5,
            device=problem.device,
            dtype=problem.dtype,
        )
    else:
        projected_initial_occ = problem.project_allowed_occupancies(initial_occ)
        heavy_real = projected_initial_occ.index_select(0, heavy_rows).index_select(
            1, heavy_species
        )
        heavy_null = torch.clamp(1.0 - heavy_real.sum(dim=1), min=0.0)
        heavy_initial_probabilities = torch.cat(
            (heavy_real, heavy_null[:, None]), dim=1
        )
        oxygen_real = projected_initial_occ.index_select(0, oxygen_rows)[
            :, problem.oxygen_species_index
        ]
        oxygen_initial_probabilities = torch.stack(
            (oxygen_real, torch.clamp(1.0 - oxygen_real, min=0.0)), dim=1
        )
    probability_floor = 1.0e-6
    heavy_initial_probabilities = (
        heavy_initial_probabilities.clamp_min(probability_floor)
    )
    heavy_initial_probabilities = (
        heavy_initial_probabilities
        / heavy_initial_probabilities.sum(dim=1, keepdim=True)
    )
    oxygen_initial_probabilities = (
        oxygen_initial_probabilities.clamp_min(probability_floor)
    )
    oxygen_initial_probabilities = (
        oxygen_initial_probabilities
        / oxygen_initial_probabilities.sum(dim=1, keepdim=True)
    )
    heavy_logits = (
        torch.log(heavy_initial_probabilities)
        + logit_noise * torch.randn_like(heavy_initial_probabilities)
    ).detach().requires_grad_(True)
    oxygen_logits = (
        torch.log(oxygen_initial_probabilities)
        + logit_noise * torch.randn_like(oxygen_initial_probabilities)
    ).detach().requires_grad_(True)
    positions = initial_z.detach().clone().requires_grad_(True)

    target_fraction_t = torch.as_tensor(
        problem.composition_target,
        device=problem.device,
        dtype=problem.dtype,
    )
    cell_lengths_t = torch.as_tensor(
        np.diag(problem.cell).copy(),
        device=problem.device,
        dtype=problem.dtype,
    )
    periodic_axes_t = torch.as_tensor(
        problem.pbc, device=problem.device, dtype=torch.bool
    )

    cell_volume_A3 = float(abs(np.linalg.det(np.asarray(problem.cell, dtype=float))))
    magpie_atomic_volumes_A3 = {}
    magpie_formula_volume_A3 = None
    magpie_continuous_formula_units = None
    if volume_prior_enable:
        module_path = Path(
            args.random_immigrant_image_prefit_magpie_volume_module
        ).expanduser().resolve()
        if not module_path.exists():
            raise FileNotFoundError(
                f"Random-immigrant Magpie volume module not found: {module_path}"
            )
        module_name = (
            "_random_immigrant_magpie_composition_volume_"
            + hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Magpie volume module {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        table = getattr(module, "ATOMIC_VOLUME_A3", None)
        if not isinstance(table, dict):
            raise ValueError(
                f"{module_path} does not define an ATOMIC_VOLUME_A3 dictionary."
            )
        missing_volumes = [
            symbol for symbol in problem.species if table.get(symbol) is None
        ]
        if missing_volumes:
            raise ValueError(
                f"No Magpie atomic volumes for species {missing_volumes}."
            )
        magpie_atomic_volumes_A3 = {
            symbol: float(table[symbol]) for symbol in problem.species
        }

    def occupancies_and_entropy():
        heavy_probabilities = torch.softmax(heavy_logits, dim=1)
        oxygen_probabilities = torch.softmax(oxygen_logits, dim=1)
        occupancy = torch.zeros(
            (problem.n_sites, problem.n_species),
            device=problem.device,
            dtype=problem.dtype,
        )
        heavy_occupancy = torch.zeros(
            (len(heavy_rows_np), problem.n_species),
            device=problem.device,
            dtype=problem.dtype,
        )
        heavy_occupancy[:, problem.heavy_species_indices] = (
            heavy_probabilities[:, :-1]
        )
        oxygen_occupancy = torch.zeros(
            (len(oxygen_rows_np), problem.n_species),
            device=problem.device,
            dtype=problem.dtype,
        )
        oxygen_occupancy[:, problem.oxygen_species_index] = (
            oxygen_probabilities[:, 0]
        )
        occupancy = occupancy.index_copy(0, heavy_rows, heavy_occupancy)
        occupancy = occupancy.index_copy(0, oxygen_rows, oxygen_occupancy)
        entropy = torch.cat(
            (
                1.0 - torch.sum(heavy_probabilities**2, dim=1),
                1.0 - torch.sum(oxygen_probabilities**2, dim=1),
            )
        ).mean()
        return occupancy, entropy, heavy_probabilities, oxygen_probabilities

    def objective_terms(occupancy, positions_raw, entropy, entropy_weight):
        wrapped_positions = torch.remainder(positions_raw, cell_lengths_t)
        predicted_image, predicted_linear, _ = (
            problem.render_differentiable_gaussian_image(
                occupancy, wrapped_positions
            )
        )
        per_voxel_image_loss = torch.nn.functional.huber_loss(
            predicted_image,
            problem.target_image_t,
            delta=problem.image_huber_delta,
            reduction="none",
        )
        image_loss = per_voxel_image_loss[problem.image_loss_mask_t].mean()
        ellipse_terms = problem.ellipse_composition_terms(
            occupancy, wrapped_positions
        )
        species_totals = occupancy.sum(dim=0)
        total_real = species_totals.sum().clamp_min(1.0e-8)
        composition_loss = torch.sum(
            (species_totals / total_real - target_fraction_t) ** 2
        )

        level_set = periodic_cubic_bspline_sample_zyx(
            problem.level_set_coefficients_t,
            wrapped_positions,
            problem.target_voxel_size_A,
        )
        heavy_mass = occupancy.index_select(1, heavy_species).sum(dim=1)
        violation = torch.nn.functional.softplus(
            problem.boundary_sharpness
            * (problem.level_set_margin_A - level_set)
        ) / problem.boundary_sharpness
        boundary_loss = torch.sum(heavy_mass * violation**2) / heavy_mass.sum().clamp_min(
            1.0e-8
        )

        delta = wrapped_positions[:, None, :] - wrapped_positions[None, :, :]
        minimum_image_delta = delta - torch.where(
            periodic_axes_t[None, None, :],
            cell_lengths_t[None, None, :]
            * torch.round(delta / cell_lengths_t[None, None, :]),
            torch.zeros_like(delta),
        )
        distance = torch.sqrt(
            torch.sum(minimum_image_delta**2, dim=2) + 1.0e-12
        )
        total_occupancy = occupancy.sum(dim=1)
        pair_weight = total_occupancy[:, None] * total_occupancy[None, :]
        upper = torch.triu(
            torch.ones_like(distance, dtype=torch.bool), diagonal=1
        )
        overlap = 0.10 * torch.nn.functional.softplus(
            (minimum_distance - distance) / 0.10
        )
        repulsion_loss = torch.sum(
            pair_weight[upper] * overlap[upper] ** 2
        ) / total_real
        objective = (
            problem.image_penalty * image_loss
            + composition_penalty_weight * composition_loss
            + problem.boundary_penalty * boundary_loss
            + problem.sample_box_confinement_terms(occupancy, wrapped_positions)[0]
            + repulsion_weight * repulsion_loss
            + float(entropy_weight) * entropy
        )
        return {
            "objective": objective,
            "image_loss": image_loss,
            "ellipse_composition_loss": ellipse_terms["loss_raw"],
            "ellipse_composition_objective": ellipse_terms["objective_term"],
            "composition_loss": composition_loss,
            "boundary_loss": boundary_loss,
            "repulsion_loss": repulsion_loss,
            "predicted_image": predicted_image,
            "predicted_linear": predicted_linear,
            "species_totals": species_totals,
            "positions": wrapped_positions,
        }

    def image_metrics(terms):
        with torch.no_grad():
            prediction = terms["predicted_image"][problem.image_loss_mask_t]
            target = problem.target_image_t[problem.image_loss_mask_t]
            prediction_centered = prediction - prediction.mean()
            target_centered = target - target.mean()
            correlation = torch.sum(
                prediction_centered * target_centered
            ) / torch.sqrt(
                torch.sum(prediction_centered**2)
                * torch.sum(target_centered**2)
                + 1.0e-20
            )
            return {
                "objective": float(terms["objective"].detach().cpu()),
                "huber_loss": float(terms["image_loss"].detach().cpu()),
                "ellipse_composition_mse": float(
                    terms["ellipse_composition_loss"].detach().cpu()
                ),
                "ellipse_composition_objective": float(
                    terms["ellipse_composition_objective"].detach().cpu()
                ),
                "correlation": float(correlation.detach().cpu()),
                "composition_loss": float(
                    terms["composition_loss"].detach().cpu()
                ),
                "boundary_loss": float(terms["boundary_loss"].detach().cpu()),
                "repulsion_loss": float(terms["repulsion_loss"].detach().cpu()),
                "soft_species_totals": terms["species_totals"]
                .detach()
                .cpu()
                .numpy()
                .tolist(),
            }

    started = time.time()
    optimizer = torch.optim.Adam(
        [
            {"params": [positions], "lr": position_lr},
            {"params": [heavy_logits, oxygen_logits], "lr": occupancy_lr},
        ]
    )
    initial_metrics = None
    terms = None
    for step in range(soft_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        occupancy, entropy, _, _ = occupancies_and_entropy()
        fraction = step / max(soft_steps, 1)
        beta = entropy_final * fraction**2
        terms = objective_terms(occupancy, positions, entropy, beta)
        if step == 0:
            initial_metrics = image_metrics(terms)
        if step == soft_steps:
            break
        terms["objective"].backward()
        torch.nn.utils.clip_grad_norm_(
            [positions, heavy_logits, oxygen_logits], max_norm=10.0
        )
        optimizer.step()
        with torch.no_grad():
            positions.remainder_(cell_lengths_t)

    with torch.no_grad():
        occupancy, _, heavy_probabilities, oxygen_probabilities = (
            occupancies_and_entropy()
        )
        soft_preround_metrics = image_metrics(terms)
        soft_preround_null_total = float(
            problem.n_sites - occupancy.sum().detach().cpu()
        )
        heavy_probabilities_np = heavy_probabilities.detach().cpu().numpy()
        oxygen_probabilities_np = oxygen_probabilities.detach().cpu().numpy()

    formula_counts = reduced_integer_formula_counts(
        args.composition_target,
        problem.n_species,
    )
    heavy_formula_count = int(
        formula_counts[problem.heavy_species_indices].sum()
    )
    oxygen_formula_count = int(
        formula_counts[problem.oxygen_species_index]
    )
    maximum_formula_units = min(
        len(heavy_rows_np) // heavy_formula_count,
        len(oxygen_rows_np) // oxygen_formula_count,
    )
    if maximum_formula_units_requested > 0:
        maximum_formula_units = min(
            maximum_formula_units, maximum_formula_units_requested
        )
    if maximum_formula_units < 1:
        raise ValueError(
            "The random-immigrant site inventories cannot hold one reduced "
            "composition formula unit."
        )

    if volume_prior_enable:
        magpie_formula_volume_A3 = float(
            sum(
                int(formula_counts[index])
                * magpie_atomic_volumes_A3[symbol]
                for index, symbol in enumerate(problem.species)
            )
        )
        if magpie_formula_volume_A3 <= 0.0:
            raise ValueError("The Magpie formula volume must be positive.")
        magpie_continuous_formula_units = (
            cell_volume_A3 / magpie_formula_volume_A3
        )

    def formula_unit_volume_prior(formula_units):
        if not volume_prior_enable:
            return {
                "magpie_predicted_volume_angstrom3": float("nan"),
                "volume_relative_error": 0.0,
                "volume_standardized_residual": 0.0,
                "volume_prior_loss": 0.0,
                "volume_prior_penalty": 0.0,
            }
        predicted_volume = float(formula_units) * magpie_formula_volume_A3
        relative_error = (predicted_volume - cell_volume_A3) / cell_volume_A3
        standardized_residual = (
            relative_error / volume_prior_relative_uncertainty
        )
        prior_loss = standardized_residual**2
        return {
            "magpie_predicted_volume_angstrom3": predicted_volume,
            "volume_relative_error": relative_error,
            "volume_standardized_residual": standardized_residual,
            "volume_prior_loss": prior_loss,
            "volume_prior_penalty": volume_prior_penalty_weight * prior_loss,
        }

    from scipy.optimize import linear_sum_assignment

    heavy_null_label = len(problem.heavy_species_indices)

    def hard_occupancy_for_counts(target_counts):
        target_counts = np.asarray(target_counts, dtype=int)
        target_heavy_counts = target_counts[problem.heavy_species_indices]
        target_oxygen_count = int(
            target_counts[problem.oxygen_species_index]
        )
        heavy_slots = np.concatenate(
            [
                np.full(int(count), local_index, dtype=int)
                for local_index, count in enumerate(target_heavy_counts)
            ]
            + [
                np.full(
                    len(heavy_rows_np) - int(target_heavy_counts.sum()),
                    heavy_null_label,
                    dtype=int,
                )
            ]
        )
        heavy_cost = -np.log(
            np.clip(
                heavy_probabilities_np[:, heavy_slots],
                1.0e-12,
                1.0,
            )
        )
        row_indices, column_indices = linear_sum_assignment(heavy_cost)
        heavy_labels = np.full(
            len(heavy_rows_np), heavy_null_label, dtype=int
        )
        heavy_labels[row_indices] = heavy_slots[column_indices]

        oxygen_slots = np.concatenate(
            (
                np.zeros(target_oxygen_count, dtype=int),
                np.ones(
                    len(oxygen_rows_np) - target_oxygen_count,
                    dtype=int,
                ),
            )
        )
        oxygen_cost = -np.log(
            np.clip(
                oxygen_probabilities_np[:, oxygen_slots],
                1.0e-12,
                1.0,
            )
        )
        row_indices, column_indices = linear_sum_assignment(oxygen_cost)
        oxygen_labels = np.ones(len(oxygen_rows_np), dtype=int)
        oxygen_labels[row_indices] = oxygen_slots[column_indices]

        hard_np = np.zeros(
            (problem.n_sites, problem.n_species), dtype=np.float32
        )
        for local_species_index, global_species_index in enumerate(
            problem.heavy_species_indices
        ):
            selected_rows = heavy_rows_np[
                heavy_labels == int(local_species_index)
            ]
            hard_np[selected_rows, int(global_species_index)] = 1.0
        selected_oxygen_rows = oxygen_rows_np[oxygen_labels == 0]
        hard_np[
            selected_oxygen_rows, problem.oxygen_species_index
        ] = 1.0
        return torch.tensor(
            hard_np, device=problem.device, dtype=problem.dtype
        )

    zero_entropy = torch.zeros((), device=problem.device, dtype=problem.dtype)
    formula_unit_trials = []
    selected_trial = None
    selected_key = None
    for formula_units in range(1, maximum_formula_units + 1):
        trial_target_counts = formula_counts * formula_units
        trial_occupancy = hard_occupancy_for_counts(trial_target_counts)
        trial_positions = positions.detach().clone().requires_grad_(True)
        trial_optimizer = torch.optim.Adam(
            [trial_positions], lr=position_lr
        )
        trial_terms = None
        for trial_step in range(k_search_steps + 1):
            trial_optimizer.zero_grad(set_to_none=True)
            trial_terms = objective_terms(
                trial_occupancy,
                trial_positions,
                zero_entropy,
                0.0,
            )
            trial_objective = (
                trial_terms["objective"]
                - composition_penalty_weight
                * trial_terms["composition_loss"]
            )
            if trial_step == k_search_steps:
                break
            trial_objective.backward()
            torch.nn.utils.clip_grad_norm_([trial_positions], max_norm=10.0)
            trial_optimizer.step()
            with torch.no_grad():
                trial_positions.remainder_(cell_lengths_t)

        trial_metrics = image_metrics(trial_terms)
        image_constraint_objective = float(trial_objective.detach().cpu())
        volume_prior_terms = formula_unit_volume_prior(formula_units)
        selection_objective = (
            image_constraint_objective
            + volume_prior_terms["volume_prior_penalty"]
        )
        trial_record = {
            "formula_units": int(formula_units),
            "target_species_counts": {
                symbol: int(trial_target_counts[index])
                for index, symbol in enumerate(problem.species)
            },
            "total_atoms": int(trial_target_counts.sum()),
            "selection_objective": selection_objective,
            "image_constraint_objective": image_constraint_objective,
            "huber_loss": trial_metrics["huber_loss"],
            "correlation": trial_metrics["correlation"],
            "boundary_loss": trial_metrics["boundary_loss"],
            "repulsion_loss": trial_metrics["repulsion_loss"],
            **volume_prior_terms,
        }
        formula_unit_trials.append(trial_record)
        trial_key = (
            trial_record["selection_objective"],
            trial_record["huber_loss"],
            int(formula_units),
        )
        if selected_key is None or trial_key < selected_key:
            selected_key = trial_key
            selected_trial = {
                "formula_units": int(formula_units),
                "target_counts": trial_target_counts.copy(),
                "occupancy": trial_occupancy.detach().clone(),
                "positions": trial_terms["positions"].detach().clone(),
            }

    assert selected_trial is not None
    selected_formula_units = int(selected_trial["formula_units"])
    prefit_target_counts = np.asarray(
        selected_trial["target_counts"], dtype=int
    )
    hard_occupancy = selected_trial["occupancy"]
    prefit_rounding_mode = "automatic_formula_unit_search"
    final_positions = selected_trial["positions"].clone().requires_grad_(True)
    final_optimizer = torch.optim.Adam([final_positions], lr=position_lr)
    final_terms = None
    for step in range(final_steps + 1):
        final_optimizer.zero_grad(set_to_none=True)
        final_terms = objective_terms(
            hard_occupancy, final_positions, zero_entropy, 0.0
        )
        # Composition is constant with fixed occupancies; remove it from the
        # scalar optimized during this positions-only polish.
        positions_objective = (
            final_terms["objective"]
            - composition_penalty_weight * final_terms["composition_loss"]
        )
        if step == final_steps:
            break
        positions_objective.backward()
        torch.nn.utils.clip_grad_norm_([final_positions], max_norm=10.0)
        final_optimizer.step()
        with torch.no_grad():
            final_positions.remainder_(cell_lengths_t)

    final_metrics = image_metrics(final_terms)
    hard_counts = hard_occupancy.sum(dim=0).detach().cpu().numpy().astype(int)
    prefit_candidate_site_count = int(problem.n_sites)
    summary = {
        "enabled": True,
        "seed": int(seed),
        "member_index": int(member_index),
        "image_transform": str(problem.image_target_transform),
        "experimental_normalization_percentile": float(
            args.target_normalization_percentile
        ),
        "clip_experimental_to_unit_interval": bool(
            args.gaussian_clip_experimental_to_unit_interval
        ),
        "relative_intensity_reference_species": str(
            problem.gaussian_relative_intensity_reference_species
        ),
        "intensity_scale_source": str(
            problem.gaussian_intensity_scale_source
        ),
        "rendered_single_atom_peak_amplitudes": {
            symbol: float(
                problem.gaussian_rendered_single_atom_peak_amplitudes[index]
            )
            for index, symbol in enumerate(problem.species)
        },
        "additional_lateral_blur_angstrom": float(
            problem.gaussian_additional_lateral_blur_A
        ),
        "depth_response_rms_angstrom": float(problem.gaussian_shared_sigma_z_A),
        "soft_steps": soft_steps,
        "k_search_position_steps": k_search_steps,
        "final_position_steps": final_steps,
        "initial_occupancy_mode": initial_mode,
        "oxygen_site_density_per_angstrom3": float(
            args.random_immigrant_image_prefit_oxygen_site_density_per_angstrom3
        ),
        "candidate_site_counts": {
            "heavy": int(len(heavy_rows_np)),
            "oxygen": int(len(oxygen_rows_np)),
        },
        "composition_penalty_enabled": composition_penalty_enable,
        "composition_penalty_weight": composition_penalty_weight,
        "rounding_mode": prefit_rounding_mode,
        "formula_counts": {
            symbol: int(formula_counts[index])
            for index, symbol in enumerate(problem.species)
        },
        "feasible_formula_units": {
            "minimum": 1,
            "maximum": int(maximum_formula_units),
        },
        "selected_formula_units": selected_formula_units,
        "selected_target_species_counts": {
            symbol: int(prefit_target_counts[index])
            for index, symbol in enumerate(problem.species)
        },
        "formula_unit_trials": formula_unit_trials,
        "initial_metrics": initial_metrics,
        "soft_preround_metrics": soft_preround_metrics,
        "soft_preround_null_total": soft_preround_null_total,
        "final_metrics": final_metrics,
        "hard_species_counts": {
            symbol: int(hard_counts[index])
            for index, symbol in enumerate(problem.species)
        },
        "hard_total_atoms": int(hard_counts.sum()),
        "candidate_site_count_before_compaction": prefit_candidate_site_count,
        "volume_prior": {
            "enabled": volume_prior_enable,
            "cell_volume_angstrom3": cell_volume_A3,
            "magpie_atomic_volumes_angstrom3": magpie_atomic_volumes_A3,
            "magpie_formula_volume_angstrom3": magpie_formula_volume_A3,
            "magpie_continuous_formula_units": magpie_continuous_formula_units,
            "relative_uncertainty": volume_prior_relative_uncertainty,
            "penalty_weight": volume_prior_penalty_weight,
            **formula_unit_volume_prior(selected_formula_units),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    append_running_log(
        args,
        "random_immigrant_image_prefit_done "
        f"member={int(member_index)} transform={problem.image_target_transform} "
        f"huber={final_metrics['huber_loss']:.9g} "
        f"correlation={final_metrics['correlation']:+.6f} "
        f"selected_k={selected_formula_units} "
        f"counts={summary['hard_species_counts']} "
        f"elapsed_seconds={summary['elapsed_seconds']:.3f}",
    )
    terminal_print(
        args,
        "[random-immigrant image prefit] "
        f"member={int(member_index)} Huber "
        f"{initial_metrics['huber_loss']:.6g} -> "
        f"{final_metrics['huber_loss']:.6g}; correlation "
        f"{initial_metrics['correlation']:+.4f} -> "
        f"{final_metrics['correlation']:+.4f}; "
        f"selected_k={selected_formula_units}; "
        f"counts={summary['hard_species_counts']}",
    )
    problem._last_aux_cache = None
    prefit_state = problem.project(
        problem.pack_y(hard_occupancy.detach(), final_positions.detach())
    )
    if compact_null_sites_before_fairchem:
        hard_real = (
            hard_occupancy.sum(dim=1).detach().cpu().numpy() > 0.5
        )
        keep_numpy = np.flatnonzero(hard_real).astype(int)
        keep_indices = torch.as_tensor(
            keep_numpy,
            device=hard_occupancy.device,
            dtype=torch.long,
        )
        compact_layout = problem.assigned_site_region_indices[keep_numpy]
        compact_occupancy = hard_occupancy.index_select(0, keep_indices)
        compact_positions = final_positions.detach().index_select(
            0, keep_indices
        )
        problem.set_site_layout(compact_layout)
        prefit_state = problem.project(
            problem.pack_y(compact_occupancy, compact_positions)
        )
        summary["null_site_compaction"] = {
            "enabled": True,
            "mode": "selected_hard_atoms_only_for_fixed_occupancy_fairchem",
            "before": prefit_candidate_site_count,
            "after": int(problem.n_sites),
            "removed": int(prefit_candidate_site_count - problem.n_sites),
            "retained_heavy_atoms": int(hard_counts[problem.heavy_species_indices].sum()),
            "retained_oxygen_atoms": int(hard_counts[problem.oxygen_species_index]),
            "selected_formula_units": int(selected_formula_units),
        }
        compaction_message = (
            "random_immigrant_exact_null_compaction "
            f"member={int(member_index)} before={prefit_candidate_site_count} "
            f"after={int(problem.n_sites)} removed="
            f"{int(prefit_candidate_site_count - problem.n_sites)} "
            f"selected_k={selected_formula_units}"
        )
        append_running_log(args, compaction_message)
        terminal_print(
            args,
            f"[random-immigrant image prefit] {compaction_message}",
        )
    else:
        summary["null_site_compaction"] = {
            "enabled": False,
            "before": prefit_candidate_site_count,
            "after": prefit_candidate_site_count,
            "removed": 0,
        }
    summary["retained_null_sites_for_fairchem_relaxation"] = int(
        problem.n_sites - hard_counts.sum()
    )
    return prefit_state, summary


def random_immigrant_image_prefit_state(problem, y0, args, seed, member_index):
    """Return a soft W/Se/null image prefit for subsequent syntropization.

    This stage intentionally does not round occupancies, remove null sites, or
    perform a hard-species position-only relaxation. The normal EAT
    syntropization path receives the complete soft candidate state, adds the
    FairChem formation-energy term, and performs the single final rounding and
    fixed-species positional relaxation.
    """
    if not bool(args.random_immigrant_image_prefit_enable):
        return y0, {"enabled": False, "reason": "disabled_by_configuration"}

    soft_steps = int(args.random_immigrant_image_prefit_soft_steps)
    configured_final_steps = int(
        args.random_immigrant_image_prefit_final_position_steps
    )
    position_lr = float(args.random_immigrant_image_prefit_position_learning_rate)
    occupancy_lr = float(args.random_immigrant_image_prefit_occupancy_learning_rate)
    entropy_final = float(args.random_immigrant_image_prefit_entropy_final_weight)
    repulsion_weight = float(args.random_immigrant_image_prefit_repulsion_weight)
    minimum_distance = float(
        args.random_immigrant_image_prefit_minimum_distance_angstrom
    )
    composition_enabled = bool(
        args.random_immigrant_image_prefit_composition_penalty_enable
    )
    composition_weight = problem.composition_penalty if composition_enabled else 0.0
    compact_nulls_requested = bool(
        args.random_immigrant_image_prefit_compact_null_sites_before_fairchem
    )
    logit_noise = float(args.random_immigrant_image_prefit_initial_logit_noise)
    initial_mode = str(
        args.random_immigrant_image_prefit_initial_occupancy_mode
    ).strip().lower()
    if min(soft_steps, configured_final_steps) < 0:
        raise ValueError("WSe2 image-prefit step counts must be nonnegative.")
    if position_lr <= 0.0 or occupancy_lr <= 0.0:
        raise ValueError("WSe2 image-prefit learning rates must be positive.")
    if min(entropy_final, repulsion_weight, minimum_distance, logit_noise) < 0.0:
        raise ValueError("WSe2 image-prefit weights and distances must be nonnegative.")
    if initial_mode not in ("uniform", "current_state"):
        raise ValueError(
            "random_immigrant_image_prefit.initial_occupancy_mode must be "
            "uniform or current_state."
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    y0 = problem.project(y0.detach().clone())
    initial_occ, initial_positions = problem.unpack_y(y0)
    if initial_mode == "uniform":
        initial_probabilities = torch.full(
            (problem.n_sites, problem.n_species + 1),
            1.0 / float(problem.n_species + 1),
            device=problem.device,
            dtype=problem.dtype,
        )
    else:
        real = problem.project_allowed_occupancies(initial_occ)
        null = torch.clamp(1.0 - real.sum(dim=1, keepdim=True), min=0.0)
        initial_probabilities = torch.cat((real, null), dim=1)
    initial_probabilities = initial_probabilities.clamp_min(1.0e-6)
    initial_probabilities /= initial_probabilities.sum(dim=1, keepdim=True)
    logits = (
        torch.log(initial_probabilities)
        + logit_noise * torch.randn_like(initial_probabilities)
    ).detach().requires_grad_(True)
    positions = initial_positions.detach().clone().requires_grad_(True)
    cell_lengths_t = torch.as_tensor(
        np.diag(problem.cell).copy(),
        device=problem.device,
        dtype=problem.dtype,
    )
    target_fraction_t = torch.as_tensor(
        problem.composition_target,
        device=problem.device,
        dtype=problem.dtype,
    )

    def occupancy_and_entropy():
        probabilities = torch.softmax(logits, dim=1)
        occupancy = probabilities[:, : problem.n_species]
        entropy = (1.0 - torch.sum(probabilities**2, dim=1)).mean()
        return occupancy, entropy, probabilities

    def objective_terms(occupancy, raw_positions, entropy, entropy_weight):
        wrapped_positions = torch.remainder(raw_positions, cell_lengths_t)
        predicted_image, predicted_linear, _ = (
            problem.render_differentiable_gaussian_image(
                occupancy, wrapped_positions
            )
        )
        per_voxel = torch.nn.functional.huber_loss(
            predicted_image,
            problem.target_image_t,
            delta=problem.image_huber_delta,
            reduction="none",
        )
        image_loss = per_voxel[problem.image_loss_mask_t].mean()
        ellipse_terms = problem.ellipse_composition_terms(
            occupancy, wrapped_positions
        )
        species_totals = occupancy.sum(dim=0)
        total_real = species_totals.sum().clamp_min(1.0e-8)
        composition_loss = torch.sum(
            (species_totals / total_real - target_fraction_t) ** 2
        )
        level_set = periodic_cubic_bspline_sample_zyx(
            problem.level_set_coefficients_t,
            wrapped_positions,
            problem.target_voxel_size_A,
        )
        real_mass = occupancy.sum(dim=1)
        violation = torch.nn.functional.softplus(
            problem.boundary_sharpness
            * (problem.level_set_margin_A - level_set)
        ) / problem.boundary_sharpness
        boundary_loss = torch.sum(real_mass * violation**2) / total_real
        (
            box_objective,
            box_loss,
            _,
            _,
        ) = problem.sample_box_confinement_terms(
            occupancy,
            wrapped_positions,
        )

        delta = wrapped_positions[:, None, :] - wrapped_positions[None, :, :]
        delta -= cell_lengths_t[None, None, :] * torch.round(
            delta / cell_lengths_t[None, None, :]
        )
        distance = torch.sqrt(torch.sum(delta**2, dim=2) + 1.0e-12)
        pair_weight = real_mass[:, None] * real_mass[None, :]
        upper = torch.triu(
            torch.ones_like(distance, dtype=torch.bool), diagonal=1
        )
        overlap = 0.10 * torch.nn.functional.softplus(
            (minimum_distance - distance) / 0.10
        )
        repulsion_loss = torch.sum(
            pair_weight[upper] * overlap[upper] ** 2
        ) / total_real
        objective = (
            problem.image_penalty * image_loss
            + float(composition_weight) * composition_loss
            + problem.boundary_penalty * boundary_loss
            + box_objective
            + repulsion_weight * repulsion_loss
            + float(entropy_weight) * entropy
        )
        return {
            "objective": objective,
            "image_loss": image_loss,
            "ellipse_composition_loss": ellipse_terms["loss_raw"],
            "ellipse_composition_objective": ellipse_terms["objective_term"],
            "composition_loss": composition_loss,
            "boundary_loss": boundary_loss,
            "box_loss_A2": box_loss,
            "box_objective": box_objective,
            "repulsion_loss": repulsion_loss,
            "predicted_image": predicted_image,
            "predicted_linear": predicted_linear,
            "species_totals": species_totals,
            "positions": wrapped_positions,
            "level_set": level_set,
        }

    def metrics(terms):
        with torch.no_grad():
            prediction = terms["predicted_image"][problem.image_loss_mask_t]
            target = problem.target_image_t[problem.image_loss_mask_t]
            pc = prediction - prediction.mean()
            tc = target - target.mean()
            correlation = torch.sum(pc * tc) / torch.sqrt(
                torch.sum(pc**2) * torch.sum(tc**2) + 1.0e-20
            )
            return {
                "objective": float(terms["objective"].detach().cpu()),
                "huber_loss": float(terms["image_loss"].detach().cpu()),
                "ellipse_composition_mse": float(
                    terms["ellipse_composition_loss"].detach().cpu()
                ),
                "ellipse_composition_objective": float(
                    terms["ellipse_composition_objective"].detach().cpu()
                ),
                "correlation": float(correlation.detach().cpu()),
                "composition_loss": float(terms["composition_loss"].detach().cpu()),
                "boundary_loss": float(terms["boundary_loss"].detach().cpu()),
                "box_loss_A2": float(terms["box_loss_A2"].detach().cpu()),
                "box_objective": float(terms["box_objective"].detach().cpu()),
                "repulsion_loss": float(terms["repulsion_loss"].detach().cpu()),
                "soft_species_totals": terms["species_totals"].detach().cpu().tolist(),
            }

    started = time.time()
    optimizer = torch.optim.Adam(
        [
            {"params": [positions], "lr": position_lr},
            {"params": [logits], "lr": occupancy_lr},
        ]
    )
    initial_metrics = None
    terms = None
    for step in range(soft_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        occupancy, entropy, probabilities = occupancy_and_entropy()
        fraction = step / max(soft_steps, 1)
        terms = objective_terms(
            occupancy,
            positions,
            entropy,
            entropy_final * fraction**2,
        )
        if step == 0:
            initial_metrics = metrics(terms)
        if step == soft_steps:
            break
        terms["objective"].backward()
        torch.nn.utils.clip_grad_norm_([positions, logits], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            positions.remainder_(cell_lengths_t)

    with torch.no_grad():
        occupancy, entropy, probabilities = occupancy_and_entropy()
        soft_handoff_metrics = metrics(terms)
        labels = torch.argmax(probabilities, dim=1)
        rounded_preview = torch.zeros_like(occupancy)
        real = labels < problem.n_species
        if bool(real.any().item()):
            rounded_preview[real, labels[real]] = 1.0
        rounded_preview_counts = (
            rounded_preview.sum(dim=0).detach().cpu().numpy().astype(int)
        )
        rounded_preview_terms = objective_terms(
            rounded_preview,
            positions,
            torch.zeros((), device=problem.device, dtype=problem.dtype),
            0.0,
        )
        rounded_preview_metrics = metrics(rounded_preview_terms)

    candidate_count = int(problem.n_sites)
    prefit_state = problem.project(
        problem.pack_y(occupancy.detach(), positions.detach())
    )
    summary = {
        "enabled": True,
        "seed": int(seed),
        "member_index": int(member_index),
        "mode": "soft_W_Se_null_prefit_then_energy_syntropization",
        "rounding_mode": "none_in_prefit_single_rounding_after_syntropization",
        "handoff": "soft_prefit_state_to_normal_energy_inclusive_syntropization",
        "image_loss_domain": "experimental_roi_only",
        "image_loss_voxel_count": int(problem.image_loss_voxel_count),
        "sample_box_confinement": {
            "penalty": float(problem.sample_box_confinement_penalty),
            "sharpness": float(problem.sample_box_confinement_sharpness),
            "margin_angstrom": float(problem.sample_box_confinement_margin_A),
            "normalization": "total_W_plus_Se_occupancy",
            "lower_xyz_angstrom": problem.sample_box_lower_A.tolist(),
            "upper_xyz_angstrom": problem.sample_box_upper_A.tolist(),
        },
        "vacuum_padding_angstrom": problem.vacuum_padding.tolist(),
        "pbc": problem.pbc.astype(bool).tolist(),
        "soft_steps": soft_steps,
        "configured_prefit_final_position_steps": configured_final_steps,
        "executed_prefit_final_position_steps": 0,
        "initial_occupancy_mode": initial_mode,
        "composition_penalty_enabled": composition_enabled,
        "composition_penalty_weight": float(composition_weight),
        "candidate_site_count": candidate_count,
        "initial_metrics": initial_metrics,
        "soft_handoff_metrics": soft_handoff_metrics,
        "soft_handoff_entropy": float(entropy.detach().cpu()),
        "rounded_preview_metrics_not_used_for_handoff": rounded_preview_metrics,
        "rounded_preview_species_counts_not_used_for_handoff": {
            symbol: int(rounded_preview_counts[index])
            for index, symbol in enumerate(problem.species)
        },
        "rounded_preview_total_atoms_not_used_for_handoff": int(
            rounded_preview_counts.sum()
        ),
        "elapsed_seconds": float(time.time() - started),
        "null_site_compaction": {
            "requested": compact_nulls_requested,
            "enabled": False,
            "reason": "all_candidate_sites_must_remain_available_to_syntropization",
            "before": candidate_count,
            "after": candidate_count,
            "removed": 0,
        },
        "candidate_sites_handed_to_syntropization": candidate_count,
    }
    append_running_log(
        args,
        "wse2_random_immigrant_image_prefit_done "
        f"member={member_index} handoff=soft_to_energy_syntropization "
        f"huber={soft_handoff_metrics['huber_loss']:.9g} "
        f"correlation={soft_handoff_metrics['correlation']:+.6f} "
        f"soft_species_totals={soft_handoff_metrics['soft_species_totals']} "
        "rounding=deferred null_compaction=deferred elapsed_seconds="
        f"{summary['elapsed_seconds']:.3f}",
    )
    return prefit_state, summary


def shock_state(problem, y, occ_sigma, pos_sigma, species_sigma=0.0):
    occ, z = problem.unpack_y(y)
    occupancy, species_probs = split_occupancy_and_species_probs(
        occ,
        fallback_species_probs=torch.tensor(
            problem.assigned_blob_species_fractions,
            device=occ.device,
            dtype=occ.dtype,
        ),
    )
    if float(occ_sigma) > 0.0:
        occupancy = occupancy + float(occ_sigma) * torch.randn_like(occupancy)
    if float(species_sigma) > 0.0:
        species_probs = project_simplex_rows(species_probs + float(species_sigma) * torch.randn_like(species_probs))
    occ = combine_occupancy_and_species_probs(occupancy, species_probs)
    if float(pos_sigma) > 0.0:
        z = z + float(pos_sigma) * torch.randn_like(z)
    return problem.project(problem.pack_y(occ, z))


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


def prune_close_pairs_state(problem, y, args):
    cutoff = float(args.ga_superposition_prune_distance_angstrom)
    if cutoff <= 0.0 or problem.n_sites < 2:
        return y, problem.assigned_site_region_indices.copy(), 0
    occ, z = problem.unpack_y(y)
    occ_sum = occ.sum(dim=1).detach().cpu().numpy()
    active = occ_sum > float(args.ga_superposition_prune_occ_thresh)
    active_indices = np.where(active)[0]
    if active_indices.size < 2:
        return y, problem.assigned_site_region_indices.copy(), 0

    z_np = z.detach().cpu().numpy()
    positions = problem.z_to_positions(z_np)
    dist2 = periodic_pair_distance2(positions[active_indices], problem.cell, problem.pbc)
    cutoff2 = cutoff * cutoff
    removed = set()
    for local_i in range(len(active_indices)):
        i = int(active_indices[local_i])
        if i in removed:
            continue
        for local_j in range(local_i + 1, len(active_indices)):
            j = int(active_indices[local_j])
            if j in removed or dist2[local_i, local_j] >= cutoff2:
                continue
            occ_i = float(occ_sum[i])
            occ_j = float(occ_sum[j])
            if occ_i < occ_j:
                removed.add(i)
                break
            if occ_j < occ_i:
                removed.add(j)
                continue
            removed.add(j if np.random.random() < 0.5 else i)
            if i in removed:
                break

    if not removed:
        return y, problem.assigned_site_region_indices.copy(), 0
    keep = np.asarray([idx for idx in range(problem.n_sites) if idx not in removed], dtype=int)
    layout = problem.assigned_site_region_indices[keep]
    occ_keep = occ[torch.as_tensor(keep, device=occ.device, dtype=torch.long)]
    z_keep = z[torch.as_tensor(keep, device=z.device, dtype=torch.long)]
    problem.set_site_layout(layout)
    return problem.project(problem.pack_y(occ_keep, z_keep)), layout.copy(), len(removed)


def shift_state_z_subcells(problem, y, site_layout, shift_subcells):
    """Apply a selected global periodic z-subcell translation to a parent."""
    site_layout = np.asarray(site_layout, dtype=int).reshape(-1)
    problem.set_site_layout(site_layout)
    nz = int(problem.diversity_subcells_xyz[2])
    shift_subcells = int(shift_subcells) % nz
    if shift_subcells == 0:
        return y, site_layout.copy()
    if not bool(problem.periodic_z_parent_alignment):
        raise RuntimeError(
            "A nonzero parent translation was requested while periodic z "
            "alignment is disabled by nonzero vacuum padding."
        )
    occ, z = problem.unpack_y(y)
    z_np = z.detach().cpu().numpy().copy()
    cell_z_length = float(np.linalg.norm(np.asarray(problem.cell, dtype=float)[2]))
    if cell_z_length <= 0.0:
        raise ValueError("The periodic cell z length must be positive.")
    z_np[:, 2] = np.mod(
        z_np[:, 2] + float(shift_subcells) * cell_z_length / float(nz),
        cell_z_length,
    )
    z_shifted = torch.as_tensor(
        z_np,
        device=z.device,
        dtype=z.dtype,
    )
    problem.set_site_layout(site_layout)
    return problem.project(problem.pack_y(occ, z_shifted)), site_layout.copy()


def compact_binary_state_null_sites(problem, y, site_layout):
    """Remove exact-null sites from a hard state without changing real atoms."""
    site_layout = np.asarray(site_layout, dtype=int).reshape(-1)
    problem.set_site_layout(site_layout)
    occ, z = problem.unpack_y(y)
    real_mass = occ.sum(dim=1)
    is_binary_mass = torch.logical_or(
        torch.isclose(real_mass, torch.zeros_like(real_mass), atol=1e-6, rtol=0.0),
        torch.isclose(real_mass, torch.ones_like(real_mass), atol=1e-6, rtol=0.0),
    )
    if not bool(torch.all(is_binary_mass).item()):
        return y, site_layout.copy(), 0
    keep = real_mass > 0.5
    if not bool(torch.any(keep).item()) or bool(torch.all(keep).item()):
        return y, site_layout.copy(), 0
    keep_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
    keep_numpy = keep_indices.detach().cpu().numpy().astype(int)
    compact_layout = site_layout[keep_numpy]
    compact_occ = occ.index_select(0, keep_indices)
    compact_z = z.index_select(0, keep_indices)
    removed = int(len(site_layout) - len(compact_layout))
    problem.set_site_layout(compact_layout)
    compact_y = problem.project(problem.pack_y(compact_occ, compact_z))
    return compact_y, compact_layout.copy(), removed


def crossover_state(problem, parent_a_y, parent_a_layout, parent_b_y, parent_b_layout, args, parent_b_z_shift_subcells):
    scale = max(0.0, min(1.0, float(args.ga_superposition_occ_scale)))
    parent_a_y, parent_a_layout, _ = compact_binary_state_null_sites(
        problem,
        parent_a_y,
        parent_a_layout,
    )
    parent_b_y, parent_b_layout, _ = compact_binary_state_null_sites(
        problem,
        parent_b_y,
        parent_b_layout,
    )
    problem.set_site_layout(parent_a_layout)
    occ_a, z_a = problem.unpack_y(parent_a_y)
    problem.set_site_layout(parent_b_layout)
    parent_b_y, parent_b_layout = shift_state_z_subcells(
        problem,
        parent_b_y,
        parent_b_layout,
        parent_b_z_shift_subcells,
    )
    occ_b, z_b = problem.unpack_y(parent_b_y)
    child_layout = np.concatenate([
        np.asarray(parent_a_layout, dtype=int),
        np.asarray(parent_b_layout, dtype=int),
    ])
    occ_child = torch.cat([scale * occ_a, scale * occ_b], dim=0)
    z_child = torch.cat([z_a, z_b], dim=0)
    problem.set_site_layout(child_layout)
    child = problem.project(problem.pack_y(occ_child, z_child))
    child = shock_state(
        problem,
        child,
        occ_sigma=float(args.ga_crossover_occ_sigma),
        pos_sigma=float(args.ga_crossover_pos_sigma),
        species_sigma=float(args.ga_crossover_species_sigma),
    )
    child, child_layout, _ = prune_close_pairs_state(problem, child, args)
    return child, child_layout


def member_history_dir(problem, args):
    return problem.output_dir / "histories"


def member_history_path(problem, args, generation, member_index, kind):
    label = safe_name(kind) or "member"
    return member_history_dir(problem, args) / f"g{generation:03d}_m{member_index:03d}_{label}.json"


def write_member_history(problem, args, result, final_snapshot, rounded_snapshot):
    if bool(args.disable_trajectory_history):
        return ""
    out_path = member_history_path(
        problem,
        args,
        int(result["generation"]),
        int(result["member_index"]),
        str(result["kind"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "kind": str(result["kind"]),
        "parent_refs": list(result.get("parent_refs", [])),
        "seed": int(result["seed"]),
        "score": float(result["score"]),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "formation_energy_difference_from_reference": float(result["formation_energy_difference_from_reference"]),
        "f_guided_no_entropy": float(result["f_guided_no_entropy"]),
        "composition_penalty": float(result["composition_penalty"]),
        "se_vacancy_bias_term": float(result.get("se_vacancy_bias_term", 0.0)),
        "se_vacancy_count_relative_to_reference": float(
            result.get("se_vacancy_count_relative_to_reference", 0.0)
        ),
        "se_occupancy_count": float(result.get("se_occupancy_count", 0.0)),
        "exact_composition_projection": copy.deepcopy(
            result.get("exact_composition_projection")
        ),
        "random_immigrant_image_prefit": copy.deepcopy(
            result.get("random_immigrant_image_prefit")
        ),
        "image_term": float(result.get("image_term", 0.0)),
        "voxel_image_term": float(result.get("voxel_image_term", 0.0)),
        "image_loss_raw": float(result.get("image_loss_raw", 0.0)),
        "ellipse_composition_term": float(
            result.get("ellipse_composition_term", 0.0)
        ),
        "ellipse_composition_loss_raw": float(
            result.get("ellipse_composition_loss_raw", 0.0)
        ),
        "image_correlation": float(result.get("image_correlation", 0.0)),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "n_sites": int(result["n_sites"]),
        "assigned_blob_indices": list(result.get("assigned_blob_indices", [])),
        "assigned_site_region_indices": list(
            result.get("assigned_site_region_indices", result.get("assigned_blob_indices", []))
        ),
        "final_snapshot": final_snapshot,
        "rounded_snapshot": rounded_snapshot,
        "stage_summaries": list(problem.stage_summaries),
        "trajectory": list(problem.history),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)


def ga_log_path(problem, args):
    if args.ga_log_file.strip():
        return Path(args.ga_log_file).expanduser().resolve()
    return problem.output_dir / "genetic_optimization_log.txt"


def format_ga_log(record):
    event = str(record.get("event", "event"))
    pieces = [event]
    for key in (
        "generation",
        "member_index",
        "kind",
        "seed",
        "parent",
        "parent_a",
        "parent_b",
        "parent_alignment_applied",
        "parent_b_z_shift_subcells",
        "parent_b_z_shift_angstrom",
        "diversity_subcells_xyz",
        "periodic_z_parent_alignment",
        "score",
        "formation_energy_per_atom",
        "f_guided_no_entropy",
        "se_vacancy_bias_term",
        "se_vacancy_count_relative_to_reference",
        "image_term",
        "image_correlation",
        "n_sites",
        "history_file",
        "best_score",
        "global_best_score",
    ):
        if key not in record:
            continue
        value = record[key]
        if value == "":
            continue
        if isinstance(value, float):
            pieces.append(f"{key}={value:+.6f}")
        else:
            pieces.append(f"{key}={value}")
    return " | ".join(pieces)


def append_ga_log(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_ga_log(record) + "\n")


def reset_ga_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def score_snapshot(snapshot, args):
    if args.ga_score == "formation":
        return float(snapshot["formation_energy_per_atom"])
    return float(snapshot["f_guided_no_entropy"])


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


def aggregate_compute_metrics(results):
    total = empty_compute_metrics()
    for result in results:
        metrics = result.get("compute_metrics", {})
        for key in (
            "model_energy_evaluations",
            "objective_evaluations",
            "accepted_optimizer_iterations",
        ):
            total[key] += int(metrics.get(key, 0))
        for key in ("model_evaluation_seconds", "member_wall_seconds"):
            total[key] += float(metrics.get(key, 0.0))
        for phase, value in metrics.get("phase_wall_seconds", {}).items():
            total["phase_wall_seconds"][phase] = (
                total["phase_wall_seconds"].get(phase, 0.0) + float(value)
            )
        for phase, value in metrics.get("by_phase", {}).items():
            entry = total["by_phase"].setdefault(phase, {"count": 0, "seconds": 0.0})
            entry["count"] += int(value.get("count", 0))
            entry["seconds"] += float(value.get("seconds", 0.0))
    return total


def structures_db_path(problem):
    return problem.output_dir / "structures.json"


def read_structures_database(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {"schema_version": 1, "structures": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"schema_version": 1, "structures": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"Structure database must be a JSON object or list: {path}")
    payload.setdefault("schema_version", 1)
    payload.setdefault("structures", [])
    if not isinstance(payload["structures"], list):
        raise ValueError(f"Structure database 'structures' must be a list: {path}")
    return payload


def write_structures_database(path, payload):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def compact_structure_record_for_database(record):
    structure = record.get("structure") or record.get("rounded_structure") or {}
    parent_refs = record.get("parent_refs", [])
    if parent_refs is None:
        parent_refs = []
    compact_structure = {
        "symbols": list(structure.get("symbols", [])),
        "cart_positions_angstrom": structure.get("cart_positions_angstrom", []),
        "cell_angstrom": structure.get("cell_angstrom", structure.get("lattice_vectors_angstrom", [])),
    }
    snapshot = record.get("continuous_snapshot", {})
    return {
        "generation": int(record.get("generation", 0)),
        "member_index": int(record.get("member_index", 0)),
        "kind": str(record.get("kind", "")),
        "parent_refs": list(parent_refs),
        "score": float(record.get("score", snapshot.get("f_guided_no_entropy", snapshot.get("formation_energy_per_atom", 0.0)))),
        "formation_energy_per_atom": float(record.get("formation_energy_per_atom", snapshot.get("formation_energy_per_atom", 0.0))),
        "composition_penalty": float(record.get("composition_penalty", snapshot.get("composition_term", 0.0))),
        "se_vacancy_bias_term": float(
            record.get("se_vacancy_bias_term", snapshot.get("se_vacancy_bias_term", 0.0))
        ),
        "se_vacancy_count_relative_to_reference": float(
            record.get(
                "se_vacancy_count_relative_to_reference",
                snapshot.get("se_vacancy_count_relative_to_reference", 0.0),
            )
        ),
        "se_occupancy_count": float(
            record.get("se_occupancy_count", snapshot.get("se_occupancy_count", 0.0))
        ),
        "image_term": float(record.get("image_term", snapshot.get("image_term", 0.0))),
        "voxel_image_term": float(
            record.get("voxel_image_term", snapshot.get("voxel_image_term", 0.0))
        ),
        "ellipse_composition_term": float(
            record.get(
                "ellipse_composition_term",
                snapshot.get("ellipse_composition_term", 0.0),
            )
        ),
        "ellipse_composition_loss_raw": float(
            record.get(
                "ellipse_composition_loss_raw",
                snapshot.get("ellipse_composition_loss_raw", 0.0),
            )
        ),
        "image_correlation": float(record.get("image_correlation", snapshot.get("image_correlation", 0.0))),
        "sample_box_term": float(
            record.get("sample_box_term", snapshot.get("sample_box_term", 0.0))
        ),
        "sample_box_active_outside_count": int(
            record.get(
                "sample_box_active_outside_count",
                snapshot.get("sample_box_active_outside_count", 0),
            )
        ),
        "compute_metrics": copy.deepcopy(record.get("compute_metrics", {})),
        "structure": compact_structure,
    }


_STRUCTURE_DATABASE_KEY_CACHE = {}


def structure_database_keys(path):
    """Read existing member keys once so append-only resume remains idempotent."""
    path = Path(path).expanduser().resolve()
    cache_key = str(path)
    if cache_key not in _STRUCTURE_DATABASE_KEY_CACHE:
        payload = read_structures_database(path)
        _STRUCTURE_DATABASE_KEY_CACHE[cache_key] = {
            (int(item.get("generation", -1)), int(item.get("member_index", -1)))
            for item in payload.get("structures", [])
        }
    return _STRUCTURE_DATABASE_KEY_CACHE[cache_key]


def append_structure_record(path, record):
    """Append one compact member without rewriting the accumulated database."""
    path = Path(path).expanduser()
    if not path.exists():
        write_structures_database(path, {"schema_version": 1, "structures": []})
    compact = compact_structure_record_for_database(record)
    keys = structure_database_keys(path)
    key = (int(compact["generation"]), int(compact["member_index"]))
    if key in keys:
        return
    rendered = json.dumps(compact, indent=2)
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
    keys.add(key)


def y_to_list(y):
    return [float(x) for x in y.detach().cpu().numpy().reshape(-1)]


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


def compact_completed_ga_result(problem, result):
    """Migrate a completed hard-state result by deleting only exact nulls."""
    compacted = dict(result)
    layout = np.asarray(
        compacted.get(
            "assigned_site_region_indices",
            compacted.get("assigned_blob_indices", []),
        ),
        dtype=int,
    ).reshape(-1)
    if layout.size <= 0 or not torch.is_tensor(compacted.get("y")):
        return compacted, 0
    y_compact, layout_compact, removed = compact_binary_state_null_sites(
        problem,
        compacted["y"],
        layout,
    )
    if removed <= 0:
        return compacted, 0

    original_count = int(len(layout))
    y_rounded = compacted.get("y_rounded")
    if torch.is_tensor(y_rounded):
        problem.set_site_layout(layout)
        rounded_occ, rounded_z = problem.unpack_y(y_rounded)
        keep = rounded_occ.sum(dim=1) > 0.5
        keep_indices = torch.nonzero(keep, as_tuple=False).reshape(-1)
        if int(keep_indices.numel()) == int(len(layout_compact)):
            problem.set_site_layout(layout_compact)
            y_rounded = problem.project(
                problem.pack_y(
                    rounded_occ.index_select(0, keep_indices),
                    rounded_z.index_select(0, keep_indices),
                )
            )
        else:
            y_rounded = y_compact.detach().clone()

    compacted["y"] = y_compact
    compacted["y_rounded"] = y_rounded
    compacted["assigned_blob_indices"] = layout_compact.astype(int).tolist()
    compacted["assigned_site_region_indices"] = layout_compact.astype(int).tolist()
    compacted["n_sites"] = int(len(layout_compact))
    for snapshot_key in ("final_snapshot", "rounded_snapshot"):
        snapshot = compacted.get(snapshot_key)
        if isinstance(snapshot, dict):
            snapshot = dict(snapshot)
            if "candidate_site_count" in snapshot:
                snapshot["candidate_site_count"] = int(len(layout_compact))
            if "n_sites" in snapshot:
                snapshot["n_sites"] = int(len(layout_compact))
            compacted[snapshot_key] = snapshot
    problem.set_site_layout(layout_compact)
    return compacted, int(original_count - len(layout_compact))


GA_CHECKPOINT_SCHEMA_VERSION = 2
# Immediately preceding production driver. This migration changes checkpoint
# handling only; its genetic operators and scientific objective are unchanged.
CHECKPOINT_COMPATIBLE_SCRIPT_SHA256 = (
    "0f94627187c7e0ad7c34e4461829b3bf7b9c0c5985725f76509cc72dacf30320",
)


def ga_checkpoint_path_from_args(args):
    return Path(args.output_dir).expanduser().resolve() / "ga_checkpoint.json"


def ga_checkpoint_path(problem):
    return problem.output_dir / "ga_checkpoint.json"


def ga_generation_checkpoint_path(problem, generation):
    return (
        problem.output_dir
        / "ga_checkpoints"
        / f"generation_{int(generation):04d}.json"
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ga_checkpoint_fingerprint(args, optimizer_script_sha256=None):
    excluded = {
        "slurm_worker_mode",
        "slurm_worker_task",
        "slurm_worker_result",
        "slurm_worker_resource_log",
        "ga_resuming_from_checkpoint",
        "_ga_checkpoint_fingerprint_override",
        "_ga_checkpoint_generation_origin",
        # This controls only how quickly the parent notices a dead local
        # process; it does not change any generated or relaxed structure.
        "worker_health_check_interval_seconds",
    }
    if optimizer_script_sha256 is None:
        fingerprint_override = str(
            getattr(args, "_ga_checkpoint_fingerprint_override", "") or ""
        )
        if fingerprint_override:
            return fingerprint_override
    resolved = {
        str(key): json_safe_value(value)
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }
    payload = {
        "resolved_parameters": resolved,
        "scientific_assets_sha256": {
            str(asset): file_sha256(asset) for asset in (
                Path(args.gaussian_element_fit_csv),
                SCRIPT_DIR / "gaussian_image_model.py",
                SCRIPT_DIR / "minimizer.py",
                resolve_script_relative_path(args.experimental_tiff),
            )
        },
        "optimizer_script_sha256": (
            file_sha256(Path(__file__).resolve())
            if optimizer_script_sha256 is None
            else str(optimizer_script_sha256)
        ),
    }
    import_path = str(getattr(args, "ga_import_structures_json", "") or "").strip()
    if import_path and Path(import_path).is_file():
        payload["import_structures_sha256"] = file_sha256(import_path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prior_run_parameters_match_current_args(args):
    """Verify scientific/runtime settings before an operational code migration."""
    path = Path(args.output_dir).expanduser().resolve() / "run_parameters.json"
    if not path.is_file():
        return False
    try:
        stored = read_json_file(path).get("resolved_parameters", {})
    except Exception:
        return False
    excluded = {
        "ga_resuming_from_checkpoint",
        "_ga_checkpoint_fingerprint_override",
        "_ga_checkpoint_generation_origin",
        "worker_health_check_interval_seconds",
    }
    stored_filtered = {
        str(key): json_safe_value(value)
        for key, value in stored.items()
        if key not in excluded
    }
    current_filtered = {
        str(key): json_safe_value(value)
        for key, value in vars(args).items()
        if key not in excluded
    }
    return stored_filtered == current_filtered


def read_ga_checkpoint_for_args(args):
    path = ga_checkpoint_path_from_args(args)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"GA checkpoint exists but could not be read: {path}. "
            "Refusing to silently restart from generation 0."
        ) from exc
    if int(payload.get("schema_version", -1)) != GA_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported GA checkpoint schema in {path}: "
            f"{payload.get('schema_version')!r}."
        )
    previous_limit = int(payload.get("generations_requested", -1))
    requested_limit = int(args.ga_generations)
    completed_count = int(payload.get("completed_generation_count", 0))
    if previous_limit <= 0 or completed_count < 0 or completed_count > previous_limit:
        raise RuntimeError("GA checkpoint has inconsistent generation counts.")
    if requested_limit < previous_limit:
        raise RuntimeError(
            f"Cannot decrease the GA generation limit from {previous_limit} "
            f"to {requested_limit}; keep it unchanged or increase it."
        )
    # Old checkpoints hashed the requested limit. Preserve that original value
    # across extensions, including a second restart after 200 was written over
    # the original 100-generation request. Completed files stay immutable.
    origin_limit = int(payload.get("fingerprint_generations_requested", previous_limit))
    if not 0 < origin_limit <= previous_limit:
        raise RuntimeError("GA checkpoint has an invalid original generation limit.")
    current_script_sha256 = file_sha256(Path(__file__).resolve())
    # Explicit script hash bypasses any in-memory override: every read still
    # verifies the current settings and scientific assets against the checkpoint.
    expected = ga_checkpoint_fingerprint(args, optimizer_script_sha256=current_script_sha256)
    actual = str(payload.get("config_fingerprint", ""))
    if actual != expected:
        origin_args = copy.copy(args)
        origin_args.ga_generations = origin_limit
        compatible = any(
            actual
            == ga_checkpoint_fingerprint(
                origin_args,
                optimizer_script_sha256=legacy_script_sha256,
            )
            for legacy_script_sha256 in (
                current_script_sha256, *CHECKPOINT_COMPATIBLE_SCRIPT_SHA256
            )
        )
        if not compatible:
            raise RuntimeError(
                f"GA checkpoint configuration/code mismatch: {path}. "
                "Use the original script and YAML to resume, or move the checkpoint "
                "aside and choose a new output directory for a fresh run."
            )
        # Continue using the old fingerprint so completed generation/member
        # checkpoint files remain valid and are not recomputed.
        if getattr(args, "_ga_checkpoint_fingerprint_override", None) != actual:
            print(
                "[ga checkpoint] verified compatible code/settings; preserving "
                f"the existing fingerprint (original limit={origin_limit}, "
                f"requested limit={requested_limit})",
                flush=True,
            )
        args._ga_checkpoint_fingerprint_override = actual
    args._ga_checkpoint_generation_origin = origin_limit
    return payload


def checkpoint_seed_for_args(args):
    payload = read_ga_checkpoint_for_args(args)
    if payload is None:
        return None
    return int(payload["seed"])


def serialize_checkpoint_generation(generation):
    return [
        json_safe_value(serialize_ga_result_for_worker(result))
        for result in generation
    ]


def write_ga_checkpoint(
    problem,
    args,
    seed,
    all_generations,
    current_generation,
    current_generation_results,
    status="running",
):
    fingerprint = ga_checkpoint_fingerprint(args)
    for generation_index, generation_results in enumerate(all_generations):
        generation_path = ga_generation_checkpoint_path(
            problem,
            generation_index,
        )
        # Completed generations are immutable and are validated when restoring
        # a checkpoint. Do not reread every old population whenever a new
        # member finishes.
        if generation_path.exists():
            continue
        serialized_generation = serialize_checkpoint_generation(
            generation_results
        )
        results_sha256 = hashlib.sha256(
            json.dumps(
                serialized_generation,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        generation_payload = {
            "schema_version": GA_CHECKPOINT_SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "seed": int(seed),
            "generation": int(generation_index),
            "population_size": int(args.ga_population_size),
            "results_sha256": results_sha256,
            "results": serialized_generation,
        }
        write_json_atomic(generation_path, generation_payload)

    payload = {
        "schema_version": GA_CHECKPOINT_SCHEMA_VERSION,
        "status": str(status),
        "updated_at": timestamp_now(),
        "config_fingerprint": fingerprint,
        "config_path": str(args.config_path),
        "optimizer_script": str(Path(__file__).resolve()),
        "seed": int(seed),
        "population_size": int(args.ga_population_size),
        "generations_requested": int(args.ga_generations),
        "fingerprint_generations_requested": int(
            getattr(args, "_ga_checkpoint_generation_origin", args.ga_generations)
        ),
        "current_generation": int(current_generation),
        "completed_generation_count": int(len(all_generations)),
        "current_generation_results": serialize_checkpoint_generation(
            current_generation_results
        ),
    }
    write_json_atomic(ga_checkpoint_path(problem), payload)


def restore_ga_checkpoint(problem, args):
    payload = read_ga_checkpoint_for_args(args)
    if payload is None:
        return None
    if int(payload.get("population_size", -1)) != int(args.ga_population_size):
        raise RuntimeError("GA checkpoint population size does not match the current run.")
    # read_ga_checkpoint_for_args already verifies unchanged/increased limits
    # and reconstructs the original fingerprint for the completed generations.
    completed = []
    completed_generation_count = int(
        payload.get("completed_generation_count", 0)
    )
    fingerprint = ga_checkpoint_fingerprint(args)
    for generation_index in range(completed_generation_count):
        generation_path = ga_generation_checkpoint_path(
            problem,
            generation_index,
        )
        if not generation_path.exists():
            raise RuntimeError(
                f"GA checkpoint is missing completed generation file: "
                f"{generation_path}"
            )
        generation_payload = read_json_file(generation_path)
        if (
            int(generation_payload.get("schema_version", -1))
            != GA_CHECKPOINT_SCHEMA_VERSION
            or str(generation_payload.get("config_fingerprint", ""))
            != fingerprint
            or int(generation_payload.get("seed", -1))
            != int(payload["seed"])
            or int(generation_payload.get("generation", -1))
            != generation_index
        ):
            raise RuntimeError(
                f"Completed GA generation checkpoint does not match the run: "
                f"{generation_path}"
            )
        serialized_results = generation_payload.get("results", [])
        actual_results_sha256 = hashlib.sha256(
            json.dumps(
                serialized_results,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if actual_results_sha256 != str(
            generation_payload.get("results_sha256", "")
        ):
            raise RuntimeError(
                f"Completed generation checkpoint failed its integrity check: "
                f"{generation_path}"
            )
        if len(serialized_results) != int(args.ga_population_size):
            raise RuntimeError(
                f"Completed generation checkpoint has "
                f"{len(serialized_results)} results instead of "
                f"{int(args.ga_population_size)}: {generation_path}"
            )
        restored_generation = []
        for result in serialized_results:
            restored = restore_ga_result_tensors(problem, result)
            restored, _ = compact_completed_ga_result(problem, restored)
            restored_generation.append(restored)
        completed.append(restored_generation)
    current = []
    migrated_current_members = 0
    removed_current_null_sites = 0
    for result in payload.get("current_generation_results", []):
        restored = restore_ga_result_tensors(problem, result)
        restored, removed = compact_completed_ga_result(problem, restored)
        migrated_current_members += int(removed > 0)
        removed_current_null_sites += int(removed)
        current.append(restored)
    if removed_current_null_sites > 0:
        message = (
            "checkpoint_null_site_migration "
            f"current_members={migrated_current_members} "
            f"removed_sites={removed_current_null_sites}"
        )
        append_running_log(args, message)
        terminal_print(args, f"[ga checkpoint] {message}")
    current_generation = int(payload.get("current_generation", len(completed)))
    if current_generation != len(completed):
        raise RuntimeError(
            "GA checkpoint is inconsistent: current_generation must equal "
            "the number of completed generations."
        )
    seen = set()
    for result in current:
        key = (int(result["generation"]), int(result["member_index"]))
        if key in seen:
            raise RuntimeError(f"Duplicate member in GA checkpoint: {key}.")
        seen.add(key)
        if key[0] != current_generation:
            raise RuntimeError(
                f"Checkpoint member {key} does not belong to current generation "
                f"{current_generation}."
            )
    return {
        "payload": payload,
        "all_generations": completed,
        "current_generation_results": current,
        "current_generation": current_generation,
        "status": str(payload.get("status", "running")),
        "seed": int(payload["seed"]),
    }


def member_checkpoint_path(problem, generation, member_index, kind):
    label = safe_name(kind) or "member"
    return (
        problem.output_dir
        / "member_checkpoints"
        / f"generation_{int(generation):03d}"
        / f"member_{int(member_index):03d}_{label}.json"
    )


def member_initial_state_fingerprint(y0, layout):
    y_array = np.asarray(
        y0.detach().cpu().numpy() if torch.is_tensor(y0) else y0,
        dtype=np.float32,
    ).reshape(-1)
    layout_array = np.asarray(layout, dtype=np.int64).reshape(-1)
    digest = hashlib.sha256()
    digest.update(y_array.tobytes())
    digest.update(layout_array.tobytes())
    return digest.hexdigest()


def member_checkpoint_identity(
    problem,
    args,
    y0,
    layout,
    seed,
    generation,
    member_index,
    kind,
):
    return {
        "config_fingerprint": ga_checkpoint_fingerprint(args),
        "initial_state_fingerprint": member_initial_state_fingerprint(y0, layout),
        "seed": int(seed),
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "path": str(member_checkpoint_path(problem, generation, member_index, kind)),
    }


def member_checkpoint_matches(payload, identity):
    return all(
        payload.get(key) == identity.get(key)
        for key in (
            "config_fingerprint",
            "initial_state_fingerprint",
            "seed",
            "generation",
            "member_index",
            "kind",
        )
    )


def read_member_checkpoint(identity):
    path = Path(identity["path"])
    if not path.exists():
        return None
    try:
        payload = read_json_file(path)
    except Exception:
        return None
    if int(payload.get("schema_version", -1)) != 1:
        return None
    if not member_checkpoint_matches(payload, identity):
        return None
    return payload


def write_active_member_progress(problem, y, stage, beta, iteration, run_kind):
    identity = getattr(problem, "active_member_checkpoint_context", None)
    if not identity or not bool(
        getattr(problem.args, "member_progress_checkpoint_enable", True)
    ):
        return
    phase = (str(run_kind), int(stage))
    iteration = int(iteration)
    now = time.time()
    iteration_interval = max(
        1,
        int(
            getattr(
                problem.args,
                "member_progress_checkpoint_interval_iterations",
                10,
            )
        ),
    )
    seconds_interval = max(
        0.0,
        float(
            getattr(
                problem.args,
                "member_progress_checkpoint_interval_seconds",
                60.0,
            )
        ),
    )
    last_phase = getattr(problem, "_member_checkpoint_last_phase", None)
    last_iteration = getattr(
        problem, "_member_checkpoint_last_iteration", None
    )
    last_time = float(
        getattr(problem, "_member_checkpoint_last_time", 0.0) or 0.0
    )
    phase_changed = phase != last_phase
    iteration_due = (
        last_iteration is None
        or iteration <= 1
        or iteration < int(last_iteration)
        or iteration - int(last_iteration) >= iteration_interval
    )
    time_due = seconds_interval > 0.0 and now - last_time >= seconds_interval
    if not (phase_changed or iteration_due or time_due):
        return
    payload = {
        "schema_version": 1,
        "status": "relaxing",
        "updated_at": timestamp_now(),
        **identity,
        "stage": int(stage),
        "beta": float(beta),
        "iteration": iteration,
        "run_kind": str(run_kind),
        "assigned_site_region_indices": problem.assigned_site_region_indices.astype(int).tolist(),
        "y": y_to_list(y),
        "exact_composition_projection": copy.deepcopy(
            problem.exact_composition_projection_summary
        ),
    }
    write_json_atomic(identity["path"], payload)
    problem._member_checkpoint_last_phase = phase
    problem._member_checkpoint_last_iteration = iteration
    problem._member_checkpoint_last_time = now


def cleanup_completed_member_checkpoints(problem, args, generation):
    """Remove redundant per-member files after the generation is safely saved."""
    if bool(getattr(args, "retain_completed_member_checkpoints", False)):
        return
    directory = (
        problem.output_dir
        / "member_checkpoints"
        / f"generation_{int(generation):03d}"
    )
    if not directory.exists():
        return
    removed = 0
    for path in directory.glob("member_*.json"):
        if path.is_file():
            path.unlink()
            removed += 1
    try:
        directory.rmdir()
    except OSError:
        pass
    if removed:
        append_running_log(
            args,
            f"member_checkpoint_cleanup generation={int(generation)} "
            f"removed_files={removed}",
        )


def restore_member_relaxation_progress(problem, identity, y0, layout):
    payload = read_member_checkpoint(identity)
    if payload is None or str(payload.get("status", "")) != "relaxing":
        return y0, np.asarray(layout, dtype=int), None
    saved_layout = np.asarray(
        payload.get("assigned_site_region_indices", []),
        dtype=int,
    ).reshape(-1)
    saved_y = np.asarray(payload.get("y", []), dtype=float).reshape(-1)
    expected_size = int(len(saved_layout)) * int(problem.n_species + 3)
    if len(saved_layout) <= 0 or saved_y.size != expected_size:
        return y0, np.asarray(layout, dtype=int), None
    problem.set_site_layout(saved_layout)
    state = torch.tensor(saved_y, device=problem.device, dtype=problem.dtype)
    problem.exact_composition_projection_summary = copy.deepcopy(
        payload.get("exact_composition_projection")
    )
    resume = {
        "stage": int(payload.get("stage", 0)),
        "beta": float(payload.get("beta", problem.args.beta0)),
        "iteration": int(payload.get("iteration", 0)),
        "run_kind": str(payload.get("run_kind", "main")),
    }
    append_running_log(
        problem.args,
        f"member_resume generation={identity['generation']} member={identity['member_index']} "
        f"kind={identity['kind']} stage={resume['stage']} iteration={resume['iteration']} "
        f"beta={resume['beta']:.6e}",
    )
    terminal_print(
        problem.args,
        f"[member checkpoint] resuming generation={identity['generation']} "
        f"member={identity['member_index']} stage={resume['stage']} "
        f"iteration={resume['iteration']}",
    )
    return state, saved_layout, resume


def restore_completed_member_result(problem, identity):
    payload = read_member_checkpoint(identity)
    if payload is None or str(payload.get("status", "")) != "complete":
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    restored = restore_ga_result_tensors(problem, result)
    restored, _ = compact_completed_ga_result(problem, restored)
    problem.set_site_layout(restored["assigned_blob_indices"])
    append_running_log(
        problem.args,
        f"member_result_recovered generation={identity['generation']} "
        f"member={identity['member_index']} kind={identity['kind']}",
    )
    return restored


def write_completed_member_result(problem, identity, result):
    payload = {
        "schema_version": 1,
        "status": "complete",
        "updated_at": timestamp_now(),
        **identity,
        "result": json_safe_value(serialize_ga_result_for_worker(result)),
    }
    write_json_atomic(identity["path"], payload)


def args_to_worker_dict(args):
    return dict(vars(args))


def state_from_realized_structure_record(problem, record):
    structure = record.get("structure") or record.get("rounded_structure")
    if not isinstance(structure, dict):
        return None, None
    symbols = list(structure.get("symbols", []))
    positions = np.asarray(structure.get("cart_positions_angstrom", []), dtype=float)
    if len(symbols) != len(positions):
        return None, None

    layout = []
    occ_rows = []
    z_rows = []
    cell = np.asarray(problem.cell, dtype=float)

    for atom_index, symbol in enumerate(symbols):
        if symbol not in problem.species:
            continue
        blob_idx = problem.effective_site_class_index
        z_row = np.mod(
            np.asarray(positions[int(atom_index)], dtype=float),
            np.diag(cell),
        )
        occ = np.zeros(problem.n_species, dtype=float)
        occ[problem.species.index(symbol)] = 1.0
        layout.append(blob_idx)
        occ_rows.append(occ)
        z_rows.append(z_row)

    if not layout:
        return None, None

    layout = np.asarray(layout, dtype=int)
    problem.set_site_layout(layout)
    occ = torch.tensor(np.asarray(occ_rows, dtype=float), device=problem.device, dtype=problem.dtype)
    z = torch.tensor(np.asarray(z_rows, dtype=float), device=problem.device, dtype=problem.dtype)
    y = problem.project(problem.pack_y(occ, z))
    return y, layout


def structure_database_record(problem, result, args, seed):
    structure = problem.realized_structure_payload(result["y_rounded"], rounded=True)
    return {
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "kind": str(result["kind"]),
        "parent_refs": list(result.get("parent_refs", [])),
        "score": float(result["score"]),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "composition_penalty": float(result["composition_penalty"]),
        "se_vacancy_bias_term": float(result.get("se_vacancy_bias_term", 0.0)),
        "se_vacancy_count_relative_to_reference": float(
            result.get("se_vacancy_count_relative_to_reference", 0.0)
        ),
        "se_occupancy_count": float(result.get("se_occupancy_count", 0.0)),
        "image_term": float(result.get("image_term", 0.0)),
        "voxel_image_term": float(result.get("voxel_image_term", 0.0)),
        "ellipse_composition_term": float(
            result.get("ellipse_composition_term", 0.0)
        ),
        "ellipse_composition_loss_raw": float(
            result.get("ellipse_composition_loss_raw", 0.0)
        ),
        "image_correlation": float(result.get("image_correlation", 0.0)),
        "compute_metrics": copy.deepcopy(
            result.get("compute_metrics", empty_compute_metrics())
        ),
        "structure": {
            "symbols": structure["symbols"],
            "cart_positions_angstrom": structure["cart_positions_angstrom"],
            "cell_angstrom": structure["cell_angstrom"],
        },
    }


def load_imported_parent_results(problem, args):
    path_text = str(args.ga_import_structures_json or "").strip()
    requested_count = max(0, int(args.ga_import_parent_count))
    if not path_text:
        return []
    count = requested_count
    if count <= 0:
        count = max(int(args.ga_population_size), int(args.ga_parent_pool_size))
    db = read_structures_database(path_text)
    candidates = []
    for index, record in enumerate(db["structures"]):
        state = record.get("continuous_state_vector") or record.get("rounded_state_vector")
        layout = None
        if state is not None:
            layout = record.get("assigned_site_region_indices")
            if layout is None:
                layout = record.get("assigned_blob_indices")
            if layout is None:
                fixed_expected_size = len(problem.initial_site_layout()) * (problem.n_species + 3)
                if len(state) != fixed_expected_size:
                    continue
                layout = problem.initial_site_layout().tolist()
            layout = np.asarray(layout, dtype=int).reshape(-1)
            expected_size = int(len(layout)) * (problem.n_species + 3)
            if len(state) != expected_size:
                continue
        score = record.get("score")
        if score is None:
            metric = str(args.ga_score)
            snapshot = record.get("continuous_snapshot", {})
            if metric == "formation":
                score = snapshot.get("formation_energy_per_atom")
            else:
                score = snapshot.get("f_guided_no_entropy")
        if score is None:
            continue
        if state is None:
            y, layout = state_from_realized_structure_record(problem, record)
            if y is None:
                continue
            state = y_to_list(y)
        candidates.append((float(score), index, record, state, np.asarray(layout, dtype=int).reshape(-1)))
    candidates.sort(key=lambda item: item[0])
    imported = []
    for rank, (score, index, record, state, layout) in enumerate(candidates[:count]):
        y = torch.tensor(state, device=problem.device, dtype=problem.dtype)
        problem.set_site_layout(layout)
        formation_energy = float(record.get("formation_energy_per_atom", record.get("continuous_snapshot", {}).get("formation_energy_per_atom", score)))
        composition_penalty = float(record.get("composition_penalty", record.get("continuous_snapshot", {}).get("composition_term", 0.0)))
        final_snapshot = record.get("continuous_snapshot") or {
            "formation_energy_per_atom": formation_energy,
            "formation_energy_difference_from_reference": float(record.get("formation_energy_difference_from_reference", 0.0)),
            "f_guided_no_entropy": float(record.get("f_guided_no_entropy", score)),
            "composition_term": composition_penalty,
        }
        imported.append({
            "generation": -1,
            "member_index": -1 - rank,
            "parent_index": None,
            "parent_refs": [f"import:{Path(path_text).name}:{index}"],
            "kind": "imported_parent",
            "seed": -1,
            "score": float(score),
            "formation_energy_per_atom": formation_energy,
            "formation_energy_difference_from_reference": float(record.get("formation_energy_difference_from_reference", final_snapshot.get("formation_energy_difference_from_reference", 0.0))),
            "f_guided_no_entropy": float(record.get("f_guided_no_entropy", final_snapshot.get("f_guided_no_entropy", score))),
            "composition_penalty": composition_penalty,
            "sumx": float(record.get("sumx", record.get("continuous_snapshot", {}).get("sumx", 0.0))),
            "entropy_avg": float(record.get("entropy_avg", record.get("continuous_snapshot", {}).get("entropy_avg", 0.0))),
            "max_min": float(record.get("max_min", record.get("continuous_snapshot", {}).get("max_min", 0.0))),
            "composition_fractions": list(record.get("composition_fractions", record.get("continuous_snapshot", {}).get("composition_fractions", []))),
            "assigned_blob_inside_count": int(record.get("assigned_blob_inside_count", record.get("continuous_snapshot", {}).get("assigned_blob_inside_count", 0))),
            "elapsed_seconds": 0.0,
            "assigned_blob_indices": layout.astype(int).tolist(),
            "assigned_site_region_indices": layout.astype(int).tolist(),
            "y": y.detach().clone(),
            "y_rounded": problem.rounded_state(y),
            "final_snapshot": final_snapshot,
            "rounded_snapshot": record.get("rounded_snapshot", {}),
            "history": [],
            "history_file": "",
            "stage_summaries": [],
        })
    if requested_count <= 0:
        terminal_print(args, f"[ga] imported {len(imported)} parent(s) from {path_text} (auto count={count})")
    else:
        terminal_print(args, f"[ga] imported {len(imported)} parent(s) from {path_text}")
    return imported


def evaluate_ga_member(
    problem,
    y0,
    assigned_blob_indices,
    args,
    seed,
    generation,
    member_index,
    parent_refs=None,
    kind="random",
    resume_state=None,
    random_immigrant_image_prefit=None,
):
    problem.set_site_layout(assigned_blob_indices)
    problem.reset_run_records()
    tracker = problem.model_evaluation_tracker
    tracker_before = tracker.snapshot()
    objective_evaluations_before = int(problem.objective_evaluation_count)
    previous_phase = tracker.phase
    member_started = time.perf_counter()
    optimization_started = time.perf_counter()
    tracker.phase = "syntropization"
    y_final = problem.run(y0, run_kind=str(kind), resume_state=resume_state)
    optimization_wall_seconds = time.perf_counter() - optimization_started
    assigned_blob_indices = problem.assigned_site_region_indices.copy()
    tracker.phase = "final_scoring"
    scoring_started = time.perf_counter()
    final_snapshot = problem.final_snapshot(y_final, f"ga_generation_{generation}_member_{member_index}")
    y_rounded = problem.rounded_state(y_final)
    rounded_snapshot = problem.final_snapshot(y_rounded, f"ga_generation_{generation}_member_{member_index}_rounded")
    final_scoring_wall_seconds = time.perf_counter() - scoring_started
    score = score_snapshot(final_snapshot, args)
    elapsed = time.perf_counter() - member_started
    tracker.phase = previous_phase
    compute_metrics = ModelEvaluationTracker.difference(tracker.snapshot(), tracker_before)
    compute_metrics.update({
        "objective_evaluations": int(
            problem.objective_evaluation_count - objective_evaluations_before
        ),
        "accepted_optimizer_iterations": int(
            sum(item.get("accepted_iterations", 0) for item in problem.stage_summaries)
        ),
        "member_wall_seconds": float(elapsed),
        "phase_wall_seconds": {
            "optimization": float(optimization_wall_seconds),
            "syntropization": float(
                sum(
                    item.get("stage_wall_seconds", 0.0)
                    for item in problem.stage_summaries
                    if not bool(item.get("positions_only", False))
                )
            ),
            "final_PBFGS": float(
                sum(
                    item.get("stage_wall_seconds", 0.0)
                    for item in problem.stage_summaries
                    if bool(item.get("positions_only", False))
                )
            ),
            "final_scoring": float(final_scoring_wall_seconds),
        },
    })
    result = {
        "generation": int(generation),
        "member_index": int(member_index),
        "parent_index": None,
        "parent_refs": [] if parent_refs is None else list(parent_refs),
        "kind": str(kind),
        "seed": int(seed),
        "score": float(score),
        "formation_energy_per_atom": float(final_snapshot["formation_energy_per_atom"]),
        "formation_energy_difference_from_reference": float(final_snapshot["formation_energy_difference_from_reference"]),
        "f_guided_no_entropy": float(final_snapshot["f_guided_no_entropy"]),
        "composition_penalty": float(final_snapshot["composition_term"]),
        "se_vacancy_bias_term": float(final_snapshot["se_vacancy_bias_term"]),
        "se_vacancy_count_relative_to_reference": float(
            final_snapshot["se_vacancy_count_relative_to_reference"]
        ),
        "se_occupancy_count": float(final_snapshot["se_occupancy_count"]),
        "image_term": float(final_snapshot["image_term"]),
        "voxel_image_term": float(final_snapshot["voxel_image_term"]),
        "image_loss_raw": float(final_snapshot["image_loss_raw"]),
        "ellipse_composition_term": float(
            final_snapshot["ellipse_composition_term"]
        ),
        "ellipse_composition_loss_raw": float(
            final_snapshot["ellipse_composition_loss_raw"]
        ),
        "image_correlation": float(final_snapshot["image_correlation"]),
        "sample_box_term": float(final_snapshot["sample_box_term"]),
        "sample_box_loss_raw_A2": float(
            final_snapshot["sample_box_loss_raw_A2"]
        ),
        "sample_box_active_outside_count": int(
            final_snapshot["sample_box_active_outside_count"]
        ),
        "sample_box_outside_occupancy_mass": float(
            final_snapshot["sample_box_outside_occupancy_mass"]
        ),
        "sumx": float(final_snapshot["sumx"]),
        "entropy_avg": float(final_snapshot["entropy_avg"]),
        "max_min": float(final_snapshot["max_min"]),
        "composition_fractions": list(final_snapshot["composition_fractions"]),
        "assigned_blob_inside_count": int(final_snapshot["assigned_blob_inside_count"]),
        "n_sites": int(problem.n_sites),
        "elapsed_seconds": float(elapsed),
        "compute_metrics": compute_metrics,
        "assigned_blob_indices": np.asarray(assigned_blob_indices, dtype=int).tolist(),
        "assigned_site_region_indices": np.asarray(assigned_blob_indices, dtype=int).tolist(),
        "y": y_final.detach().clone(),
        "y_rounded": y_rounded.detach().clone(),
        "final_snapshot": final_snapshot,
        "rounded_snapshot": rounded_snapshot,
        "history": [],
        "history_file": "",
        "stage_summaries": list(problem.stage_summaries),
        "exact_composition_projection": copy.deepcopy(
            problem.exact_composition_projection_summary
        ),
        "random_immigrant_image_prefit": copy.deepcopy(
            random_immigrant_image_prefit
            if random_immigrant_image_prefit is not None
            else {"enabled": False, "reason": "not_requested"}
        ),
    }
    result["history_file"] = write_member_history(problem, args, result, final_snapshot, rounded_snapshot)
    return result


def evaluate_candidate_member(problem, y0, assigned_blob_indices, args, seed, generation, member_index, parent_refs=None, kind="random"):
    identity = member_checkpoint_identity(
        problem,
        args,
        y0,
        assigned_blob_indices,
        seed,
        generation,
        member_index,
        kind,
    )
    cached = restore_completed_member_result(problem, identity)
    if cached is not None:
        problem.write_member_slice_comparisons(cached)
        return cached

    y0, assigned_blob_indices, resume_state = (
        restore_member_relaxation_progress(
            problem,
            identity,
            y0,
            assigned_blob_indices,
        )
    )
    prefit_summary = {
        "enabled": False,
        "reason": "not_random_immigrant",
    }
    if (
        resume_state is None
        and str(kind) == "random_immigrant"
        and bool(args.random_immigrant_image_prefit_enable)
    ):
        problem.set_site_layout(assigned_blob_indices)
        y0, prefit_summary = random_immigrant_image_prefit_state(
            problem,
            y0,
            args,
            seed,
            member_index,
        )
        assigned_blob_indices = problem.assigned_site_region_indices.copy()
    elif (
        resume_state is not None
        and str(kind) == "random_immigrant"
        and bool(args.random_immigrant_image_prefit_enable)
    ):
        prefit_summary = {
            "enabled": True,
            "reason": "energy_syntropization_resumed_after_completed_random_immigrant_prefit",
            "handoff": "restored_syntropization_checkpoint",
        }
    problem.active_member_checkpoint_context = identity
    try:
        result = evaluate_ga_member(
            problem,
            y0,
            assigned_blob_indices,
            args,
            seed,
            generation,
            member_index,
            parent_refs=parent_refs,
            kind=kind,
            resume_state=resume_state,
            random_immigrant_image_prefit=prefit_summary,
        )
    finally:
        problem.active_member_checkpoint_context = None
    problem.write_member_slice_comparisons(result)
    write_completed_member_result(problem, identity, result)
    return result


def parse_parallel_gpu_ids(text):
    ids = [part.strip() for part in str(text).split(",") if part.strip()]
    return ids or ["0"]


def move_problem_tensor_state_to_device(problem, device):
    """Move coordinator-owned tensors without touching the FairChem model.

    Local GA workers own CUDA.  The coordinating process only performs
    checkpointing, selection, crossover, and plotting, so keeping its state on
    CUDA wastes a third model/context on GPU 0 and makes checkpoint writes
    vulnerable to unrelated asynchronous CUDA failures.
    """
    target = torch.device(device)
    for name, value in list(vars(problem).items()):
        if torch.is_tensor(value):
            setattr(problem, name, value.detach().to(target))
    problem.device = target
    problem._last_aux_cache = None
    if target.type == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_fatal_cuda_exception(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "acceleratorerror",
            "device-side assert",
            "unspecified launch failure",
        )
    )


def ga_worker_loop(worker_index, gpu_id, args_dict, task_queue, result_queue):
    args = SimpleNamespace(**args_dict)
    stop_event = None
    monitor_thread = None
    try:
        if str(gpu_id).lower() != "cpu":
            if torch.cuda.is_available():
                torch.cuda.set_device(int(gpu_id))
            else:
                terminal_print(args, f"[ga worker {worker_index}] CUDA not available; falling back to CPU")
        resource_path = (
            Path(args.output_dir).expanduser().resolve()
            / "resources"
            / f"local_worker_{int(worker_index):02d}_gpu_{safe_name(gpu_id)}.tsv"
        )
        stop_event, monitor_thread = start_resource_monitor(
            resource_path,
            getattr(args, "local_resource_log_interval_seconds", 0.0),
            f"local_worker_{int(worker_index)}_gpu_{gpu_id}",
        )
        worker_seed = int(args.seed or 0) + 7919 * (int(worker_index) + 1)
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed % 2**32)
        problem = WSe2VolumeProblem(args)
        terminal_print(args, f"[ga worker {worker_index}] ready gpu={gpu_id} device={problem.device}")
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = task["task_id"]
            result = None
            y0 = None
            fatal_cuda_failure = False
            try:
                member_seed = int(task["seed"])
                torch.manual_seed(member_seed)
                np.random.seed(member_seed % 2**32)
                layout0 = np.asarray(task["assigned_blob_indices"], dtype=int).reshape(-1)
                problem.set_site_layout(layout0)
                y0 = torch.tensor(task["y0"], device=problem.device, dtype=problem.dtype)
                result = evaluate_candidate_member(
                    problem,
                    y0,
                    layout0,
                    args,
                    member_seed,
                    int(task["generation"]),
                    int(task["member_index"]),
                    parent_refs=task.get("parent_refs", []),
                    kind=str(task.get("kind", "random")),
                )
                # Force asynchronous kernel failures to surface in the worker
                # that launched them, before success is reported upstream.
                if problem.device.type == "cuda":
                    torch.cuda.synchronize(problem.device)
                serialized_result = serialize_ga_result_for_worker(result)
                result_queue.put({
                    "task_id": task_id,
                    "ok": True,
                    "worker_index": int(worker_index),
                    "gpu_id": str(gpu_id),
                    "result": serialized_result,
                })
            except Exception as exc:
                fatal_cuda_failure = is_fatal_cuda_exception(exc)
                result_queue.put({
                    "task_id": task_id,
                    "ok": False,
                    "worker_index": int(worker_index),
                    "gpu_id": str(gpu_id),
                    "traceback": traceback.format_exc(),
                })
            finally:
                # Drop per-member tensors before the next structure changes
                # graph size. This keeps the persistent calculator/model but
                # releases unused cached CUDA blocks.
                result = None
                y0 = None
                problem._last_aux_cache = None
                if torch.cuda.is_available() and not fatal_cuda_failure:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        fatal_cuda_failure = True
            if fatal_cuda_failure:
                # A launch failure poisons this process's CUDA context.  Do not
                # accept another task and masquerade as a live worker.
                try:
                    result_queue.close()
                    result_queue.join_thread()
                except (AttributeError, OSError, ValueError):
                    pass
                return
    except Exception:
        result_queue.put({
            "task_id": "__worker_init__",
            "ok": False,
            "worker_index": int(worker_index),
            "gpu_id": str(gpu_id),
            "traceback": traceback.format_exc(),
        })
    finally:
        stop_resource_monitor(stop_event, monitor_thread)


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
            name=f"ga-worker-{int(worker_index)}-gpu-{safe_name(gpu_id)}",
            daemon=True,
        )
        process.start()
        workers.append(process)
        terminal_print(args, f"[ga parallel] started worker={worker_index} gpu={gpu_id} pid={process.pid}")
        append_running_log(
            args,
            f"local_worker_started worker={worker_index} gpu={gpu_id} "
            f"pid={process.pid}",
        )
    return task_queue, result_queue, workers


def describe_ga_worker_exit(process):
    exitcode = process.exitcode
    if exitcode is None:
        return f"name={process.name} pid={process.pid} exitcode=unknown"
    if int(exitcode) < 0:
        signal_number = -int(exitcode)
        try:
            signal_name = signal.Signals(signal_number).name
        except (ValueError, AttributeError):
            signal_name = f"signal_{signal_number}"
        return (
            f"name={process.name} pid={process.pid} exitcode={exitcode} "
            f"signal={signal_name}"
        )
    return f"name={process.name} pid={process.pid} exitcode={exitcode}"


def dead_ga_workers(workers):
    dead = []
    for process in workers:
        if process.is_alive():
            continue
        # Refresh multiprocessing's cached exit status without blocking.
        process.join(timeout=0.0)
        dead.append(process)
    return dead


def terminate_ga_workers(args, workers, reason):
    """Bounded emergency cleanup used when one local worker fails."""
    append_running_log(args, f"local_worker_pool_terminating reason={reason}")
    for process in workers:
        if process.is_alive():
            process.terminate()
    for process in workers:
        process.join(timeout=5.0)
    for process in workers:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
    for process in workers:
        process.join(timeout=5.0)
        append_running_log(
            args,
            f"local_worker_stopped {describe_ga_worker_exit(process)}",
        )


def close_ga_queues(task_queue, result_queue):
    for queue_object in (task_queue, result_queue):
        if queue_object is None:
            continue
        try:
            queue_object.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            queue_object.close()
        except (AttributeError, OSError, ValueError):
            pass


def stop_ga_workers(args, task_queue, workers, result_queue=None):
    if task_queue is not None:
        for process in workers:
            if not process.is_alive():
                continue
            task_queue.put(None)
    for process in workers:
        process.join(timeout=30.0)
    stragglers = [process for process in workers if process.is_alive()]
    if stragglers:
        terminate_ga_workers(args, stragglers, "normal shutdown timed out")
    for process in workers:
        terminal_print(args, f"[ga parallel] stopped pid={process.pid} exitcode={process.exitcode}")
        append_running_log(
            args,
            f"local_worker_stopped {describe_ga_worker_exit(process)}",
        )
    close_ga_queues(task_queue, result_queue)


def emergency_stop_active_ga_pool(problem, args, reason):
    pool = getattr(problem, "_active_local_ga_pool", None)
    if not pool:
        return
    task_queue, result_queue, workers = pool
    try:
        terminate_ga_workers(args, workers, reason)
    except Exception as cleanup_error:
        print(
            f"[ga parallel] worker cleanup also failed: {cleanup_error}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        close_ga_queues(task_queue, result_queue)
        problem._active_local_ga_pool = None


def make_evaluation_task(task_id, y0, layout0, member_seed, generation, member_index, kind, parent_refs=None):
    return {
        "task_id": str(task_id),
        "y0": y_to_list(y0),
        "assigned_blob_indices": np.asarray(layout0, dtype=int).reshape(-1).astype(int).tolist(),
        "seed": int(member_seed),
        "generation": int(generation),
        "member_index": int(member_index),
        "kind": str(kind),
        "parent_refs": [] if parent_refs is None else list(parent_refs),
    }


def record_completed_ga_result(problem, args, seed, db_path, log_path, all_generations, generation_results, result, event="member_done"):
    problem.set_site_layout(result["assigned_blob_indices"])
    if (
        not bool(result.get("slice_comparison_written", False))
        and not bool(result.get("slice_comparison_disabled", False))
    ):
        problem.write_member_slice_comparisons(result)
    generation_results.append(result)
    append_structure_record(db_path, structure_database_record(problem, result, args, seed))
    append_ga_log(log_path, ga_member_log_record(result, event))
    metrics = result.get("compute_metrics", {})
    append_running_log(
        args,
        f"{event} generation={result['generation']} member={result['member_index']} "
        f"kind={result['kind']} score={result['score']:+.6f} "
        f"energy_evaluations={int(metrics.get('model_energy_evaluations', 0))} "
        f"energy_eval_s={float(metrics.get('model_evaluation_seconds', 0.0)):.2f} "
        f"member_wall_s={float(metrics.get('member_wall_seconds', result.get('elapsed_seconds', 0.0))):.2f}",
    )
    write_ga_checkpoint(
        problem,
        args,
        seed,
        all_generations,
        int(result["generation"]),
        generation_results,
        status="running",
    )
    update_live_ga_plot(problem, args, all_generations, generation_results)


def copy_imported_parent_to_generation(imported_parent, generation, member_index):
    copied = dict(imported_parent)
    copied.pop("slice_comparison_dir", None)
    copied.pop("slice_comparison_written", None)
    copied["generation"] = int(generation)
    copied["member_index"] = int(member_index)
    copied["parent_refs"] = [parent_ref(imported_parent)]
    copied["kind"] = "imported_seed"
    copied["y"] = imported_parent["y"].detach().clone()
    copied["y_rounded"] = imported_parent["y_rounded"].detach().clone()
    copied["assigned_blob_indices"] = list(imported_parent["assigned_blob_indices"])
    copied["history"] = list(imported_parent.get("history", []))
    copied["stage_summaries"] = list(imported_parent.get("stage_summaries", []))
    copied["compute_metrics"] = empty_compute_metrics()
    copied["elapsed_seconds"] = 0.0
    return copied


def collect_parallel_ga_tasks(
    problem,
    args,
    seed,
    db_path,
    log_path,
    all_generations,
    generation_results,
    result_queue,
    workers,
    expected_count,
):
    completed = 0
    last_main_heartbeat = 0.0
    interval = float(getattr(args, "heartbeat_interval_seconds", 0.0) or 0.0)
    health_interval = max(
        0.25,
        float(
            getattr(args, "worker_health_check_interval_seconds", 10.0)
            or 10.0
        ),
    )
    while completed < int(expected_count):
        try:
            message = result_queue.get(timeout=health_interval)
        except queue_module.Empty:
            dead = dead_ga_workers(workers)
            if dead:
                details = "; ".join(
                    describe_ga_worker_exit(process) for process in dead
                )
                reason = (
                    f"worker death detected with completed={completed}/"
                    f"{int(expected_count)}: {details}"
                )
                append_running_log(args, f"local_worker_death {reason}")
                terminate_ga_workers(args, workers, reason)
                raise RuntimeError(
                    "A local GA worker exited before returning all member "
                    f"results ({completed}/{int(expected_count)} complete). "
                    f"{details}. The completed-member GA checkpoint and any "
                    "mid-member checkpoints remain valid; rerun the same "
                    "command to resume instead of waiting indefinitely."
                )
            now = time.time()
            if interval > 0.0 and now - last_main_heartbeat >= interval:
                append_running_log(
                    args,
                    f"main waiting for generation members "
                    f"completed={completed}/{int(expected_count)}; "
                    f"workers_alive={len(workers)}",
                )
                last_main_heartbeat = now
            continue
        if not message.get("ok", False):
            reason = (
                f"worker-reported failure task={message.get('task_id')} "
                f"worker={message.get('worker_index')} "
                f"gpu={message.get('gpu_id')}"
            )
            append_running_log(args, f"local_worker_failure {reason}")
            terminate_ga_workers(args, workers, reason)
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


def finish_ga_task_batch(problem, args, seed, db_path, log_path, all_generations,
                         generation_results, generation, tasks, submitted_count,
                         result_queue, workers, phase=""):
    """Collect a completed batch before its results are used as mutation donors."""
    if submitted_count <= 0:
        return
    if str(args.ga_execution_backend).strip().lower() == "slurm":
        previous_phase = getattr(problem, "_slurm_task_phase", "")
        problem._slurm_task_phase = phase
        try:
            write_slurm_task_files(problem, generation, tasks)
            job_id, _ = submit_slurm_generation(problem, args, generation, submitted_count)
            collect_slurm_ga_tasks(problem, args, seed, db_path, log_path,
                                   all_generations, generation_results, generation, tasks, job_id)
        finally:
            problem._slurm_task_phase = previous_phase
    else:
        collect_parallel_ga_tasks(problem, args, seed, db_path, log_path,
                                  all_generations, generation_results, result_queue,
                                  workers, submitted_count)


def current_mutation_donors(generation_results, generation, first_index, stop_index):
    """Use only completed random slots, ordered independently of worker timing."""
    donors = sorted((r for r in generation_results
                     if int(r["generation"]) == generation
                     and r["kind"] == "random_immigrant"
                     and first_index <= int(r["member_index"]) < stop_index),
                    key=lambda r: int(r["member_index"]))
    if [int(r["member_index"]) for r in donors] != list(range(first_index, stop_index)) or not donors:
        raise RuntimeError("All current-generation random immigrants must finish before constructing mutations.")
    return donors


def slurm_root_dir(problem):
    return problem.output_dir / "slurm"


def slurm_generation_dirs(problem, generation):
    root = slurm_root_dir(problem)
    # Separate artifacts for the donor and offspring batches of one generation.
    phase = getattr(problem, "_slurm_task_phase", "")
    if phase:
        root = root / phase
    generation_name = f"generation_{int(generation):04d}"
    return {
        "root": root,
        "tasks": root / "tasks" / generation_name,
        "results": root / "results" / generation_name,
        "scripts": root / "scripts",
        "out": root / "slurm_out",
        "resources": root / "resources" / generation_name,
    }


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_slurm_task_files(problem, generation, tasks):
    dirs = slurm_generation_dirs(problem, generation)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for directory, pattern in (
        (dirs["tasks"], "task_*.json"),
        (dirs["results"], "result_*.json"),
        (dirs["resources"], "resource_*.tsv"),
    ):
        for stale_path in directory.glob(pattern):
            stale_path.unlink()
    for array_index, task in enumerate(tasks, start=1):
        write_json_atomic(dirs["tasks"] / f"task_{array_index:05d}.json", task)
    return dirs


def slurm_directive(name, value):
    text = str(value).strip()
    if not text:
        return ""
    return f"#SBATCH --{name}={text}"


def write_slurm_array_script(problem, args, generation, task_count):
    dirs = slurm_generation_dirs(problem, generation)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    script_path = dirs["scripts"] / f"submit_generation_{int(generation):04d}.sh"
    job_name = safe_name(f"{args.slurm_job_name}_g{int(generation):04d}")[:64] or f"blobGA_g{int(generation):04d}"
    script_file = Path(__file__).resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    venv = str(args.slurm_venv or "").strip()
    python_executable = str(args.slurm_python or "").strip() or "python"
    launch_directory = Path.cwd().resolve()
    directives = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -N {int(args.slurm_nodes)}",
        f"#SBATCH -c {int(args.slurm_cpus_per_task)}",
        slurm_directive("partition", args.slurm_partition),
        slurm_directive("account", args.slurm_account),
        slurm_directive("constraint", args.slurm_constraint),
        slurm_directive("time", args.slurm_time),
        slurm_directive("mem", args.slurm_mem),
        f"#SBATCH --array=1-{int(task_count)}",
        f"#SBATCH --output={dirs['out']}/g{int(generation):04d}_%A_%a.out",
        f"#SBATCH --error={dirs['out']}/g{int(generation):04d}_%A_%a.err",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(launch_directory))}",
        "export PYTHONUNBUFFERED=1",
        "export CUDA_VISIBLE_DEVICES=\"\"",
        "if [ -n \"${SLURM_CPUS_PER_TASK:-}\" ]; then",
        "  export OMP_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export MKL_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export OPENBLAS_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export NUMEXPR_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "fi",
    ]
    task_pattern = str(dirs["tasks"] / "task_%05d.json")
    result_pattern = str(dirs["results"] / "result_%05d.json")
    resource_pattern = str(dirs["resources"] / "resource_%05d.tsv")
    directives.extend([
        f"TASK_FILE=$(printf {shlex.quote(task_pattern)} \"$SLURM_ARRAY_TASK_ID\")",
        f"RESULT_FILE=$(printf {shlex.quote(result_pattern)} \"$SLURM_ARRAY_TASK_ID\")",
        f"RESOURCE_LOG=$(printf {shlex.quote(resource_pattern)} \"$SLURM_ARRAY_TASK_ID\")",
        "mkdir -p \"$(dirname \"$RESULT_FILE\")\" \"$(dirname \"$RESOURCE_LOG\")\"",
        "write_shell_failure_result() {",
        "  status=\"$?\"",
        "  if [ ! -s \"$RESULT_FILE\" ]; then",
        "    cat > \"$RESULT_FILE\" <<JSON",
        "{\"ok\": false, \"traceback\": \"Slurm shell failed before the Python worker produced a result JSON. exit_status=${status} job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-unknown} host=$(hostname). Check the matching .out/.err file.\"}",
        "JSON",
        "  fi",
        "}",
        "trap 'write_shell_failure_result' ERR",
        "echo \"[$(date +'%F %T')] host=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}\"",
    ])
    if venv:
        directives.append(f"source {shlex.quote(venv)}")
    directives.extend([
        f"PYTHON_EXE={shlex.quote(python_executable)}",
        "if [ ! -x \"$PYTHON_EXE\" ] && ! command -v \"$PYTHON_EXE\" >/dev/null 2>&1; then",
        "  echo \"Configured Python executable not found: $PYTHON_EXE\" >&2",
        "  false",
        "fi",
        (
            f"\"$PYTHON_EXE\" -u {shlex.quote(str(script_file))} "
            f"--config {shlex.quote(str(config_path))} "
            "--slurm-worker-task \"$TASK_FILE\" "
            "--slurm-worker-result \"$RESULT_FILE\" "
            "--slurm-worker-resource-log \"$RESOURCE_LOG\""
        ),
        "echo \"[$(date +'%F %T')] done task=${SLURM_ARRAY_TASK_ID}\"",
        "",
    ])
    script_path.write_text("\n".join(line for line in directives if line != "") + "\n", encoding="utf-8")
    return script_path


def write_slurm_single_job_script(problem, args, generation, task_index):
    dirs = slurm_generation_dirs(problem, generation)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    script_path = dirs["scripts"] / f"submit_generation_{int(generation):04d}_task_{int(task_index):05d}.sh"
    job_name = safe_name(f"{args.slurm_job_name}_g{int(generation):04d}_m{int(task_index):05d}")[:64] or f"blobGA_g{int(generation):04d}_m{int(task_index):05d}"
    script_file = Path(__file__).resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    venv = str(args.slurm_venv or "").strip()
    python_executable = str(args.slurm_python or "").strip() or "python"
    launch_directory = Path.cwd().resolve()
    task_file = dirs["tasks"] / f"task_{int(task_index):05d}.json"
    result_file = dirs["results"] / f"result_{int(task_index):05d}.json"
    resource_log = dirs["resources"] / f"resource_{int(task_index):05d}.tsv"
    directives = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -N {int(args.slurm_nodes)}",
        "#SBATCH -n 1",
        f"#SBATCH -c {int(args.slurm_cpus_per_task)}",
        slurm_directive("partition", args.slurm_partition),
        slurm_directive("account", args.slurm_account),
        slurm_directive("constraint", args.slurm_constraint),
        slurm_directive("time", args.slurm_time),
        slurm_directive("mem", args.slurm_mem),
        f"#SBATCH --output={dirs['out']}/g{int(generation):04d}_%j_m{int(task_index):05d}.out",
        f"#SBATCH --error={dirs['out']}/g{int(generation):04d}_%j_m{int(task_index):05d}.err",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(launch_directory))}",
        "export PYTHONUNBUFFERED=1",
        "export CUDA_VISIBLE_DEVICES=\"\"",
        "if [ -n \"${SLURM_CPUS_PER_TASK:-}\" ]; then",
        "  export OMP_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export MKL_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export OPENBLAS_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "  export NUMEXPR_NUM_THREADS=\"$SLURM_CPUS_PER_TASK\"",
        "fi",
        f"TASK_FILE={shlex.quote(str(task_file))}",
        f"RESULT_FILE={shlex.quote(str(result_file))}",
        f"RESOURCE_LOG={shlex.quote(str(resource_log))}",
        "mkdir -p \"$(dirname \"$RESULT_FILE\")\" \"$(dirname \"$RESOURCE_LOG\")\"",
        "write_shell_failure_result() {",
        "  status=\"$?\"",
        "  if [ ! -s \"$RESULT_FILE\" ]; then",
        "    cat > \"$RESULT_FILE\" <<JSON",
        "{\"ok\": false, \"traceback\": \"Slurm shell failed before the Python worker produced a result JSON. exit_status=${status} job=${SLURM_JOB_ID:-unknown} host=$(hostname). Check the matching .out/.err file.\"}",
        "JSON",
        "  fi",
        "}",
        "trap 'write_shell_failure_result' ERR",
        "echo \"[$(date +'%F %T')] host=$(hostname) job=${SLURM_JOB_ID:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset}\"",
    ]
    if venv:
        directives.append(f"source {shlex.quote(venv)}")
    directives.extend([
        f"PYTHON_EXE={shlex.quote(python_executable)}",
        "if [ ! -x \"$PYTHON_EXE\" ] && ! command -v \"$PYTHON_EXE\" >/dev/null 2>&1; then",
        "  echo \"Configured Python executable not found after venv activation: $PYTHON_EXE\" >&2",
        "  false",
        "fi",
        (
            f"\"$PYTHON_EXE\" -u {shlex.quote(str(script_file))} "
            f"--config {shlex.quote(str(config_path))} "
            "--slurm-worker-task \"$TASK_FILE\" "
            "--slurm-worker-result \"$RESULT_FILE\" "
            "--slurm-worker-resource-log \"$RESOURCE_LOG\""
        ),
        "echo \"[$(date +'%F %T')] done job=${SLURM_JOB_ID:-unknown}\"",
        "",
    ])
    script_path.write_text("\n".join(line for line in directives if line != "") + "\n", encoding="utf-8")
    return script_path


def submit_slurm_generation(problem, args, generation, task_count):
    if bool(getattr(args, "slurm_use_array", False)):
        script_path = write_slurm_array_script(problem, args, generation, task_count)
        script_paths = [script_path]
    else:
        script_paths = [write_slurm_single_job_script(problem, args, generation, task_index) for task_index in range(1, int(task_count) + 1)]
    job_ids = []
    for script_path in script_paths:
        command = ["sbatch", "--parsable", str(script_path)]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Could not find sbatch. Run Slurm backend from a Slurm login/submit node.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"sbatch failed for {script_path}:\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}") from exc
        job_id = completed.stdout.strip().split(";")[0]
        if not job_id:
            raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {completed.stdout!r}")
        job_ids.append(job_id)
    job_id_text = ",".join(job_ids)
    append_running_log(args, f"slurm_submitted generation={generation} job_ids={job_id_text} tasks={task_count} scripts={len(script_paths)} use_array={bool(getattr(args, 'slurm_use_array', False))}")
    return job_id_text, script_paths[0]


def slurm_job_is_active(job_id):
    try:
        completed = subprocess.run(["squeue", "-h", "-j", str(job_id)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False
    return bool(completed.stdout.strip())


def command_output(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return ""
    text = (completed.stdout or "") + (completed.stderr or "")
    return text.strip()


def tail_text_file(path, max_lines=80):
    path = Path(path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(lines[-int(max_lines):])


def slurm_failure_diagnostics(problem, generation, job_id, missing_indices):
    dirs = slurm_generation_dirs(problem, generation)
    pieces = []
    sacct_text = command_output([
        "sacct",
        "-j",
        str(job_id),
        "--format=JobID,JobName%24,Partition,State,ExitCode,Reason%40,Elapsed,NodeList%32,MaxRSS,ReqMem,AllocCPUS",
        "--parsable2",
    ])
    if sacct_text:
        pieces.append("sacct:\n" + sacct_text)
    script_paths = sorted(dirs["scripts"].glob(f"submit_generation_{int(generation):04d}*.sh"))
    for script_path in script_paths[:4]:
        script_text = tail_text_file(script_path, max_lines=200)
        if script_text:
            pieces.append(f"generated script {script_path}:\n{script_text}")
    job_ids = [part for part in str(job_id).split(",") if part]
    for array_index in list(missing_indices)[:3]:
        out_glob = []
        err_glob = []
        for single_job_id in job_ids:
            out_glob.extend(sorted(dirs["out"].glob(f"g{int(generation):04d}_{single_job_id}_{int(array_index)}.out")))
            err_glob.extend(sorted(dirs["out"].glob(f"g{int(generation):04d}_{single_job_id}_{int(array_index)}.err")))
            out_glob.extend(sorted(dirs["out"].glob(f"g{int(generation):04d}_{single_job_id}_m{int(array_index):05d}.out")))
            err_glob.extend(sorted(dirs["out"].glob(f"g{int(generation):04d}_{single_job_id}_m{int(array_index):05d}.err")))
        for path in out_glob + err_glob:
            text = tail_text_file(path, max_lines=120)
            if text:
                pieces.append(f"tail {path}:\n{text}")
    if not pieces:
        pieces.append(
            f"No sacct output or Slurm .out/.err files were found under {dirs['out']}. "
            "The array may have been rejected/cancelled before tasks launched, or accounting may be delayed."
        )
    return "\n\n".join(pieces)


def slurm_result_path(problem, generation, array_index):
    return slurm_generation_dirs(problem, generation)["results"] / f"result_{int(array_index):05d}.json"


def collect_slurm_ga_tasks(problem, args, seed, db_path, log_path, all_generations, generation_results, generation, expected_tasks, job_id):
    expected_tasks = list(expected_tasks)
    expected_count = len(expected_tasks)
    completed_indices = set()
    last_main_heartbeat = 0.0
    inactive_since = None
    heartbeat_interval = float(getattr(args, "heartbeat_interval_seconds", 0.0) or 0.0)
    poll_seconds = max(1.0, float(getattr(args, "slurm_poll_seconds", 30.0) or 30.0))
    inactive_grace_seconds = max(60.0, 2.0 * poll_seconds)
    while len(completed_indices) < int(expected_count):
        made_progress = False
        for array_index in range(1, int(expected_count) + 1):
            if array_index in completed_indices:
                continue
            path = slurm_result_path(problem, generation, array_index)
            if not path.exists():
                continue
            payload = read_json_file(path)
            if not payload.get("ok", False):
                raise RuntimeError(
                    f"Slurm worker failed for generation={generation} array_index={array_index} "
                    f"job_id={job_id}:\n{payload.get('traceback', '')}"
                )
            result = restore_ga_result_tensors(problem, payload["result"])
            expected_member_index = int(
                expected_tasks[array_index - 1]["member_index"]
            )
            if (
                int(result.get("generation", -1)) != int(generation)
                or int(result.get("member_index", -1)) != expected_member_index
            ):
                append_running_log(
                    args,
                    f"ignored_stale_slurm_result generation={generation} "
                    f"array_index={array_index} expected_member={expected_member_index} "
                    f"found_generation={result.get('generation')} "
                    f"found_member={result.get('member_index')}",
                )
                path.unlink(missing_ok=True)
                continue
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
            completed_indices.add(array_index)
            made_progress = True
            append_running_log(
                args,
                f"slurm_member_done generation={generation} completed={len(completed_indices)}/{int(expected_count)} "
                f"member_index={result['member_index']} score={result['score']:+.6f}",
            )
        if len(completed_indices) >= int(expected_count):
            break
        now = time.time()
        if heartbeat_interval > 0.0 and now - last_main_heartbeat >= heartbeat_interval:
            append_running_log(
                args,
                f"slurm_wait generation={generation} job_id={job_id} completed={len(completed_indices)}/{int(expected_count)}",
            )
            last_main_heartbeat = now
        if not made_progress and not slurm_job_is_active(job_id):
            if inactive_since is None:
                inactive_since = time.time()
            elif time.time() - inactive_since >= inactive_grace_seconds:
                missing = [idx for idx in range(1, int(expected_count) + 1) if idx not in completed_indices]
                diagnostics = slurm_failure_diagnostics(problem, generation, job_id, missing)
                raise RuntimeError(
                    f"Slurm job {job_id} is no longer active but results are missing for array indices {missing[:20]}.\n\n"
                    f"{diagnostics}"
                )
        elif made_progress:
            inactive_since = None
        time.sleep(poll_seconds)


def run_slurm_worker(config_path, task_path, result_path, resource_log_path):
    args = get_settings(config_path)
    args.slurm_worker_mode = True
    # Use the controller's preserved fingerprint for per-member restart files.
    read_ga_checkpoint_for_args(args)
    configure_cpu_threads(args)
    task = read_json_file(task_path)
    result_path = Path(result_path).expanduser()
    stop_event, monitor_thread = start_resource_monitor(
        resource_log_path,
        getattr(args, "slurm_resource_log_interval_seconds", 0.0),
        f"g{int(task['generation'])}_m{int(task['member_index'])}_{task.get('kind', 'member')}",
    )
    try:
        member_seed = int(task["seed"])
        torch.manual_seed(member_seed)
        np.random.seed(member_seed % 2**32)
        problem = WSe2VolumeProblem(args)
        layout0 = np.asarray(task["assigned_blob_indices"], dtype=int).reshape(-1)
        problem.set_site_layout(layout0)
        y0 = torch.tensor(task["y0"], device=problem.device, dtype=problem.dtype)
        result = evaluate_candidate_member(
            problem,
            y0,
            layout0,
            args,
            member_seed,
            int(task["generation"]),
            int(task["member_index"]),
            parent_refs=task.get("parent_refs", []),
            kind=str(task.get("kind", "random")),
        )
        write_json_atomic(result_path, {"ok": True, "result": serialize_ga_result_for_worker(result)})
    except Exception:
        write_json_atomic(result_path, {"ok": False, "traceback": traceback.format_exc()})
        raise
    finally:
        stop_resource_monitor(stop_event, monitor_thread)


def ga_member_log_record(result, event):
    metrics = result.get("compute_metrics", {})
    phases = metrics.get("by_phase", {})
    return {
        "event": str(event),
        "generation": int(result["generation"]),
        "member_index": int(result["member_index"]),
        "parent": ",".join(result.get("parent_refs", [])),
        "kind": result["kind"],
        "score": float(result["score"]),
        "formation_energy_per_atom": float(result["formation_energy_per_atom"]),
        "formation_energy_difference_from_reference": float(result["formation_energy_difference_from_reference"]),
        "f_guided_no_entropy": float(result["f_guided_no_entropy"]),
        "composition_penalty": float(result.get("composition_penalty", 0.0)),
        "se_vacancy_bias_term": float(result.get("se_vacancy_bias_term", 0.0)),
        "se_vacancy_count_relative_to_reference": float(
            result.get("se_vacancy_count_relative_to_reference", 0.0)
        ),
        "se_occupancy_count": float(result.get("se_occupancy_count", 0.0)),
        "image_term": float(result.get("image_term", 0.0)),
        "voxel_image_term": float(result.get("voxel_image_term", 0.0)),
        "ellipse_composition_term": float(
            result.get("ellipse_composition_term", 0.0)
        ),
        "ellipse_composition_loss_raw": float(
            result.get("ellipse_composition_loss_raw", 0.0)
        ),
        "image_loss_raw": float(result.get("image_loss_raw", 0.0)),
        "image_correlation": float(result.get("image_correlation", 0.0)),
        "sample_box_term": float(result.get("sample_box_term", 0.0)),
        "sample_box_loss_raw_A2": float(
            result.get("sample_box_loss_raw_A2", 0.0)
        ),
        "sample_box_active_outside_count": int(
            result.get("sample_box_active_outside_count", 0)
        ),
        "sumx": float(result["sumx"]),
        "entropy_avg": float(result["entropy_avg"]),
        "max_min": float(result["max_min"]),
        "composition_fractions": result["composition_fractions"],
        "assigned_blob_inside_count": int(result["assigned_blob_inside_count"]),
        "n_sites": int(result.get("n_sites", len(result.get("assigned_blob_indices", [])))),
        "history_file": str(result.get("history_file", "")),
        "slice_comparison_dir": str(result.get("slice_comparison_dir", "")),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "energy_evaluations": int(metrics.get("model_energy_evaluations", 0)),
        "energy_evaluation_seconds": float(metrics.get("model_evaluation_seconds", 0.0)),
        "member_wall_seconds": float(metrics.get("member_wall_seconds", result.get("elapsed_seconds", 0.0))),
        "objective_evaluations": int(metrics.get("objective_evaluations", 0)),
        "accepted_optimizer_iterations": int(metrics.get("accepted_optimizer_iterations", 0)),
        "syntropization_energy_evaluations": int(phases.get("syntropization", {}).get("count", 0)),
        "syntropization_energy_evaluation_seconds": float(phases.get("syntropization", {}).get("seconds", 0.0)),
        "final_PBFGS_energy_evaluations": int(phases.get("final_PBFGS", {}).get("count", 0)),
        "final_PBFGS_energy_evaluation_seconds": float(phases.get("final_PBFGS", {}).get("seconds", 0.0)),
        "final_scoring_energy_evaluations": int(phases.get("final_scoring", {}).get("count", 0)),
        "final_scoring_energy_evaluation_seconds": float(phases.get("final_scoring", {}).get("seconds", 0.0)),
        "random_immigrant_image_prefit": copy.deepcopy(
            result.get("random_immigrant_image_prefit")
        ),
    }


def write_ga_summary(problem, args, log_path, generations, best):
    total_compute = aggregate_compute_metrics(
        member for generation in generations for member in generation
    )
    payload = {
        "generator": str(Path(__file__).resolve()),
        "config_path": str(args.config_path),
        "experimental_tiff": str(problem.blob_path),
        "experimental_crop_origin_zyx": problem.experimental["crop_origin"].tolist(),
        "sample_cell_lengths_A": problem.experimental["sample_cell_lengths_A"].tolist(),
        "padded_cell_lengths_A": np.diag(problem.cell).tolist(),
        "requested_vacuum_padding_angstrom": (
            problem.requested_vacuum_padding.tolist()
        ),
        "vacuum_padding_angstrom": problem.vacuum_padding.tolist(),
        "pbc": problem.pbc.astype(bool).tolist(),
        "target_shape_zyx": list(problem.target_shape_zyx),
        "sample_shape_zyx": list(problem.experimental["sample_shape_zyx"]),
        "sample_roi_start_zyx": problem.experimental[
            "sample_roi_start_zyx"
        ].tolist(),
        "sample_roi_stop_zyx": problem.experimental[
            "sample_roi_stop_zyx"
        ].tolist(),
        "image_loss_voxel_count": int(problem.image_loss_voxel_count),
        "target_voxel_size_A": problem.target_voxel_size_A.tolist(),
        "sample_box_confinement_penalty": float(
            problem.sample_box_confinement_penalty
        ),
        "sample_box_confinement_sharpness": float(
            problem.sample_box_confinement_sharpness
        ),
        "sample_box_confinement_margin_angstrom": float(
            problem.sample_box_confinement_margin_A
        ),
        "sample_box_lower_xyz_angstrom": problem.sample_box_lower_A.tolist(),
        "sample_box_upper_xyz_angstrom": problem.sample_box_upper_A.tolist(),
        "sample_box_confinement_normalization": "total_W_plus_Se_occupancy",
        "heavy_threshold_value": float(problem.experimental["threshold_value"]),
        "heavy_allowed_fraction": float(problem.experimental["heavy_allowed_fraction"]),
        "heavy_allowed_volume_angstrom3": float(
            problem.feasible_volume_angstrom3
        ),
        "candidate_site_density_per_angstrom3": float(
            problem.heavy_site_density_per_angstrom3
        ),
        "initialization_scheme": "intensity_prioritized_cylinder_with_angstrom_jitter",
        "initial_cylinder_lateral_radius_angstrom": float(
            problem.initial_cylinder_lateral_radius_A
        ),
        "initial_cylinder_z_radius_angstrom": float(
            problem.initial_cylinder_z_radius_A
        ),
        "initial_jitter_angstrom": float(problem.initial_jitter_A),
        "initial_candidate_sites": int(problem.initial_heavy_region_sites),
        "feasible_sample_volume_angstrom3": float(problem.feasible_volume_angstrom3),
        "sample_volume_angstrom3": float(problem.sample_volume_angstrom3),
        "padded_cell_volume_angstrom3": float(problem.cell_volume_angstrom3),
        "initial_n_sites": int(problem.initial_n_sites),
        "se_vacancy_bias_enable": bool(problem.se_vacancy_bias_enable),
        "se_vacancy_bias_alpha_eV_per_vacancy": float(
            problem.se_vacancy_bias_alpha_eV
        ),
        "se_vacancy_reference_atom_count": float(
            problem.se_vacancy_reference_atom_count
        ),
        "se_vacancy_reference_se_count": float(
            problem.se_vacancy_reference_se_count
        ),
        "se_vacancy_bias_definition": (
            "-alpha*(reference_Se_count-summed_Se_occupancy)/reference_atom_count"
        ),
        "image_penalty": float(problem.image_penalty),
        "image_huber_delta": float(problem.image_huber_delta),
        "ellipse_composition_penalty": float(
            problem.ellipse_composition_penalty
        ),
        "ellipse_region_count": int(problem.ellipse_region_count),
        "ellipse_I_Se": float(problem.ellipse_I_Se),
        "ellipse_I_W": float(problem.ellipse_I_W),
        "ellipse_W_to_Se_ratio": float(problem.ellipse_I_W / problem.ellipse_I_Se),
        "ellipse_membership_boundary_softness": float(
            problem.ellipse_membership_boundary_softness
        ),
        "ellipse_membership_background_score": float(
            problem.ellipse_membership_background_score
        ),
        "ellipse_region_definition": (
            "removed"
        ),
        "ellipse_loss_definition": (
            "removed; compatibility metrics are zero"
        ),
        "image_forward_model": "species_resolved_double_xy_gaussian_gaussian_depth_no_envelope",
        "image_target_transform": problem.image_target_transform,
        "image_target_domain": (
            f"{problem.image_target_transform}_of_background_subtracted_"
            "normalized_linear_TIFF"
        ),
        "image_transform_epsilon": float(problem.gaussian_transform_epsilon),
        "gaussian_total_integrated_amplitudes": {
            symbol: float(problem.gaussian_amplitudes[index])
            for index, symbol in enumerate(problem.species)
        },
        "gaussian_component_integrated_amplitudes": {
            symbol: {
                "core": float(problem.gaussian_component_amplitudes[index, 0]),
                "tail": float(problem.gaussian_component_amplitudes[index, 1]),
            }
            for index, symbol in enumerate(problem.species)
        },
        "gaussian_component_sigma_xy_angstrom": {
            symbol: {
                "core": float(problem.gaussian_sigma_xy_A[index, 0]),
                "tail": float(problem.gaussian_sigma_xy_A[index, 1]),
            }
            for index, symbol in enumerate(problem.species)
        },
        "gaussian_fitted_component_sigma_xy_angstrom": {
            symbol: {
                "core": float(problem.gaussian_fitted_sigma_xy_A[index, 0]),
                "tail": float(problem.gaussian_fitted_sigma_xy_A[index, 1]),
            }
            for index, symbol in enumerate(problem.species)
        },
        "gaussian_additional_lateral_blur_angstrom": float(
            problem.gaussian_additional_lateral_blur_A
        ),
        "gaussian_image_model": problem.image_model_metadata,
        "gaussian_element_fit_csv": str(problem.gaussian_element_fit_csv),
        "gaussian_relative_intensity_reference_species": problem.gaussian_relative_intensity_reference_species,
        "gaussian_intensity_scale_source": problem.gaussian_intensity_scale_source,
        "gaussian_rendered_single_atom_peak_amplitudes": {
            symbol: float(
                problem.gaussian_rendered_single_atom_peak_amplitudes[index]
            )
            for index, symbol in enumerate(problem.species)
        },
        "gaussian_reference_integrated_amplitude": float(
            problem.gaussian_reference_integrated_amplitude
        ),
        "gaussian_phase_table_to_image_amplitude_scale": float(
            problem.gaussian_phase_table_to_image_amplitude_scale
        ),
        "depth_response_rms_angstrom": float(
            problem.gaussian_shared_sigma_z_A
        ),
        "random_immigrant_image_prefit": {
            "enable": bool(args.random_immigrant_image_prefit_enable),
            "initial_occupancy_mode": str(
                args.random_immigrant_image_prefit_initial_occupancy_mode
            ),
            "initial_logit_noise": float(
                args.random_immigrant_image_prefit_initial_logit_noise
            ),
            "soft_steps": int(args.random_immigrant_image_prefit_soft_steps),
            "final_position_steps": int(
                args.random_immigrant_image_prefit_final_position_steps
            ),
            "position_learning_rate": float(
                args.random_immigrant_image_prefit_position_learning_rate
            ),
            "occupancy_learning_rate": float(
                args.random_immigrant_image_prefit_occupancy_learning_rate
            ),
            "entropy_final_weight": float(
                args.random_immigrant_image_prefit_entropy_final_weight
            ),
            "repulsion_weight": float(
                args.random_immigrant_image_prefit_repulsion_weight
            ),
            "minimum_distance_angstrom": float(
                args.random_immigrant_image_prefit_minimum_distance_angstrom
            ),
            "composition_penalty_enable": bool(
                args.random_immigrant_image_prefit_composition_penalty_enable
            ),
            "compact_null_sites_before_fairchem": bool(
                args.random_immigrant_image_prefit_compact_null_sites_before_fairchem
            ),
            "handoff": "soft_prefit_to_energy_inclusive_syntropization",
            "rounding": "once_after_syntropization_before_final_positions",
            "image_loss_domain": "experimental_roi_only",
        },
        "ga_enabled": True,
        "ga_population_size": int(args.ga_population_size),
        "ga_generations": int(args.ga_generations),
        "ga_elite_fraction": float(args.ga_elite_fraction),
        "ga_random_fraction": float(args.ga_random_fraction),
        "ga_crossover_fraction": float(args.ga_crossover_fraction),
        "ga_mutation_fraction": float(args.ga_mutation_fraction),
        "ga_parent_pool_size": int(args.ga_parent_pool_size),
        "ga_crossover_pos_sigma": float(args.ga_crossover_pos_sigma),
        "ga_crossover_occ_sigma": float(args.ga_crossover_occ_sigma),
        "ga_crossover_species_sigma": float(args.ga_crossover_species_sigma),
        "ga_parent_a_tournament_size": int(args.ga_parent_a_tournament_size),
        "ga_parent_b_tournament_size": int(args.ga_parent_b_tournament_size),
        "ga_parent_selection": "two_independent_best_of_tournament",
        "ga_elite_selection": "best_score",
        "ga_superposition_occ_scale": float(args.ga_superposition_occ_scale),
        "ga_superposition_prune_distance_angstrom": float(args.ga_superposition_prune_distance_angstrom),
        "ga_superposition_prune_occ_thresh": float(args.ga_superposition_prune_occ_thresh),
        "ga_score": str(args.ga_score),
        "ga_execution_backend": str(args.ga_execution_backend),
        "ga_parallel_workers": int(args.ga_parallel_workers),
        "ga_parallel_gpu_ids": str(args.ga_parallel_gpu_ids),
        "requested_vacuum_padding_angstrom": (
            problem.requested_vacuum_padding.tolist()
        ),
        "vacuum_padding_angstrom": problem.vacuum_padding.tolist(),
        "pbc": problem.pbc.astype(bool).tolist(),
        "legacy_diversity_subcells_xyz": problem.diversity_subcells_xyz.tolist(),
        "crossover_parent_alignment": "none",
        "slurm_partition": str(args.slurm_partition),
        "slurm_cpus_per_task": int(args.slurm_cpus_per_task),
        "slurm_time": str(args.slurm_time),
        "slurm_mem": str(args.slurm_mem),
        "slurm_venv": str(args.slurm_venv),
        "slurm_resource_log_interval_seconds": float(args.slurm_resource_log_interval_seconds),
        "terminal_verbose": bool(args.terminal_verbose),
        "heartbeat_interval_seconds": float(args.heartbeat_interval_seconds),
        "worker_health_check_interval_seconds": float(
            args.worker_health_check_interval_seconds
        ),
        "running_log_file": str(running_log_path_from_args(args)),
        "ga_log_file": str(log_path),
        "ga_live_plot_file": str(live_plot_path(problem, args)),
        "structures_database": str(structures_db_path(problem)),
        "trajectory_history_disabled": bool(args.disable_trajectory_history),
        "compute_metrics": total_compute,
        "ga_wall_seconds_this_process": float(
            getattr(problem, "ga_wall_seconds_this_process", 0.0)
        ),
        "best": ga_member_log_record(best, "best"),
        "generations": [
            [ga_member_log_record(member, "member_summary") for member in generation]
            for generation in generations
        ],
    }
    summary_path = problem.output_dir / "genetic_optimization_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote GA summary: {summary_path}")


def parent_ref(parent):
    refs = parent.get("parent_refs", [])
    if parent.get("kind") == "imported_parent" and refs:
        return refs[0]
    return f"g{int(parent['generation'])}:m{int(parent['member_index'])}:{parent['kind']}"


def allocate_generation_counts(population_size, args):
    fractions = np.asarray([
        float(args.ga_elite_fraction),
        float(args.ga_random_fraction),
        float(args.ga_crossover_fraction),
        float(getattr(args, "ga_mutation_fraction", 0.0)),
    ], dtype=float)
    if not np.isfinite(fractions).all() or np.any(fractions < 0.0):
        raise ValueError("GA elite/random/crossover/mutation fractions must be finite and non-negative.")
    if fractions[3] > 0 and fractions[1] <= 0:
        raise ValueError("A positive mutation_fraction requires a positive random_fraction for current-generation donors.")
    total = float(fractions.sum())
    if total <= 0.0:
        raise ValueError("At least one GA operator fraction must be positive.")
    fractions = fractions / total
    raw = fractions * int(population_size)
    counts = np.floor(raw).astype(int)
    remainder = int(population_size) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    # A configured nonzero operator must actually appear. Largest-remainder
    # rounding alone can silently assign zero members to a small positive
    # random_fraction in a finite population.
    positive = np.flatnonzero(fractions > 0.0)
    if int(population_size) < int(len(positive)):
        raise ValueError(
            "Population size is too small to allocate at least one member to "
            "every GA operator with a nonzero fraction."
        )
    for recipient in positive:
        recipient = int(recipient)
        if counts[recipient] > 0:
            continue
        donors = [int(index) for index in positive if counts[int(index)] > 1]
        if not donors:
            raise RuntimeError("Could not preserve all nonzero GA operator fractions.")
        donor = max(donors, key=lambda index: (counts[index], raw[index]))
        counts[donor] -= 1
        counts[recipient] += 1
    return {"elite": int(counts[0]), "random": int(counts[1]), "crossover": int(counts[2]), "mutation": int(counts[3])}


def choose_tournament_parent(parent_pool, tournament_size, exclude=None):
    if not parent_pool:
        raise ValueError("Cannot choose from an empty parent pool.")
    candidates = [parent for parent in parent_pool if parent is not exclude]
    if not candidates:
        candidates = list(parent_pool)
    size = min(max(1, int(tournament_size)), len(candidates))
    indices = np.random.choice(len(candidates), size=size, replace=False)
    tournament = [candidates[int(index)] for index in indices]
    return min(tournament, key=lambda parent: float(parent["score"]))


def member_state_arrays(problem, member):
    layout = np.asarray(
        member.get("assigned_site_region_indices", member.get("assigned_blob_indices", [])),
        dtype=int,
    ).reshape(-1)
    y = member.get("y_rounded", member.get("y"))
    if y is None or layout.size == 0:
        return layout, np.zeros((0, problem.n_species), dtype=float), np.zeros((0, 3), dtype=float)
    y_np = y.detach().cpu().numpy().reshape(-1) if torch.is_tensor(y) else np.asarray(y, dtype=float).reshape(-1)
    occ_count = int(layout.size) * int(problem.n_species)
    if y_np.size < occ_count + int(layout.size) * 3:
        return layout, np.zeros((layout.size, problem.n_species), dtype=float), np.zeros((layout.size, 3), dtype=float)
    occ = y_np[:occ_count].reshape(int(layout.size), int(problem.n_species))
    z = y_np[occ_count:occ_count + int(layout.size) * 3].reshape(int(layout.size), 3)
    return layout, occ, z


def member_subcell_species_histogram(problem, member):
    """Return the x,y,z subcell occupancy histogram used for GA diversity."""
    _, occ, z = member_state_arrays(problem, member)
    nx, ny, nz = (int(value) for value in problem.diversity_subcells_xyz)
    histogram = np.zeros((nx, ny, nz, problem.n_species), dtype=float)
    if len(z) == 0:
        return histogram
    positions = problem.z_to_positions(z)
    # Vacuum must not consume descriptor resolution. Bin all coordinates
    # relative to the measured sample box; atoms entering any vacuum region
    # fall into the corresponding sample-edge bin.
    sample_origin_xyz = np.asarray(problem.vacuum_position_shift, dtype=float)
    sample_lengths_xyz = np.diag(problem.original_cell).astype(float)
    frac = np.clip(
        (positions - sample_origin_xyz[None, :])
        / sample_lengths_xyz[None, :],
        0.0,
        np.nextafter(1.0, 0.0),
    )
    subcells = np.floor(
        frac * np.asarray([nx, ny, nz], dtype=float)[None, :]
    ).astype(int)
    subcells = np.minimum(
        subcells,
        np.asarray([nx - 1, ny - 1, nz - 1], dtype=int)[None, :],
    )
    np.add.at(
        histogram,
        (subcells[:, 0], subcells[:, 1], subcells[:, 2]),
        occ,
    )
    return histogram


def member_diversity_alignment(problem, member_a, member_b):
    """Return subcell L1 distance and the best allowed periodic z shift of B."""
    histogram_a = member_subcell_species_histogram(problem, member_a)
    histogram_b = member_subcell_species_histogram(problem, member_b)
    nz = int(problem.diversity_subcells_xyz[2])
    shifts = range(nz) if bool(problem.periodic_z_parent_alignment) else (0,)
    best_distance = float("inf")
    best_shift = 0
    for shift in shifts:
        shifted_b = np.roll(histogram_b, int(shift), axis=2)
        distance = float(np.sum(np.abs(histogram_a - shifted_b)))
        if distance < best_distance:
            best_distance = distance
            best_shift = int(shift)
    return best_distance, best_shift


def member_diversity_distance(problem, member_a, member_b):
    return float(member_diversity_alignment(problem, member_a, member_b)[0])


def choose_tournament_pair(
    parent_pool,
    parent_a_tournament_size,
    parent_b_tournament_size,
):
    """Choose two distinct parents as the best of two score tournaments.

    This matches the fixed-scaffold GEAT rule: tournament B is sampled after
    excluding the winning parent A. Other non-winning contestants from the
    first tournament remain eligible for the second tournament.
    """
    if len(parent_pool) < 2:
        raise ValueError("Need at least two parents for crossover.")
    parent_a = choose_tournament_parent(
        parent_pool,
        parent_a_tournament_size,
    )
    parent_b = choose_tournament_parent(
        parent_pool,
        parent_b_tournament_size,
        exclude=parent_a,
    )
    return parent_a, parent_b


def live_plot_path(problem, args):
    path_text = str(args.ga_live_plot_file or "").strip()
    if path_text:
        return Path(path_text).expanduser().resolve()
    return problem.output_dir / "genetic_optimization_live_scores_by_generation.png"


def stable_generation_jitter(generation, member_index, jitter):
    seed = (int(generation) * 1000003 + int(member_index) * 9176 + 7) % 2**32
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-float(jitter), float(jitter)))


def update_live_ga_plot(problem, args, generations, current_generation=None):
    if not bool(args.ga_live_plot_enable):
        return
    metric = str(args.ga_live_plot_metric)
    if metric not in ("score", "formation_energy_per_atom", "f_guided_no_entropy", "composition_penalty"):
        raise ValueError("ga_live_plot_metric must be score, formation_energy_per_atom, f_guided_no_entropy, or composition_penalty.")

    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ga plot warning] could not import matplotlib: {exc}")
        return

    kind_style = {
        "random": ("random", "red"),
        "random_immigrant": ("random immigrant", "red"),
        "mutated_child": ("mutation", "purple"),
        "mutation": ("mutation", "purple"),
        "crossover_child": ("crossover", "green"),
        "crossover": ("crossover", "green"),
        "elite_copy": ("elite/imported", "0.25"),
        "imported_parent": ("elite/imported", "0.25"),
        "imported_seed": ("elite/imported", "0.25"),
    }
    groups = {}
    best_by_generation = []
    plot_generations = list(generations)
    if current_generation is not None:
        plot_generations.append(list(current_generation))
    for generation_index, generation in enumerate(plot_generations):
        values = []
        for member in generation:
            if metric not in member:
                continue
            kind = str(member.get("kind", "unknown"))
            label, color = kind_style.get(kind, ("other", "0.65"))
            member_generation = int(member.get("generation", generation_index))
            member_index = int(member.get("member_index", 0))
            x = float(member_generation) + stable_generation_jitter(member_generation, member_index, args.ga_live_plot_jitter)
            y = float(member[metric])
            groups.setdefault(label, {"x": [], "y": [], "color": color})
            groups[label]["x"].append(x)
            groups[label]["y"].append(y)
            values.append(y)
        if values:
            best_by_generation.append((generation_index, min(values)))

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for label in (
        "random immigrant",
        "random",
        "mutation",
        "crossover",
        "elite/imported",
        "other",
    ):
        if label not in groups:
            continue
        group = groups[label]
        ax.scatter(
            group["x"],
            group["y"],
            s=46,
            c=group["color"],
            alpha=(
                max(0.85, float(args.ga_live_plot_alpha))
                if label == "random immigrant"
                else float(args.ga_live_plot_alpha)
            ),
            edgecolors="none",
            label=label,
        )
    if best_by_generation:
        x_best, y_best = zip(*best_by_generation)
        ax.plot(x_best, y_best, color="black", linewidth=1.6, alpha=0.75, label="best per generation")
    ax.set_xlabel("Generation", fontsize=18)
    ax.set_ylabel(metric.replace("_", " "), fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=13)
    ax.set_title("GA Population Scores by Generation", fontsize=18, pad=14)
    live_compute = aggregate_compute_metrics(
        member for generation in plot_generations for member in generation
    )
    subtitle = (
        f"live update   ranking={args.ga_score}   "
        f"energy evaluations={live_compute['model_energy_evaluations']:,}   "
        f"model time={live_compute['model_evaluation_seconds'] / 3600.0:.2f} h"
    )
    ax.text(0.0, -0.14, subtitle, transform=ax.transAxes, fontsize=10, color="0.35")
    path = live_plot_path(problem, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_genetic_algorithm(problem, args, seed):
    ga_wall_started = time.perf_counter()
    population_size = int(args.ga_population_size)
    generations_count = int(args.ga_generations)
    parent_pool_size = max(1, int(args.ga_parent_pool_size))
    counts = allocate_generation_counts(population_size, args)
    if population_size <= 0:
        raise ValueError("--ga-population-size must be positive.")
    if generations_count <= 0:
        raise ValueError("--ga-generations must be positive.")
    if int(args.member_progress_checkpoint_interval_iterations) <= 0:
        raise ValueError(
            "checkpointing.member_progress_interval_iterations must be positive."
        )
    if float(args.member_progress_checkpoint_interval_seconds) < 0.0:
        raise ValueError(
            "checkpointing.member_progress_interval_seconds must be nonnegative."
        )
    if float(args.local_resource_log_interval_seconds) < 0.0:
        raise ValueError(
            "runtime.local_resource_log_interval_seconds must be nonnegative."
        )
    if float(args.worker_health_check_interval_seconds) <= 0.0:
        raise ValueError(
            "runtime.worker_health_check_interval_seconds must be positive."
        )
    for name in (
        "ga_crossover_pos_sigma",
        "ga_crossover_occ_sigma",
        "ga_crossover_species_sigma",
        "ga_superposition_prune_distance_angstrom",
        "ga_superposition_prune_occ_thresh",
        "ga_live_plot_alpha",
        "ga_live_plot_jitter",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if not (0.0 <= float(args.ga_superposition_occ_scale) <= 1.0):
        raise ValueError("ga_superposition_occ_scale must be in [0, 1].")
    if int(args.ga_parent_a_tournament_size) <= 0:
        raise ValueError("ga_parent_a_tournament_size must be positive.")
    execution_backend = str(getattr(args, "ga_execution_backend", "local")).strip().lower()
    if execution_backend not in ("local", "slurm"):
        raise ValueError("genetic.execution_backend must be 'local' or 'slurm'.")
    parallel_workers = max(1, int(args.ga_parallel_workers))
    if parallel_workers > population_size:
        parallel_workers = population_size
    use_slurm = execution_backend == "slurm"
    if use_slurm:
        parallel_workers = 1
        if float(args.slurm_poll_seconds) <= 0.0:
            raise ValueError("slurm.poll_seconds must be positive.")
        if int(args.slurm_nodes) <= 0 or int(args.slurm_cpus_per_task) <= 0:
            raise ValueError("slurm.nodes and slurm.cpus_per_task must be positive.")

    local_parallel = bool(not use_slurm and parallel_workers > 1)
    if local_parallel:
        # Do this before loading a checkpoint: otherwise every historical GA
        # state is needlessly restored onto GPU 0 before being moved back.
        problem.release_calculator()
        move_problem_tensor_state_to_device(problem, "cpu")
        append_running_log(
            args,
            "local_parallel_coordinator_moved_to_cpu; CUDA reserved for GA workers",
        )
        terminal_print(
            args,
            "[ga parallel] coordinator moved to CPU; CUDA is worker-only",
        )

    log_path = ga_log_path(problem, args)
    db_path = structures_db_path(problem)
    running_path = running_log_path_from_args(args)
    checkpoint = restore_ga_checkpoint(problem, args)
    resuming = checkpoint is not None
    if resuming and int(checkpoint["seed"]) != int(seed):
        raise RuntimeError(
            f"Checkpoint seed {checkpoint['seed']} does not match startup seed {seed}."
        )
    ga_start_record = {
        "event": "ga_start",
        "seed": int(seed),
        "population_size": population_size,
        "generations": generations_count,
        "parent_pool_size": parent_pool_size,
        "score": str(args.ga_score),
        "elite_fraction": float(args.ga_elite_fraction),
        "random_fraction": float(args.ga_random_fraction),
        "crossover_fraction": float(args.ga_crossover_fraction),
        "mutation_fraction": float(args.ga_mutation_fraction),
        "elite_count": int(counts["elite"]),
        "random_count": int(counts["random"]),
        "crossover_count": int(counts["crossover"]),
        "mutation_count": int(counts["mutation"]),
        "mutation_pairing": "previous_population_tournament_parent+uniform_current_generation_relaxed_random_immigrant",
        "parent_a_tournament_size": int(args.ga_parent_a_tournament_size),
        "parent_b_tournament_size": int(args.ga_parent_b_tournament_size),
        "random_immigrant_image_prefit_enable": bool(
            args.random_immigrant_image_prefit_enable
        ),
        "random_immigrant_image_prefit_soft_steps": int(
            args.random_immigrant_image_prefit_soft_steps
        ),
        "random_immigrant_image_prefit_final_position_steps": int(
            args.random_immigrant_image_prefit_final_position_steps
        ),
        "random_immigrant_image_prefit_rounding": (
            "none_before_syntropization_single_rounding_after_syntropization"
        ),
        "image_loss_domain": "experimental_roi_only",
        "image_loss_voxel_count": int(problem.image_loss_voxel_count),
        "random_immigrant_fairchem_relaxation": (
            "soft_prefit_then_energy_syntropization_then_fixed_species_final_positions"
        ),
        "parent_selection": "two_independent_best_of_tournament",
        "crossover_parent_alignment": "none",
        "elite_selection": "best_score",
        "crossover_pos_sigma": float(args.ga_crossover_pos_sigma),
        "crossover_occ_sigma": float(args.ga_crossover_occ_sigma),
        "crossover_species_sigma": float(args.ga_crossover_species_sigma),
        "superposition_occ_scale": float(args.ga_superposition_occ_scale),
        "superposition_prune_distance_angstrom": float(args.ga_superposition_prune_distance_angstrom),
        "execution_backend": str(execution_backend),
        "parallel_workers": int(parallel_workers),
        "parallel_gpu_ids": str(args.ga_parallel_gpu_ids),
        "worker_health_check_interval_seconds": float(
            args.worker_health_check_interval_seconds
        ),
        "slurm_partition": str(args.slurm_partition),
        "slurm_cpus_per_task": int(args.slurm_cpus_per_task),
        "slurm_time": str(args.slurm_time),
    }
    if resuming:
        append_running_log(
            args,
            f"ga_resume seed={int(seed)} current_generation={checkpoint['current_generation']} "
            f"completed_generations={len(checkpoint['all_generations'])} "
            f"completed_current_members={len(checkpoint['current_generation_results'])}",
        )
        append_ga_log(
            log_path,
            {
                "event": "ga_resume",
                "seed": int(seed),
                "generation": int(checkpoint["current_generation"]),
                "completed_current_members": len(
                    checkpoint["current_generation_results"]
                ),
            },
        )
        print(
            f"[ga checkpoint] resuming {ga_checkpoint_path(problem)} at "
            f"generation={checkpoint['current_generation']} with "
            f"{len(checkpoint['current_generation_results'])}/"
            f"{population_size} members already complete"
        )
    else:
        reset_ga_log(log_path)
        running_path.parent.mkdir(parents=True, exist_ok=True)
        running_path.write_text(
            f"{timestamp_now()} | ga_start seed={int(seed)}\n",
            encoding="utf-8",
        )
        append_ga_log(log_path, ga_start_record)
    terminal_print(args, f"[ga] writing progress log to {log_path}")
    terminal_print(args, f"[ga] writing every relaxed member to {db_path}")
    if bool(args.ga_live_plot_enable):
        terminal_print(args, f"[ga] updating live score plot at {live_plot_path(problem, args)}")
    if use_slurm:
        terminal_print(args, f"[ga slurm] enabled partition={args.slurm_partition} cpus_per_task={args.slurm_cpus_per_task}")
        append_running_log(
            args,
            f"slurm_enabled partition={args.slurm_partition} cpus_per_task={args.slurm_cpus_per_task} "
            f"time={args.slurm_time} mem={args.slurm_mem}",
        )
    if parallel_workers > 1:
        terminal_print(args, f"[ga parallel] enabled workers={parallel_workers} gpu_ids={parse_parallel_gpu_ids(args.ga_parallel_gpu_ids)}")

    all_generations = (
        list(checkpoint["all_generations"]) if resuming else []
    )
    resume_generation_results = (
        list(checkpoint["current_generation_results"]) if resuming else []
    )
    start_generation = (
        int(checkpoint["current_generation"]) if resuming else 0
    )
    previous_population = list(all_generations[-1]) if all_generations else []
    imported_parents = load_imported_parent_results(problem, args)
    # Imports occupy the first generation only, regardless of how random
    # immigrants are constructed. Thereafter they survive via normal selection.
    seed_generation_from_imports = len(imported_parents) > 0
    completed_members = [
        member for generation_items in all_generations for member in generation_items
    ]
    best = min(completed_members, key=lambda item: item["score"]) if completed_members else None
    member_seed_base = int(seed)
    task_queue = None
    result_queue = None
    workers = []
    if resuming and bool(args.ga_live_plot_enable):
        update_live_ga_plot(
            problem,
            args,
            all_generations,
            resume_generation_results,
        )
    has_remaining_generations = start_generation < generations_count
    if has_remaining_generations and (use_slurm or parallel_workers > 1):
        if not local_parallel:
            problem.release_calculator()
        terminal_print(args, "[ga parallel] released main-process FAIRChem calculator while workers evaluate members")
    if has_remaining_generations and parallel_workers > 1:
        task_queue, result_queue, workers = start_ga_workers(args, parallel_workers)
        problem._active_local_ga_pool = (task_queue, result_queue, workers)

    for generation in range(start_generation, generations_count):
        generation_wall_started = time.perf_counter()
        is_resumed_generation = bool(
            resuming and generation == start_generation
        )
        terminal_print(args, "\n" + "#" * 88)
        terminal_print(args, f"[ga generation {generation + 1}/{generations_count}]")
        terminal_print(args, "#" * 88)
        generation_results = (
            list(resume_generation_results) if is_resumed_generation else []
        )
        completed_member_indices = {
            int(result["member_index"]) for result in generation_results
        }
        if is_resumed_generation and generation_results:
            append_running_log(
                args,
                f"generation_resume generation={generation} "
                f"completed={len(generation_results)}/{population_size}",
            )
            append_ga_log(
                log_path,
                {
                    "event": "generation_resume",
                    "generation": generation,
                    "completed_members": len(generation_results),
                },
            )
        else:
            append_running_log(args, f"generation_start generation={generation} population_size={population_size}")
            append_ga_log(log_path, {"event": "generation_start", "generation": generation})
        write_ga_checkpoint(
            problem,
            args,
            seed,
            all_generations,
            generation,
            generation_results,
            status="running",
        )

        if generation == 0:
            submitted_tasks = 0
            slurm_tasks = []
            member_index = 0
            if seed_generation_from_imports:
                imports_to_use = imported_parents[:population_size]
                append_ga_log(log_path, {
                    "event": "generation0_import_seed_start",
                    "generation": generation,
                    "imported_count": len(imports_to_use),
                    "population_size": population_size,
                })
                for imported in imports_to_use:
                    if member_index in completed_member_indices:
                        member_index += 1
                        continue
                    copied = copy_imported_parent_to_generation(imported, generation, member_index)
                    record_completed_ga_result(
                        problem,
                        args,
                        seed,
                        db_path,
                        log_path,
                        all_generations,
                        generation_results,
                        copied,
                        event="imported_seed_used",
                    )
                    member_index += 1

            while member_index < population_size:
                if member_index in completed_member_indices:
                    member_index += 1
                    continue
                member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                torch.manual_seed(member_seed)
                np.random.seed(member_seed)
                immigrant_kind = "random_immigrant"
                terminal_print(
                    args,
                    f"[ga] generation={generation} member={member_index} "
                    f"kind={immigrant_kind} seed={member_seed}",
                )
                append_ga_log(log_path, {
                    "event": "member_start",
                    "generation": generation,
                    "member_index": member_index,
                    "kind": immigrant_kind,
                    "seed": int(member_seed),
                })
                layout0 = (
                    random_immigrant_image_prefit_site_layout(problem, args)
                    if bool(args.random_immigrant_image_prefit_enable)
                    else problem.initial_site_layout()
                )
                problem.set_site_layout(layout0)
                y0 = make_initial_state(problem, args)
                task = make_evaluation_task(
                    f"g{generation}_m{member_index}",
                    y0,
                    layout0,
                    member_seed,
                    generation,
                    member_index,
                    immigrant_kind,
                )
                if use_slurm:
                    slurm_tasks.append(task)
                    submitted_tasks += 1
                elif parallel_workers > 1:
                    task_queue.put(task)
                    submitted_tasks += 1
                else:
                    result = evaluate_candidate_member(
                        problem,
                        y0,
                        layout0,
                        args,
                        member_seed,
                        generation,
                        member_index,
                        kind=immigrant_kind,
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
                member_index += 1
            if use_slurm and submitted_tasks > 0:
                write_slurm_task_files(problem, generation, slurm_tasks)
                job_id, _ = submit_slurm_generation(problem, args, generation, submitted_tasks)
                collect_slurm_ga_tasks(
                    problem,
                    args,
                    seed,
                    db_path,
                    log_path,
                    all_generations,
                    generation_results,
                    generation,
                    slurm_tasks,
                    job_id,
                )
            elif parallel_workers > 1:
                collect_parallel_ga_tasks(
                    problem,
                    args,
                    seed,
                    db_path,
                    log_path,
                    all_generations,
                    generation_results,
                    result_queue,
                    workers,
                    submitted_tasks,
                )
        else:
            base_parents = previous_population
            parents = sorted(base_parents, key=lambda item: item["score"])
            parent_pool = parents[: min(parent_pool_size, len(parents))]
            elites = parents[: counts["elite"]]
            for member_index, elite in enumerate(elites):
                if member_index in completed_member_indices:
                    continue
                copied = dict(elite)
                copied.pop("slice_comparison_dir", None)
                copied.pop("slice_comparison_written", None)
                copied["generation"] = int(generation)
                copied["member_index"] = int(member_index)
                copied["parent_refs"] = [parent_ref(elite)]
                copied["kind"] = "elite_copy"
                copied["y"] = elite["y"].detach().clone()
                copied["y_rounded"] = elite["y_rounded"].detach().clone()
                copied["assigned_blob_indices"] = list(elite["assigned_blob_indices"])
                copied["history"] = list(elite.get("history", []))
                copied["stage_summaries"] = list(elite.get("stage_summaries", []))
                copied["compute_metrics"] = empty_compute_metrics()
                copied["elapsed_seconds"] = 0.0
                terminal_print(
                    args,
                    f"[ga] generation={generation} member={member_index} kind=elite_copy "
                    f"parent={parent_ref(elite)} score={elite['score']:+.6f}"
                )
                record_completed_ga_result(
                    problem,
                    args,
                    seed,
                    db_path,
                    log_path,
                    all_generations,
                    generation_results,
                    copied,
                    event="elite_copied",
                )

            member_index = int(counts["elite"])
            submitted_tasks = 0
            slurm_tasks = []
            random_stop = min(population_size, member_index + int(counts["random"]))
            while member_index < random_stop:
                if member_index in completed_member_indices:
                    member_index += 1
                    continue
                member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                torch.manual_seed(member_seed)
                np.random.seed(member_seed)
                terminal_print(args, f"[ga] generation={generation} member={member_index} kind=random_immigrant seed={member_seed}")
                append_ga_log(log_path, {
                    "event": "member_start",
                    "generation": generation,
                    "member_index": member_index,
                    "kind": "random_immigrant",
                    "seed": int(member_seed),
                })
                layout0 = (
                    random_immigrant_image_prefit_site_layout(problem, args)
                    if bool(args.random_immigrant_image_prefit_enable)
                    else problem.initial_site_layout()
                )
                problem.set_site_layout(layout0)
                y0 = make_initial_state(problem, args)
                task = make_evaluation_task(
                    f"g{generation}_m{member_index}",
                    y0,
                    layout0,
                    member_seed,
                    generation,
                    member_index,
                    "random_immigrant",
                )
                if use_slurm:
                    slurm_tasks.append(task)
                    submitted_tasks += 1
                elif parallel_workers > 1:
                    task_queue.put(task)
                    submitted_tasks += 1
                else:
                    result = evaluate_candidate_member(
                        problem,
                        y0,
                        layout0,
                        args,
                        member_seed,
                        generation,
                        member_index,
                        kind="random_immigrant",
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
                member_index += 1

            mutation_donors = []
            if counts["mutation"] > 0:
                finish_ga_task_batch(
                    problem, args, seed, db_path, log_path, all_generations,
                    generation_results, generation, slurm_tasks, submitted_tasks,
                    result_queue, workers, phase="random_donors")
                submitted_tasks = 0
                slurm_tasks = []
                mutation_donors = current_mutation_donors(
                    generation_results, generation, int(counts["elite"]), random_stop)

            crossover_stop = random_stop + int(counts["crossover"])
            while member_index < population_size:
                if member_index in completed_member_indices:
                    member_index += 1
                    continue
                member_seed = (member_seed_base + generation * 1000003 + member_index * 9176) % 2**32
                torch.manual_seed(member_seed)
                np.random.seed(member_seed)
                if member_index < crossover_stop:
                    child_kind = "crossover_child"
                    selection = "two_independent_best_of_tournament"
                    parent_a, parent_b = choose_tournament_pair(
                        parent_pool, args.ga_parent_a_tournament_size,
                        args.ga_parent_b_tournament_size)
                else:
                    child_kind = "mutated_child"
                    selection = "tournament_parent+uniform_current_generation_random_immigrant"
                    parent_a = choose_tournament_parent(parent_pool, args.ga_parent_a_tournament_size)
                    parent_b = mutation_donors[int(np.random.randint(len(mutation_donors)))]
                # Tournament selection is intentionally independent of the
                # legacy subcell-diversity metric. Parents are superposed in
                # their stored common cell without a diversity-derived shift.
                parent_b_z_shift = 0
                terminal_print(
                    args,
                    f"[ga] generation={generation} member={member_index} kind={child_kind} "
                    f"parents={parent_ref(parent_a)} + {parent_ref(parent_b)} "
                    f"parent_selection={selection} "
                    f"seed={member_seed}"
                )
                append_ga_log(log_path, {
                    "event": "member_start",
                    "generation": generation,
                    "member_index": member_index,
                    "kind": child_kind,
                    "parent_a": parent_ref(parent_a),
                    "parent_b": parent_ref(parent_b),
                    "parent_selection": selection,
                    "parent_a_tournament_size": int(args.ga_parent_a_tournament_size),
                    "parent_b_tournament_size": (
                        int(args.ga_parent_b_tournament_size) if child_kind == "crossover_child" else None
                    ),
                    "seed": int(member_seed),
                })
                y0, layout0 = crossover_state(
                    problem,
                    parent_a["y"],
                    parent_a["assigned_blob_indices"],
                    parent_b["y"],
                    parent_b["assigned_blob_indices"],
                    args,
                    parent_b_z_shift_subcells=parent_b_z_shift,
                )
                task = make_evaluation_task(
                    f"g{generation}_m{member_index}",
                    y0,
                    layout0,
                    member_seed,
                    generation,
                    member_index,
                    child_kind,
                    parent_refs=[parent_ref(parent_a), parent_ref(parent_b)],
                )
                if use_slurm:
                    slurm_tasks.append(task)
                    submitted_tasks += 1
                elif parallel_workers > 1:
                    task_queue.put(task)
                    submitted_tasks += 1
                else:
                    result = evaluate_candidate_member(
                        problem,
                        y0,
                        layout0,
                        args,
                        member_seed,
                        generation,
                        member_index,
                        parent_refs=[parent_ref(parent_a), parent_ref(parent_b)],
                        kind=child_kind,
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
                member_index += 1
            finish_ga_task_batch(
                problem, args, seed, db_path, log_path, all_generations,
                generation_results, generation, slurm_tasks, submitted_tasks,
                result_queue, workers,
                phase="offspring" if counts["mutation"] > 0 else "")

        member_indices = [int(item["member_index"]) for item in generation_results]
        if len(generation_results) != population_size or set(member_indices) != set(
            range(population_size)
        ):
            missing = sorted(set(range(population_size)) - set(member_indices))
            duplicates = sorted(
                index for index in set(member_indices) if member_indices.count(index) > 1
            )
            raise RuntimeError(
                f"Generation {generation} is incomplete after evaluation: "
                f"results={len(generation_results)}/{population_size}, "
                f"missing={missing}, duplicates={duplicates}."
            )
        generation_results.sort(key=lambda item: item["score"])
        previous_population = generation_results
        all_generations.append(generation_results)
        resume_generation_results = []
        generation_best = generation_results[0]
        if best is None or generation_best["score"] < best["score"]:
            best = generation_best
        generation_compute = aggregate_compute_metrics(generation_results)
        generation_wall_seconds = time.perf_counter() - generation_wall_started
        append_ga_log(log_path, {
            "event": "generation_done",
            "generation": generation,
            "best_score": float(generation_best["score"]),
            "best_member_index": int(generation_best["member_index"]),
            "global_best_score": float(best["score"]),
            "global_best_generation": int(best["generation"]),
            "global_best_member_index": int(best["member_index"]),
            "energy_evaluations": int(generation_compute["model_energy_evaluations"]),
            "energy_evaluation_seconds": float(generation_compute["model_evaluation_seconds"]),
            "objective_evaluations": int(generation_compute["objective_evaluations"]),
            "accepted_optimizer_iterations": int(generation_compute["accepted_optimizer_iterations"]),
            "evaluated_member_wall_seconds_sum": float(generation_compute["member_wall_seconds"]),
            "generation_wall_seconds": float(generation_wall_seconds),
        })
        terminal_print(
            args,
            f"[ga generation {generation} done] best_score={generation_best['score']:+.6f} "
            f"global_best={best['score']:+.6f} "
            f"(generation={best['generation']} member={best['member_index']})"
        )
        append_running_log(
            args,
            f"generation_done generation={generation} best_score={generation_best['score']:+.6f} "
            f"global_best={best['score']:+.6f} "
            f"energy_evaluations={generation_compute['model_energy_evaluations']} "
            f"energy_eval_s={generation_compute['model_evaluation_seconds']:.2f} "
            f"generation_wall_s={generation_wall_seconds:.2f}",
        )
        write_ga_checkpoint(
            problem,
            args,
            seed,
            all_generations,
            generation + 1,
            [],
            status="running",
        )
        cleanup_completed_member_checkpoints(problem, args, generation)

    problem.ga_wall_seconds_this_process = time.perf_counter() - ga_wall_started
    total_compute = aggregate_compute_metrics(
        member for generation_items in all_generations for member in generation_items
    )
    terminal_print(
        args,
        f"[ga compute] energy_evaluations={total_compute['model_energy_evaluations']} "
        f"energy_evaluation_time={total_compute['model_evaluation_seconds']:.2f} s "
        f"member_wall_time_sum={total_compute['member_wall_seconds']:.2f} s "
        f"ga_wall_time_this_process={problem.ga_wall_seconds_this_process:.2f} s",
    )
    append_ga_log(log_path, ga_member_log_record(best, "ga_done_best"))
    write_ga_summary(problem, args, log_path, all_generations, best)
    write_ga_checkpoint(
        problem,
        args,
        seed,
        all_generations,
        generations_count,
        [],
        status="complete",
    )
    problem.history = [] if bool(args.disable_trajectory_history) else list(best.get("history", []))
    problem.stage_summaries = list(best.get("stage_summaries", []))
    problem.exact_composition_projection_summary = copy.deepcopy(
        best.get("exact_composition_projection")
    )
    problem.set_site_layout(best["assigned_blob_indices"])
    if workers:
        stop_ga_workers(args, task_queue, workers, result_queue=result_queue)
        problem._active_local_ga_pool = None
    if local_parallel and torch.cuda.is_available():
        # Worker processes are gone, so GPU 0 is safe to reuse for the single
        # final diagnostic/output pass.
        move_problem_tensor_state_to_device(problem, "cuda")
        best["y"] = best["y"].to(problem.device)
        best["y_rounded"] = best["y_rounded"].to(problem.device)
    return best["y"]


def main():
    cli_args = parse_cli_args()
    if cli_args.slurm_worker_task:
        run_slurm_worker(cli_args.config, cli_args.slurm_worker_task, cli_args.slurm_worker_result, cli_args.slurm_worker_resource_log)
        return
    args = get_settings(cli_args.config)
    args.slurm_worker_mode = False
    resume_seed = checkpoint_seed_for_args(args) if bool(args.ga_enable) else None
    args.ga_resuming_from_checkpoint = resume_seed is not None
    params_path = write_run_parameters(args)
    configure_cpu_threads(args)
    seed = (
        int(resume_seed)
        if resume_seed is not None
        else (int(time.time_ns() % 2**32) if args.seed is None else int(args.seed))
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bool(getattr(args, "exit_after_initial_diagnostics", False)):
        diagnostic_output_dir = Path(args.output_dir).expanduser().resolve()
        prepare_experimental_volume(
            args,
            diagnostic_output_dir,
            write_diagnostics=True,
        )
        print(f"[config] {args.config_path}")
        print(f"[run parameters] {params_path}")
        print(f"[initial diagnostics] {diagnostic_output_dir / 'initial_diagnostics'}")
        print("[initial diagnostics] exit_after_initial_diagnostics=true; optimization not started")
        return
    main_resource_stop, main_resource_thread = start_resource_monitor(
        Path(args.output_dir).expanduser().resolve()
        / "resources"
        / "local_main.tsv",
        getattr(args, "local_resource_log_interval_seconds", 0.0),
        "local_main",
    )
    problem = WSe2VolumeProblem(args)

    print(f"[config] {args.config_path}")
    print(f"[run parameters] {params_path}")
    print(f"[seed] {seed}")
    print(f"[device] {problem.device}")
    print(f"[experimental TIFF] {problem.blob_path}")
    if bool(getattr(args, "initial_diagnostics_enable", True)):
        print(f"[initial diagnostics] {problem.output_dir / 'initial_diagnostics'}")
    print(
        f"[experimental crop] origin_zyx={problem.experimental['crop_origin'].tolist()} "
        f"source_voxel_zyx_A={problem.experimental['source_voxel_size_A'].tolist()}"
    )
    print(
        f"[periodic padded cell] sample_lengths_xyz_A="
        f"{problem.experimental['sample_cell_lengths_A'].tolist()} "
        f"padded_lengths_xyz_A={np.diag(problem.cell).tolist()} "
        f"requested_vacuum_each_side_xyz_A="
        f"{problem.requested_vacuum_padding.tolist()} "
        f"realized_vacuum_each_side_xyz_A={problem.vacuum_padding.tolist()} "
        f"full_shape_zyx={problem.target_shape_zyx} "
        f"sample_shape_zyx={problem.experimental['sample_shape_zyx']} "
        f"sample_roi_zyx=[{problem.experimental['sample_roi_start_zyx'].tolist()}, "
        f"{problem.experimental['sample_roi_stop_zyx'].tolist()}) "
        f"target_voxel_zyx_A={problem.target_voxel_size_A.tolist()}"
    )
    print(
        f"[W/Se feasible region] mode={problem.experimental['threshold_mode']} "
        f"threshold={problem.experimental['threshold_value']:.6g} "
        f"volume_fraction={problem.experimental['heavy_allowed_fraction']:.4f} "
        f"level_set_penalty={problem.boundary_penalty:.3e}"
    )
    print(
        f"[sample-box confinement] lower_xyz_A={problem.sample_box_lower_A.tolist()} "
        f"upper_xyz_A={problem.sample_box_upper_A.tolist()} "
        f"penalty={problem.sample_box_confinement_penalty:.3e} "
        f"sharpness={problem.sample_box_confinement_sharpness:.3e} 1/A "
        f"margin={problem.sample_box_confinement_margin_A:.3f} A "
        "normalization=total_W_plus_Se_occupancy"
    )
    print(f"[pbc] {problem.pbc.astype(bool).tolist()}")
    print(f"[species] {problem.species} atomic_numbers={problem.atomic_numbers}")
    print(
        "[species constraints] one effective site class allows W/Se/null; "
        "all real occupancy uses the feasible-region level set"
    )
    print(
        f"[composition] target={problem.composition_target.tolist()} "
        f"quadratic_penalty={problem.composition_penalty:.3e} "
        "exact_projection_after_rounding="
        f"{bool(args.exact_composition_enforcement_enable)}"
    )
    print(
        f"[Se-vacancy bias] enabled={problem.se_vacancy_bias_enable} "
        f"alpha={problem.se_vacancy_bias_alpha_eV:.6f} eV/vacancy "
        f"reference_atoms={problem.se_vacancy_reference_atom_count:.3f} "
        f"reference_Se={problem.se_vacancy_reference_se_count:.6f} "
        "term=-alpha*(reference_Se-sum_Se_occupancy)/reference_atoms"
    )
    print(
        f"[effective atoms] candidate_density={problem.heavy_site_density_per_angstrom3:.6f} sites/A^3 "
        f"candidate_sites={problem.initial_heavy_region_sites} "
        f"feasible_sample_volume={problem.feasible_volume_angstrom3:.3f} A^3 "
        f"sample_volume={problem.sample_volume_angstrom3:.3f} A^3 "
        f"padded_cell_volume={problem.cell_volume_angstrom3:.3f} A^3 "
        f"initial_sites={problem.initial_n_sites} "
        f"cylinder_lateral_radius={problem.initial_cylinder_lateral_radius_A:.3f} A "
        f"cylinder_z_radius={problem.initial_cylinder_z_radius_A:.3f} A "
        f"jitter=+/-{problem.initial_jitter_A:.3f} A"
    )
    print(
        f"[phase-table double-Gaussian {problem.image_target_transform}-domain image] "
        f"penalty={problem.image_penalty:.3e} "
        f"Huber_delta={problem.image_huber_delta:.3f} "
        f"total_amplitudes={dict(zip(problem.species, problem.gaussian_amplitudes.tolist()))} "
        f"component_amplitudes={dict(zip(problem.species, problem.gaussian_component_amplitudes.tolist()))} "
        f"fitted_component_sigma_xy_A={dict(zip(problem.species, problem.gaussian_fitted_sigma_xy_A.tolist()))} "
        f"effective_component_sigma_xy_A={dict(zip(problem.species, problem.gaussian_sigma_xy_A.tolist()))} "
        f"additional_lateral_blur_A={problem.gaussian_additional_lateral_blur_A:.6g} "
        f"reference={problem.gaussian_relative_intensity_reference_species} "
        f"scale_source={problem.gaussian_intensity_scale_source} "
        f"rendered_single_atom_peaks={dict(zip(problem.species, problem.gaussian_rendered_single_atom_peak_amplitudes.tolist()))} "
        f"depth_response_rms_A={problem.gaussian_shared_sigma_z_A:.6g}"
    )
    print("[depth model] analytic Gaussian response; no envelope; ellipse penalty removed")
    print(f"[phase-table double-Gaussian CSV] {problem.gaussian_element_fit_csv}")
    for ref in problem.element_reference_metadata:
        print(f"[element reference] {ref['symbol']} {ref['material_id']} E/atom={ref['computed_energy_per_atom_eV']:+.6f} eV")
    print(f"[compound reference] composition={problem.compound_reference_composition.tolist()} Eform/atom={problem.compound_reference_formation_energy_per_atom:+.6f} eV")
    print(
        "[assignment] one permanent W/Se/null site class; "
        f"each site has {problem.n_species} real species occupancies plus "
        "implicit null occupancy"
    )
    print("[constraints] for each ghost: x_alpha >= 0 and sum_alpha x_alpha <= 1")
    print(f"[entropy] Tsallis q={problem.tsallis_q:.6g} using real species plus implicit null species")
    print(
        f"[final PBFGS] enabled={bool(args.final_pbfgs_enable)} "
        f"fixed_occupancies_positions_only=true beta=0 "
        f"tol={float(args.final_pbfgs_tol):.3e} "
        f"maxit={int(args.final_pbfgs_maxit)} "
        f"cost_plateau_tol={float(args.final_pbfgs_cost_plateau_tol):.3e} "
        f"cost_plateau_patience={int(args.final_pbfgs_cost_plateau_patience)} "
        f"cost_plateau_min_iterations={int(args.final_pbfgs_cost_plateau_min_iterations)}"
    )
    print(
        "[objective] Eform + beta*Tsallis + soft WSe2 composition + Se-vacancy bias "
        "+ periodic W/Se level-set boundary "
        "+ occupancy-normalized sample-box confinement "
        f"+ image_penalty*masked_Huber({problem.image_target_transform}(Gaussian field), "
        f"{problem.image_target_transform}(normalized linear experiment)); image mask voxels="
        f"{problem.image_loss_voxel_count} "
        "guided GA score uses the same no-entropy objective"
    )
    print(
        f"[initialization] mode={args.init_species_mode} total_occ={args.init_occ_total:.3f} "
        f"occ_noise={args.init_occ_noise:.3e} dirichlet_alpha={args.init_species_dirichlet_alpha:.3f} "
        "W/Se/null proposal positions sampled inside the shifted feasible sample region"
    )
    print(
        f"[random-immigrant image prefit] enabled={bool(args.random_immigrant_image_prefit_enable)} "
        f"transform={problem.image_target_transform} "
        f"soft_steps={int(args.random_immigrant_image_prefit_soft_steps)} "
        f"final_position_steps={int(args.random_immigrant_image_prefit_final_position_steps)} "
        f"initial_occupancy_mode={args.random_immigrant_image_prefit_initial_occupancy_mode} "
        f"composition_penalty_enable={bool(args.random_immigrant_image_prefit_composition_penalty_enable)} "
        f"compact_nulls_before_fairchem={bool(args.random_immigrant_image_prefit_compact_null_sites_before_fairchem)} "
        "handoff=soft_prefit_to_energy_syntropization "
        "rounding=once_after_syntropization "
        "loss_domain=experimental_ROI_only "
        "fairchem_relaxation=energy_syntropization_then_fixed_species_positions "
        f"lateral_blur_A={problem.gaussian_additional_lateral_blur_A:.6g} "
        f"depth_response_rms_A={problem.gaussian_shared_sigma_z_A:.6g}"
    )
    print("[relaxation shocks] disabled; shocks occur only before EAT relaxation in GA crossover")
    print(
        f"[crossover shock] pos_sigma={args.ga_crossover_pos_sigma:.3e} "
        f"occ_sigma={args.ga_crossover_occ_sigma:.3e} "
        f"species_sigma={args.ga_crossover_species_sigma:.3e}"
    )
    print(
        f"[tournament selection] parent_a_size={args.ga_parent_a_tournament_size} "
        f"parent_b_size={args.ga_parent_b_tournament_size} "
        "parent_a=best_score parent_b=best_score parent_b_excludes_parent_a=true "
        "diversity_selection=false parent_alignment=none"
    )
    print("[elite selection] best_score")
    print(
        f"[superposition crossover] occ_scale={args.ga_superposition_occ_scale:.3f} "
        f"prune_distance={args.ga_superposition_prune_distance_angstrom:.3f} Angstrom "
        f"active_thresh={args.ga_superposition_prune_occ_thresh:.3e}"
    )
    print(
        "[crossover sites] inherited from parent A + parent B only; "
        "close-pair pruning may delete sites; null-site padding/reseeding is disabled"
    )
    print(f"[history] trajectory_history_disabled={args.disable_trajectory_history} history_every={args.history_every}")
    print(
        f"[ga] enabled={args.ga_enable} population={args.ga_population_size} generations={args.ga_generations} "
        f"generation0=imported_seeds+remaining_random_immigrants "
        f"later_generations=elite+random_immigrant+crossover+mutation "
        f"random_immigrant_constructor={'soft_image_prefit_then_energy_syntropization_then_fixed_species_positions' if bool(args.random_immigrant_image_prefit_enable) else 'random_then_fairchem'} "
        f"fractions=(elite={args.ga_elite_fraction}, random={args.ga_random_fraction}, crossover={args.ga_crossover_fraction}, mutation={args.ga_mutation_fraction}) "
        f"parent_pool={args.ga_parent_pool_size} score={args.ga_score}"
    )
    print(f"[ga execution] backend={args.ga_execution_backend}")
    print(f"[ga parallel] workers={args.ga_parallel_workers} gpu_ids={args.ga_parallel_gpu_ids}")
    print(
        f"[checkpointing] member_progress_enable={bool(args.member_progress_checkpoint_enable)} "
        f"iterations={int(args.member_progress_checkpoint_interval_iterations)} "
        f"seconds={float(args.member_progress_checkpoint_interval_seconds):.1f} "
        f"retain_completed_members={bool(args.retain_completed_member_checkpoints)} "
        f"local_resource_log_seconds={float(args.local_resource_log_interval_seconds):.1f} "
        f"worker_health_check_seconds={float(args.worker_health_check_interval_seconds):.1f}"
    )
    print(
        f"[ga slurm] partition={args.slurm_partition} cpus_per_task={args.slurm_cpus_per_task} "
        f"time={args.slurm_time} mem={args.slurm_mem} venv={args.slurm_venv}"
    )
    print(f"[stop] entropy/site target={args.entropy_stop_avg:.6e} beta_max_stages={args.beta_max_stages} also_require_binary_stop={args.also_require_binary_stop}")

    try:
        if args.ga_enable:
            y_cont = run_genetic_algorithm(problem, args, seed)
        else:
            y0 = make_initial_state(problem, args)
            y_cont = problem.run(y0)
        y_round = problem.rounded_state(y_cont)
        problem.write_outputs(y_cont, y_round, seed)
    except BaseException:
        emergency_stop_active_ga_pool(
            problem,
            args,
            "unhandled coordinator exception",
        )
        raise
    finally:
        stop_resource_monitor(main_resource_stop, main_resource_thread)


if __name__ == "__main__":
    main()
